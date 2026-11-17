"""Perception theo frame: detector (YOLOX), tracker (ByteTrack-DMA-LTC + ReID), pose (YOLOv8-pose), hand (RTMW)."""
from .common import *  # noqa: F401,F403
from .prof import span as _span


def _low_light_cfg(cfg=None):
    """Đọc cấu hình low_light_boost (hỗ trợ dict hoặc bool)."""
    base = dict(enabled=True, luma_thr=75.0, clip_limit=2.0, tile_grid=8)
    raw = MODELS.get('detector', {}).get('low_light_boost') if cfg is None else cfg
    if isinstance(raw, dict):
        base.update(raw)
    elif isinstance(raw, bool):
        base['enabled'] = raw
    elif raw is None:
        base['enabled'] = False
    return base


def apply_low_light_boost(img, cfg=None):
    """Tăng sáng nhẹ bằng CLAHE trên kênh L (LAB) khi độ sáng trung bình < ngưỡng; trả ảnh gốc nếu đủ sáng hoặc tắt."""
    if img is None or getattr(img, 'ndim', 0) != 3 or img.shape[2] < 3:
        return img
    c = _low_light_cfg(cfg)
    if not c.get('enabled', True):
        return img
    thr = float(c.get('luma_thr', 75.0))
    h, w = img.shape[:2]
    if max(h, w) > 320:
        sample = img[::4, ::4]
        mean_l = float(cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
        if mean_l >= thr:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        if float(lab[:, :, 0].mean()) >= thr:
            return img
    tg = c.get('tile_grid', 8)
    grid = (max(1, int(tg[0])), max(1, int(tg[1]))) if isinstance(tg, (tuple, list)) else (max(1, int(tg)), max(1, int(tg)))
    clahe = cv2.createCLAHE(clipLimit=float(c.get('clip_limit', 2.0)), tileGridSize=grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class Detector:
    def warmup(self):
        for _ in range(2):
            self.detect(np.zeros((int(CFG['width'] * 9 / 16), CFG['width'], 3), np.uint8))


def detector_preprocess(frame, size):
    # BGR 0-255, letterbox pad 114, không chuẩn hoá = ValTransform(legacy=False) của Megvii YOLOX (như lúc train YOLOX-s)
    ih, iw = size
    h, w = frame.shape[:2]
    ratio = min(ih / h, iw / w)
    image = np.full((ih, iw, 3), 114, np.float32)
    image[:int(h * ratio), :int(w * ratio)] = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32), ratio


@register('detector', 'yolox')
class YoloxDetector(Detector):
    def __init__(self, device=GPU_A, conf=0.01, nms=0.7, exp=None, depth=0.33, width=0.5, num_classes=1,
                 test_size=(608, 1088), ckpt='best_ckpt.pth', fp16=True, fuse=True, gpu_preproc=True,
                 low_light_boost=None, **_):
        ensure_tracker_repo()
        from yolox.exp import Exp, get_exp
        from yolox.utils import postprocess
        self.postprocess, self.device, self.conf, self.nms = postprocess, device, conf, nms
        self.low_light_boost = low_light_boost
        if exp:      # file exp trong repo tracker (vd YOLOX-nano MOT17)
            self.exp = get_exp(str(settings.TRACKER_REPO / exp), None)
        else:        # checkpoint tự train: dựng YOLOX từ depth / width / num_classes (YOLOX-s = 0.33 / 0.5)
            self.exp = Exp()
            self.exp.depth, self.exp.width, self.exp.num_classes = depth, width, num_classes
        self.exp.test_size = tuple(test_size)
        self.model = self.exp.get_model().to(device).eval()
        ckpt_path = resolve_weight(ckpt, required=True, allow_dir=True)
        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar
        # checkpoint lưu bằng numpy 2.x ghi tên 'numpy._core...', numpy 1.x ghi 'numpy.core...': đăng ký cả hai
        np_globals = [(np_scalar, 'numpy.core.multiarray.scalar'), (np_scalar, 'numpy._core.multiarray.scalar'),
                      np_scalar, np.dtype]
        np_globals += [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32')]
        with torch.serialization.safe_globals(np_globals):
            state = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
        state = {k.replace('module.', ''): v for k, v in state.get('model', state).items()}
        self.model.load_state_dict(state, strict=True)
        if fuse:                      # gộp Conv + BN (như make --fuse)
            from yolox.utils import fuse_model
            self.model = fuse_model(self.model)
        self.half = bool(fp16 and 'cuda' in str(device))
        if self.half:                 # như make --fp16
            self.model = self.model.half()
        torch.backends.cudnn.benchmark = True    # như track_video_bench.py (kích thước ảnh vào cố định)
        # resize + pad + đổi kiểu trên GPU: CPU chỉ chép ảnh uint8 (CPU Kaggle chậm, lại phải chia cho 6 stage khác)
        self.gpu_preproc = bool(gpu_preproc and 'cuda' in str(device))
        name = exp or f'depth={self.exp.depth} width={self.exp.width} classes={self.exp.num_classes}'
        print(f'    YOLOX {name} | ckpt={ckpt_path} | test_size={self.exp.test_size} | fp16={self.half} fuse={fuse} '
              f'conf={conf} nms={nms}')

    def _preprocess_gpu(self, frame):
        """= detector_preprocess (letterbox, pad 114, BGR 0-255) nhưng chạy trên GPU."""
        ih, iw = self.exp.test_size
        h, w = frame.shape[:2]
        r = min(ih / h, iw / w)
        nh, nw = int(h * r), int(w * r)
        with torch.no_grad():
            img = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device, non_blocking=True)
            img = img.permute(2, 0, 1).unsqueeze(0).float()
            img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)
            x = torch.full((1, 3, ih, iw), 114.0, device=self.device)
            x[:, :, :nh, :nw] = img
        return x, r

    def detect(self, frame):
        frame = apply_low_light_boost(frame, getattr(self, 'low_light_boost', None))
        fh, fw = frame.shape[:2]
        with _span('detect.preprocess'):
            if self.gpu_preproc:
                x, ratio = self._preprocess_gpu(frame)
            else:
                inp, ratio = detector_preprocess(frame, self.exp.test_size)
                x = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        with _span('detect.model_nms'), torch.no_grad():     # GPU bất đồng bộ: phần chờ GPU dồn vào detect.to_cpu_post
            dets = self.postprocess(self.model(x.half() if self.half else x), self.exp.num_classes, self.conf, self.nms)[0]
        out = []
        with _span('detect.to_cpu_post'):
            if dets is not None:
                for det in dets.float().cpu().numpy():
                    x1, y1, x2, y2, s1, s2, cls = det[:7]
                    if int(cls) != 0:
                        continue
                    x1, x2 = np.clip([x1 / ratio, x2 / ratio], 0, fw - 1)
                    y1, y2 = np.clip([y1 / ratio, y2 / ratio], 0, fh - 1)
                    out.append([float(x1), float(y1), float(x2), float(y2), float(s1 * s2)])
        return out


@register('detector', 'ultralytics')
class UltralyticsDetector(Detector):
    """YOLOv8 / YOLO11 / YOLO12 / RT-DETR (tên model chính thức tự tải, hoặc .pt tự train)."""

    def __init__(self, weights='yolo11n.pt', device=GPU_A, conf=0.1, nms=0.5, imgsz=640, classes=(0,), half=True,
                 low_light_boost=None, **_):
        from ultralytics import YOLO, RTDETR
        w = resolve_weight(weights)
        self.model = (RTDETR if 'rtdetr' in Path(str(w)).name.lower() else YOLO)(str(w))
        self.kw = dict(conf=conf, iou=nms, imgsz=imgsz, classes=list(classes) if classes is not None else None,
                       device=device, half=bool(half and 'cuda' in str(device)), verbose=False)
        self.low_light_boost = low_light_boost
        print(f'    Ultralytics {w} | imgsz={imgsz} classes={classes}')

    def detect(self, frame):
        frame = apply_low_light_boost(frame, getattr(self, 'low_light_boost', None))
        boxes = self.model.predict(frame, **self.kw)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy, conf = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
        return [[float(a), float(b), float(c), float(d), float(s)] for (a, b, c, d), s in zip(xyxy, conf)]


def _unpack_track_item(item):
    """Chuẩn hoá phần tử tracked thành (tid, box, meta_dict)."""
    if isinstance(item, dict):
        return int(item['id']), item['box'], item
    if len(item) >= 3:
        return int(item[0]), item[1], item[2] or {}
    return int(item[0]), item[1], {}


def _attach_scores(
    tracked,
    dets,
    min_iou=0.30,
    keep_predicted_frames=8,
    stable_min_iou=0.15,
    stable_frames=10,
    score_decay=0.9,
    track_state=None,
):
    """Gắn điểm detector cho track; dùng min_iou thấp hơn cho track ổn định và giữ track dự đoán tối đa keep_predicted_frames."""
    out = []
    keep_lim = max(0, int(keep_predicted_frames or 0))
    decay = max(0.0, min(1.0, float(score_decay)))
    stable_thr = int(stable_frames)
    seen_ids = set()

    for item in tracked:
        tid, tb, meta = _unpack_track_item(item)
        seen_ids.add(tid)
        st = track_state.get(tid) if track_state is not None else None
        seen_hits = max(int(st.get('hits', 0)) if st is not None else 0, int(meta.get('hits') or 0))
        thr = (
            min(float(min_iou), float(stable_min_iou))
            if (stable_min_iou is not None and seen_hits >= stable_thr)
            else float(min_iou)
        )

        best, best_iou = None, 0.0
        if not meta.get('is_lost'):
            for d in (dets or ()):
                iou = box_iou(tb, d[:4])
                if best is None or iou > best_iou:
                    best, best_iou = d, iou

        box_list = [float(v) for v in tb]
        if best is not None and best_iou >= thr:
            score = round(float(best[4]), 4)
            if track_state is not None:
                track_state[tid] = dict(
                    hits=seen_hits + 1,
                    score=score,
                    base_score=score,
                    pred_frames=0,
                    absent=0,
                    box=box_list,
                )
            out.append(dict(id=tid, box=box_list, score=score, predicted=False, predicted_frames=0, kp=None, kc=None))
            continue

        prev_pred = int(st.get('pred_frames', 0)) if st is not None else 0
        pred_frames = int(meta['lost_frames']) if meta.get('lost_frames') is not None else (prev_pred + 1)
        if keep_lim > 0 and 1 <= pred_frames <= keep_lim:
            if st is not None and st.get('base_score') is not None:
                base_score = float(st['base_score'])
            elif meta.get('score') is not None:
                base_score = float(meta['score'])
            elif st is not None and st.get('score') is not None:
                base_score = float(st['score'])
            else:
                base_score = 0.5
            score = round(max(1e-4, base_score * (decay ** pred_frames)), 4)
            if track_state is not None:
                track_state[tid] = dict(
                    hits=seen_hits,
                    score=score,
                    base_score=base_score,
                    pred_frames=pred_frames,
                    absent=0,
                    box=box_list,
                )
            out.append(dict(
                id=tid,
                box=box_list,
                score=score,
                predicted=True,
                predicted_frames=pred_frames,
                kp=None,
                kc=None,
            ))
        elif track_state is not None and st is not None:
            st['pred_frames'] = pred_frames
            st['absent'] = 0

    if track_state is not None:
        stale = []
        for tid, st in track_state.items():
            if tid not in seen_ids:
                st['absent'] = int(st.get('absent', 0)) + 1
                if st['absent'] > 120:
                    stale.append(tid)
        for tid in stale:
            track_state.pop(tid, None)
    return out


def extract_track_embedding(track):
    """Lấy vector embedding ReID (ưu tiên smooth_feat, sau đó curr_feat/reid_feat/embedding) và chuẩn hoá L2."""
    if track is None:
        return None
    for attr in ('smooth_feat', 'curr_feat', 'reid_feat', 'embedding', 'feat'):
        val = track.get(attr) if isinstance(track, dict) else getattr(track, attr, None)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            return (arr / norm).tolist()
    return None


def torso_hsv_histogram(img, box):
    """Fallback đặc trưng ngoại hình khi tracker không lộ embedding OSNet: histogram màu HSV vùng thân (chuẩn hoá L2)."""
    if img is None or getattr(img, 'ndim', 0) != 3 or img.shape[2] < 3 or box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    w, h = x2 - x1, y2 - y1
    if w <= 0.0 or h <= 0.0:
        return None
    ih, iw = img.shape[:2]
    tx1 = max(0, int(round(x1 + 0.20 * w)))
    tx2 = min(iw, int(round(x2 - 0.20 * w)))
    ty1 = max(0, int(round(y1 + 0.18 * h)))
    ty2 = min(ih, int(round(y1 + 0.60 * h)))
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        tx1, tx2 = max(0, int(round(x1))), min(iw, int(round(x2)))
        ty1, ty2 = max(0, int(round(y1))), min(ih, int(round(y2)))
    if tx2 - tx1 < 1 or ty2 - ty1 < 1:
        return None
    hsv = cv2.cvtColor(img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 6, 4], [0, 180, 0, 256, 0, 256]).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-6:
        return None
    return hist / norm


def cosine_similarity(a, b):
    """Cosine similarity giữa hai vector đặc trưng 1D (trả 0.0 nếu rỗng hoặc khác kích thước)."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or va.shape != vb.shape:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-6 or not np.isfinite(denom):
        return 0.0
    val = float(np.dot(va, vb) / denom)
    return val if math.isfinite(val) else 0.0


def _ensure_reid_weights(folder: Path) -> Path:
    weights = folder / 'osnet_x1_0_imagenet.pth'
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        import gdown
        gdown.download(id='1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY', output=str(weights), quiet=False)
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        raise RuntimeError('Tải trọng số OSNet ReID thất bại.')
    return weights


def _single_thread_gbm(tracker):
    """LightGBM (DMA) predict mặc định dùng OpenMP trên mọi core mỗi lần gọi. Batch chỉ vài chục dòng nên song song hoá
    vô ích, còn khi nhiều thread stage cùng tranh 4 vCPU (Kaggle) thì OpenMP quá tải thread: đo 0.07 ms -> ~110 ms / lần.
    Ép num_threads=1 (không sửa repo tracker bên ngoài)."""
    booster = getattr(getattr(getattr(tracker, 'dma', None), 'model', None), 'booster', None)
    if booster is None or getattr(booster, '_ef_single_thread', False):
        return
    orig = booster.predict

    def predict(*a, **k):
        k.setdefault('num_threads', 1)
        return orig(*a, **k)
    booster.predict, booster._ef_single_thread = predict, True


def _wrap_span(obj, attr, name):
    """Bọc obj.attr bằng đo thời gian (không sửa repo tracker bên ngoài). Bọc 1 lần cho mỗi đối tượng."""
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, '_ef_prof', False):
        return

    def wrapped(*a, **k):
        with _span(name):
            return fn(*a, **k)
    wrapped._ef_prof = True
    setattr(obj, attr, wrapped)


def _wrap_span_sync(obj, attr, name):
    """Như _wrap_span nhưng có torch.cuda.synchronize() sau lời gọi khi EXHIBITFLOW_PROF_SYNC=1: span gồm cả thời gian GPU tính
    (mặc định tắt vì sync làm đổi nhịp chạy bất đồng bộ; chỉ bật khi muốn tách "phóng kernel" khỏi "GPU tính")."""
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, '_ef_prof', False):
        return
    sync = os.environ.get('EXHIBITFLOW_PROF_SYNC') == '1' and torch.cuda.is_available()

    def wrapped(*a, **k):
        with _span(name):
            out = fn(*a, **k)
            if sync:
                torch.cuda.synchronize()
            return out
    wrapped._ef_prof = True
    setattr(obj, attr, wrapped)


def _profile_tracker(tr):
    """Chia nhỏ track.update. Tổng: reid_extract = reid_crop (cắt ảnh) + reid_model (phóng kernel OSNet, hoặc cả GPU tính khi
    EXHIBITFLOW_PROF_SYNC=1) + phần còn lại (resize từng crop, chép lên GPU, chờ kết quả về CPU). ltc_refine = ltc_history + ltc_model + còn lại."""
    rx = getattr(tr, 'reid_extractor', None)
    _wrap_span(rx, 'extract', 'track.reid_extract')
    _wrap_span_sync(getattr(rx, 'model', None), 'forward', 'track.reid_model')
    try:                                                # crop_tlbrs là hàm module-level, extract() gọi qua tên toàn cục của module reid
        import yolox.tracker.reid as _reid
        _wrap_span(_reid, 'crop_tlbrs', 'track.reid_crop')
    except Exception:                                   # noqa: BLE001
        pass
    _wrap_span(getattr(tr, 'dma', None), 'fuse', 'track.dma_fuse')
    mp = getattr(tr, 'motion_predictor', None)
    _wrap_span(mp, 'refine', 'track.ltc_refine')
    _wrap_span(mp, '_history_tensor', 'track.ltc_history')
    _wrap_span_sync(getattr(mp, 'model', None), 'forward', 'track.ltc_model')
    _wrap_span(tr, '_fuse_reid', 'track.fuse_reid_total')          # gồm embedding_distance + dma_fuse


@register('tracker', 'bytetrack_dma_ltc')
class ByteTrackDMALTC:
    """Tham số mặc định = `make track-video-full-torch` (tools/track_video_bench.py), tinh chỉnh giảm ID switch và giữ track dự đoán."""

    def __init__(self, device=GPU_A, with_reid=True, track_thresh=0.5, track_buffer=75, match_thresh=0.9,
                 reid_model='osnet_x0_25', reid_ckpt='osnet_x0_25.pth', reid_weight=0.5, reid_thresh=0.6,
                 ltc_ckpt='ltc_motion_mot17.pth', ltc_min_history=10, dma_ckpt='dma_gbm_mot17.gbm',
                 min_box_area=100, max_aspect=1.6, keep_predicted_frames=8,
                 attach_min_iou=0.30, attach_stable_min_iou=0.15, attach_stable_frames=10,
                 predicted_score_decay=0.9, extra_args=None, **_):
        ensure_tracker_repo()
        if with_reid:
            # reid_ckpt=None + osnet_x1_0 -> tải osnet_x1_0_imagenet (cấu hình notebook gốc)
            reid_path = (_ensure_reid_weights(settings.CACHE) if reid_ckpt is None and reid_model == 'osnet_x1_0' else
                         resolve_weight(reid_ckpt) if reid_ckpt else None)
            if reid_ckpt and not Path(str(reid_path)).exists():
                raise RuntimeError(
                    f'Không tìm thấy trọng số ReID "{reid_ckpt}" trong WEIGHT_ROOTS. File này là OSNet đã train ReID (MSMT17), '
                    'repo trên GitHub không có -> chép external/ByteTrack-DMA-LTC-Motion-Tracker/pretrained/'
                    f'{reid_ckpt} vào thư mục weights/ (hoặc dataset Kaggle cùng best_ckpt.pth). '
                    "Hoặc đổi MODELS['tracker']: reid_model='osnet_x1_0', reid_ckpt=None (OSNet ImageNet, tự tải — cấu hình cũ) "
                    'hoặc with_reid=False.')
        self.args = SimpleNamespace(
            track_thresh=track_thresh, track_buffer=track_buffer, match_thresh=match_thresh, mot20=False,
            ltc_motion_ckpt=str(resolve_weight(ltc_ckpt, required=True)), ltc_device=device,
            ltc_history_len=16, ltc_input_dim=12, ltc_min_history=ltc_min_history, ltc_hidden_size=128,
            ltc_num_layers=2, ltc_covariance_scale=1., ltc_max_abs_residual=256.,
            with_reid=with_reid, reid_device=device, reid_weight=reid_weight, reid_thresh=reid_thresh, reid_alpha=.9,
            reid_backend='deep', reid_model=reid_model,
            reid_model_path=str(reid_path) if with_reid and reid_path else '',
            ml='gbm', ml_weights=str(resolve_weight(dma_ckpt, required=True)), dma_device=device,
            **(extra_args or {}))
        self.min_box_area, self.max_aspect = min_box_area, max_aspect
        self.keep_predicted_frames = int(keep_predicted_frames)
        self.attach_min_iou = float(attach_min_iou)
        self.attach_stable_min_iou = float(attach_stable_min_iou)
        self.attach_stable_frames = int(attach_stable_frames)
        self.predicted_score_decay = float(predicted_score_decay)
        self._track_state = {}
        self.tracker = None
        print(f"    ByteTrack-DMA-LTC | ReID {reid_model if with_reid else 'off'} ({self.args.reid_model_path or '-'}) | "
              f"DMA {self.args.ml_weights} | LTC min_history={ltc_min_history} | "
              f"thresh={track_thresh} match={match_thresh} buffer={track_buffer} "
              f"reid_w={reid_weight} reid_thr={reid_thresh} keep_pred={self.keep_predicted_frames}")

    def reset(self, fps):
        from yolox.tracker.byte_tracker import BYTETracker
        from yolox.tracker.basetrack import BaseTrack
        BaseTrack._count = 0
        self.tracker = BYTETracker(self.args, frame_rate=int(round(fps)))
        self._track_state = {}
        _single_thread_gbm(self.tracker)
        _profile_tracker(self.tracker)
        for name, ok in (('DMA GBM', getattr(self.tracker, 'dma', None) is not None),
                         ('LTC motion', getattr(getattr(self.tracker, 'motion_predictor', None), 'enabled', False)),
                         ('ReID', getattr(self.tracker, 'reid_extractor', None) is not None or not self.args.with_reid)):
            if not ok:
                print(f'    CẢNH BÁO: {name} không được nạp trong BYTETracker')

    def _valid_track_box(self, t):
        w, h = float(t.tlwh[2]), float(t.tlwh[3])
        return w * h > self.min_box_area and not (self.max_aspect and w / max(h, 1e-6) > self.max_aspect)

    def update(self, dets, frame, native=None, S=1.0):
        # như make: tracker (ReID cắt ảnh người) chạy trên ảnh gốc -> đổi box sang toạ độ ảnh gốc rồi đổi lại
        img, s = (native, S) if native is not None else (frame, 1.0)
        fh, fw = img.shape[:2]
        arr = np.array(dets, np.float32).reshape(-1, 5) if dets else np.empty((0, 5), np.float32)
        arr[:, :4] *= s
        with _span('track.update'):                       # ReID + LTC + DMA + gán ID
            online = self.tracker.update(arr, (fh, fw), (fh, fw), frame=img)
        keep_pred = int(getattr(self, 'keep_predicted_frames', MODELS['tracker'].get('keep_predicted_frames', 8)))
        track_state = getattr(self, '_track_state', None)
        if track_state is None:
            track_state = self._track_state = {}

        keep, feats, online_ids = [], {}, set()
        for t in online:
            if not self._valid_track_box(t):
                continue
            tid = int(t.track_id)
            online_ids.add(tid)
            meta = dict(
                is_lost=False,
                score=getattr(t, 'score', None),
                hits=getattr(t, 'tracklet_len', None),
            )
            keep.append((tid, [float(v) / s for v in t.tlbr], meta))
            feat = extract_track_embedding(t)
            if feat is not None:
                feats[tid] = feat

        if keep_pred > 0:
            cur_fid = getattr(self.tracker, 'frame_id', None)
            for t in getattr(self.tracker, 'lost_stracks', ()) or ():
                tid = int(t.track_id)
                if tid in online_ids or not getattr(t, 'is_activated', True) or not self._valid_track_box(t):
                    continue
                miss = getattr(t, 'missing_count', None)
                if miss is not None and int(miss) > 0:
                    lost_n = int(miss)
                elif cur_fid is not None and hasattr(t, 'end_frame'):
                    lost_n = max(1, int(cur_fid) - int(t.end_frame))
                else:
                    lost_n = None
                meta = dict(
                    is_lost=True,
                    lost_frames=lost_n,
                    score=getattr(t, 'score', None),
                    hits=getattr(t, 'tracklet_len', None),
                )
                keep.append((tid, [float(v) / s for v in t.tlbr], meta))
                feat = extract_track_embedding(t)
                if feat is not None:
                    feats[tid] = feat

        out = _attach_scores(
            keep,
            dets,
            min_iou=getattr(self, 'attach_min_iou', MODELS['tracker'].get('attach_min_iou', 0.30)),
            keep_predicted_frames=keep_pred,
            stable_min_iou=getattr(self, 'attach_stable_min_iou', MODELS['tracker'].get('attach_stable_min_iou', 0.15)),
            stable_frames=getattr(self, 'attach_stable_frames', MODELS['tracker'].get('attach_stable_frames', 10)),
            score_decay=getattr(self, 'predicted_score_decay', MODELS['tracker'].get('predicted_score_decay', 0.9)),
            track_state=track_state,
        )
        if feats:
            for r in out:
                if r['id'] in feats:
                    r['reid_feat'] = feats[r['id']]
        return out


class _RemoteProc:
    """Chạy 1 model (tracker / hand / pose) trong TIẾN TRÌNH RIÊNG (exhibitflow/tracker_worker.py). Các stage track / hand là phần nặng mã
    Python nhỏ (nhiều kernel / lời gọi nhỏ phát từ CPU): trong cùng tiến trình chúng tranh GIL với các stage còn lại. Tiến trình riêng
    có GIL riêng nên không bị các stage kia làm chậm (và ngược lại). Đổi lại: thêm 1 CUDA context (~0.5 GB VRAM) và vài ms chép ảnh
    qua shared memory mỗi frame."""

    def __init__(self, kind, spec, startup_timeout=300, threads=1):
        import atexit
        import secrets
        import shutil
        import subprocess
        import tempfile
        import threading
        from multiprocessing.connection import Listener
        self._lock, self._shm, self._closed, self._kind = threading.Lock(), None, False, kind
        self._dir = tempfile.mkdtemp(prefix='ef_proc_')
        key = secrets.token_bytes(16)
        addr = os.path.join(self._dir, 'w.sock')
        listener = Listener(addr, family='AF_UNIX', authkey=key)
        env = dict(os.environ, EF_TRK_KEY=key.hex(),
                   PYTHONPATH=os.pathsep.join([str(settings.PROJECT_DIR), os.environ.get('PYTHONPATH', '')]))
        self.proc = subprocess.Popen([sys.executable, '-m', 'exhibitflow.tracker_worker', addr], env=env)
        box = {}
        t = threading.Thread(target=lambda: box.update(conn=listener.accept()), daemon=True)
        t.start()
        t0 = time.perf_counter()
        while t.is_alive() and self.proc.poll() is None and time.perf_counter() - t0 < 60:
            t.join(0.2)
        listener.close()
        if 'conn' not in box:
            self._kill()
            shutil.rmtree(self._dir, ignore_errors=True)
            raise RuntimeError(f'Tiến trình {kind} không khởi động được (mã {self.proc.poll()})')
        self.conn = box['conn']
        atexit.register(self.close)
        self._call(('init', dict(kind=kind, spec=spec, weight_roots=[str(q) for q in settings.WEIGHT_ROOTS],
                                 work=str(settings.WORK), threads=threads, cfg=dict(settings.CFG))), timeout=startup_timeout)
        print(f'    {kind} chạy ở tiến trình riêng (pid {self.proc.pid})')

    def _kill(self):
        try:
            self.proc.kill()
            self.proc.wait(5)
        except Exception:
            pass

    def _call(self, msg, timeout=120):
        with self._lock:
            if self._closed:
                raise RuntimeError('Tiến trình con đã đóng')
            self.conn.send(msg)
            t0 = time.perf_counter()
            _t_rpc = t0
            while not self.conn.poll(1.0):
                if self.proc.poll() is not None:
                    raise RuntimeError(f'Tiến trình con đã chết (mã {self.proc.returncode})')
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f'Tiến trình con không trả lời sau {timeout}s')
            status, val = self.conn.recv()
        if msg[0] not in ('prof', 'shm', 'init', 'quit'):
            from .prof import add
            add(f'{self._kind}.rpc_roundtrip', time.perf_counter() - _t_rpc)   # gửi + đợi worker + nhận (gồm thời gian worker tính)
        if status == 'err':
            raise RuntimeError('Lỗi trong tiến trình con:\n' + val)
        return val

    def _put_image(self, img):
        """Chép ảnh vào vùng nhớ chung (cấp lại khi ảnh lớn hơn); trả shape."""
        from multiprocessing import shared_memory
        _t_cp = time.perf_counter()
        img = np.ascontiguousarray(img)
        if self._shm is None or self._shm.size < img.nbytes:
            old = self._shm
            self._shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
            self._call(('shm', self._shm.name))
            if old is not None:
                old.close()
                old.unlink()
        np.ndarray(img.shape, img.dtype, buffer=self._shm.buf)[...] = img
        from .prof import add
        add(f'{self._kind}.shm_copy', time.perf_counter() - _t_cp)
        return img.shape

    def reset_prof(self):
        try:
            self._call(('prof',), timeout=10)
        except Exception:
            pass

    def collect_prof(self):
        """Lấy bộ đếm thời gian của tiến trình con về tiến trình chính (cuối video)."""
        from .prof import merge
        try:
            merge(self._call(('prof',), timeout=10))
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        try:
            self._call(('quit',), timeout=10)
        except Exception:
            pass
        with self._lock:
            self._closed = True
        try:
            self.conn.close()
        except Exception:
            pass
        self._kill()
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@register('tracker', 'bytetrack_dma_ltc_proc')
class RemoteTracker(_RemoteProc):
    """ByteTrack-DMA-LTC ở tiến trình riêng. Giao diện giống ByteTrackDMALTC (reset / update)."""

    def __init__(self, startup_timeout=300, threads=1, **spec):
        super().__init__('tracker', dict(spec, backend='bytetrack_dma_ltc'), startup_timeout, threads)

    def reset(self, fps):
        self._call(('reset', float(fps)))

    def update(self, dets, frame, native=None, S=1.0):
        img, s = (native, S) if native is not None else (frame, 1.0)
        shape = self._put_image(img)
        return self._call(('upd', shape, [list(map(float, d)) for d in dets] if dets else [], float(s), native is not None))


@register('hand', 'rtmw_proc')
class RemoteHand(_RemoteProc):
    """RTMW hand ở tiến trình riêng. Giao diện giống RTMWHand (gọi như hàm: hand(native, recs, S))."""
    enabled = True

    def __init__(self, startup_timeout=300, threads=1, **spec):
        super().__init__('hand', dict(spec, backend='rtmw'), startup_timeout, threads)

    def __call__(self, native, recs, S):
        idx = [i for i, r in enumerate(recs) if r.get('kp') is not None]   # chỉ người đã có khung xương mới có bàn tay
        if not idx:
            return
        small = [dict(box=recs[i]['box'], kp=1, pose_crop=recs[i].get('pose_crop'), pose_box=recs[i].get('pose_box')) for i in idx]
        shape = self._put_image(native)
        for i, res in zip(idx, self._call(('hand', shape, small, float(S)))):
            if res:
                recs[i].update(res)


@register('pose', 'ultralytics_pose_proc')
class RemotePose(_RemoteProc):
    """YOLOv8-pose ở tiến trình riêng (thoát GIL: pose cắt ảnh / ghép keypoint bằng Python nên tranh GIL với read / detect / sink).
    Giao diện giống UltralyticsPose (gọi như hàm: pose(native, recs, S)). Chỉ gửi box của từng người, nhận lại kp / kc / pose_crop / pose_box."""

    def __init__(self, startup_timeout=300, threads=4, **spec):
        super().__init__('pose', dict(spec, backend='ultralytics_pose'), startup_timeout, threads)
        self._smoother = PoseSmoother()

    def reset_prof(self):
        super().reset_prof()
        self._smoother.reset()

    def __call__(self, native, recs, S):
        if not recs:
            return
        shape = self._put_image(native)
        for rec, res in zip(recs, self._call(('pose', shape, [list(map(float, r['box'])) for r in recs], float(S)))):
            if res:
                rec.update(res)
        self._smoother.update(recs)


class _UltralyticsTracker:
    TYPE = 'bytetrack'

    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6, track_buffer=30,
                 match_thresh=0.8, fuse_score=True, gmc_method='sparseOptFlow', proximity_thresh=0.5,
                 appearance_thresh=0.25, **_):
        self.args = SimpleNamespace(tracker_type=self.TYPE, track_high_thresh=track_high_thresh,
                                    track_low_thresh=track_low_thresh, new_track_thresh=new_track_thresh,
                                    track_buffer=track_buffer, match_thresh=match_thresh, fuse_score=fuse_score,
                                    gmc_method=gmc_method, proximity_thresh=proximity_thresh,
                                    appearance_thresh=appearance_thresh, with_reid=False, model='auto')
        self.tracker = None

    def reset(self, fps):
        if self.TYPE == 'bytetrack':
            from ultralytics.trackers.byte_tracker import BYTETracker as T
        else:
            from ultralytics.trackers.bot_sort import BOTSORT as T
        self.tracker = T(self.args, frame_rate=int(round(fps)))

    def update(self, dets, frame, **_):
        from ultralytics.engine.results import Boxes
        data = np.array([[*d[:5], 0] for d in dets], np.float32).reshape(-1, 6)
        out = self.tracker.update(Boxes(data, frame.shape[:2]), frame)
        return _attach_scores([(int(r[4]), [float(v) for v in r[:4]]) for r in out], dets)


@register('tracker', 'ul_bytetrack')
class UltralyticsByteTrack(_UltralyticsTracker):
    TYPE = 'bytetrack'


@register('tracker', 'ul_botsort')
class UltralyticsBoTSORT(_UltralyticsTracker):
    TYPE = 'botsort'


@register('tracker', 'iou')
class IoUTracker:
    """Baseline: ghép Hungarian theo IoU, giữ track mất tối đa max_age frame."""

    def __init__(self, iou_thr=0.3, max_age=30, min_score=0.3, keep_predicted_frames=0,
                 attach_min_iou=0.30, attach_stable_min_iou=0.15, attach_stable_frames=10,
                 predicted_score_decay=0.9, **_):
        self.iou_thr, self.max_age, self.min_score = iou_thr, max_age, min_score
        self.keep_predicted_frames = int(keep_predicted_frames)
        self.attach_min_iou = float(attach_min_iou)
        self.attach_stable_min_iou = float(attach_stable_min_iou)
        self.attach_stable_frames = int(attach_stable_frames)
        self.predicted_score_decay = float(predicted_score_decay)
        self._track_state = {}

    def reset(self, fps):
        self.tracks, self.next_id, self._track_state = {}, 1, {}

    def update(self, dets, frame, **_):
        from scipy.optimize import linear_sum_assignment
        dets = [d for d in dets if d[4] >= self.min_score]
        ids = list(self.tracks)
        M = iou_matrix([self.tracks[t]['box'] for t in ids], [d[:4] for d in dets])
        matched_t, matched_d, out = set(), set(), []
        if M.size:
            for r, c in zip(*linear_sum_assignment(-M)):
                if M[r, c] >= self.iou_thr:
                    matched_t.add(ids[r]), matched_d.add(c)
                    st = self.tracks[ids[r]]
                    st.update(box=dets[c][:4], age=0, score=float(dets[c][4]), hits=int(st.get('hits', 0)) + 1)
                    out.append((ids[r], dets[c][:4], dict(is_lost=False, score=st['score'], hits=st['hits'])))
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks[self.next_id] = dict(box=d[:4], age=0, score=float(d[4]), hits=1)
                out.append((self.next_id, d[:4], dict(is_lost=False, score=float(d[4]), hits=1)))
                self.next_id += 1
        for t in ids:
            if t not in matched_t:
                st = self.tracks[t]
                st['age'] += 1
                if st['age'] > self.max_age:
                    del self.tracks[t]
                elif self.keep_predicted_frames > 0 and st['age'] <= self.keep_predicted_frames:
                    out.append((t, st['box'], dict(is_lost=True, lost_frames=st['age'], score=st.get('score'), hits=st.get('hits', 0))))
        return _attach_scores(
            out,
            dets,
            min_iou=self.attach_min_iou,
            keep_predicted_frames=self.keep_predicted_frames,
            stable_min_iou=self.attach_stable_min_iou,
            stable_frames=self.attach_stable_frames,
            score_decay=self.predicted_score_decay,
            track_state=self._track_state,
        )


HAND_WB = {'left': slice(91, 112), 'right': slice(112, 133)}
ARM_KP = (7, 8, 9, 10)
TORSO_KP = (5, 6, 11, 12)
WRIST_KP = (('left', 9), ('right', 10))


class OneEuroFilter:
    """Bộ lọc One-Euro cho toạ độ keypoint: mượt khi đứng yên, ít trễ khi chuyển động nhanh."""

    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        t = float(t)
        if self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return self._x_prev.copy()
        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev.copy()
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat.copy()


def interpolate_wrist_keypoints(seq, conf_thr=None, max_gap=None):
    """Nội suy tuyến tính keypoint cổ tay cho khoảng mất <= max_gap frame trên chuỗi record theo frame."""
    if not CFG.get('pose_smooth', True):
        return seq
    thr = float(CFG['pose_conf_thr'] if conf_thr is None else conf_thr)
    gap_lim = int(CFG.get('pose_wrist_interp_max_gap', 8) if max_gap is None else max_gap)
    if gap_lim <= 0:
        return seq

    items = list(seq.items()) if isinstance(seq, dict) else [
        (it[0], it[1]) if isinstance(it, tuple) and len(it) == 2 else (idx, it)
        for idx, it in enumerate(seq)
    ]
    by_track = defaultdict(list)
    for fidx, rec in items:
        if rec is None:
            continue
        tid = rec.get('id', 0)
        by_track[int(tid)].append((int(fidx), rec))

    for tid, track_seq in by_track.items():
        track_seq.sort(key=lambda pair: pair[0])
        for side, widx in WRIST_KP:
            valid = [
                (pos, fidx, np.asarray(rec['kp'][widx], dtype=np.float64))
                for pos, (fidx, rec) in enumerate(track_seq)
                if rec.get('kp') is not None and rec.get('kc') is not None and rec['kc'][widx] >= thr
            ]
            for (pos_a, fa, pa), (pos_b, fb, pb) in zip(valid, valid[1:]):
                missing = fb - fa - 1
                if missing <= 0 or missing > gap_lim:
                    continue
                span = float(fb - fa)
                for pos in range(pos_a + 1, pos_b):
                    f_cur, rec = track_seq[pos]
                    if rec.get('kp') is None or rec.get('kc') is None or rec['kc'][widx] >= thr:
                        continue
                    alpha = (f_cur - fa) / span
                    interp = (1.0 - alpha) * pa + alpha * pb
                    kp = [list(map(float, pt)) for pt in rec['kp']]
                    kp[widx] = [float(interp[0]), float(interp[1])]
                    rec['kp'] = kp
                    rec.setdefault('wrist_interp', {})[side] = True
    return seq


class PoseSmoother:
    """Làm mượt One-Euro theo track_id (chỉ khi kc >= pose_conf_thr) và bù cổ tay mất <= 8 frame."""

    def __init__(self, min_cutoff=None, beta=None, d_cutoff=None, max_gap=None):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_gap = max_gap
        self._filters = {}      # (tid, k) -> OneEuroFilter
        self._last_k_frame = {} # (tid, k) -> int
        self._wrist_hist = {}   # (tid, side) -> deque([(frame_idx, np.ndarray)], maxlen=2)
        self._track_seen = {}   # tid -> last frame_idx
        self._step = -1

    def reset(self):
        self._filters.clear()
        self._last_k_frame.clear()
        self._wrist_hist.clear()
        self._track_seen.clear()
        self._step = -1

    def _prune(self, fidx, keep_frames):
        stale = [tid for tid, last_f in self._track_seen.items() if fidx - last_f > keep_frames]
        if not stale:
            return
        stale_set = set(stale)
        for tid in stale:
            self._track_seen.pop(tid, None)
        for key in [k for k in self._filters if k[0] in stale_set]:
            self._filters.pop(key, None)
            self._last_k_frame.pop(key, None)
        for key in [k for k in self._wrist_hist if k[0] in stale_set]:
            self._wrist_hist.pop(key, None)

    def update(self, recs, frame_idx=None, fps=25.0):
        if not CFG.get('pose_smooth', True) or not recs:
            return recs
        if frame_idx is None:
            self._step += 1
            fidx = self._step
        else:
            fidx = int(frame_idx)
        fps_val = max(float(fps), 1e-3)
        t = fidx / fps_val
        thr = float(CFG['pose_conf_thr'])
        min_c = float(CFG.get('pose_smooth_min_cutoff', 1.5) if self.min_cutoff is None else self.min_cutoff)
        beta = float(CFG.get('pose_smooth_beta', 0.02) if self.beta is None else self.beta)
        d_c = float(CFG.get('pose_smooth_d_cutoff', 1.0) if self.d_cutoff is None else self.d_cutoff)
        max_gap = int(CFG.get('pose_wrist_interp_max_gap', 8) if self.max_gap is None else self.max_gap)
        reset_gap = max(max_gap * 3, int(round(fps_val)))

        for rec in recs:
            tid = rec.get('id')
            if tid is None or rec.get('_pose_smoothed') or rec.get('kp') is None or rec.get('kc') is None:
                continue
            tid = int(tid)
            self._track_seen[tid] = fidx
            kp = [list(map(float, pt)) for pt in rec['kp']]
            kc = [float(c) for c in rec['kc']]

            for k in range(len(kp)):
                if k >= len(kc) or kc[k] < thr:
                    continue
                fk = (tid, k)
                flt = self._filters.get(fk)
                if flt is None:
                    flt = self._filters[fk] = OneEuroFilter(min_c, beta, d_c)
                else:
                    flt.min_cutoff, flt.beta, flt.d_cutoff = min_c, beta, d_c
                    if fidx - self._last_k_frame.get(fk, fidx) > reset_gap:
                        flt.reset()
                sm = flt(kp[k], t)
                kp[k] = [float(sm[0]), float(sm[1])]
                self._last_k_frame[fk] = fidx

            if max_gap > 0:
                box = rec.get('box')
                box_h = max(float(box[3] - box[1]), 1.0) if box is not None else 100.0
                for side, widx in WRIST_KP:
                    if widx >= len(kp) or widx >= len(kc):
                        continue
                    wk = (tid, side)
                    if kc[widx] >= thr:
                        hist = self._wrist_hist.get(wk)
                        if hist is None:
                            hist = self._wrist_hist[wk] = deque(maxlen=2)
                        hist.append((fidx, np.asarray(kp[widx], dtype=np.float64)))
                    else:
                        hist = self._wrist_hist.get(wk)
                        if not hist:
                            continue
                        f1, p1 = hist[-1]
                        gap = fidx - f1
                        if 1 <= gap <= max_gap:
                            est = p1
                            if len(hist) >= 2:
                                f0, p0 = hist[0]
                                if f1 > f0 and (f1 - f0) <= max_gap:
                                    delta = ((p1 - p0) / float(f1 - f0)) * float(gap)
                                    norm = float(np.linalg.norm(delta))
                                    max_shift = 0.5 * box_h
                                    if norm > max_shift and norm > 1e-6:
                                        delta = delta * (max_shift / norm)
                                    est = p1 + delta
                            kp[widx] = [float(est[0]), float(est[1])]
                            rec.setdefault('wrist_interp', {})[side] = True

            rec['kp'] = kp
            rec['_pose_smoothed'] = True

        self._prune(fidx, max(300, max_gap * 10))
        return recs


def _crop_box(rec, S, nw, nh):
    x1, y1, x2, y2 = [v * S for v in rec['box']]
    bw, bh = x2 - x1, y2 - y1
    pad_frac = float(CFG['pose_crop_pad_frac'])
    if rec.get('predicted') and CFG.get('pose_predicted_expand', True):
        pad_frac += float(CFG['pose_crop_pad_frac'])
    dx, dy = bw * pad_frac, bh * pad_frac * 0.5
    return (x1, y1, x2, y2), (max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(nw, int(x2 + dx)), min(nh, int(y2 + dy)))


def arm_leaves_crop(kp, kc, crop, nw):
    cx1, _, cx2, _ = crop
    edge = CFG['pose_wide_edge_frac'] * (cx2 - cx1)
    thr = 0.5 * CFG['pose_conf_thr']
    for k in ARM_KP:
        if kc[k] < thr:
            continue
        x = kp[k][0]
        if (cx1 > 0 and x <= edge) or (cx2 < nw and x >= (cx2 - cx1) - edge):
            return True
    return False


def _wide_crop(rec, S, nw, nh):
    x1, y1, x2, y2 = [v * S for v in rec['box']]
    bh = y2 - y1
    dx, dd = CFG['pose_wide_pad_x'] * bh, CFG['pose_wide_pad_down'] * bh
    dy = bh * CFG['pose_crop_pad_frac'] * 0.5
    return max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(nw, int(x2 + dx)), min(nh, int(y2 + dd))


class _PoseBoxes:
    """Giống phần dùng của ultralytics Boxes: .xyxy (tensor CPU) và len()."""
    def __init__(self, xyxy):
        self.xyxy = xyxy

    def __len__(self):
        return len(self.xyxy)


@register('pose', 'ultralytics_pose')
class UltralyticsPose:
    def __init__(self, weights='yolov8n-pose.pt', device=GPU_A, batch=True, wide_retry=None, gpu_preproc=False, imgsz=640, **_):
        from ultralytics import YOLO
        self.model = YOLO(str(resolve_weight(weights)))
        self.device, self.batch = device, batch
        self.wide = CFG['pose_wide_retry'] if wide_retry is None else wide_retry
        self.imgsz = int(imgsz)
        self._smoother = PoseSmoother()
        # gpu_preproc: letterbox + resize + chuẩn hoá + NMS chạy trên GPU thay vì ở CPU của Ultralytics (cv2.resize từng crop). Chiếm ~44% thời gian pose
        self.gpu_pp = bool(gpu_preproc and 'cuda' in str(device))
        if self.gpu_pp:
            try:                                        # Ultralytics mới chuyển NMS sang utils.nms; bản cũ ở utils.ops
                from ultralytics.utils.nms import non_max_suppression as _nms
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression as _nms
            self._nms = _nms
            self.model.fuse()                           # giống Predictor của Ultralytics (gộp conv + BN)
            self._net = self.model.model.to(device).eval()
            self._kpt_shape = tuple(getattr(self._net, 'kpt_shape', (17, 3)))
            self._nc = len(getattr(self._net, 'names', {0: 'person'}))
            if getattr(self._net, 'end2end', False):    # model NMS-free (end2end) có đầu ra khác: dùng đường CPU của Ultralytics
                print('    pose: model end2end, không hỗ trợ GPU preproc -> dùng đường mặc định')
                self.gpu_pp = False
            print(f'    pose: tiền xử lý + NMS trên GPU (imgsz={self.imgsz})')

    def _predict_gpu(self, crops):
        """Tương đương self.model(crops): letterbox giữa ảnh (pad 114) + resize bilinear + /255 + RGB, forward, NMS, đổi toạ độ về crop."""
        from types import SimpleNamespace
        import torch.nn.functional as F
        nms = self._nms
        dev, sz = torch.device(self.device), self.imgsz
        with _span('pose.gpu_preprocess'):
            canvas = torch.full((len(crops), 3, sz, sz), 114.0 / 255.0, device=dev)
            meta = []
            for i, c in enumerate(crops):
                h, w = c.shape[:2]
                r = min(sz / h, sz / w)
                nw_, nh_ = int(round(w * r)), int(round(h * r))
                top, left = int(round((sz - nh_) / 2 - 0.1)), int(round((sz - nw_) / 2 - 0.1))
                t = torch.from_numpy(np.ascontiguousarray(c)).to(dev, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
                t = F.interpolate(t, size=(nh_, nw_), mode='bilinear', align_corners=False)
                canvas[i, :, top:top + nh_, left:left + nw_] = t[0].flip(0) / 255.0        # BGR -> RGB
                meta.append((r, left, top, w, h))
        with _span('pose.gpu_forward_nms'), torch.inference_mode():                        # NMS có đồng bộ -> gồm thời gian GPU tính
            preds = nms(self._net(canvas), 0.25, 0.7, nc=self._nc, max_det=300)
        out = []
        with _span('pose.gpu_to_cpu'):
            for p, (r, left, top, w, h) in zip(preds, meta):
                p = p.clone()
                p[:, [0, 2]] = ((p[:, [0, 2]] - left) / r).clamp_(0, w)
                p[:, [1, 3]] = ((p[:, [1, 3]] - top) / r).clamp_(0, h)
                k = p[:, 6:].view(len(p), *self._kpt_shape)
                xy = k[..., :2].clone()
                xy[..., 0] = ((xy[..., 0] - left) / r).clamp_(0, w)
                xy[..., 1] = ((xy[..., 1] - top) / r).clamp_(0, h)
                conf = k[..., 2] if self._kpt_shape[-1] == 3 else None
                out.append(SimpleNamespace(boxes=_PoseBoxes(p[:, :4].cpu()),
                                           keypoints=SimpleNamespace(xy=xy.cpu(), conf=None if conf is None else conf.cpu())))
        return out

    def _predict(self, crops):
        if self.gpu_pp:
            with _span('pose.predict'):
                return self._predict_gpu(crops)
        with _span('pose.predict'):                      # tiền xử lý Ultralytics (CPU) + suy luận + NMS; .cpu() ở sau mới chờ GPU xong
            res = self.model(crops, verbose=False, device=self.device)
        from .prof import add                            # Ultralytics tự đo từng ảnh: speed = {preprocess, inference, postprocess} (ms)
        for r in res:
            for k, v in (getattr(r, 'speed', None) or {}).items():
                add(f'pose.ultra_{k}', float(v) / 1000.0)
        return res

    def warmup(self):
        for _ in range(2):
            self._predict([np.zeros((256, 128, 3), np.uint8)] * 2)

    def reset_prof(self):
        self._smoother.reset()

    def _wide_retry(self, native, rec, kp_native, kc, S):
        nh, nw = native.shape[:2]
        wx1, wy1, wx2, wy2 = _wide_crop(rec, S, nw, nh)
        res = self._predict([native[wy1:wy2, wx1:wx2]])[0]
        if res.keypoints is None or len(res.boxes) == 0:
            return None
        ref = [k for k in TORSO_KP if kc[k] >= CFG['pose_conf_thr']]
        if len(ref) < 2:
            return None
        bh = (rec['box'][3] - rec['box'][1]) * S
        kps = res.keypoints.xy.cpu().numpy() + np.array([wx1, wy1])
        kcs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kps.shape[:2])
        best, best_d = None, None
        for j in range(len(kps)):
            common = [k for k in ref if kcs[j][k] >= CFG['pose_conf_thr']]
            if len(common) < 2:
                continue
            dist = float(np.mean([np.linalg.norm(kps[j][k] - kp_native[k]) for k in common]))
            if best_d is None or dist < best_d:
                best, best_d = j, dist
        if best is None or best_d > 0.15 * bh:
            return None
        pb = res.boxes.xyxy.cpu().numpy()[best] + np.array([wx1, wy1, wx1, wy1])
        return kps[best], kcs[best], [wx1, wy1, wx2, wy2], pb.tolist()

    def __call__(self, native, recs, S):
        with _span('pose.call'):
            self._run(native, recs, S)

    def _run(self, native, recs, S):
        nh, nw = native.shape[:2]
        jobs = []
        for rec in recs:
            (x1, y1, x2, y2), (cx1, cy1, cx2, cy2) = _crop_box(rec, S, nw, nh)
            if (cx2 - cx1) >= CFG['pose_min_crop_px'] and (cy2 - cy1) >= CFG['pose_min_crop_px']:
                jobs.append((rec, (x1, y1, x2, y2), (cx1, cy1, cx2, cy2)))
        if not jobs:
            return
        crops = [native[c[1]:c[3], c[0]:c[2]] for _, _, c in jobs]
        results = self._predict(crops) if self.batch else [self._predict([c])[0] for c in crops]
        for (rec, (x1, y1, x2, y2), (cx1, cy1, cx2, cy2)), res in zip(jobs, results):
            if res.keypoints is None or len(res.boxes) == 0:
                continue
            tb = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
            best_iou, best_j, second_iou = 0.0, None, 0.0
            for j, pb in enumerate(res.boxes.xyxy.cpu().numpy()):
                iou = box_iou(tb, pb.tolist())
                if iou > best_iou:
                    second_iou, best_iou, best_j = best_iou, iou, j
                elif iou > second_iou:
                    second_iou = iou
            if best_j is None or best_iou < CFG['pose_conf_thr'] or (best_iou - second_iou) < CFG['pose_margin_thr']:
                continue
            kp = res.keypoints.xy[best_j].cpu().numpy()
            kc = res.keypoints.conf[best_j].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(kp))
            kp_native = kp + np.array([cx1, cy1])
            if self.wide and arm_leaves_crop(kp, kc, (cx1, cy1, cx2, cy2), nw):
                with _span('pose.wide_retry'):
                    wide = self._wide_retry(native, rec, kp_native, kc, S)
                if wide is not None:
                    kp_native, kc, rec['pose_crop'], rec['pose_box'] = wide
            rec['kp'] = (kp_native / S).tolist()
            rec['kc'] = np.asarray(kc).tolist()
        self._smoother.update(recs)


def _hand_quality(kp_hand, kc_hand, conf_thr, min_diag_px):
    visible = kc_hand >= conf_thr
    n_visible = int(visible.sum())
    if n_visible < 5:
        return 'occluded_or_missing', n_visible
    pts = kp_hand[visible]
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))) if len(pts) >= 2 else 0.0
    return ('too_small' if diag < min_diag_px else 'ok'), n_visible


@register('hand', 'rtmw')
class RTMWHand:
    """RTMW wholebody (rtmlib, onnxruntime) trên crop ảnh gốc quanh track."""
    enabled = True

    BUCKETS = (1, 2, 4, 8, 16)       # kích thước lô cố định: onnxruntime CUDA dò thuật toán lại mỗi khi gặp shape mới

    def __init__(self, mode='lightweight', device='cuda', require_gpu=False, batch=True, **_):
        import onnxruntime as ort
        from rtmlib import Wholebody
        self.batch = bool(batch)         # True: gom crop của mọi người trong frame thành 1 lượt chạy model (nhanh hơn nhiều)
        want_gpu = 'cuda' in str(device)
        if want_gpu and hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls()          # nạp CUDA/cuDNN từ gói nvidia-* (onnxruntime-gpu[cuda,cudnn])
            except Exception as e:
                print(f'    onnxruntime preload_dlls lỗi: {e}')
        self.pose_model = Wholebody(mode=mode, backend='onnxruntime', device='cuda' if want_gpu else 'cpu').pose_model
        sess = getattr(self.pose_model, 'session', None)
        providers = sess.get_providers() if sess is not None else []
        self.on_gpu = bool(providers) and providers[0] == 'CUDAExecutionProvider'
        print(f'    RTMW onnxruntime {ort.__version__} | session providers: {providers}')
        if want_gpu and not self.on_gpu:
            msg = ('RTMW đang chạy trên CPU (onnxruntime không dùng được CUDA) -> rất chậm và chiếm CPU của các module khác. '
                   f'Providers có sẵn: {ort.get_available_providers()}. '
                   "Kiểm tra cell cài đặt, hoặc dùng hand=dict(backend='none').")
            if require_gpu:
                raise RuntimeError(msg)
            print('    !!! ' + msg)

    def warmup(self):
        img = np.zeros((256, 192, 3), np.uint8)
        self.pose_model(img, bboxes=np.array([[0, 0, 192, 256]], np.float32))
        if self.batch and self._can_batch():          # làm nóng từng mức lô để lần gặp đầu không bị chậm
            for b in self.BUCKETS:
                self._infer_batch([(img, [0, 0, 192, 256])] * b)

    def _can_batch(self):
        pm = self.pose_model
        return (getattr(pm, 'backend', None) == 'onnxruntime' and getattr(pm, 'session', None) is not None
                and not getattr(pm, 'to_openpose', False))

    def _infer_batch(self, items):
        """items: [(crop, bbox_trong_crop)] -> [(kp_all, kc_all)] giống pose_model(crop, bboxes=[bbox]) từng cái một, nhưng chỉ 1 lần
        session.run cho cả lô (ONNX có trục batch động). Lô được pad lên mức cố định bằng ảnh trắng; các dòng độc lập nhau nên kết
        quả của dòng thật không đổi."""
        pm = self.pose_model
        pre = [pm.preprocess(crop, bbox) for crop, bbox in items]
        x = np.stack([np.ascontiguousarray(r[0].transpose(2, 0, 1), dtype=np.float32) for r in pre])
        n = len(pre)
        b = next((k for k in self.BUCKETS if k >= n), n)
        if b > n:
            x = np.concatenate([x, np.zeros((b - n,) + x.shape[1:], np.float32)])
        sess = pm.session
        outs = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: x})
        res = []
        for i, (_, center, scale) in enumerate(pre):
            res.append(pm.postprocess([outs[0][i:i + 1], outs[1][i:i + 1]], center, scale))
        return res

    def _pose_all(self, items):
        """Chạy pose cho mọi crop; trả list (kp_all, kc_all) hoặc None nếu crop đó lỗi (giống try/except của vòng lặp từng người)."""
        if self.batch and self._can_batch():
            try:
                out = []
                for i in range(0, len(items), self.BUCKETS[-1]):
                    out += self._infer_batch(items[i:i + self.BUCKETS[-1]])
                return out
            except Exception:
                pass                                  # lỗi cả lô -> thử từng người để cô lập crop hỏng
        out = []
        for crop, bbox in items:
            try:
                out.append(self.pose_model(crop, bboxes=np.array([bbox], dtype=np.float32)))
            except Exception:
                out.append(None)
        return out

    def __call__(self, native, recs, S):
        with _span('hand.call'):
            self._run(native, recs, S)

    def _run(self, native, recs, S):
        nh, nw = native.shape[:2]
        work = []
        for rec in recs:
            if rec.get('kp') is None:
                continue
            if rec.get('pose_crop'):
                cx1, cy1, cx2, cy2 = rec['pose_crop']
                px1, py1, px2, py2 = rec['pose_box']
                bbox = [px1 - cx1, py1 - cy1, px2 - cx1, py2 - cy1]
            else:
                _, (cx1, cy1, cx2, cy2) = _crop_box(rec, S, nw, nh)
                bbox = None
            if (cx2 - cx1) < CFG['pose_min_crop_px'] or (cy2 - cy1) < CFG['pose_min_crop_px']:
                continue
            crop = native[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]
            work.append((rec, crop, bbox or [0, 0, cw, ch], cx1, cy1))
        if not work:
            return
        for (rec, _, _, cx1, cy1), res in zip(work, self._pose_all([(w[1], w[2]) for w in work])):
            if res is None:
                continue
            kp_all, kc_all = res
            if kp_all is None or len(kp_all) == 0 or kp_all.shape[1] < 133:
                continue
            hand_kp, hand_kc, hand_quality = {}, {}, {}
            for side, sl in HAND_WB.items():
                kp_native = kp_all[0][sl] + np.array([cx1, cy1])
                kc_hand = kc_all[0][sl]
                q, n_vis = _hand_quality(kp_native, kc_hand, CFG['hand_conf_thr'], CFG['hand_min_bbox_diag_px'])
                hand_kp[side] = (kp_native / S).tolist()
                hand_kc[side] = kc_hand.tolist()
                hand_quality[side] = dict(status=q, n_visible=n_vis)
            rec.update(hand_kp=hand_kp, hand_kc=hand_kc, hand_quality=hand_quality)


@register('pose', 'rtmw_wholebody')
class RTMWWholebodyPose(RTMWHand):
    """1 model RTMW (133 điểm COCO-WholeBody) cho cả thân lẫn bàn tay: 17 điểm đầu = khung xương COCO17 (thay YOLO-pose),
    điểm 91-132 = 2 bàn tay. Dùng kèm hand=dict(backend='none') -> bỏ hẳn 1 lần chạy model / người / frame."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._smoother = PoseSmoother()

    def reset_prof(self):
        self._smoother.reset()

    def __call__(self, native, recs, S):
        nh, nw = native.shape[:2]
        for rec in recs:
            (x1, y1, x2, y2), _ = _crop_box(rec, S, nw, nh)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(nw), x2), min(float(nh), y2)
            if (x2 - x1) < CFG['pose_min_crop_px'] or (y2 - y1) < CFG['pose_min_crop_px']:
                continue
            try:   # rtmlib tự cắt + affine theo bbox (đã nới 1.25) trên ảnh gốc
                kp_all, kc_all = self.pose_model(native, bboxes=np.array([[x1, y1, x2, y2]], dtype=np.float32))
            except Exception:
                continue
            if kp_all is None or len(kp_all) == 0 or kp_all.shape[1] < 133:
                continue
            kp_native, kc = np.asarray(kp_all[0], np.float32), np.asarray(kc_all[0], np.float32)
            rec['kp'] = (kp_native[:17] / S).tolist()
            rec['kc'] = kc[:17].tolist()
            hand_kp, hand_kc, hand_quality = {}, {}, {}
            for side, sl in HAND_WB.items():
                q, n_vis = _hand_quality(kp_native[sl], kc[sl], CFG['hand_conf_thr'], CFG['hand_min_bbox_diag_px'])
                hand_kp[side] = (kp_native[sl] / S).tolist()
                hand_kc[side] = kc[sl].tolist()
                hand_quality[side] = dict(status=q, n_visible=n_vis)
            rec.update(hand_kp=hand_kp, hand_kc=hand_kc, hand_quality=hand_quality)
        self._smoother.update(recs)


@register('hand', 'none')
class NoHand:
    enabled = False

    def __init__(self, **_):
        pass

    def __call__(self, native, recs, S):
        return None


def check_person_box(rec, cfg=None, work_width=None):
    """Kiểm tra score, diện tích quy đổi theo ảnh 960px và tỷ lệ w/h của một track box."""
    pf = dict(RT.get('person_filter') or {})
    if cfg is not None:
        pf.update(cfg)
    min_score = pf.get('display_min_score', 0.5)
    score = rec.get('score')
    if min_score is not None and score is not None and float(score) < float(min_score):
        return False
    box = rec.get('box')
    if box is None or len(box) < 4:
        return False
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    if w <= 0.0 or h <= 0.0:
        return False
    ref_w = float(CFG.get('width', 960) if work_width is None else work_width)
    scale = 960.0 / max(ref_w, 1.0)
    area_960 = (w * scale) * (h * scale)
    if area_960 < float(pf.get('min_box_area', 600)):
        return False
    aspect_range = pf.get('aspect_range')
    if aspect_range is not None:
        aspect = w / h
        if aspect < float(aspect_range[0]) or aspect > float(aspect_range[1]):
            return False
    return True


def count_valid_keypoints(rec, conf_thr=None):
    """Đếm số keypoint có độ tin cậy >= pose_conf_thr."""
    thr = float(CFG['pose_conf_thr'] if conf_thr is None else conf_thr)
    kp, kc = rec.get('kp'), rec.get('kc')
    if kp is None or kc is None:
        return 0
    return sum(1 for c in kc if float(c) >= thr)


class PersonTrackFilter:
    """Bộ lọc track/box giả sau tracker (không ảnh hưởng đầu vào conf=0.01 của ByteTrack)."""

    def __init__(self, cfg=None, work_width=None, pose_conf_thr=None):
        self._cfg_override = cfg
        self.work_width = work_width
        self.pose_conf_thr = pose_conf_thr
        self._hits = {}
        self._last_seen = {}
        self._pose_hist = {}
        self._step = -1

    def reset(self):
        self._hits.clear()
        self._last_seen.clear()
        self._pose_hist.clear()
        self._step = -1

    def _cfg(self):
        base = dict(RT.get('person_filter') or {})
        if self._cfg_override:
            base.update(self._cfg_override)
        return base

    def _next_frame(self, frame_idx):
        if frame_idx is None:
            self._step += 1
            return self._step
        return int(frame_idx)

    def _prune(self, fidx, keep_frames=300):
        stale = [tid for tid, last_f in self._last_seen.items() if fidx - last_f > keep_frames]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._hits.pop(tid, None)
            self._pose_hist.pop(tid, None)

    def filter_pre_pose(self, tracks, frame_idx=None, work_width=None):
        """Lọc trước khi chạy pose: score, diện tích 960px, tỷ lệ w/h và min_hits."""
        cfg = self._cfg()
        if not cfg.get('enabled', True):
            return list(tracks or [])
        if not tracks:
            return []
        fidx = self._next_frame(frame_idx)
        w_ref = work_width if work_width is not None else (
            self.work_width if self.work_width is not None else CFG.get('width', 960)
        )
        min_hits = max(1, int(cfg.get('min_hits', 3)))
        kept = []
        pred_min_score = cfg.get('predicted_min_score', 0.05)
        for rec in tracks:
            tid = int(rec['id'])
            self._last_seen[tid] = fidx
            if rec.get('predicted'):
                if self._hits.get(tid, 0) < min_hits:
                    continue
                pred_cfg = dict(cfg, display_min_score=pred_min_score)
                if not check_person_box(rec, cfg=pred_cfg, work_width=w_ref):
                    continue
                kept.append(rec)
                continue
            if not check_person_box(rec, cfg=cfg, work_width=w_ref):
                continue
            hits = self._hits.get(tid, 0) + 1
            self._hits[tid] = hits
            if hits >= min_hits:
                kept.append(rec)
        self._prune(fidx)
        return kept

    def filter_post_pose(self, tracks, posed_ids=None, frame_idx=None):
        """Lọc sau khi chạy pose: yêu cầu >= min_valid_kpts trong >= 50% frame gần nhất khi đã có pose."""
        cfg = self._cfg()
        if not cfg.get('enabled', True):
            return list(tracks or [])
        if not tracks:
            return []
        min_kpts = int(cfg.get('min_valid_kpts', 5))
        min_frac = float(cfg.get('pose_min_valid_frac', 0.5))
        win = max(1, int(cfg.get('pose_window', 10)))
        conf_thr = float(CFG['pose_conf_thr'] if self.pose_conf_thr is None else self.pose_conf_thr)
        posed_set = None if posed_ids is None else {int(t) for t in posed_ids}
        kept = []
        for rec in tracks:
            tid = int(rec['id'])
            has_pose_eval = (
                (tid in posed_set) or (rec.get('kp') is not None and rec.get('kc') is not None)
                if posed_set is not None
                else (rec.get('kp') is not None and rec.get('kc') is not None)
            )
            if has_pose_eval:
                n_valid = count_valid_keypoints(rec, conf_thr=conf_thr)
                hist = self._pose_hist.get(tid)
                if hist is None or hist.maxlen != win:
                    hist = self._pose_hist[tid] = deque(hist or (), maxlen=win)
                hist.append(n_valid >= min_kpts)
            hist = self._pose_hist.get(tid)
            if hist:
                frac = sum(1 for ok in hist if ok) / float(len(hist))
                if frac < min_frac:
                    continue
            kept.append(rec)
        return kept
