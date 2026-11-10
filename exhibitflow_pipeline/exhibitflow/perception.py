"""Perception theo frame: detector (YOLOX), tracker (ByteTrack-DMA-LTC + ReID), pose (YOLOv8-pose), hand (RTMW)."""
from .common import *  # noqa: F401,F403


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
                 test_size=(608, 1088), ckpt='best_ckpt.pth', fp16=True, fuse=True, gpu_preproc=True, **_):
        ensure_tracker_repo()
        from yolox.exp import Exp, get_exp
        from yolox.utils import postprocess
        self.postprocess, self.device, self.conf, self.nms = postprocess, device, conf, nms
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
        fh, fw = frame.shape[:2]
        if self.gpu_preproc:
            x, ratio = self._preprocess_gpu(frame)
        else:
            inp, ratio = detector_preprocess(frame, self.exp.test_size)
            x = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dets = self.postprocess(self.model(x.half() if self.half else x), self.exp.num_classes, self.conf, self.nms)[0]
        out = []
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

    def __init__(self, weights='yolo11n.pt', device=GPU_A, conf=0.1, nms=0.5, imgsz=640, classes=(0,), half=True, **_):
        from ultralytics import YOLO, RTDETR
        w = resolve_weight(weights)
        self.model = (RTDETR if 'rtdetr' in Path(str(w)).name.lower() else YOLO)(str(w))
        self.kw = dict(conf=conf, iou=nms, imgsz=imgsz, classes=list(classes) if classes is not None else None,
                       device=device, half=bool(half and 'cuda' in str(device)), verbose=False)
        print(f'    Ultralytics {w} | imgsz={imgsz} classes={classes}')

    def detect(self, frame):
        boxes = self.model.predict(frame, **self.kw)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy, conf = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
        return [[float(a), float(b), float(c), float(d), float(s)] for (a, b, c, d), s in zip(xyxy, conf)]


def _attach_scores(tracked, dets, min_iou=0.30):
    """Chỉ giữ track khớp một detection ở frame này (như notebook gốc); score = conf detector."""
    out = []
    for tid, tb in tracked:
        best = max(dets, key=lambda x: box_iou(tb, x[:4]), default=None)
        if best is not None and box_iou(tb, best[:4]) >= min_iou:
            out.append(dict(id=int(tid), box=[float(v) for v in tb], score=round(float(best[4]), 4), kp=None, kc=None))
    return out


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


@register('tracker', 'bytetrack_dma_ltc')
class ByteTrackDMALTC:
    """Tham số mặc định = `make track-video-full-torch` (tools/track_video_bench.py)."""

    def __init__(self, device=GPU_A, with_reid=True, track_thresh=0.6, track_buffer=30, match_thresh=0.9,
                 reid_model='osnet_x0_25', reid_ckpt='osnet_x0_25.pth', reid_weight=0.35, reid_thresh=0.7,
                 ltc_ckpt='ltc_motion_mot17.pth', ltc_min_history=16, dma_ckpt='dma_gbm_mot17.gbm',
                 min_box_area=100, max_aspect=1.6, extra_args=None, **_):
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
        self.tracker = None
        print(f"    ByteTrack-DMA-LTC | ReID {reid_model if with_reid else 'off'} ({self.args.reid_model_path or '-'}) | "
              f"DMA {self.args.ml_weights} | LTC min_history={ltc_min_history} | "
              f"thresh={track_thresh} match={match_thresh} buffer={track_buffer}")

    def reset(self, fps):
        from yolox.tracker.byte_tracker import BYTETracker
        from yolox.tracker.basetrack import BaseTrack
        BaseTrack._count = 0
        self.tracker = BYTETracker(self.args, frame_rate=int(round(fps)))
        _single_thread_gbm(self.tracker)
        for name, ok in (('DMA GBM', getattr(self.tracker, 'dma', None) is not None),
                         ('LTC motion', getattr(getattr(self.tracker, 'motion_predictor', None), 'enabled', False)),
                         ('ReID', getattr(self.tracker, 'reid_extractor', None) is not None or not self.args.with_reid)):
            if not ok:
                print(f'    CẢNH BÁO: {name} không được nạp trong BYTETracker')

    def update(self, dets, frame, native=None, S=1.0):
        # như make: tracker (ReID cắt ảnh người) chạy trên ảnh gốc -> đổi box sang toạ độ ảnh gốc rồi đổi lại
        img, s = (native, S) if native is not None else (frame, 1.0)
        fh, fw = img.shape[:2]
        arr = np.array(dets, np.float32).reshape(-1, 5) if dets else np.empty((0, 5), np.float32)
        arr[:, :4] *= s
        online = self.tracker.update(arr, (fh, fw), (fh, fw), frame=img)
        keep = []
        for t in online:
            w, h = t.tlwh[2], t.tlwh[3]
            if w * h > self.min_box_area and not (self.max_aspect and w / max(h, 1e-6) > self.max_aspect):
                keep.append((t.track_id, [float(v) / s for v in t.tlbr]))
        return _attach_scores(keep, dets)


@register('tracker', 'bytetrack_dma_ltc_proc')
class RemoteTracker:
    """ByteTrack-DMA-LTC chạy trong TIẾN TRÌNH RIÊNG (exhibitflow/tracker_worker.py). Stage `track` là phần nặng mã Python nhất
    (ReID + LTC + DMA, nhiều kernel nhỏ phát từ CPU): trong cùng tiến trình nó tranh GIL với 6 stage khác. Tiến trình riêng có GIL
    riêng nên không bị các stage kia làm chậm (và ngược lại). Giao diện giống ByteTrackDMALTC (reset / update).
    Đổi lại: thêm 1 CUDA context (~0.5 GB VRAM) và ~2-4 ms chép ảnh qua shared memory mỗi frame."""

    def __init__(self, startup_timeout=300, threads=1, **spec):
        import atexit
        import secrets
        import shutil
        import subprocess
        import tempfile
        import threading
        from multiprocessing.connection import Listener
        self._lock, self._shm, self._closed = threading.Lock(), None, False
        self._dir = tempfile.mkdtemp(prefix='ef_trk_')
        key = secrets.token_bytes(16)
        addr = os.path.join(self._dir, 'trk.sock')
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
            raise RuntimeError(f'Tiến trình tracker không khởi động được (mã {self.proc.poll()})')
        self.conn = box['conn']
        atexit.register(self.close)
        spec = dict(spec, backend='bytetrack_dma_ltc')
        self._call(('init', dict(spec=spec, weight_roots=[str(p) for p in settings.WEIGHT_ROOTS], work=str(settings.WORK),
                                 threads=threads)), timeout=startup_timeout)
        print(f'    Tracker chạy ở tiến trình riêng (pid {self.proc.pid})')

    def _kill(self):
        try:
            self.proc.kill()
            self.proc.wait(5)
        except Exception:
            pass

    def _call(self, msg, timeout=120):
        with self._lock:
            if self._closed:
                raise RuntimeError('Tiến trình tracker đã đóng')
            self.conn.send(msg)
            t0 = time.perf_counter()
            while not self.conn.poll(1.0):
                if self.proc.poll() is not None:
                    raise RuntimeError(f'Tiến trình tracker đã chết (mã {self.proc.returncode})')
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f'Tiến trình tracker không trả lời sau {timeout}s')
            status, val = self.conn.recv()
        if status == 'err':
            raise RuntimeError('Lỗi trong tiến trình tracker:\n' + val)
        return val

    def reset(self, fps):
        self._call(('reset', float(fps)))

    def update(self, dets, frame, native=None, S=1.0):
        from multiprocessing import shared_memory
        img, s = (native, S) if native is not None else (frame, 1.0)
        img = np.ascontiguousarray(img)
        if self._shm is None or self._shm.size < img.nbytes:      # (cấp lại) vùng nhớ chung khi ảnh lớn hơn
            old = self._shm
            self._shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
            self._call(('shm', self._shm.name))
            if old is not None:
                old.close()
                old.unlink()
        np.ndarray(img.shape, img.dtype, buffer=self._shm.buf)[...] = img
        return self._call(('upd', img.shape, [list(map(float, d)) for d in dets] if dets else [], float(s), native is not None))

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

    def __init__(self, iou_thr=0.3, max_age=30, min_score=0.3, **_):
        self.iou_thr, self.max_age, self.min_score = iou_thr, max_age, min_score

    def reset(self, fps):
        self.tracks, self.next_id = {}, 1

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
                    self.tracks[ids[r]].update(box=dets[c][:4], age=0)
                    out.append((ids[r], dets[c][:4]))
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks[self.next_id] = dict(box=d[:4], age=0)
                out.append((self.next_id, d[:4]))
                self.next_id += 1
        for t in ids:
            if t not in matched_t:
                self.tracks[t]['age'] += 1
                if self.tracks[t]['age'] > self.max_age:
                    del self.tracks[t]
        return _attach_scores(out, dets)


HAND_WB = {'left': slice(91, 112), 'right': slice(112, 133)}
ARM_KP = (7, 8, 9, 10)
TORSO_KP = (5, 6, 11, 12)


def _crop_box(rec, S, nw, nh):
    x1, y1, x2, y2 = [v * S for v in rec['box']]
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * CFG['pose_crop_pad_frac'], bh * CFG['pose_crop_pad_frac'] * 0.5
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


@register('pose', 'ultralytics_pose')
class UltralyticsPose:
    def __init__(self, weights='yolov8n-pose.pt', device=GPU_A, batch=True, wide_retry=None, **_):
        from ultralytics import YOLO
        self.model = YOLO(str(resolve_weight(weights)))
        self.device, self.batch = device, batch
        self.wide = CFG['pose_wide_retry'] if wide_retry is None else wide_retry

    def _predict(self, crops):
        return self.model(crops, verbose=False, device=self.device)

    def warmup(self):
        for _ in range(2):
            self._predict([np.zeros((256, 128, 3), np.uint8)] * 2)

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
                wide = self._wide_retry(native, rec, kp_native, kc, S)
                if wide is not None:
                    kp_native, kc, rec['pose_crop'], rec['pose_box'] = wide
            rec['kp'] = (kp_native / S).tolist()
            rec['kc'] = np.asarray(kc).tolist()


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

    def __init__(self, mode='lightweight', device='cuda', require_gpu=False, **_):
        import onnxruntime as ort
        from rtmlib import Wholebody
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

    def __call__(self, native, recs, S):
        nh, nw = native.shape[:2]
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
            try:
                kp_all, kc_all = self.pose_model(crop, bboxes=np.array([bbox or [0, 0, cw, ch]], dtype=np.float32))
            except Exception:
                continue
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


@register('hand', 'none')
class NoHand:
    enabled = False

    def __init__(self, **_):
        pass

    def __call__(self, native, recs, S):
        return None
