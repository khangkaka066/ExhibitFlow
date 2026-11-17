"""Segment hiện vật (Grounding-DINO + SAM) 1 lần / cảnh trước pipeline, cache theo video."""
from .common import *  # noqa: F401,F403


def _fix_mmgdino_decoder_heads(model, model_id):
    """Khôi phục decoder heads không share của MM-Grounding-DINO."""
    import glob
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    cfg = model.config
    if cfg.model_type != 'mm-grounding-dino' or (cfg.decoder_bbox_embed_share and cfg.decoder_cls_embed_share):
        return
    model.bbox_embed = torch.nn.ModuleList(copy.deepcopy(m) for m in model.bbox_embed)
    model.class_embed = torch.nn.ModuleList(copy.deepcopy(m) for m in model.class_embed)
    model.model.decoder.bbox_embed = model.bbox_embed
    model.model.decoder.class_embed = model.class_embed
    ckpt_dir = model_id if Path(model_id).is_dir() else snapshot_download(model_id, allow_patterns=['*.safetensors'])
    pattern = re.compile(r'^(?:model\.decoder\.)?(bbox_embed|class_embed)\.(\d+)\.(.+)$')
    weights = {}
    for path in glob.glob(str(Path(ckpt_dir) / '*.safetensors')):
        with safe_open(path, framework='pt') as f:
            for k in f.keys():
                m = pattern.match(k)
                if m:
                    name, idx, rest = m.groups()
                    weights[f'{name}.{idx}.{rest}'] = f.get_tensor(k)
    if not weights:
        raise RuntimeError('MM-Grounding-DINO checkpoint không có decoder head weights để khôi phục.')
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f'Unexpected MM-Grounding-DINO head keys: {result.unexpected_keys[:5]}')


def duplicate_mask_of(mask, kept, thr):
    area = mask.sum()
    for kid, km in kept:
        if (mask & km).sum() / max(min(area, km.sum()), 1) >= thr:
            return kid
    return None


def artifact_is_person(mask, box, person_boxes, sample_frames, tracks, ref_idx):
    area = max(int(mask.sum()), 1)
    for pb in person_boxes:
        x1, y1, x2, y2 = int(max(0, pb[0])), int(max(0, pb[1])), int(math.ceil(pb[2])), int(math.ceil(pb[3]))
        inside = mask[y1:y2, x1:x2].sum() / area
        if inside >= CFG['artifact_max_person_overlap'] and box_iou(box, pb) >= CFG['artifact_min_person_iou']:
            return f'trung_bbox_nguoi(inside={inside:.2f})'
    if CFG.get('artifact_max_motion') is None or len(sample_frames) < 2:
        return None
    ref = cv2.cvtColor(sample_frames[ref_idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
    changes = []
    for f, img in sample_frames.items():
        if f == ref_idx:
            continue
        m = mask.copy()
        for rec in tracks.get(f, []):
            bx = rec['box']
            if box_iou(bx, box) >= CFG['artifact_min_person_iou']:
                continue
            m[max(0, int(bx[1])):int(math.ceil(bx[3])), max(0, int(bx[0])):int(math.ceil(bx[2]))] = False
        if m.sum() < 0.3 * area:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        changes.append(float((np.abs(g - ref)[m] > 25).mean()))
    if changes and float(np.median(changes)) > CFG['artifact_max_motion']:
        return f'di_chuyen(median_change={np.median(changes):.2f})'
    return None


def _pedestal_words():
    return set(SEGMENTATION_CFG['pedestal_phrases']) | set(SEGMENTATION_CFG.get('fallback_pedestal_phrases', []))


def _is_pedestal(label):
    return any(w in str(label) for w in _pedestal_words())


def _is_3d(label):
    words = set(SEGMENTATION_CFG['sculpture_phrases']) | _pedestal_words() | set(SEGMENTATION_CFG.get('fallback_object_phrases', []))
    return any(w in str(label) for w in words)


def merge_standalone_pedestals(artifacts, masks):
    for p in [a for a in artifacts if _is_pedestal(a['label'])]:
        pm = masks[p['id']]
        px1, py1, px2, py2 = p['box']
        ph = py2 - py1
        best = None
        for a in artifacts:
            if a is p or _is_pedestal(a['label']) or a['id'] not in masks:
                continue
            cover = (pm & masks[a['id']]).sum() / max(pm.sum(), 1)
            ax1, ay1, ax2, ay2 = a['box']
            sits_on = (px1 <= (ax1 + ax2) / 2 <= px2 and py1 - 0.15 * ph <= ay2 <= py1 + 0.25 * ph
                       and (ax2 - ax1) <= (px2 - px1))
            if cover >= SEGMENTATION_CFG['pedestal_cover_merge'] or sits_on:
                if best is None or cover > best[0]:
                    best = (cover, a)
        if best is None:
            continue
        a = best[1]
        mk = masks[a['id']] | pm
        ys, xs = np.nonzero(mk)
        masks[a['id']] = mk
        a.update(box=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], pedestal=True)
        artifacts.remove(p)
        masks.pop(p['id'])


@register('segmenter', 'gdino_sam')
class GroundingDinoSAM:
    def __init__(self, dino='openmmlab-community/mm_grounding_dino_large_all', sam='sam2.1_b.pt', device=GPU_B,
                 fp16=True, dino_size='auto', **_):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from ultralytics import SAM
        self.device, self.fp16 = device, bool(fp16 and 'cuda' in str(device))
        self.dino_size = self._resolve_size(dino_size, device)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(dino)
        _fix_mmgdino_decoder_heads(model, dino)
        self.dino = (model.half() if self.fp16 else model).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(dino)
        self.sam = SAM(str(resolve_weight(sam)))
        self.lock = threading.Lock()

    @staticmethod
    def _resolve_size(size, device):
        """Kích thước ảnh vào Grounding-DINO. None = mặc định của model (800x1333, cần > 5.6 GB VRAM).
        'auto' = mặc định, trừ khi GPU < 8 GB thì thu nhỏ (cạnh ngắn 640) để không hết VRAM. Hoặc dict(shortest_edge=, longest_edge=)."""
        if size != 'auto':
            return size or None
        try:
            if 'cuda' in str(device) and torch.cuda.get_device_properties(torch.device(device)).total_memory < 8 * 2**30:
                print('  [segmenter] GPU < 8 GB: thu nhỏ ảnh vào Grounding-DINO (cạnh ngắn 640) để không hết VRAM '
                      "(đặt segmenter dino_size=None để dùng kích thước gốc)")
                return dict(shortest_edge=640, longest_edge=1066)
        except Exception:
            pass
        return None

    def warmup(self):
        img = np.zeros((540, 960, 3), np.uint8)
        self.detect(img, ['statue'], 0.3, 0.25)
        self.sam_box(img, [100, 100, 300, 300])

    def detect(self, rgb, phrases, thr, text_thr):
        inputs = self.processor(images=rgb, text=' '.join(f'{p}.' for p in phrases), return_tensors='pt',
                                **(dict(size=self.dino_size) if self.dino_size else {})).to(self.device)
        with torch.no_grad():
            if self.fp16:
                inputs['pixel_values'] = inputs['pixel_values'].half()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.dino(**inputs)
            else:
                outputs = self.dino(**inputs)
        post = self.processor.post_process_grounded_object_detection
        box_kw = 'box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
        res = post(outputs, inputs.input_ids, **{box_kw: thr}, text_threshold=text_thr, target_sizes=[rgb.shape[:2]])[0]
        labels = [str(x) for x in res.get('text_labels', res['labels'])]
        return res['boxes'].cpu().numpy().tolist(), labels, res['scores'].float().cpu().numpy()

    def _sam(self, rgb, **kw):
        res = self.sam.predict(rgb, verbose=False, device=self.device, **kw)[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None
        return res.masks.data[0].cpu().numpy().astype(bool)

    def sam_box(self, rgb, box):
        return self._sam(rgb, bboxes=[box])

    def segment_pedestal(self, rgb, box, mask, person_mask):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        py = y2 + SEGMENTATION_CFG['pedestal_point_frac'] * h
        if py >= rgb.shape[0]:
            return None
        ped = self._sam(rgb, points=[[(x1 + x2) / 2, py]], labels=[1])
        if ped is None:
            return None
        ped = ped & ~mask
        ys, xs = np.nonzero(ped)
        if len(ys) == 0:
            return None
        ok = (ped.sum() <= SEGMENTATION_CFG['pedestal_max_area_ratio'] * max(mask.sum(), 1)
              and ys.min() - y2 <= 0.15 * h and ys.max() > y2 + 0.10 * h and xs.max() - xs.min() <= 2.5 * w
              and (ped & person_mask).sum() / ped.sum() < SEGMENTATION_CFG['pedestal_max_person_frac'])
        return ped if ok else None

    def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
        with self.lock:
            return self._segment(frame, ref_idx, person_boxes, sample_frames, tracks)

    def _segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
        C = SEGMENTATION_CFG
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, scores = self.detect(rgb, C['sculpture_phrases'] + C['painting_phrases'] + C['pedestal_phrases'],
                                            C['full_thr'], C['text_thr'])
        person_mask = np.zeros(rgb.shape[:2], bool)
        for x1, y1, x2, y2 in person_boxes:
            person_mask[max(0, int(y1)):int(math.ceil(y2)), max(0, int(x1)):int(math.ceil(x2))] = True
        artifacts, masks, kept, person_like = [], {}, [], []

        def add_candidates(boxes, labels, scores, id_offset, max_keep=None, max_area=None, fallback=False):
            added = 0
            H, W = rgb.shape[:2]
            for i in np.argsort(-scores).tolist():
                if max_keep is not None and added >= max_keep:
                    break
                box, label = boxes[i], labels[i] or 'object'
                if max_area is not None and (box[2] - box[0]) * (box[3] - box[1]) > max_area * W * H:
                    continue
                mask = self.sam_box(rgb, box)
                if mask is None or mask.sum() / max(1.0, (box[2] - box[0]) * (box[3] - box[1])) < C['min_mask_fill']:
                    continue
                aid = id_offset + i
                why = artifact_is_person(mask, box, person_boxes, sample_frames, tracks, ref_idx)
                if why is not None:
                    person_like.append(dict(id=aid, label=label, box=box, reason=why))
                    continue
                if duplicate_mask_of(mask, kept, C['mask_overlap']) is not None:
                    continue
                ped = None
                if not fallback and C['pedestal_merge'] and label in set(C['sculpture_phrases']):
                    ped = self.segment_pedestal(rgb, box, mask, person_mask)
                    if ped is not None:
                        mask = mask | ped
                        ys, xs = np.nonzero(mask)
                        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                kept.append((aid, mask))
                artifacts.append(dict(id=aid, box=box, label=label, score=float(scores[i]), pedestal=ped is not None,
                                      fallback=fallback))
                masks[aid] = mask
                added += 1

        if boxes:
            add_candidates(boxes, labels, scores, 0)
        if C.get('fallback_when_no_3d') and not any(_is_3d(a['label']) for a in artifacts):
            fb_thr = (C.get('fallback_empty_thr', 0.15)
                      if (C.get('fallback_lower_when_empty', True) and not artifacts)
                      else C['fallback_thr'])
            fb = self.detect(rgb, C['fallback_object_phrases'] + C['fallback_pedestal_phrases'],
                             fb_thr, C['fallback_text_thr'])
            if fb[0]:
                add_candidates(*fb, 500, max_keep=C['fallback_max_keep'], max_area=C['fallback_max_area_frac'], fallback=True)
        if C.get('merge_standalone_pedestal'):
            merge_standalone_pedestals(artifacts, masks)
        return artifacts, masks, person_like

    def manual(self, frame, box, aid):
        mask = self.sam_box(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box)
        if mask is None:
            mask = np.zeros(frame.shape[:2], bool)
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
        return dict(id=aid, box=[float(v) for v in box], label='manual', score=1.0, pedestal=False), mask

    def release(self):
        self.dino = self.sam = self.processor = None


def draw_artifacts(img, artifacts, masks, alpha=0.30, color=(255, 120, 0)):
    vis = img.copy()
    over = vis.copy()
    for a in artifacts:
        over[masks[a['id']]] = color
    vis = cv2.addWeighted(over, alpha, vis, 1 - alpha, 0)
    for a in artifacts:
        for c in mask_contours(masks[a['id']]):
            cv2.polylines(vis, [c.astype(np.int32)], True, (255, 90, 0), 1)
        put_label(vis, f"{a['id']}:{a['label'][:10]}{'+ped' if a.get('pedestal') else ''}",
                  (int(a['box'][0]), int(a['box'][1])), (200, 70, 0), 0.45)
    return vis


class ArtifactResult:
    """Mask hiện vật của 1 video: tính 1 lần trước pipeline, dùng chung cho mọi lần chạy video đó."""

    def __init__(self, key):
        self.key = key
        self.error = self.review = self.ref_idx = None
        self.artifacts, self.masks, self.contours = [], {}, {}     # gộp mọi cảnh (id duy nhất trong video)
        self.scenes = []                                               # [dict(start, ref_idx, artifacts, masks)]
        self.timing = dict(source=None, seconds=None)


class SegmenterSession:
    """Nạp segmenter lười (chỉ khi có video chưa có cache), dùng cho mọi video, rồi close() để trả VRAM cho VLM."""

    def __init__(self, spec):
        self.spec, self.model, self.load_times = spec, None, dict(load_s=None, warmup_s=None)

    def get(self):
        if self.model is None:
            self.model = build('segmenter', self.spec)
            self.load_times = dict(load_s=self.model.load_s, warmup_s=self.model.warmup_s)
        return self.model

    def close(self):
        if self.model is None:
            return
        if hasattr(self.model, 'release'):
            self.model.release()
        self.model = None
        free_gpu()
        if settings.N_GPU:
            used = ', '.join(f'cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.2f}GB' for i in range(settings.N_GPU))
            print(f'  [segmenter] đã giải phóng | VRAM còn dùng: {used}')


def cuda_check(where):
    """Lỗi CUDA được báo bất đồng bộ: đồng bộ để lỗi (nếu có) hiện ra đúng chỗ thay vì ở lệnh CUDA kế tiếp."""
    for i in range(settings.N_GPU):
        try:
            torch.cuda.synchronize(i)
        except Exception as e:
            raise RuntimeError(f'Lỗi CUDA trên cuda:{i} sinh ra trong {where}: {e}. CUDA context đã hỏng -> '
                               'phải Restart kernel.') from e


SCENE_ID_STEP = 10000    # id hiện vật của cảnh k = id gốc + k * 10000


def artifact_cache_dir(key, seg_spec, ref_idx, sample_idxs=None):
    refs = [int(i) for i in (sample_idxs if sample_idxs is not None else [ref_idx])]
    tag = spec_tag(dict(
        v=3,
        seg=seg_spec,
        cfg=SEGMENTATION_CFG,
        ref=int(ref_idx),
        refs=refs,
        w=CFG['width'],
        mf=RT.get('seg_multi_frame'),
        iou=RT.get('seg_merge_iou', 0.5),
        mv=RT.get('seg_merge_min_votes', 1),
        kr=RT.get('seg_merge_keep_ref', True),
        mve=RT.get('seg_merge_min_votes_extra', 1),
    ))
    return settings.CACHE / f'{key}_artifacts_{tag}'


def mask_iou(a, b):
    """Tính IoU giữa 2 mask boolean 2D."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def merge_masks(frame_results, iou_thr=0.5, min_votes=1, keep_ref=True, min_votes_extra=1, ref_frame=None):
    """Hợp nhất (union) mask qua nhiều frame trong 1 cảnh theo IoU >= iou_thr.
    frame_results: list [(frame_idx, artifacts, masks)] hoặc [(artifacts, masks)].
    Khi keep_ref=True: mọi mask ở frame tham chiếu (frame đầu cảnh / ref_frame) luôn được giữ nguyên,
    các frame khác chỉ bổ sung mask mới (IoU < iou_thr với mọi mask đã có) đạt >= min_votes_extra phiếu."""
    if not frame_results:
        return [], {}
    first_item = frame_results[0]
    ref_fidx = int(ref_frame) if ref_frame is not None else (int(first_item[0]) if len(first_item) == 3 else 0)

    candidates = []
    for idx, item in enumerate(frame_results):
        if len(item) == 3:
            fidx, arts, masks = item
        else:
            fidx, (arts, masks) = idx, item
        for a in arts:
            mk = masks.get(a['id'])
            if mk is None or not np.any(mk):
                continue
            candidates.append((int(fidx), dict(a), np.asarray(mk, dtype=bool)))
    if not candidates:
        return [], {}

    clusters = []

    def _add_to_clusters(items, from_ref_flag):
        for fidx, art, mk in items:
            best_cl, best_iou = None, -1.0
            for cl in clusters:
                iou = mask_iou(mk, cl['rep_mask'])
                if iou >= iou_thr and iou > best_iou:
                    best_cl, best_iou = cl, iou
            if best_cl is not None:
                if fidx not in best_cl['frames']:
                    best_cl['members'].append((fidx, art, mk))
                    best_cl['frames'].add(fidx)
            else:
                clusters.append(dict(rep_mask=mk, members=[(fidx, art, mk)], frames={fidx}, from_ref=from_ref_flag))

    if keep_ref:
        ref_cands = sorted(
            [c for c in candidates if c[0] == ref_fidx],
            key=lambda x: (-float(x[1].get('score', 0.0)), int(x[1].get('id', 0))),
        )
        other_cands = sorted(
            [c for c in candidates if c[0] != ref_fidx],
            key=lambda x: (-float(x[1].get('score', 0.0)), x[0], int(x[1].get('id', 0))),
        )
        _add_to_clusters(ref_cands, True)
        _add_to_clusters(other_cands, False)
    else:
        all_cands = sorted(
            candidates,
            key=lambda x: (-float(x[1].get('score', 0.0)), x[0], int(x[1].get('id', 0))),
        )
        _add_to_clusters(all_cands, False)

    min_votes = max(1, int(min_votes))
    min_votes_extra = max(1, int(min_votes_extra))
    kept_clusters = []
    for cl in clusters:
        votes = len(cl['frames'])
        if keep_ref and cl['from_ref']:
            kept_clusters.append(cl)
        else:
            req = min_votes if cl['from_ref'] else max(min_votes, min_votes_extra)
            if votes >= req:
                kept_clusters.append(cl)

    merged_arts, merged_masks = [], {}
    for new_id, cl in enumerate(kept_clusters):
        members = cl['members']
        n = len(members)
        if keep_ref and cl['from_ref']:
            mk = members[0][2].copy()
        elif n >= 3:
            votes_px = np.sum([m.astype(np.int32) for _, _, m in members], axis=0)
            mk = votes_px >= ((n + 1) // 2)
            if not mk.any():
                mk = members[0][2].copy()
        else:
            mk = members[0][2].copy()
        ys, xs = np.nonzero(mk)
        if len(ys) == 0:
            continue
        top = max(members, key=lambda m: float(m[1].get('score', 0.0)))
        best = dict(top[1])
        best.update(
            id=new_id,
            box=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            pedestal=any(bool(a.get('pedestal')) for _, a, _ in members),
            fallback=all(bool(a.get('fallback')) for _, a, _ in members),
            votes=len(cl['frames']),
        )
        merged_arts.append(best)
        merged_masks[new_id] = mk
    return merged_arts, merged_masks


def scene_sample_frames(start, ref, end, fps, is_live=False, explicit_ref=False):
    """Chọn các frame lấy mẫu để segment trong 1 cảnh [start, end] theo RT['seg_multi_frame']."""
    mf = RT.get('seg_multi_frame') or {}
    if is_live or explicit_ref or not mf.get('enabled', True):
        return [int(ref)]
    step_sec = float(mf.get('step_sec', 2.0) or 2.0)
    max_f = max(1, int(mf.get('max_frames_per_scene', 4) or 1))
    step = max(1, int(round(step_sec * (fps or 25.0))))
    idxs = list(range(int(ref), int(end) + 1, step))[:max_f]
    return idxs or [int(ref)]


def read_frames(d, idxs):
    """Đọc đúng các frame idxs (đọc tuần tự, không seek -> chính xác với mọi codec), resize về ảnh làm việc."""
    from .source import open_capture
    want, out = {int(i) for i in idxs}, {}
    cap = open_capture(d['path'])
    try:
        i = 0
        while len(out) < len(want) and i <= max(want):
            if not cap.grab():
                break
            if i in want:
                ok, native = cap.retrieve()
                if ok:
                    out[i] = cv2.resize(native, (d['W'], d['H']))
            i += 1
    finally:
        cap.release()
    if len(out) < len(want):
        raise RuntimeError(f"không đọc được frame {sorted(want - out.keys())} của {d['path']}")
    return out


def _color_hist_hsv(bgr_small):
    """Histogram màu (H, S) chuẩn hoá L1 trên ảnh nhỏ; gán H=0 khi S < 12 để tránh nhiễu xám."""
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    h[s < 12] = 0
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).astype(np.float32)
    tot = float(hist.sum())
    return hist / tot if tot > 0 else hist


def detect_scenes(path, diff_thr=25.0, min_scene_sec=1.0, ref_offset_sec=0.5, max_frames=None, step_sec=0.0,
                  color_hist_thr=None):
    """Tách video thành các cảnh (cắt cảnh / đổi camera) -> [(start_frame, ref_frame)].
    Cắt cảnh = chênh lệch xám trung bình (ảnh 64x36) >= diff_thr hoặc khoảng cách L1/2 histogram màu (H, S) >= color_hist_thr.
    Cảnh ngắn hơn min_scene_sec được gộp vào cảnh sau; ref = đầu cảnh + ref_offset_sec.
    step_sec > 0: vẫn giải mã mọi frame (grab) nhưng chỉ chuyển màu + resize + so sánh mỗi step_sec giây -> nhanh hơn nhiều
    trên video dài. An toàn vì cảnh ngắn hơn min_scene_sec bị gộp; mốc cắt lệch tối đa step_sec."""
    if color_hist_thr is None:
        color_hist_thr = float(RT.get('scene_color_hist_thr', 0.35)) if RT.get('scene_color_cut', True) else 0.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_len = max(1, int(min_scene_sec * fps))
    step = max(1, int(round(step_sec * fps))) if step_sec and step_sec > 0 else 1
    segs, prev_g, prev_h, idx = [[0, 0]], None, None, 0          # [frame bắt đầu, mốc đặt ref]
    try:
        while max_frames is None or idx < max_frames:
            if step > 1 and idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, f = cap.read()
            if not ok:
                break
            small = cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h = _color_hist_hsv(small) if color_hist_thr and color_hist_thr > 0 else None
            gray_cut = prev_g is not None and float(np.abs(g - prev_g).mean()) >= diff_thr
            color_cut = (prev_h is not None and h is not None
                         and 0.5 * float(np.abs(h - prev_h).sum()) >= color_hist_thr)
            if gray_cut or color_cut:
                if idx - segs[-1][1] < min_len:
                    segs[-1][1] = idx                 # cảnh trước quá ngắn -> chỉ dời mốc ref
                else:
                    segs.append([idx, idx])
            prev_g, prev_h, idx = g, h, idx + 1
    finally:
        cap.release()
    out = []
    for k, (s, base) in enumerate(segs):
        end = segs[k + 1][0] - 1 if k + 1 < len(segs) else max(idx - 1, 0)
        out.append((s, int(min(base + ref_offset_sec * fps, end))))
    return out


def _seg_ref_spec(key, d):
    p = Path(str(d['path']))
    return next((SEG_REF_FRAMES[k] for k in (key, p.name, p.stem) if k in SEG_REF_FRAMES), RT.get('seg_ref_default', 'first'))


def video_scenes(key, d):
    """Các cảnh của video -> [(start_frame, ref_frame)] theo SEG_REF_FRAMES (khoá: key / tên file / stem) hoặc RT['seg_ref_default']."""
    spec = _seg_ref_spec(key, d)
    n = max(1, d['n'])
    from .source import is_stream
    if is_stream(d['path']):
        spec = 'first'                              # camera: không quét được cả video để tách cảnh; segment trên khung đầu
    if spec == 'auto':
        color_thr = float(RT.get('scene_color_hist_thr', 0.35)) if RT.get('scene_color_cut', True) else 0.0
        scenes = detect_scenes(d['path'], RT['scene_cut_diff'], RT['scene_min_sec'], RT['scene_ref_offset_sec'], max_frames=n,
                               step_sec=float(RT.get('scene_step_sec', 0) or 0), color_hist_thr=color_thr)
    elif spec in (None, 'first'):
        scenes = [(0, 0)]
    elif isinstance(spec, (int, np.integer)):
        scenes = [(0, int(spec))]
    else:
        scenes = [(int(s[0]), int(s[1])) if isinstance(s, (list, tuple)) else (int(s), int(s)) for s in spec]
    scenes = sorted(dict(scenes).items())
    scenes[0] = (0, scenes[0][1])                   # cảnh đầu luôn bắt đầu ở frame 0
    return [(s, min(max(r, 0), n - 1)) for s, r in scenes if s < n]


def cached_video_scenes(ckey, key, d):
    """video_scenes() có cache theo nội dung video: chạy lại cùng video không phải quét tách cảnh nữa."""
    tag = spec_tag(dict(cut=RT['scene_cut_diff'], step=RT.get('scene_step_sec'), mn=RT['scene_min_sec'], off=RT['scene_ref_offset_sec'],
                        ref=RT.get('seg_ref_default'), n=d['n'], k=_seg_ref_spec(key, d),
                        color_cut=RT.get('scene_color_cut', True), color_thr=RT.get('scene_color_hist_thr', 0.35)))
    f = settings.CACHE / f'{ckey}_scenes_{tag}.json'
    if RT['artifact_cache'] and f.exists():
        return [tuple(x) for x in json.loads(f.read_text())]
    scenes = video_scenes(key, d)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(scenes))
    return scenes


def segment_scene(key, M, seg, frame, ref_idx, sid, cache_key=None, frames=None, sample_idxs=None):
    """Segment hiện vật trên 1 hoặc nhiều frame của 1 cảnh (cache theo video + frame + segmenter) -> artifacts, masks (id đã cộng sid*SCENE_ID_STEP)."""
    if sample_idxs is not None:
        s_idxs = [int(i) for i in sample_idxs]
    elif frames:
        s_idxs = sorted(int(i) for i in frames)
    else:
        s_idxs = [int(ref_idx)]
    cache_dir = artifact_cache_dir(cache_key or key, seg.spec, ref_idx, sample_idxs=s_idxs)
    reg = cache_dir / 'artifacts.json'
    if RT['artifact_cache'] and reg.exists():
        registry, source = json.loads(reg.read_text()), 'cache'
    else:
        sample_items = [(fi, frames[fi]) for fi in s_idxs] if frames else [(int(ref_idx), frame)]
        per_frame, person_like, n_person_boxes = [], [], 0
        for fidx, img in sample_items:
            # box người trên frame segment (chỉ detector) -> loại các 'hiện vật' thực chất là người
            person_boxes = ([list(b[:4]) for b in M.detector.detect(img) if b[4] >= RT['artifact_person_min_score']]
                            if RT['artifact_person_filter'] else [])
            n_person_boxes = max(n_person_boxes, len(person_boxes))
            arts_f, masks_f, pl_f = seg.get().segment(mask_regions(img), 0, person_boxes, {0: img},
                                                      {0: [dict(box=b) for b in person_boxes]})
            cuda_check(f'segmentation {key} frame {fidx}')
            per_frame.append((fidx, arts_f, masks_f))
            for pl in pl_f:
                person_like.append(dict(pl, frame_idx=fidx))
        if len(per_frame) == 1:
            arts, masks = per_frame[0][1], per_frame[0][2]
        else:
            arts, masks = merge_masks(
                per_frame,
                iou_thr=float(RT.get('seg_merge_iou', 0.5)),
                min_votes=int(RT.get('seg_merge_min_votes', 1)),
                keep_ref=bool(RT.get('seg_merge_keep_ref', True)),
                min_votes_extra=int(RT.get('seg_merge_min_votes_extra', 1)),
                ref_frame=int(ref_idx),
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        for aid, mk in masks.items():
            np.save(cache_dir / f'artifact_{aid}_mask.npy', mk.astype(np.uint8))
        registry = dict(reference=dict(frame_idx=int(ref_idx), sample_frames=s_idxs, n_person_boxes=n_person_boxes),
                        artifacts=arts, person_like_removed=person_like)
        reg.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
        source = 'computed'
    off, excl = sid * SCENE_ID_STEP, set(EXCLUDE_ARTIFACTS.get(key, []))
    arts, masks = [], {}
    for a in registry['artifacts']:
        if a['id'] + off in excl:
            continue
        mk = np.load(cache_dir / f"artifact_{a['id']}_mask.npy").astype(bool)
        a = dict(a, id=a['id'] + off)
        arts.append(a)
        masks[a['id']] = mk
    if sid == 0:
        for j, box in enumerate(MANUAL_ARTIFACTS.get(key, [])):
            a, mk = seg.get().manual(frame, box, 1000 + j)
            arts.append(a)
            masks[a['id']] = mk
    return arts, masks, source


def prepare_artifacts(key, d, M, seg):
    """Bước chuẩn bị (ngoài pipeline song song): tách cảnh, segment hiện vật theo 1 hoặc nhiều frame mỗi cảnh.
    Cache hit -> không cần nạp segmenter."""
    res = ArtifactResult(key)
    t0 = time.perf_counter()
    try:
        ckey = f"v{video_fingerprint(d['path'])}"   # cache theo NỘI DUNG video: video khác trùng tên không dùng nhầm, upload lại không sinh bản mới
        scenes = cached_video_scenes(ckey, key, d)
        from .source import is_stream
        is_live = is_stream(d['path'])
        spec = _seg_ref_spec(key, d)
        explicit_ref = isinstance(spec, (int, np.integer, list, tuple))
        scene_samples = []
        for sid, (start, ref) in enumerate(scenes):
            end = scenes[sid + 1][0] - 1 if sid + 1 < len(scenes) else max(d['n'] - 1, 0)
            scene_samples.append(scene_sample_frames(start, ref, end, d['fps'], is_live=is_live, explicit_ref=explicit_ref))
        need_idxs = {r for _, r in scenes}
        for sid, (start, ref) in enumerate(scenes):
            s_idxs = scene_samples[sid]
            reg = artifact_cache_dir(ckey, seg.spec, ref, sample_idxs=s_idxs) / 'artifacts.json'
            if not (RT['artifact_cache'] and reg.exists()):
                need_idxs.update(s_idxs)
        frames = read_frames(d, sorted(need_idxs))
        sources, reviews = [], []
        for sid, (start, ref) in enumerate(scenes):
            s_idxs = scene_samples[sid]
            sc_frames = {fi: frames[fi] for fi in s_idxs if fi in frames}
            arts, masks, src = segment_scene(key, M, seg, frames[ref], ref, sid, cache_key=ckey,
                                             frames=sc_frames, sample_idxs=s_idxs)
            sources.append(src)
            res.scenes.append(dict(start=start, ref_idx=ref, artifacts=arts, masks=masks))
            res.artifacts += arts
            res.masks.update(masks)
            rv = draw_artifacts(frames[ref], arts, masks)
            extra = f' ({len(s_idxs)}f)' if len(s_idxs) > 1 else ''
            put_label(rv, f'scene {sid}: frame {start}+ | segment frame {ref}{extra} | {len(arts)} hien vat', (8, 24), (0, 0, 0), 0.6)
            reviews.append(rv)
        res.ref_idx = scenes[0][1]
        res.contours = {aid: mask_contours(mk) for aid, mk in res.masks.items()}
        res.review = np.vstack(reviews)
        res.timing['source'] = 'cache' if all(s == 'cache' for s in sources) else 'computed'
    except Exception:
        res.error = traceback.format_exc()
        print(f'!!! segmentation {key} lỗi:\n{res.error[-1500:]}')
    finally:
        res.timing['seconds'] = round(time.perf_counter() - t0, 3)
    return res
