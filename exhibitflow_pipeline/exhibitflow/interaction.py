"""Pairing người - hiện vật, state machine chạm, gom event, cắt ảnh cho VLM, quyết định từ VLM."""
from .common import *  # noqa: F401,F403
from .perception import PoseSmoother


WRIST_IDX = {'left': 9, 'right': 10}
HIP_IDX = {'left': 11, 'right': 12}
SHOULDER_IDX = {'left': 5, 'right': 6}
TORSO_IDX = (5, 6, 11, 12)
HEAD_IDX = (0, 1, 2, 3, 4)

_BASE_EVENT_MIN_FRAMES = 12
_BASE_MIN_PERSON_H_PX = 80
_BASE_HOLD_SECONDS = 0.25
_BASE_CONTACT_THR = 0.35


def effective_event_min_frames():
    """Số frame tối thiểu của một event (8 khi relaxed_gate=True nếu chưa ghi đè event_min_frames)."""
    base = CFG['event_min_frames']
    if CFG.get('relaxed_gate', False) and base == _BASE_EVENT_MIN_FRAMES:
        return int(CFG.get('relaxed_event_min_frames', 8))
    return int(base)


def effective_min_person_h_px(work_width=None):
    """Chiều cao người tối thiểu (quy đổi theo work_width so với chuẩn 960px)."""
    base = CFG['state_min_person_h_px']
    if CFG.get('relaxed_gate', False) and base == _BASE_MIN_PERSON_H_PX:
        base = CFG.get('relaxed_min_person_h_px', 60)
    w = work_width if work_width is not None else CFG.get('width', 960)
    return float(base) * (float(w) / 960.0)


def effective_hold_seconds():
    """Thời gian giữ liên tục để vào CONTACT_LIKE (0.15s khi relaxed_gate=True nếu chưa ghi đè)."""
    base = CFG['state_hold_seconds']
    if CFG.get('relaxed_gate', False) and base == _BASE_HOLD_SECONDS:
        return float(CFG.get('relaxed_hold_seconds', 0.15))
    return float(base)


def effective_contact_thr(is_pedestal=False):
    """Ngưỡng khoảng cách chạm chuẩn hoá (nới lên 0.45 cho mask bệ pedestal khi relaxed_gate=True)."""
    base = float(CFG['state_contact_thr'])
    if is_pedestal and CFG.get('relaxed_gate', False) and base == _BASE_CONTACT_THR:
        return float(CFG.get('relaxed_pedestal_contact_thr', 0.45))
    return base


def _periodic_vlm_cfg():
    """Đọc cấu hình RT['periodic_vlm'] (hỗ trợ dict hoặc bool)."""
    cfg = dict(
        enabled=False,
        min_near_sec=1.0,
        cooldown_sec=3.0,
        max_near_gap_frames=5,
        flush_on_end=True,
        flush_min_sec=1.0,
        flush_max_per_track=2,
    )
    raw = RT.get('periodic_vlm')
    if isinstance(raw, dict):
        cfg.update(raw)
    elif isinstance(raw, bool):
        cfg['enabled'] = raw
    return cfg


def _missing_detect_cfg():
    """Đọc cấu hình RT['missing_detect'] (hỗ trợ dict hoặc bool)."""
    cfg = dict(
        enabled=True,
        check_sec=1.0,
        sustain_sec=1.5,
        max_side=160,
        change_thr=0.35,
        edge_weight=0.5,
        edge_diff_scale=64.0,
        color_diff_scale=48.0,
        max_person_occlude=0.25,
        person_range_norm=2.0,
        global_luma_diff=25.0,
        global_hist_thr=0.35,
        cooldown_sec=4.0,
        carry_detect=True,
        carry_change_thr=0.42,
        carry_sustain_sec=1.5,
    )
    raw = RT.get('missing_detect')
    if isinstance(raw, dict):
        cfg.update(raw)
    elif isinstance(raw, bool):
        cfg['enabled'] = raw
    return cfg


def hand_at_own_head(rec, anchor):
    box_h = max(rec['box'][3] - rec['box'][1], 1e-3)
    return any(math.dist(anchor, rec['kp'][k]) <= CFG['self_touch_head_frac'] * box_h
               for k in HEAD_IDX if rec['kc'][k] >= CFG['pose_conf_thr'])


def arm_at_rest(rec, hand, anchor):
    hp = HIP_IDX[hand]
    torso = [rec['kp'][k][0] for k in TORSO_IDX if rec['kc'][k] >= CFG['pose_conf_thr']]
    if rec['kc'][hp] < CFG['pose_conf_thr'] or len(torso) < 2:
        return False
    box_h = max(rec['box'][3] - rec['box'][1], 1e-3)
    (wx, wy), hy = anchor, rec['kp'][hp][1]
    margin = CFG['rest_arm_max_dx'] * box_h
    return min(torso) - margin <= wx <= max(torso) + margin and (wy - hy) / box_h > CFG['rest_arm_min_dy']


def rtmw_hand_center(rec, hand):
    hq = (rec.get('hand_quality') or {}).get(hand)
    if not hq or not rec.get('hand_kp'):
        return None, None
    pts = np.asarray(rec['hand_kp'][hand], np.float32)
    conf = np.asarray(rec['hand_kc'][hand], np.float32)
    vis = conf >= CFG['hand_conf_thr']
    if hq['status'] == 'ok' and vis.any():
        return pts[vis].mean(axis=0), 'hand'
    if (hq['status'] != 'too_small' or not CFG['hand_small_use_center'] or vis.sum() < 5
            or conf[vis].mean() < CFG['hand_small_min_conf']):
        return None, None
    c = pts[vis].mean(axis=0)
    sh = SHOULDER_IDX[hand]
    if rec.get('kp') is not None and rec['kc'][sh] >= CFG['pose_conf_thr']:
        box_h = max(rec['box'][3] - rec['box'][1], 1e-3)
        if np.linalg.norm(c - np.asarray(rec['kp'][sh], np.float32)) > CFG['hand_small_max_reach'] * box_h:
            return None, None
    return c, 'hand_small'


def hand_anchor_point(rec, hand):
    c, source = rtmw_hand_center(rec, hand)
    if c is not None:
        return c, source
    idx = WRIST_IDX[hand]
    if rec['kc'][idx] >= CFG['pose_conf_thr']:
        return np.array(rec['kp'][idx]), 'wrist'
    if CFG.get('pose_smooth', True) and (rec.get('wrist_interp') or {}).get(hand):
        return np.array(rec['kp'][idx]), 'wrist_interp'
    return None, None


def person_foot(rec):
    x1, y1, x2, y2 = rec['box']
    bh = max(y2 - y1, 1e-3)
    kp, kc = rec.get('kp'), rec.get('kc')
    if not CFG.get('foot_from_pose') or kp is None:
        return y2, bh
    thr = CFG['pose_conf_thr']
    ys = lambda idx: [kp[k][1] for k in idx if kc[k] >= thr]
    ank, knee, hip, sh = ys((15, 16)), ys((13, 14)), ys((11, 12)), ys((5, 6))
    est = None
    if ank:
        est = max(ank)
    elif knee and hip:
        est = max(knee) + 1.15 * (max(knee) - float(np.mean(hip)))
    elif hip and sh:
        est = float(np.mean(hip)) + 1.7 * (float(np.mean(hip)) - float(np.mean(sh)))
    if est is None or est - y2 < CFG['foot_extend_min_frac'] * bh:
        return y2, bh
    est = min(est, y2 + CFG['foot_extend_max_frac'] * bh)
    return est, est - y1


def contact_gate(rec, hand, anchor, aid, pedestal_foot, work_width=None):
    box_h = max(rec['box'][3] - rec['box'][1], 1e-3)
    if box_h < effective_min_person_h_px(work_width):
        return 'too_small'
    if aid in pedestal_foot:
        foot_y, full_h = person_foot(rec)
        if abs(foot_y - pedestal_foot[aid]) / full_h > CFG['state_ground_max_gap']:
            return 'depth_mismatch'
    if hand_at_own_head(rec, anchor):
        return 'self_touch'
    if CFG['state_reject_rest_arm'] and arm_at_rest(rec, hand, anchor):
        return 'arm_at_rest'
    return None


def norm_distance(xy, dist_map, box_h):
    x, y = int(round(xy[0])), int(round(xy[1]))
    h, w = dist_map.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    return float(dist_map[y, x]) / max(box_h, 1e-3)


class PairState:
    def __init__(self):
        self.state = 'FAR'
        self.history = deque()
        self.lost_counter = 0
        self.min_dist_since_low = None

    def update(self, norm_dist, frame_idx, fps, contact_allowed=True, is_pedestal=False):
        if norm_dist is None:
            self.lost_counter += 1
            if self.lost_counter > CFG['state_lost_grace_frames']:
                self.state = 'OCCLUDED'
            return self.state, False, False
        self.lost_counter = 0
        hyst = float(CFG.get('state_hysteresis_mult', 1.3)) if CFG.get('state_hysteresis', True) else 1.0
        near_thr = CFG['state_near_thr'] * (hyst if self.state not in ('FAR', 'OCCLUDED') else 1.0)
        contact_enter_thr = effective_contact_thr(is_pedestal=is_pedestal)
        contact_stay_thr = contact_enter_thr * hyst
        in_contact_zone = self.state in ('NEAR', 'CONTACT_LIKE')
        low_enter = norm_dist <= contact_enter_thr and contact_allowed
        low = norm_dist <= (contact_stay_thr if in_contact_zone else contact_enter_thr) and contact_allowed
        self.history.append((frame_idx, norm_dist, low))
        while self.history and frame_idx - self.history[0][0] > CFG['state_history_seconds'] * fps:
            self.history.popleft()
        need = math.ceil(effective_hold_seconds() * fps)
        recent_low = sum(1 for _, _, l in self.history if l)
        dists = [x for _, x, _ in self.history]
        approach = len(dists) >= 3 and (dists[0] - dists[-1]) > 0.02 and dists[-1] < dists[-2]
        withdraw = bool(len(dists) >= 2 and self.min_dist_since_low is not None
                        and norm_dist > self.min_dist_since_low + 0.02 and dists[-1] > dists[-2])
        if norm_dist > near_thr:
            self.state, self.min_dist_since_low = 'FAR', None
        elif (self.state == 'CONTACT_LIKE' and low and recent_low >= need) or (low_enter and recent_low >= need):
            self.state = 'CONTACT_LIKE'
            self.min_dist_since_low = norm_dist if self.min_dist_since_low is None else min(self.min_dist_since_low, norm_dist)
        elif low:
            self.state = 'NEAR'
            self.min_dist_since_low = norm_dist if self.min_dist_since_low is None else min(self.min_dist_since_low, norm_dist)
        elif withdraw:
            self.state = 'WITHDRAWING'
        elif approach or self.state in ('FAR', 'APPROACHING'):
            self.state = 'APPROACHING'
        else:
            self.state = 'WITHDRAWING'
        return self.state, approach, withdraw


# ---------------- Ảnh cho VLM (cắt từ ring buffer) ----------------
def plan_crop(g, ctx):
    Ss = ctx.Sst   # px ring buffer / px ảnh làm việc
    ws, we = g['window']
    boxes = [ctx.artifact_by_id[a]['box'] for a in g['artifact_ids'] if a in ctx.artifact_by_id]
    boxes += [r['box'] for fi in range(ws, we + 1) for r in ctx.tracks.get(fi, []) if r['id'] == g['tid']]
    if not boxes:
        sw, sh = ctx.store.size
        b = [0, 0, sw, sh]
        return b, EXPORT['crop_out_long'] / max(sw, sh, 1)
    b = [min(x[0] for x in boxes) * Ss, min(x[1] for x in boxes) * Ss, max(x[2] for x in boxes) * Ss, max(x[3] for x in boxes) * Ss]
    px, py = (b[2] - b[0]) * EXPORT['crop_pad'] / 2, (b[3] - b[1]) * EXPORT['crop_pad'] / 2
    b = [b[0] - px, b[1] - py, b[2] + px, b[3] + py]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if bw < 0.6 * bh:
        e = (0.6 * bh - bw) / 2
        b = [b[0] - e, b[1], b[2] + e, b[3]]
    elif bh < 0.6 * bw:
        e = (0.6 * bw - bh) / 2
        b = [b[0], b[1] - e, b[2], b[3] + e]
    sw, sh = ctx.store.size
    b = [int(max(0, b[0])), int(max(0, b[1])), int(min(sw, b[2])), int(min(sh, b[3]))]
    return b, EXPORT['crop_out_long'] / max(b[2] - b[0], b[3] - b[1], 1)


def render_crop(img_store, fi, g, ctx, marked=True, before=False):
    x1, y1, x2, y2 = g['crop']
    sc = g['scale']
    img = cv2.resize(img_store[y1:y2, x1:x2], (max(1, int((x2 - x1) * sc)), max(1, int((y2 - y1) * sc))),
                     interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    if not marked:
        return img
    T = lambda pts: (np.asarray(pts, np.float32) * ctx.Sst - np.array([x1, y1], np.float32)) * sc
    th = max(2, EXPORT['crop_out_long'] // 300)
    fs = max(0.7, EXPORT['crop_out_long'] / 900)
    for aid in g['artifact_ids']:
        if aid not in ctx.artifact_by_id:
            continue
        for c in ctx.art.contours.get(aid, []):
            cv2.polylines(img, [T(c).astype(np.int32)], True, (255, 90, 0), th)
        a = ctx.artifact_by_id[aid]
        p = T([[a['box'][0], a['box'][3]]])[0]
        put_label(img, f'EXHIBIT {aid}', (p[0], p[1] + 6 + int(22 * fs)), (200, 70, 0), fs)
    if before:
        put_label(img, 'BEFORE', (6, int(28 * fs) + 6), (40, 40, 40), fs)
        return img
    r = next((r for r in ctx.tracks.get(fi, []) if r['id'] == g['tid']), None)
    if r is not None:
        b = T([r['box'][:2], r['box'][2:]]).astype(int)
        cv2.rectangle(img, tuple(map(int, b[0])), tuple(map(int, b[1])), (0, 0, 255), th)
        put_label(img, f"ID {g['tid']}", (b[0][0], b[0][1]), (0, 0, 200), fs, 2)
    else:
        put_label(img, f"ID {g['tid']} (lost)", (8, img.shape[0] - 8), (0, 0, 200), fs)
    put_label(img, f"t={(fi - g['window'][0]) / ctx.fps:+.2f}s", (6, int(28 * fs) + 6), (40, 40, 40), fs)
    return img


def _scene_ref_frame(ctx, frame_idx):
    if hasattr(ctx, 'get_ref_frame'):
        return ctx.get_ref_frame(frame_idx)
    if hasattr(ctx, 'scene_at'):
        sc = ctx.scene_at(frame_idx)
        ref_idx = int(sc.get('ref_idx', sc.get('start', 0)))
        img = getattr(ctx, 'ref_frames', {}).get(ref_idx)
        if img is None and hasattr(ctx, 'store'):
            img = ctx.store.get(ref_idx)
        if img is None:
            img = sc.get('ref_frame')
        if img is not None and hasattr(ctx, 'store'):
            sw, sh = ctx.store.size
            if img.shape[1] != sw or img.shape[0] != sh:
                img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return ref_idx, img
    return None, None


def export_task(ctx, g):
    """Cắt ảnh cho 1 đoạn VLM từ ring buffer (ảnh gốc thu nhỏ), lưu ra đĩa, trả task cho VLMWorker."""
    t0 = time.perf_counter()
    ws, we = g['window']
    ws = max(ws, ctx.store.oldest())
    g['window'] = [ws, we]
    g['samples'] = sorted(set(np.linspace(ws, we, min(EXPORT['n_sample_frames'], we - ws + 1)).round().astype(int).tolist()))
    g['crop'], g['scale'] = plan_crop(g, ctx)
    gdir = ctx.dir / 'vlm_inputs' / g['name']
    (gdir / 'marked').mkdir(parents=True, exist_ok=True)
    (gdir / 'clean').mkdir(exist_ok=True)
    tiles, images, times, used = [], [], [], []
    has_before, ref_idx = False, None
    if EXPORT.get('before_ref', True):
        ref_idx, ref_img = _scene_ref_frame(ctx, g['start'])
        if ref_img is not None:
            ref_img = mask_regions(ref_img)
            b_marked = render_crop(ref_img, ref_idx, g, ctx, marked=True, before=True)
            b_clean = render_crop(ref_img, ref_idx, g, ctx, marked=False, before=True)
            pb_m = gdir / 'marked' / f'00_before_f{ref_idx:05d}.jpg'
            pb_c = gdir / 'clean' / f'00_before_f{ref_idx:05d}.jpg'
            cv2.imwrite(str(pb_m), b_marked, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(pb_c), b_clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
            tiles.append(b_marked)
            images.append(str(pb_m if settings.VLM_IMAGE_SET == 'marked' else pb_c))
            times.append(None)
            has_before = True
    off = 1 if has_before else 0
    for k, fi in enumerate(g['samples']):
        img = ctx.store.get(fi)
        if img is None:
            continue
        img = mask_regions(img)
        marked = render_crop(img, fi, g, ctx)
        p_m, p_c = gdir / 'marked' / f'{k + off:02d}_f{fi:05d}.jpg', gdir / 'clean' / f'{k + off:02d}_f{fi:05d}.jpg'
        cv2.imwrite(str(p_m), marked, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(p_c), render_crop(img, fi, g, ctx, False), [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(marked)
        images.append(str(p_m if settings.VLM_IMAGE_SET == 'marked' else p_c))
        times.append(round((fi - ws) / ctx.fps, 3))
        used.append(fi)
    g['samples'] = used
    g['has_before'] = has_before
    g['before_frame'] = ref_idx if has_before else None
    sheet = contact_sheet(tiles)
    if sheet is not None:
        cv2.imwrite(str(gdir / 'sheet.jpg'), sheet)
    dt = (used[1] - used[0]) / ctx.fps if len(used) > 1 else 0.0
    before_note = getattr(settings, 'BEFORE_NOTE', '') if has_before else ''
    prompt = settings.PROMPT_TEMPLATE.format(n=len(images), dt=dt, tid=g['tid'], before_note=before_note)
    (gdir / 'prompt.txt').write_text(prompt, encoding='utf-8')
    g['dir'] = gdir
    g['task_id'] = f"{ctx.exp}/{ctx.key}/{g['name']}"
    g['export_ms'] = round((time.perf_counter() - t0) * 1e3, 2)
    return dict(task_id=g['task_id'], video_key=ctx.key, segment=g['name'], tid=g['tid'], images=images, times=times,
                has_before=has_before, before_frame=ref_idx if has_before else None,
                prompt=prompt, system=settings.SYSTEM_PROMPT, t_emit=time.perf_counter())


# ---------------- Quyết định từ VLM (như Phase 7) ----------------
def decide(r):
    if r is None:
        return 'CANDIDATE', None, None
    parsed = r.get('parsed') or {}
    label, conf = r.get('label'), parsed.get('confidence')
    if label in settings.ANOMALY_LABELS and (conf is None or conf >= settings.VLM_MIN_CONF):
        return 'ANOMALY', label, conf
    if label == 'normal':
        return 'NORMAL', label, conf
    return 'CANDIDATE', label, conf


def vlm_contact_indices(parsed, n):
    idx = []
    cf = parsed.get('contact_frames')
    if isinstance(cf, list):
        for k in cf:
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= n:
                idx.append(k)
    if not idx and parsed.get('evidence'):
        m = re.search(r'frames?\s+(\d+)', str(parsed['evidence']), re.I)
        if m and 1 <= int(m.group(1)) <= n:
            idx.append(int(m.group(1)))
    return sorted(set(idx))


def alarm_onset(e, g, vidx):
    first_contact = e['contact_frames'][0]
    if settings.ALARM_ONSET == 'event':
        return e['start_frame'], 'event'
    if settings.ALARM_ONSET == 'contact' or not vidx:
        return first_contact, 'contact_2d'
    k = vidx[0]
    confirmed = g['samples'][k - 1]
    prev = g['samples'][k - 2] if k >= 2 else -1
    between = [f for f in e['contact_frames'] if prev < f <= confirmed]
    return (between[0], 'vlm+contact_2d') if between else (confirmed, 'vlm')


def _resize_small(img, max_side=160):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w, 1))
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    if sw == w and sh == h:
        return img.copy(), 1.0, 1.0
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return small, sw / float(w), sh / float(h)


def _frame_features_small(small):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 4.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).ravel()
    tot = float(hist.sum())
    if tot > 0:
        hist /= tot
    return dict(bgr=small.astype(np.float32), gray=gray, grad=grad, hist=hist)


def _patch_change_score(cur_bgr, cur_grad, ref_bgr, ref_grad, cfg, mask=None, edge_mask=None):
    if mask is not None and not np.any(mask):
        return 0.0
    if edge_mask is None or not np.any(edge_mask):
        edge_mask = mask
    if mask is None:
        ed = float(np.abs(cur_grad - ref_grad).mean())
        cd = float(np.abs(cur_bgr - ref_bgr).mean())
    else:
        ed = float(np.abs(cur_grad[edge_mask] - ref_grad[edge_mask]).mean())
        cd = float(np.abs(cur_bgr[mask] - ref_bgr[mask]).mean())
    edge_diff = min(1.0, ed / max(float(cfg.get('edge_diff_scale', 64.0)), 1e-3))
    color_diff = min(1.0, cd / max(float(cfg.get('color_diff_scale', 48.0)), 1e-3))
    w_edge = float(cfg.get('edge_weight', 0.5))
    return w_edge * edge_diff + (1.0 - w_edge) * color_diff


def _carry_patch(feats, box, sx, sy, luma_shift=0.0, size=(24, 24)):
    sh, sw = feats['gray'].shape[:2]
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx1 = max(0, int(math.floor((x1 - 0.10 * bw) * sx)))
    cy1 = max(0, int(math.floor((y1 + 0.15 * bh) * sy)))
    cx2 = min(sw, int(math.ceil((x2 + 0.10 * bw) * sx)))
    cy2 = min(sh, int(math.ceil((y1 + 0.85 * bh) * sy)))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    bgr = cv2.resize(feats['bgr'][cy1:cy2, cx1:cx2] - float(luma_shift), size, interpolation=cv2.INTER_AREA)
    grad = cv2.resize(feats['grad'][cy1:cy2, cx1:cx2], size, interpolation=cv2.INTER_AREA)
    return bgr, grad


class MissingArtifactDetector:
    """Phát hiện hiện vật biến mất khỏi vùng mask hoặc người mang vật đi ra (CPU nhẹ ở ~160px)."""

    def __init__(self, fps=25.0):
        self.fps = max(float(fps or 25.0), 1e-3)
        self.reset_scene([], {}, ref_idx=0, scene_start=0, ref_frame=None)

    def reset_scene(self, artifacts, masks, ref_idx=0, scene_start=0, ref_frame=None):
        self.artifacts = list(artifacts or [])
        self.art_boxes = {int(a['id']): a['box'] for a in self.artifacts if a.get('box') is not None}
        self.masks = {int(aid): mk.astype(bool) for aid, mk in (masks or {}).items()}
        self.ref_idx = int(ref_idx or 0)
        self.scene_start = int(scene_start or 0)
        self.small_shape = None
        self.small_masks = {}
        self.union_small = None
        self.ref_feats = None
        self.prev_feats = None
        self.art_refs = {}
        self.streaks = {}
        self.near_hist = defaultdict(deque)
        self.last_emit_aid = {}
        self.last_emit_tid = {}
        self.last_check_frame = None
        self.carry_tracks = {}
        if ref_frame is not None:
            self._init_from_ref_frame(ref_frame)

    def _init_small_masks(self, sh, sw, sx, sy):
        self.small_shape = (sh, sw, sx, sy)
        self.small_masks = {}
        union = np.zeros((sh, sw), bool)
        kernel = np.ones((3, 3), np.uint8)
        for aid, mk in self.masks.items():
            if mk.shape[:2] == (sh, sw):
                mk_s = mk.copy()
            else:
                mk_s = cv2.resize(mk.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) >= 0.3
                if not mk_s.any() and mk.any():
                    bx = self.art_boxes.get(aid)
                    if bx is not None:
                        x1 = max(0, min(sw - 1, int(bx[0] * sx)))
                        y1 = max(0, min(sh - 1, int(bx[1] * sy)))
                        x2 = max(x1 + 1, min(sw, int(math.ceil(bx[2] * sx))))
                        y2 = max(y1 + 1, min(sh, int(math.ceil(bx[3] * sy))))
                        mk_s[y1:y2, x1:x2] = True
            mk_edge = cv2.dilate(mk_s.astype(np.uint8), kernel, iterations=1).astype(bool)
            self.small_masks[aid] = (mk_s, mk_edge)
            union |= mk_s
        self.union_small = union

    def _init_from_ref_frame(self, ref_frame):
        cfg = _missing_detect_cfg()
        small, sx, sy = _resize_small(ref_frame, max_side=int(cfg.get('max_side', 160)))
        sh, sw = small.shape[:2]
        self._init_small_masks(sh, sw, sx, sy)
        feats = _frame_features_small(small)
        self.ref_feats = feats
        self.prev_feats = feats
        for aid in self.small_masks:
            self.art_refs[aid] = dict(bgr=feats['bgr'].copy(), grad=feats['grad'].copy())

    @staticmethod
    def _person_mask_small(recs, sh, sw, sx, sy):
        pm = np.zeros((sh, sw), bool)
        for rec in recs or ():
            x1, y1, x2, y2 = rec['box']
            xa = max(0, int(math.floor(x1 * sx)))
            ya = max(0, int(math.floor(y1 * sy)))
            xb = min(sw, int(math.ceil(x2 * sx)))
            yb = min(sh, int(math.ceil(y2 * sy)))
            if xb > xa and yb > ya:
                pm[ya:yb, xa:xb] = True
        return pm

    @staticmethod
    def _person_artifact_norm_dist(rec, aid, art_box, dist_maps):
        x1, y1, x2, y2 = rec['box']
        bh = max(float(y2 - y1), 1.0)
        dm = dist_maps.get(aid) if dist_maps else None
        if dm is not None:
            H, W = dm.shape
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
            if xb > xa and yb > ya:
                return float(dm[ya:yb, xa:xb].min()) / bh
        if art_box is not None:
            ax1, ay1, ax2, ay2 = art_box
            dx = max(0.0, float(ax1) - float(x2), float(x1) - float(ax2))
            dy = max(0.0, float(ay1) - float(y2), float(y1) - float(ay2))
            return math.hypot(dx, dy) / bh
        return None

    def _update_near_history(self, frame_idx, recs, dist_maps, cfg, fps):
        max_range = float(cfg.get('person_range_norm', 2.0))
        keep_frames = max(1, int(round((float(cfg.get('sustain_sec', 1.5)) + 1.5) * fps)))
        for aid in self.masks:
            art_box = self.art_boxes.get(aid)
            best = None
            for rec in recs or ():
                nd = self._person_artifact_norm_dist(rec, aid, art_box, dist_maps)
                if nd is not None and nd <= max_range:
                    cand = (nd, int(rec['id']))
                    if best is None or cand[0] < best[0]:
                        best = cand
            dq = self.near_hist[aid]
            if best is not None:
                dq.append((frame_idx, best[0], best[1]))
            while dq and frame_idx - dq[0][0] > keep_frames:
                dq.popleft()

    def _find_near_person(self, aid, streak_start, frame_idx, recs, dist_maps, cfg, fps):
        max_range = float(cfg.get('person_range_norm', 2.0))
        art_box = self.art_boxes.get(aid)
        best = None
        for rec in recs or ():
            nd = self._person_artifact_norm_dist(rec, aid, art_box, dist_maps)
            if nd is not None and nd <= max_range:
                cand = (nd, int(rec['id']))
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is not None:
            return best[1], best[0]
        lookback = streak_start - int(round(0.5 * fps))
        for f_idx, nd, tid in reversed(self.near_hist.get(aid, ())):
            if lookback <= f_idx <= frame_idx and nd <= max_range:
                if best is None or nd < best[0]:
                    best = (nd, tid)
        return (best[1], best[0]) if best is not None else (None, None)

    def _update_carry_positions(self, frame_idx, recs, dist_maps, cfg):
        if not cfg.get('carry_detect', True):
            return
        max_range = float(cfg.get('person_range_norm', 2.0))
        for rec in recs or ():
            tid = int(rec['id'])
            x1, y1, x2, y2 = rec['box']
            cx, cy, bh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(float(y2 - y1), 1.0)
            st = self.carry_tracks.get(tid)
            if st is None:
                st = self.carry_tracks[tid] = dict(
                    entry_frame=frame_idx,
                    entry_patch=None,
                    approached=False,
                    anchor_xy=None,
                    anchor_aid=None,
                    min_nd=None,
                    dwell_xy=(cx, cy),
                    dwell_frames=1,
                    streak_start=None,
                    last_frame=frame_idx,
                    checks=0,
                    peak_frame=frame_idx,
                    peak_score=0.0,
                )
            else:
                st['last_frame'] = frame_idx
                move_norm = math.hypot(cx - st['dwell_xy'][0], cy - st['dwell_xy'][1]) / bh
                if move_norm <= 0.25:
                    st['dwell_frames'] += 1
                else:
                    st['dwell_xy'] = (cx, cy)
                    st['dwell_frames'] = 1

            best = None
            for aid in self.masks:
                nd = self._person_artifact_norm_dist(rec, aid, self.art_boxes.get(aid), dist_maps)
                if nd is not None and nd <= max_range:
                    if best is None or nd < best[0]:
                        best = (nd, aid)
            if best is not None:
                nd, aid = best
                if not st['approached'] or st['min_nd'] is None or nd <= st['min_nd']:
                    st['approached'] = True
                    st['anchor_xy'] = (cx, cy)
                    st['anchor_aid'] = aid
                    st['min_nd'] = nd
            elif not self.masks and not st['approached'] and st['dwell_frames'] >= max(3, int(round(0.5 * self.fps))):
                st['approached'] = True
                st['anchor_xy'] = st['dwell_xy']

    def step(self, frame_idx, recs, work=None, dist_maps=None):
        cfg = _missing_detect_cfg()
        if not cfg.get('enabled', True) or work is None:
            return []
        fps = self.fps
        recs = recs or []
        self._update_near_history(frame_idx, recs, dist_maps or {}, cfg, fps)
        self._update_carry_positions(frame_idx, recs, dist_maps or {}, cfg)

        if frame_idx < self.ref_idx:
            return []

        check_step = max(1, int(round(float(cfg.get('check_sec', 1.0)) * fps)))
        sustain_frames = max(1, int(round(float(cfg.get('sustain_sec', 1.5)) * fps)))
        carry_sustain_frames = max(1, int(round(float(cfg.get('carry_sustain_sec', 1.5)) * fps)))
        active_step = max(1, min(check_step, max(1, min(sustain_frames, carry_sustain_frames) // 2)))
        has_active = bool(self.streaks) or any(st.get('streak_start') is not None for st in self.carry_tracks.values())
        step_needed = active_step if has_active else check_step

        need_ref = (self.ref_feats is None) or (len(self.art_refs) < len(self.masks))
        need_carry_entry = bool(cfg.get('carry_detect', True)) and any(
            self.carry_tracks[int(r['id'])].get('entry_patch') is None for r in recs
        )
        is_check = (self.last_check_frame is None) or (frame_idx - self.last_check_frame >= step_needed)
        if not (is_check or need_ref or need_carry_entry):
            return []

        small, sx, sy = _resize_small(work, max_side=int(cfg.get('max_side', 160)))
        sh, sw = small.shape[:2]
        if self.small_shape != (sh, sw, sx, sy):
            self._init_small_masks(sh, sw, sx, sy)
        feats = _frame_features_small(small)
        person_mask = self._person_mask_small(recs, sh, sw, sx, sy)

        if self.ref_feats is None:
            self.ref_feats = feats
            self.prev_feats = feats

        bg = (~self.union_small) & (~person_mask)
        if int(bg.sum()) < max(16, int(0.1 * sh * sw)):
            bg = ~person_mask
        if int(bg.sum()) < 8:
            bg = np.ones((sh, sw), bool)

        luma_shift_ref = float(feats['gray'][bg].mean() - self.ref_feats['gray'][bg].mean())
        luma_diff_ref = abs(luma_shift_ref)
        hist_diff_ref = 0.5 * float(np.abs(feats['hist'] - self.ref_feats['hist']).sum())
        luma_diff_prev = abs(float(feats['gray'][bg].mean() - self.prev_feats['gray'][bg].mean())) if self.prev_feats else 0.0
        hist_diff_prev = 0.5 * float(np.abs(feats['hist'] - self.prev_feats['hist']).sum()) if self.prev_feats else 0.0
        self.prev_feats = feats

        global_changed = (
            luma_diff_ref >= float(cfg.get('global_luma_diff', 25.0))
            or hist_diff_ref >= float(cfg.get('global_hist_thr', 0.35))
            or luma_diff_prev >= float(cfg.get('global_luma_diff', 25.0))
            or hist_diff_prev >= float(cfg.get('global_hist_thr', 0.35))
        )
        if global_changed:
            self.streaks.clear()
            for st in self.carry_tracks.values():
                st['streak_start'] = None
                st['checks'] = 0
            if is_check:
                self.last_check_frame = frame_idx
            return []

        max_occ = float(cfg.get('max_person_occlude', 0.25))
        for aid, (mk_s, _) in self.small_masks.items():
            if aid not in self.art_refs:
                occ = float((mk_s & person_mask).sum()) / max(int(mk_s.sum()), 1)
                if occ <= max_occ:
                    self.art_refs[aid] = dict(
                        bgr=feats['bgr'] - luma_shift_ref,
                        grad=feats['grad'].copy(),
                    )

        if cfg.get('carry_detect', True):
            for rec in recs:
                tid = int(rec['id'])
                st = self.carry_tracks.get(tid)
                if st is not None and st['entry_patch'] is None:
                    st['entry_patch'] = _carry_patch(feats, rec['box'], sx, sy, luma_shift=luma_shift_ref)

        if not is_check:
            return []
        self.last_check_frame = frame_idx

        events = []
        cooldown_frames = max(1, int(round(float(cfg.get('cooldown_sec', 4.0)) * fps)))
        cur_bgr_comp = feats['bgr'] - luma_shift_ref
        change_thr = float(cfg.get('change_thr', 0.35))

        for aid, art_ref in self.art_refs.items():
            mk_s, mk_edge = self.small_masks[aid]
            occ = float((mk_s & person_mask).sum()) / max(int(mk_s.sum()), 1)
            if occ > max_occ:
                st = self.streaks.get(aid)
                if st is not None and frame_idx - st['last_frame'] > max(sustain_frames * 2, int(round(5.0 * fps))):
                    self.streaks.pop(aid, None)
                continue

            m = mk_s & (~person_mask)
            me = mk_edge & (~person_mask)
            if not m.any():
                m = mk_s
            if not me.any():
                me = mk_edge
            score = _patch_change_score(cur_bgr_comp, feats['grad'], art_ref['bgr'], art_ref['grad'], cfg, mask=m, edge_mask=me)
            if score > change_thr:
                st = self.streaks.get(aid)
                if st is None:
                    st = self.streaks[aid] = dict(
                        start_frame=frame_idx,
                        last_frame=frame_idx,
                        checks=1,
                        peak_frame=frame_idx,
                        peak_score=score,
                    )
                else:
                    st['last_frame'] = frame_idx
                    st['checks'] += 1
                    if score > st['peak_score']:
                        st['peak_score'] = score
                        st['peak_frame'] = frame_idx

                if (frame_idx - st['start_frame']) >= sustain_frames and (st['checks'] >= 2 or sustain_frames <= 1):
                    last_aid = self.last_emit_aid.get(aid)
                    if last_aid is None or (frame_idx - last_aid) >= cooldown_frames:
                        tid, min_nd = self._find_near_person(aid, st['start_frame'], frame_idx, recs, dist_maps, cfg, fps)
                        if tid is not None:
                            events.append(dict(
                                track_id=int(tid),
                                artifact_id=int(aid),
                                start_frame=int(st['start_frame']),
                                end_frame=int(frame_idx),
                                peak_frame=int(st['peak_frame']),
                                min_norm_dist=min_nd,
                                score=round(float(st['peak_score']), 4),
                                carry=False,
                            ))
                            self.last_emit_aid[aid] = frame_idx
                            self.last_emit_tid[tid] = frame_idx
                            self.streaks.pop(aid, None)
            else:
                self.streaks.pop(aid, None)

        if cfg.get('carry_detect', True):
            carry_thr = float(cfg.get('carry_change_thr', 0.42))
            for rec in recs:
                tid = int(rec['id'])
                st = self.carry_tracks.get(tid)
                if st is None or st['entry_patch'] is None or not st['approached'] or st['anchor_xy'] is None:
                    continue
                last_tid = self.last_emit_tid.get(tid)
                if last_tid is not None and (frame_idx - last_tid) < cooldown_frames:
                    st['streak_start'] = None
                    st['checks'] = 0
                    continue
                x1, y1, x2, y2 = rec['box']
                cx, cy, bh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(float(y2 - y1), 1.0)
                dist_away = math.hypot(cx - st['anchor_xy'][0], cy - st['anchor_xy'][1]) / bh
                if dist_away < 0.4:
                    st['streak_start'] = None
                    st['checks'] = 0
                    continue
                cur_patch = _carry_patch(feats, rec['box'], sx, sy, luma_shift=luma_shift_ref)
                if cur_patch is None:
                    continue
                c_bgr, c_grad = cur_patch
                e_bgr, e_grad = st['entry_patch']
                c_score = _patch_change_score(c_bgr, c_grad, e_bgr, e_grad, cfg)
                if c_score > carry_thr:
                    if st['streak_start'] is None:
                        st['streak_start'] = frame_idx
                        st['checks'] = 1
                        st['peak_frame'] = frame_idx
                        st['peak_score'] = c_score
                    else:
                        st['checks'] += 1
                        if c_score > st['peak_score']:
                            st['peak_score'] = c_score
                            st['peak_frame'] = frame_idx
                    if (frame_idx - st['streak_start']) >= carry_sustain_frames and (st['checks'] >= 2 or carry_sustain_frames <= 1):
                        aid = st['anchor_aid'] if st['anchor_aid'] is not None else -1
                        events.append(dict(
                            track_id=int(tid),
                            artifact_id=int(aid),
                            start_frame=int(st['streak_start']),
                            end_frame=int(frame_idx),
                            peak_frame=int(st['peak_frame']),
                            min_norm_dist=st['min_nd'],
                            score=round(float(st['peak_score']), 4),
                            carry=True,
                        ))
                        self.last_emit_tid[tid] = frame_idx
                        st['streak_start'] = None
                        st['checks'] = 0
                else:
                    st['streak_start'] = None
                    st['checks'] = 0

        return events


class OnlineInteraction:
    """Pairing + state machine + gom event + lập đoạn VLM, gọi step() mỗi frame."""

    def __init__(self, ctx):
        self.ctx, self.fps = ctx, ctx.fps
        self.pair_states, self.open_ev, self.hist = {}, {}, OrderedDict()
        self.events, self.pending, self.tasks = [], [], []
        self.merge_gap = max(1, math.ceil(CFG['event_merge_gap_seconds'] * self.fps))
        self.hist_max = int(RT['ring_seconds'] * self.fps) + 10
        self.dist_maps, self.pedestal_foot, self.pedestal_ids = {}, {}, set()
        self.work_width = (getattr(ctx, 'd', None) or {}).get('W', CFG.get('width', 960))
        self.gate_rejects = Counter()                  # (track_id, reason) -> count
        self.scene_gate_rejects = defaultdict(Counter) # scene_id -> Counter(reason -> count)
        self._rejects_scene_track = Counter()          # (scene_id, track_id, reason) -> count
        self.ambiguous_resolved = 0
        self.track_stats = {}                          # track_id -> stats dict
        self.pose_smoother = PoseSmoother()
        self._near_streaks = {}
        self._frame_near_cands = {}
        self._flush_counts = Counter()
        self.missing_detector = MissingArtifactDetector(fps=self.fps)

    def _scene_id(self, frame_idx=None):
        if frame_idx is not None and hasattr(self.ctx, 'scene_index'):
            return int(self.ctx.scene_index(frame_idx))
        return int(getattr(self.ctx, 'cur_scene', 0))

    def _reject(self, frame_idx, tid, reason):
        tid = int(tid)
        sc = self._scene_id(frame_idx)
        self.gate_rejects[(tid, reason)] += 1
        self.scene_gate_rejects[sc][reason] += 1
        self._rejects_scene_track[(sc, tid, reason)] += 1

    def set_artifacts(self, artifacts, masks, ref_idx=0, scene_start=0, ref_frame=None):
        self.dist_maps = {aid: cv2.distanceTransform((1 - mk).astype(np.uint8), cv2.DIST_L2, 5) for aid, mk in masks.items()}
        self.pedestal_ids = {a['id'] for a in artifacts if a.get('pedestal')}
        self.pedestal_foot = {a['id']: a['box'][3] for a in artifacts if a.get('pedestal') and a.get('box') is not None}
        self.missing_detector.reset_scene(artifacts, masks, ref_idx=ref_idx, scene_start=scene_start, ref_frame=ref_frame)

    def frame_states(self, i, recs):
        self.pose_smoother.update(recs, frame_idx=i, fps=self.fps)
        out = []
        near_cands = defaultdict(list)
        sc = self._scene_id(i)
        for rec in recs:
            tid = int(rec['id'])
            ts = self.track_stats.setdefault(
                tid, dict(scene_id=sc, frames_seen=0, near_frames=0, contact_frames=0, min_norm_dist=None)
            )
            ts['frames_seen'] += 1
            if not self.dist_maps:
                self._reject(i, tid, 'no_artifacts')
                continue
            if rec.get('kp') is None:
                self._reject(i, tid, 'no_pose')
                continue
            box_h = rec['box'][3] - rec['box'][1]
            saw_near = False
            saw_contact = False
            for hand in WRIST_IDX:
                anchor, src = hand_anchor_point(rec, hand)
                if anchor is None:
                    self._reject(i, tid, 'no_hand_anchor')
                    continue
                all_dists = []
                for aid, dm in self.dist_maps.items():
                    nd = norm_distance(anchor, dm, box_h)
                    if nd is not None:
                        all_dists.append((nd, aid))
                if all_dists:
                    min_nd = min(nd for nd, _ in all_dists)
                    if ts['min_norm_dist'] is None or min_nd < ts['min_norm_dist']:
                        ts['min_norm_dist'] = round(float(min_nd), 4)
                cands = sorted((nd, aid) for nd, aid in all_dists if nd <= CFG['pairing_max_norm_dist'])
                if not cands:
                    self._reject(i, tid, 'no_artifact_in_range')
                    continue
                d0, aid0 = cands[0]
                if d0 <= CFG['state_near_thr']:
                    saw_near = True
                ambiguous_thr = float(CFG.get('ambiguous_diff_thr', 0.15))
                ambiguous = len(cands) > 1 and (cands[1][0] - d0) < ambiguous_thr
                ps = self.pair_states.setdefault((rec['id'], aid0, hand), PairState())
                is_ped = aid0 in self.pedestal_ids
                contact_thr = effective_contact_thr(is_pedestal=is_ped)
                gate = contact_gate(rec, hand, anchor, aid0, self.pedestal_foot, work_width=self.work_width)
                state, _, _ = ps.update(d0, i, self.fps, contact_allowed=gate is None, is_pedestal=is_ped)
                if gate and d0 <= contact_thr:
                    state = gate.upper()
                    self._reject(i, tid, gate)
                elif ambiguous and state in ('NEAR', 'CONTACT_LIKE'):
                    if str(CFG.get('ambiguous_policy', 'nearest')).lower() == 'reject':
                        state = 'AMBIGUOUS'
                        self._reject(i, tid, 'ambiguous')
                    else:
                        self.ambiguous_resolved += 1
                if state == 'CONTACT_LIKE':
                    saw_contact = True
                if (state not in ('FAR', 'OCCLUDED', 'TOO_SMALL', 'DEPTH_MISMATCH', 'SELF_TOUCH', 'ARM_AT_REST')
                        and gate not in ('too_small', 'depth_mismatch')):
                    near_cands[tid].append((d0, aid0, hand, state))
                out.append(dict(track_id=rec['id'], artifact_id=aid0, hand=hand, norm_dist=d0, state=state,
                                anchor_source=src, anchor_xy=(float(anchor[0]), float(anchor[1]))))
            if saw_near:
                ts['near_frames'] += 1
            if saw_contact:
                ts['contact_frames'] += 1
        self._frame_near_cands = near_cands
        return out

    def step(self, i, recs, work=None):
        fs = self.frame_states(i, recs)
        self.hist[i] = fs
        while len(self.hist) > self.hist_max:
            self.hist.popitem(last=False)
        self._update_events(i, fs)
        self._update_periodic_vlm(i)
        self._update_missing_detect(i, recs, work=work)
        self._emit_ready(i)
        return fs

    def _has_contact_event_for_track(self, tid, start_frame, end_frame):
        if any(p[0] == tid for p in self.open_ev):
            return True
        for e in self.events:
            if e.get('periodic') or e.get('missing') or e.get('track_id') != tid:
                continue
            if e['end_frame'] >= start_frame - self.merge_gap and e['start_frame'] <= end_frame:
                return True
        return False

    def _update_missing_detect(self, i, recs, work=None):
        """Đường dự phòng stealing: hiện vật biến mất khỏi mask hoặc người mang vật đi -> tạo event suspect_missing gửi VLM."""
        hits = self.missing_detector.step(i, recs, work=work, dist_maps=self.dist_maps)
        if not hits:
            return
        fps = max(float(self.fps or 25.0), 1e-3)
        max_win_frames = max(1, int(EXPORT['max_window_sec'] * fps))
        pre_frames = int(EXPORT['pre_sec'] * fps)
        art_by_id = getattr(self.ctx, 'artifact_by_id', None) or {}
        for h in hits:
            tid = int(h['track_id'])
            aid = int(h['artifact_id'])
            e = dict(
                eid=len(self.events),
                track_id=tid,
                artifact_id=aid,
                hand='suspect_carry' if h.get('carry') else 'suspect_missing',
                event_type='suspect_missing',
                missing=True,
                carry=bool(h.get('carry')),
                start_frame=int(h['start_frame']),
                end_frame=int(h['end_frame']),
                contact_frames=[int(h['peak_frame'])],
                min_norm_dist=round(float(h['min_norm_dist']), 4) if h.get('min_norm_dist') is not None else None,
                peak=int(h['peak_frame']),
                missing_score=h.get('score'),
                group=None,
            )
            self.events.append(e)
            g = self._new_group(e)
            if aid >= 0 and (not art_by_id or aid in art_by_id):
                g['artifact_ids'] = [aid]
            else:
                g['artifact_ids'] = []
            g['missing'] = True
            g['event_type'] = 'suspect_missing'
            ws = max(0, e['start_frame'] - pre_frames, i - max_win_frames + 1)
            g['window'] = [ws, i]
            self.pending.append(g)

    def _flush_near_streak(self, tid, st, pcfg):
        """Gửi 1 đoạn VLM cuối cho chuỗi NEAR khi track mất > max_gap hoặc hết video."""
        if not pcfg.get('enabled', False) or not pcfg.get('flush_on_end', True):
            return False
        if not st.get('active', False) or st.get('start_frame') is None:
            return False
        fps = max(float(self.fps or 25.0), 1e-3)
        flush_min_frames = max(1, math.ceil(float(pcfg.get('flush_min_sec', 1.0)) * fps))
        if st.get('near_count', 0) < flush_min_frames:
            return False
        max_flushes = max(0, int(pcfg.get('flush_max_per_track', 2)))
        if self._flush_counts[tid] >= max_flushes:
            return False
        if self._has_contact_event_for_track(tid, st['start_frame'], st['last_frame']):
            return False
        if len(self.tasks) + len(self.pending) >= int(CFG['max_tasks_per_video']):
            return False

        e = dict(
            eid=len(self.events),
            track_id=tid,
            artifact_id=st['aid'],
            hand='+'.join(sorted(st['hands'])),
            start_frame=st['start_frame'],
            end_frame=st['last_frame'],
            contact_frames=[st['peak']],
            min_norm_dist=round(float(st['min_norm_dist']), 4),
            peak=st['peak'],
            periodic=True,
            flush=True,
            group=None,
        )
        self.events.append(e)
        g = self._new_group(e)
        g['artifact_ids'] = sorted(st['aids'])
        g['periodic'] = True
        g['flush'] = True
        self._set_window(g)
        self.pending.append(g)

        self._flush_counts[tid] += 1
        st['active'] = False
        st['last_emit'] = st['last_frame']
        st['start_frame'] = None
        st['near_count'] = 0
        st['min_norm_dist'] = 9.9
        st['aids'] = set()
        st['hands'] = set()
        return True

    def _update_periodic_vlm(self, i):
        """Đường dự phòng: người ở trong vùng NEAR liên tục >= min_near_sec mà chưa có event -> gửi VLM."""
        pcfg = _periodic_vlm_cfg()
        if not pcfg.get('enabled', False):
            return
        fps = max(float(self.fps or 25.0), 1e-3)
        min_frames = max(1, math.ceil(float(pcfg.get('min_near_sec', 1.0)) * fps))
        cooldown_frames = max(1, math.ceil(float(pcfg.get('cooldown_sec', 3.0)) * fps))
        max_gap = max(0, int(pcfg.get('max_near_gap_frames', 5)))

        for tid, st in list(self._near_streaks.items()):
            if tid not in self._frame_near_cands and i - st['last_frame'] > max_gap:
                if st.get('active', False):
                    self._flush_near_streak(tid, st, pcfg)
                st['active'] = False

        for tid, cands in self._frame_near_cands.items():
            d0, aid0, hand, _ = min(cands, key=lambda x: x[0])
            st = self._near_streaks.get(tid)
            if st is None or not st.get('active', False) or i - st['last_frame'] > max_gap + 1:
                last_emit = st.get('last_emit') if st else None
                st = self._near_streaks[tid] = dict(
                    active=True,
                    start_frame=i,
                    last_frame=i,
                    near_count=1,
                    min_norm_dist=d0,
                    peak=i,
                    aid=aid0,
                    aids={aid0},
                    hands={hand},
                    last_emit=last_emit,
                )
            else:
                if st['start_frame'] is None:
                    st['start_frame'] = i
                st['last_frame'] = i
                st['near_count'] += 1
                st['aids'].add(aid0)
                st['hands'].add(hand)
                if d0 < st['min_norm_dist']:
                    st['min_norm_dist'] = d0
                    st['peak'] = i
                    st['aid'] = aid0

            if self._has_contact_event_for_track(tid, st['start_frame'], i):
                st['active'] = False
                continue

            if st['near_count'] < min_frames:
                continue
            if st['last_emit'] is not None and (i - st['last_emit']) < cooldown_frames:
                continue

            e = dict(
                eid=len(self.events),
                track_id=tid,
                artifact_id=st['aid'],
                hand='+'.join(sorted(st['hands'])),
                start_frame=st['start_frame'],
                end_frame=i,
                contact_frames=[st['peak']],
                min_norm_dist=round(float(st['min_norm_dist']), 4),
                peak=st['peak'],
                periodic=True,
                group=None,
            )
            self.events.append(e)
            g = self._new_group(e)
            g['artifact_ids'] = sorted(st['aids'])
            g['periodic'] = True
            max_win_frames = max(1, int(EXPORT['max_window_sec'] * fps))
            ws = max(0, st['start_frame'] - int(EXPORT['pre_sec'] * fps), i - max_win_frames + 1)
            g['window'] = [ws, i]
            self.pending.append(g)

            st['last_emit'] = i
            st['start_frame'] = None
            st['near_count'] = 0
            st['min_norm_dist'] = 9.9
            st['peak'] = i
            st['aids'] = set()
            st['hands'] = set()

    def _update_events(self, i, fs):
        per_pair = defaultdict(list)
        for s in fs:
            if s['state'] == 'CONTACT_LIKE':
                per_pair[(s['track_id'], s['artifact_id'])].append(s)
        for pair, ss in per_pair.items():
            e = self.open_ev.get(pair)
            if e is None:
                start = i    # CONTACT_LIKE bật sau state_hold_seconds -> lùi về chuỗi NEAR liền trước
                while (start - 1) in self.hist and any((s['track_id'], s['artifact_id']) == pair and s['state'] == 'NEAR'
                                                       for s in self.hist[start - 1]):
                    start -= 1
                e = self.open_ev[pair] = dict(track_id=pair[0], artifact_id=pair[1], start_frame=start, end_frame=i,
                                              contact_frames=[], _hands=set(), min_norm_dist=9.9, peak=i, group=None)
            e['end_frame'] = i
            e['contact_frames'].append(i)
            e['_hands'].update(s['hand'] for s in ss)
            dmin = min(s['norm_dist'] for s in ss)
            if dmin < e['min_norm_dist']:
                e['min_norm_dist'], e['peak'] = round(dmin, 4), i
            early = RT.get('vlm_early_sec')
            if early and e['group'] is None and (i - e['contact_frames'][0]) >= early * self.fps:
                g = self._new_group(e)
                g['window'] = [max(0, e['start_frame'] - int(EXPORT['pre_sec'] * self.fps)), i]
                g['early'] = True
                self._submit(g)
        for pair in list(self.open_ev):
            if i - self.open_ev[pair]['end_frame'] > self.merge_gap:
                self._close(self.open_ev.pop(pair))

    def _close(self, e):
        e['hand'] = '+'.join(sorted(e.pop('_hands')))
        if e['group'] is None and e['end_frame'] - e['start_frame'] + 1 < effective_event_min_frames():
            self._reject(e['start_frame'], e['track_id'], 'event_too_short')
            return
        e['eid'] = len(self.events)
        self.events.append(e)
        if e['group'] is not None:
            return
        for g in self.pending:   # gộp event cùng người chồng thời gian (như group_tasks)
            if g['tid'] == e['track_id'] and (min(g['end'], e['end_frame']) - max(g['start'], e['start_frame']) >
                                              0.3 * min(g['end'] - g['start'] + 1, e['end_frame'] - e['start_frame'] + 1)):
                g['events'].append(e)
                e['group'] = g
                g['artifact_ids'] = sorted(set(g['artifact_ids']) | {e['artifact_id']})
                g['start'], g['end'] = min(g['start'], e['start_frame']), max(g['end'], e['end_frame'])
                if len(e['contact_frames']) > len(g['_main']['contact_frames']):
                    g['peak'], g['_main'] = e['peak'], e
                self._set_window(g)
                return
        g = self._new_group(e)
        self._set_window(g)
        self.pending.append(g)

    def _new_group(self, e):
        g = dict(tid=e['track_id'], events=[e], artifact_ids=[e['artifact_id']], start=e['start_frame'],
                 end=e['end_frame'], peak=e['peak'], _main=e, early=False)
        e['group'] = g
        return g

    def _set_window(self, g):
        fps, n = self.fps, self.ctx.n
        ws, we = max(0, g['start'] - int(EXPORT['pre_sec'] * fps)), min(n - 1, g['end'] + int(EXPORT['post_sec'] * fps))
        if we - ws + 1 > EXPORT['max_window_sec'] * fps:
            half = int(EXPORT['max_window_sec'] * fps / 2)
            ws, we = max(0, g['peak'] - half), min(n - 1, g['peak'] + half)
        g['window'] = [ws, we]

    def _emit_ready(self, i, final=False):
        for g in [g for g in self.pending if final or i >= g['window'][1]]:
            self.pending.remove(g)
            if final:
                g['window'][1] = min(g['window'][1], i)
            self._submit(g)

    def _submit(self, g):
        k = len(self.tasks)
        ex_tag = '-'.join(map(str, g['artifact_ids'])) or 'none'
        g['name'] = f"seg_{k:03d}_id{g['tid']}_ex{ex_tag}_f{g['start']}-{g['end']}"
        self.tasks.append(g)
        if k >= CFG['max_tasks_per_video']:
            g['vlm_status'] = 'skipped_max_tasks'
            self._reject(g['start'], g['tid'], 'skipped_max_tasks')
            return
        task = export_task(self.ctx, g)
        g['t_emit'] = task['t_emit']
        g['emit_frame'] = self.ctx.last_frame
        self.ctx.vlm.submit(task, self.ctx.on_vlm_result)

    def flush(self, last_idx):
        if last_idx is None or last_idx < 0:
            return
        for pair in list(self.open_ev):
            self._close(self.open_ev.pop(pair))
        pcfg = _periodic_vlm_cfg()
        if pcfg.get('enabled', False) and pcfg.get('flush_on_end', True):
            for tid, st in sorted(self._near_streaks.items()):
                if st.get('active', False):
                    self._flush_near_streak(tid, st, pcfg)
        self._emit_ready(last_idx, final=True)

    def diagnostics(self):
        by_track = defaultdict(dict)
        for (tid, reason), cnt in sorted(self.gate_rejects.items()):
            by_track[str(tid)][reason] = int(cnt)
        by_scene = {
            str(sc): dict(sorted((r, int(c)) for r, c in reasons.items()))
            for sc, reasons in sorted(self.scene_gate_rejects.items())
        }
        totals = Counter()
        for reasons in self.scene_gate_rejects.values():
            totals.update(reasons)
        tracks = {}
        for tid, st in sorted(self.track_stats.items()):
            tkey = str(tid)
            tracks[tkey] = dict(
                track_id=int(tid),
                frames_seen=int(st['frames_seen']),
                near_frames=int(st['near_frames']),
                contact_frames=int(st['contact_frames']),
                min_norm_dist=st['min_norm_dist'],
                gate_rejects=dict(by_track.get(tkey, {})),
            )
        return dict(
            gate_rejects={tkey: dict(reasons) for tkey, reasons in by_track.items()},
            gate_rejects_by_scene=by_scene,
            gate_rejects_total=dict(sorted((r, int(c)) for r, c in totals.items())),
            tracks=tracks,
            ambiguous_resolved=int(self.ambiguous_resolved),
        )

    def diagnostic_rows(self):
        rows = []
        seen_tids = set()
        for (sc, tid, reason), cnt in sorted(self._rejects_scene_track.items()):
            seen_tids.add(tid)
            st = self.track_stats.get(tid, {})
            rows.append(dict(
                scene_id=int(sc),
                track_id=int(tid),
                reason=reason,
                count=int(cnt),
                frames_seen=int(st.get('frames_seen', 0)),
                near_frames=int(st.get('near_frames', 0)),
                contact_frames=int(st.get('contact_frames', 0)),
                min_norm_dist=st.get('min_norm_dist'),
            ))
        for tid, st in sorted(self.track_stats.items()):
            if tid not in seen_tids:
                rows.append(dict(
                    scene_id=int(st.get('scene_id', 0)),
                    track_id=int(tid),
                    reason='',
                    count=0,
                    frames_seen=int(st.get('frames_seen', 0)),
                    near_frames=int(st.get('near_frames', 0)),
                    contact_frames=int(st.get('contact_frames', 0)),
                    min_norm_dist=st.get('min_norm_dist'),
                ))
        return rows
