"""Pairing người - hiện vật, state machine chạm, gom event, cắt ảnh cho VLM, quyết định từ VLM."""
from .common import *  # noqa: F401,F403


WRIST_IDX = {'left': 9, 'right': 10}
HIP_IDX = {'left': 11, 'right': 12}
SHOULDER_IDX = {'left': 5, 'right': 6}
TORSO_IDX = (5, 6, 11, 12)
HEAD_IDX = (0, 1, 2, 3, 4)


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


def contact_gate(rec, hand, anchor, aid, pedestal_foot):
    box_h = max(rec['box'][3] - rec['box'][1], 1e-3)
    if box_h < CFG['state_min_person_h_px']:
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

    def update(self, norm_dist, frame_idx, fps, contact_allowed=True):
        if norm_dist is None:
            self.lost_counter += 1
            if self.lost_counter > CFG['state_lost_grace_frames']:
                self.state = 'OCCLUDED'
            return self.state, False, False
        self.lost_counter = 0
        low = norm_dist <= CFG['state_contact_thr'] and contact_allowed
        self.history.append((frame_idx, norm_dist, low))
        while self.history and frame_idx - self.history[0][0] > CFG['state_history_seconds'] * fps:
            self.history.popleft()
        need = math.ceil(CFG['state_hold_seconds'] * fps)
        recent_low = sum(1 for _, _, l in self.history if l)
        dists = [x for _, x, _ in self.history]
        approach = len(dists) >= 3 and (dists[0] - dists[-1]) > 0.02 and dists[-1] < dists[-2]
        withdraw = bool(len(dists) >= 2 and self.min_dist_since_low is not None
                        and norm_dist > self.min_dist_since_low + 0.02 and dists[-1] > dists[-2])
        if norm_dist > CFG['state_near_thr']:
            self.state, self.min_dist_since_low = 'FAR', None
        elif low and recent_low >= need:
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
    boxes = [ctx.artifact_by_id[a]['box'] for a in g['artifact_ids']]
    boxes += [r['box'] for fi in range(ws, we + 1) for r in ctx.tracks.get(fi, []) if r['id'] == g['tid']]
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


def render_crop(img_store, fi, g, ctx, marked=True):
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
        for c in ctx.art.contours[aid]:
            cv2.polylines(img, [T(c).astype(np.int32)], True, (255, 90, 0), th)
        a = ctx.artifact_by_id[aid]
        p = T([[a['box'][0], a['box'][3]]])[0]
        put_label(img, f'EXHIBIT {aid}', (p[0], p[1] + 6 + int(22 * fs)), (200, 70, 0), fs)
    r = next((r for r in ctx.tracks.get(fi, []) if r['id'] == g['tid']), None)
    if r is not None:
        b = T([r['box'][:2], r['box'][2:]]).astype(int)
        cv2.rectangle(img, tuple(map(int, b[0])), tuple(map(int, b[1])), (0, 0, 255), th)
        put_label(img, f"ID {g['tid']}", (b[0][0], b[0][1]), (0, 0, 200), fs, 2)
    else:
        put_label(img, f"ID {g['tid']} (lost)", (8, img.shape[0] - 8), (0, 0, 200), fs)
    put_label(img, f"t={(fi - g['window'][0]) / ctx.fps:+.2f}s", (6, int(28 * fs) + 6), (40, 40, 40), fs)
    return img


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
    for k, fi in enumerate(g['samples']):
        img = ctx.store.get(fi)
        if img is None:
            continue
        img = mask_regions(img)
        marked = render_crop(img, fi, g, ctx)
        p_m, p_c = gdir / 'marked' / f'{k:02d}_f{fi:05d}.jpg', gdir / 'clean' / f'{k:02d}_f{fi:05d}.jpg'
        cv2.imwrite(str(p_m), marked, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(p_c), render_crop(img, fi, g, ctx, False), [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(marked)
        images.append(str(p_m if settings.VLM_IMAGE_SET == 'marked' else p_c))
        times.append(round((fi - ws) / ctx.fps, 3))
        used.append(fi)
    g['samples'] = used
    sheet = contact_sheet(tiles)
    if sheet is not None:
        cv2.imwrite(str(gdir / 'sheet.jpg'), sheet)
    dt = (used[1] - used[0]) / ctx.fps if len(used) > 1 else 0.0
    prompt = settings.PROMPT_TEMPLATE.format(n=len(used), dt=dt, tid=g['tid'])
    (gdir / 'prompt.txt').write_text(prompt, encoding='utf-8')
    g['dir'] = gdir
    g['task_id'] = f"{ctx.exp}/{ctx.key}/{g['name']}"
    g['export_ms'] = round((time.perf_counter() - t0) * 1e3, 2)
    return dict(task_id=g['task_id'], video_key=ctx.key, segment=g['name'], tid=g['tid'], images=images, times=times,
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


class OnlineInteraction:
    """Pairing + state machine + gom event + lập đoạn VLM, gọi step() mỗi frame."""

    def __init__(self, ctx):
        self.ctx, self.fps = ctx, ctx.fps
        self.pair_states, self.open_ev, self.hist = {}, {}, OrderedDict()
        self.events, self.pending, self.tasks = [], [], []
        self.merge_gap = max(1, math.ceil(CFG['event_merge_gap_seconds'] * self.fps))
        self.hist_max = int(RT['ring_seconds'] * self.fps) + 10
        self.dist_maps, self.pedestal_foot = {}, {}

    def set_artifacts(self, artifacts, masks):
        self.dist_maps = {aid: cv2.distanceTransform((1 - mk).astype(np.uint8), cv2.DIST_L2, 5) for aid, mk in masks.items()}
        self.pedestal_foot = {a['id']: a['box'][3] for a in artifacts if a.get('pedestal')}

    def frame_states(self, i, recs):
        out = []
        for rec in recs:
            if rec.get('kp') is None:
                continue
            box_h = rec['box'][3] - rec['box'][1]
            for hand in WRIST_IDX:
                anchor, src = hand_anchor_point(rec, hand)
                if anchor is None:
                    continue
                cands = sorted((nd, aid) for aid, dm in self.dist_maps.items()
                               if (nd := norm_distance(anchor, dm, box_h)) is not None and nd <= CFG['pairing_max_norm_dist'])
                if not cands:
                    continue
                d0, aid0 = cands[0]
                ambiguous = len(cands) > 1 and (cands[1][0] - d0) < 0.15
                ps = self.pair_states.setdefault((rec['id'], aid0, hand), PairState())
                gate = contact_gate(rec, hand, anchor, aid0, self.pedestal_foot)
                state, _, _ = ps.update(d0, i, self.fps, contact_allowed=gate is None)
                if gate and d0 <= CFG['state_contact_thr']:
                    state = gate.upper()
                elif ambiguous and state in ('NEAR', 'CONTACT_LIKE'):
                    state = 'AMBIGUOUS'
                out.append(dict(track_id=rec['id'], artifact_id=aid0, hand=hand, norm_dist=d0, state=state,
                                anchor_source=src, anchor_xy=(float(anchor[0]), float(anchor[1]))))
        return out

    def step(self, i, recs):
        fs = self.frame_states(i, recs)
        self.hist[i] = fs
        while len(self.hist) > self.hist_max:
            self.hist.popitem(last=False)
        self._update_events(i, fs)
        self._emit_ready(i)
        return fs

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
        if e['group'] is None and e['end_frame'] - e['start_frame'] + 1 < CFG['event_min_frames']:
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
        g['name'] = f"seg_{k:03d}_id{g['tid']}_ex{'-'.join(map(str, g['artifact_ids']))}_f{g['start']}-{g['end']}"
        self.tasks.append(g)
        if k >= CFG['max_tasks_per_video']:
            g['vlm_status'] = 'skipped_max_tasks'
            return
        task = export_task(self.ctx, g)
        g['t_emit'] = task['t_emit']
        g['emit_frame'] = self.ctx.last_frame
        self.ctx.vlm.submit(task, self.ctx.on_vlm_result)

    def flush(self, last_idx):
        for pair in list(self.open_ev):
            self._close(self.open_ev.pop(pair))
        self._emit_ready(last_idx, final=True)
