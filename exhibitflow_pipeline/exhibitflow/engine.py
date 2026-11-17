"""Engine realtime: các stage (detect -> track -> pose -> hand -> interact -> sink) chạy song song bằng thread + queue."""
import types

from .common import *  # noqa: F401,F403
from .interaction import (OnlineInteraction, WRIST_IDX, alarm_onset, decide, hand_anchor_point,  # noqa: F401
                          vlm_contact_indices)
from .perception import PersonTrackFilter, cosine_similarity, extract_track_embedding, torso_hsv_histogram


EOS = object()


class Packet:
    __slots__ = ('idx', 't_cap', 't_ready', 'work', 'native', 'dets', 'tracks', 'states', 'n_pose')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class LatencyRecorder:
    def __init__(self):
        self.rows = []    # list.append là thread-safe trong CPython

    def add(self, stage, idx, wait_ms, compute_ms):
        self.rows.append((stage, idx, wait_ms, compute_ms))

    def df(self):
        return pd.DataFrame(self.rows, columns=['stage', 'frame', 'wait_ms', 'compute_ms'])


class FrameStore:
    """Ring buffer ảnh gốc (thu nhỏ về store_long) để cắt ảnh VLM sau khi event kết thúc."""

    def __init__(self, capacity, native_wh, store_long):
        nw, nh = native_wh
        self.scale = min(1.0, store_long / max(nw, nh))
        self.size = (int(round(nw * self.scale)), int(round(nh * self.scale)))
        self.capacity, self.buf, self.lock = max(2, int(capacity)), OrderedDict(), threading.Lock()

    def put(self, idx, native):
        img = native if self.scale == 1.0 else cv2.resize(native, self.size, interpolation=cv2.INTER_AREA)
        with self.lock:
            self.buf[idx] = img
            while len(self.buf) > self.capacity:
                self.buf.popitem(last=False)

    def get(self, idx):
        with self.lock:
            return self.buf.get(idx)

    def oldest(self):
        with self.lock:
            return next(iter(self.buf), 0)


class AlarmBoard:
    """Bảng theo dõi các track đang cảnh báo / đã bị gắn cờ ANOMALY."""

    _SEVERITY = {'stealing': 4, 'holding': 3, 'touching': 2, 'normal': 1, 'uncertain': 0}

    def __init__(self, persist=None):
        self.lock, self.alarms = threading.Lock(), []
        self.flagged = {}
        self.inherited_from = {}
        self.flag_roots = {}
        self.persist = persist

    def add(self, alarm):
        with self.lock:
            self.alarms.append(alarm)
            tid = alarm['tid']
            shown_from = int(alarm.get('shown_from', 0))
            behavior = alarm.get('behavior')
            prev = self.flagged.get(tid)
            if prev is None:
                self.flagged[tid] = dict(behavior=behavior, first_frame=shown_from)
            else:
                if behavior:
                    cur_r = self._SEVERITY.get(str(prev.get('behavior') or '').lower(), -1)
                    new_r = self._SEVERITY.get(str(behavior).lower(), -1)
                    if new_r >= cur_r:
                        prev['behavior'] = behavior
                prev['first_frame'] = min(prev['first_frame'], shown_from)
            self.flag_roots.setdefault(tid, tid)

    def inherit(self, new_tid, src_tid, frame):
        """Kế thừa cờ ANOMALY từ src_tid sang new_tid tại frame (giữ nguyên self.alarms cho báo động gốc)."""
        new_tid, src_tid, frame = int(new_tid), int(src_tid), int(frame)
        with self.lock:
            src = self.flagged.get(src_tid)
            if src is None or new_tid in self.flagged:
                return False
            self.flagged[new_tid] = dict(behavior=src.get('behavior'), first_frame=frame)
            self.inherited_from[new_tid] = src_tid
            self.flag_roots[new_tid] = self.flag_roots.get(src_tid, src_tid)
            return True

    def flagged_items(self):
        """Bản sao an toàn luồng của các track đang mang cờ kèm root_tid."""
        with self.lock:
            return {
                tid: dict(info, root_tid=self.flag_roots.get(tid, tid))
                for tid, info in self.flagged.items()
            }

    def active(self, frame, hold_frames, persist=None):
        use_persist = bool(RT.get('alarm_persist', True)) if (persist is None and self.persist is None) else (
            bool(self.persist) if persist is None else bool(persist))
        with self.lock:
            if use_persist:
                out = {a['tid']: a for a in self.alarms if a.get('shown_from', 0) <= frame}
                for tid, info in self.flagged.items():
                    if tid not in out and info.get('first_frame', 0) <= frame:
                        out[tid] = dict(tid=tid, behavior=info.get('behavior'), shown_from=info.get('first_frame', 0))
                return out
            out = {a['tid']: a for a in self.alarms if a.get('shown_from', 0) <= frame <= a.get('shown_from', 0) + hold_frames}
            for tid in self.inherited_from:
                if tid not in out and tid in self.flagged:
                    info = self.flagged[tid]
                    ff = info.get('first_frame', 0)
                    if ff <= frame <= ff + hold_frames:
                        out[tid] = dict(tid=tid, behavior=info.get('behavior'), shown_from=ff)
            return out


def _box_center_h(box):
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(y2 - y1, 1.0)


def _ema_norm(prev, cur, alpha=0.7):
    if cur is None:
        return prev
    arr = np.asarray(cur, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return prev
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return prev
    arr = arr / norm
    if prev is None or prev.shape != arr.shape:
        return arr
    blended = alpha * prev + (1.0 - alpha) * arr
    bn = float(np.linalg.norm(blended))
    return (blended / bn) if bn > 1e-6 else arr


class FlagInheritor:
    """Kế thừa cờ ANOMALY qua đổi ID bằng OSNet ReID (nếu tracker lộ embedding) hoặc fallback HSV vùng thân + vị trí.

    Ràng buộc: chỉ kế thừa khi track cũ đã mất <= max_gap_sec và mỗi cờ kế thừa tối đa 1 track tại một thời điểm.
    """

    def __init__(self, fps=25.0, cfg=None):
        self.fps = max(float(fps or 25.0), 1e-3)
        self._cfg_override = cfg
        self._tracks = {}

    def reset(self):
        self._tracks.clear()

    def _cfg(self):
        base = dict(RT.get('flag_inherit') or {})
        if self._cfg_override:
            base.update(self._cfg_override)
        return base

    def _prune(self, frame_idx, keep_frames, flagged):
        stale = [
            tid for tid, st in self._tracks.items()
            if (frame_idx - st['last_frame'] > keep_frames) and (tid not in flagged)
        ]
        for tid in stale:
            self._tracks.pop(tid, None)

    def step(self, frame_idx, tracks, work, board):
        cfg = self._cfg()
        if not cfg.get('enabled', True):
            return {}
        fidx = int(frame_idx)
        tracks = tracks or []
        current_tids = set()

        for rec in tracks:
            tid = int(rec['id'])
            current_tids.add(tid)
            box = [float(v) for v in rec['box']]
            reid_raw = extract_track_embedding(rec)
            # Fallback khi tracker không lộ embedding OSNet: histogram màu HSV vùng thân trên ảnh work
            hsv_raw = torso_hsv_histogram(work, box) if work is not None else None
            st = self._tracks.get(tid)
            if st is None:
                reid_vec = _ema_norm(None, reid_raw)
                hsv_vec = _ema_norm(None, hsv_raw)
                self._tracks[tid] = dict(
                    first_frame=fidx,
                    last_frame=fidx,
                    first_box=box,
                    box=box,
                    first_reid_feat=reid_vec,
                    reid_feat=reid_vec,
                    first_hsv_feat=hsv_vec,
                    hsv_feat=hsv_vec,
                )
            else:
                st['last_frame'] = fidx
                st['box'] = box
                st['reid_feat'] = _ema_norm(st['reid_feat'], reid_raw)
                st['hsv_feat'] = _ema_norm(st['hsv_feat'], hsv_raw)
                if st['first_reid_feat'] is None and st['reid_feat'] is not None:
                    st['first_reid_feat'] = st['reid_feat']
                if st['first_hsv_feat'] is None and st['hsv_feat'] is not None:
                    st['first_hsv_feat'] = st['hsv_feat']

        flagged = board.flagged_items()
        if not flagged or not current_tids:
            return {}

        max_gap_frames = max(0, int(round(float(cfg.get('max_gap_sec', 3.0)) * self.fps)))
        max_dist = float(cfg.get('max_center_dist_norm', 2.0))
        min_cos = float(cfg.get('reid_cos', 0.7))

        # Mỗi cờ (root_tid) chỉ gắn tối đa 1 track hoạt động tại một thời điểm
        active_roots = {info['root_tid'] for tid, info in flagged.items() if tid in current_tids}
        donors_by_root = {}
        for tid, info in flagged.items():
            root = info['root_tid']
            if root in active_roots or tid not in self._tracks:
                continue
            prev = donors_by_root.get(root)
            if prev is None or self._tracks[tid]['last_frame'] > self._tracks[prev]['last_frame']:
                donors_by_root[root] = tid

        candidates = [tid for tid in current_tids if tid not in flagged and tid in self._tracks]
        if not donors_by_root or not candidates:
            self._prune(fidx, max(300, max_gap_frames * 2), flagged)
            return {}

        scored = []
        for root, src_tid in donors_by_root.items():
            src_st = self._tracks[src_tid]
            src_last = src_st['last_frame']
            gap_now = fidx - src_last
            if gap_now < 1 or gap_now > max_gap_frames:
                continue
            acx, acy, ah = _box_center_h(src_st['box'])

            for cand_tid in candidates:
                cand_st = self._tracks[cand_tid]
                # Chỉ kế thừa cho track mới xuất hiện SAU khi track cũ mất
                appear_gap = cand_st['first_frame'] - src_last
                if appear_gap < 1 or appear_gap > max_gap_frames:
                    continue
                bcx, bcy, bh = _box_center_h(cand_st['first_box'])
                dist_norm = math.hypot(bcx - acx, bcy - acy) / max(0.5 * (ah + bh), 1.0)
                if dist_norm > max_dist:
                    continue

                cand_reid = cand_st['first_reid_feat'] if cand_st['first_reid_feat'] is not None else cand_st['reid_feat']
                cand_hsv = cand_st['first_hsv_feat'] if cand_st['first_hsv_feat'] is not None else cand_st['hsv_feat']
                if src_st['reid_feat'] is not None and cand_reid is not None and src_st['reid_feat'].shape == cand_reid.shape:
                    sim = cosine_similarity(src_st['reid_feat'], cand_reid)
                    method = 'reid'
                elif src_st['hsv_feat'] is not None and cand_hsv is not None:
                    sim = cosine_similarity(src_st['hsv_feat'], cand_hsv)
                    method = 'hsv_fallback'
                else:
                    continue

                if sim >= min_cos:
                    scored.append((sim, -dist_norm, root, src_tid, cand_tid, method, dist_norm, gap_now))

        scored.sort(reverse=True)
        used_roots, used_cands, inherited = set(), set(), {}
        for sim, _, root, src_tid, cand_tid, method, dist_norm, gap_now in scored:
            if root in used_roots or cand_tid in used_cands:
                continue
            if board.inherit(cand_tid, src_tid, frame=fidx):
                used_roots.add(root)
                used_cands.add(cand_tid)
                inherited[cand_tid] = dict(
                    src_tid=src_tid, root_tid=root, method=method,
                    sim=round(sim, 4), dist_norm=round(dist_norm, 4), gap_frames=gap_now,
                )

        self._prune(fidx, max(300, max_gap_frames * 2), flagged)
        return inherited


class GpuMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True, name='gpu-monitor')
        self.interval, self.samples, self._stop_ev = interval, [], threading.Event()

    def run(self):
        exe = shutil.which('nvidia-smi')
        if not exe:
            return
        while not self._stop_ev.is_set():
            try:
                out = subprocess.run([exe, '--query-gpu=index,utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
                t = time.perf_counter()
                for line in out.strip().splitlines():
                    i, u, m = [x.strip() for x in line.split(',')]
                    self.samples.append(dict(t=t, gpu=int(i), util=float(u), mem_mb=float(m)))
            except Exception:
                pass
            self._stop_ev.wait(self.interval)

    def stop(self):
        self._stop_ev.set()


class Stage(threading.Thread):
    def __init__(self, name, fn, in_q, out_q, ctx, flush=None, after=None):
        super().__init__(daemon=True, name=f'stage-{name}')
        self.sname, self.fn, self.in_q, self.out_q, self.ctx, self.flush = name, fn, in_q, out_q, ctx, flush
        self.after = after          # gọi sau khi frame đã chuyển tiếp (không tính vào thời gian xử lý của stage)
        self.busy_s, self.n = 0.0, 0

    def _emit(self, outs):
        for o in outs:
            o.t_ready = time.perf_counter()
            self.out_q.put(o)

    def run(self):
        failed = False
        while True:
            pkt = self.in_q.get()
            if pkt is EOS:
                if not failed and self.flush is not None:
                    try:
                        self._emit(self.flush() or [])
                        if self.after is not None:
                            self.after()
                    except Exception:
                        self.ctx.errors.append((self.sname, traceback.format_exc()))
                self.out_q.put(EOS)
                return
            if failed or self.ctx.stop.is_set():
                continue     # stage đã lỗi / người dùng bấm dừng: bỏ frame còn trong queue, vẫn rút queue để stage trước không bị kẹt
            try:
                t0 = time.perf_counter()
                outs = self.fn(pkt)
                t1 = time.perf_counter()
                self.ctx.rec.add(self.sname, pkt.idx, (t0 - pkt.t_ready) * 1e3, (t1 - t0) * 1e3)
                self.busy_s += t1 - t0
                self.n += 1
                self._emit([pkt] if outs is None else outs)
                if self.after is not None:
                    self.after()
            except Exception:
                failed = True
                self.ctx.errors.append((self.sname, traceback.format_exc()))
                self.ctx.stop.set()


class VideoRun:
    """Trạng thái của 1 lần chạy (1 video × 1 cấu hình model × 1 mode)."""

    def __init__(self, exp, key, d, M, vlm, mode, rt, art):
        self.exp, self.key, self.d, self.M, self.vlm, self.mode, self.rt = exp, key, d, M, vlm, mode, rt
        self.fps, self.n, self.S = d['fps'], d['n'], d['S']
        self.dir = settings.OUT / key
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        cap = self.n + 1 if mode == 'sequential' else rt['ring_seconds'] * self.fps
        self.store = FrameStore(cap, (d['NW'], d['NH']), rt['store_long'])
        self.Sst = self.store.size[0] / d['W']
        self.rec, self.board = LatencyRecorder(), AlarmBoard(persist=rt.get('alarm_persist', True))
        self.flagged = self.board.flagged
        self.stop, self.errors = threading.Event(), []
        self.tracks, self.t_cap, self.frame_log = {}, {}, []
        self.dropped = []
        self.last_frame = -1
        self.vlm_hold = threading.Event()   # đang dừng luồng video để đợi VLM (vlm_sync): reader không đọc frame mới
        self.pauses, self.last_pkt, self.last_alarm_n = [], None, 0
        self.art = art                      # mask hiện vật đã tính ở bước chuẩn bị (prepare_artifacts)
        self.person_filter = PersonTrackFilter(cfg=rt.get('person_filter'), work_width=d['W'])
        self.flag_inheritor = FlagInheritor(fps=self.fps, cfg=rt.get('flag_inherit'))
        self.inter = OnlineInteraction(self)
        if art.error:
            print(f'  {key}: không có hiện vật (segmentation lỗi) -> không có event chạm')
        self.artifact_by_id = {a['id']: a for a in art.artifacts}     # mọi cảnh (id duy nhất)
        self.scenes = []
        for sc in (art.scenes or [dict(start=0, ref_idx=0, artifacts=art.artifacts, masks=art.masks)]):
            union = np.zeros((d['H'], d['W']), bool)
            for mk in sc['masks'].values():
                union |= mk
            # khoảng cách (px, ảnh work) tới hiện vật gần nhất -> lọc người cần chạy pose/hand
            dist = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, 5) if union.any() else None
            self.scenes.append(dict(sc, union=union, dist=dist))
        self.scene_starts = [sc['start'] for sc in self.scenes]
        self._ref_idxs = {int(sc.get('ref_idx', sc.get('start', 0))) for sc in self.scenes}
        self.ref_frames = {}
        self.cur_scene = 0
        sc0 = self.scenes[0]
        self.inter.set_artifacts(
            sc0['artifacts'],
            sc0['masks'],
            ref_idx=sc0.get('ref_idx', sc0.get('start', 0)),
            scene_start=sc0.get('start', 0),
            ref_frame=sc0.get('ref_frame'),
        )
        self.sink_times = deque(maxlen=30)
        self.writer = None

    def pause_overlap(self, a, b):
        """Tổng thời gian dừng chờ VLM nằm trong khoảng [a, b] (perf_counter) - trừ ra khi tính FPS / latency."""
        return sum(max(0.0, min(e, b) - max(s, a)) for s, e in list(self.pauses))

    def vlm_sync_wait(self):
        """Gọi sau mỗi frame của stage interact: có đoạn đang chờ VLM -> dừng luồng đến khi VLM trả lời xong."""
        vlm = self.vlm
        if vlm.pending() <= 0:
            return
        t0 = time.perf_counter()
        deadline = t0 + float(self.rt.get('vlm_sync_timeout_s') or 180.0)
        self.vlm_hold.set()
        try:
            while vlm.pending() > 0 and not self.stop.is_set() and time.perf_counter() < deadline:
                time.sleep(0.03)
        finally:
            self.pauses.append((t0, time.perf_counter()))
            self.vlm_hold.clear()

    def scene_index(self, idx):
        return max(0, sum(s <= idx for s in self.scene_starts) - 1)

    def scene_at(self, idx):
        return self.scenes[self.scene_index(idx)]

    def capture_ref_frame(self, idx, native):
        """Giữ bản thu nhỏ (kích thước store) của frame tham chiếu mỗi cảnh để không bị ring buffer ghi đè."""
        if idx in self._ref_idxs and idx not in self.ref_frames:
            sw, sh = self.store.size
            nh, nw = native.shape[:2]
            self.ref_frames[idx] = (native.copy() if (nw, nh) == (sw, sh)
                                    else cv2.resize(native, (sw, sh), interpolation=cv2.INTER_AREA))

    def get_ref_frame(self, frame_idx):
        """Trả (ref_idx, img_store) của cảnh chứa frame_idx; trả (ref_idx, None) nếu chưa có ảnh."""
        sc = self.scene_at(frame_idx)
        ref_idx = int(sc.get('ref_idx', sc.get('start', 0)))
        img = self.ref_frames.get(ref_idx)
        if img is None:
            img = self.store.get(ref_idx)
        if img is None:
            img = sc.get('ref_frame')
        if img is not None:
            sw, sh = self.store.size
            if img.shape[1] != sw or img.shape[0] != sh:
                img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return ref_idx, img

    # ---- callback từ VLMWorker (thread VLM) ----
    def on_vlm_result(self, task, rec):
        decision, behavior, conf = decide(rec)
        rec['decision'] = decision
        if decision == 'ANOMALY':
            g = next((g for g in self.inter.tasks if g.get('task_id') == task['task_id']), None)
            self.board.add(dict(tid=task['tid'], behavior=behavior, conf=conf, segment=task['segment'],
                                shown_from=max(self.last_frame, 0), t_alarm=rec['t_done'],
                                artifact_ids=g['artifact_ids'] if g else []))
            hook = self.rt.get('alert_hook')
            if hook is not None:
                try:
                    hook(self, task, rec, behavior, conf)
                except Exception:
                    pass


def _stage_fns(ctx):
    M = ctx.M

    S = ctx.S                                   # px ảnh gốc / px ảnh làm việc

    def detect(p):
        # như make: detector chạy trên ảnh gốc (resize thẳng về test_size), box đổi về ảnh làm việc
        p.dets = [[b[0] / S, b[1] / S, b[2] / S, b[3] / S, b[4]] for b in M.detector.detect(p.native)]

    def track(p):
        tracks = M.tracker.update(p.dets, p.work, native=p.native, S=S)
        p.tracks = ctx.person_filter.filter_pre_pose(tracks, frame_idx=p.idx)

    gate = ctx.rt.get('pose_gate_norm_dist')

    def near_artifact(rec, dist):
        if dist is None:
            return False                      # không có hiện vật -> không thể có event chạm
        x1, y1, x2, y2 = rec['box']
        H, W = dist.shape
        xa, ya, xb, yb = max(0, int(x1)), max(0, int(y1)), min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
        if xb <= xa or yb <= ya:
            return False
        return float(dist[ya:yb, xa:xb].min()) <= gate * max(y2 - y1, 1.0)

    def pose(p):
        dist = ctx.scene_at(p.idx)['dist']
        recs = p.tracks if gate is None else [r for r in p.tracks if near_artifact(r, dist)]
        p.n_pose = len(recs)
        if recs:
            M.pose(p.native, recs, ctx.S)
        p.tracks = ctx.person_filter.filter_post_pose(p.tracks, posed_ids=[r['id'] for r in recs], frame_idx=p.idx)

    def hand(p):
        M.hand(p.native, p.tracks, ctx.S)
        p.native = None    # giải phóng ảnh gốc (ring buffer đã giữ bản thu nhỏ)

    def interact(p):
        k = ctx.scene_index(p.idx)
        if k != ctx.cur_scene:                # sang cảnh mới -> state machine dùng mask hiện vật của cảnh đó
            ctx.cur_scene = k
            sc = ctx.scenes[k]
            ctx.inter.set_artifacts(
                sc['artifacts'],
                sc['masks'],
                ref_idx=sc.get('ref_idx', sc.get('start', 0)),
                scene_start=sc.get('start', 0),
                ref_frame=sc.get('ref_frame'),
            )
        ctx.tracks[p.idx] = p.tracks
        ctx.last_frame = max(ctx.last_frame, p.idx)
        ctx.flag_inheritor.step(p.idx, p.tracks, p.work, ctx.board)
        p.states = ctx.inter.step(p.idx, p.tracks, work=p.work)

    def interact_flush():
        if ctx.last_frame >= 0:
            ctx.inter.flush(ctx.last_frame)
        return []

    def sink(p):
        t = time.perf_counter()
        ctx.sink_times.append(t)
        e2e_ms = (t - p.t_cap - ctx.pause_overlap(p.t_cap, t)) * 1e3
        ctx.frame_log.append(dict(frame=p.idx, t_cap=p.t_cap, t_out=t, e2e_ms=e2e_ms,
                                  n_people=len(p.tracks or []), n_pose=p.n_pose, n_contact=sum(s['state'] == 'CONTACT_LIKE' for s in p.states or [])))
        if ctx.rt['render_live'] or ctx.rt.get('frame_hook'):
            render_live(ctx, p, e2e_ms)
            ctx.last_pkt = types.SimpleNamespace(idx=p.idx, tracks=p.tracks, states=p.states, work=p.work, t_cap=p.t_cap)
            ctx.last_e2e = e2e_ms
        p.work = None
        return []     # stage cuối: không chuyển tiếp

    return [('detect', detect, None), ('track', track, None), ('pose', pose, None), ('hand', hand, None),
            ('interact', interact, interact_flush), ('sink', sink, None)]


def reader_loop(ctx, out_q):
    d, rt = ctx.d, ctx.rt
    from .source import is_stream, open_capture
    stream = is_stream(d['path'])
    realtime = rt['realtime'] and ctx.mode == 'parallel' and not stream      # camera: nguồn tự chạy theo nhịp thật, không giả lập
    cap = open_capture(d['path'])
    t0 = time.perf_counter()
    ctx.t_start = t0
    try:
        for idx in range(ctx.n):
            if ctx.stop.is_set():
                break
            if ctx.vlm_hold.is_set():              # đang đợi VLM: giữ nguyên, rồi dời mốc thời gian để không bị coi là trễ (không bỏ frame)
                tw = time.perf_counter()
                while ctx.vlm_hold.is_set() and not ctx.stop.is_set():
                    time.sleep(0.01)
                t0 += time.perf_counter() - tw
            if realtime:
                delay = t0 + idx / ctx.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            t_r0 = time.perf_counter()
            ok, native = cap.read()
            if not ok:
                break
            work = cv2.resize(native, (d['W'], d['H']))
            ctx.store.put(idx, native)
            ctx.capture_ref_frame(idx, native)
            t_r1 = time.perf_counter()
            ctx.t_cap[idx] = t_r0
            ctx.rec.add('read', idx, 0.0, (t_r1 - t_r0) * 1e3)
            pkt = Packet(idx=idx, t_cap=t_r0, t_ready=t_r1, work=work, native=native)
            if realtime and rt['drop_frames']:
                try:
                    out_q.put_nowait(pkt)
                except queue.Full:
                    ctx.dropped.append(idx)
            else:
                out_q.put(pkt)
    finally:
        cap.release()
        ctx.n_read = len(ctx.t_cap)
        if stream:                                 # camera dừng sớm: số khung thật (kết quả / timeline không được ghi hàng giờ)
            ctx.n = ctx.n_read
        out_q.put(EOS)


# ---------------- Video live (sink) ----------------
IDLE_PERSON, CONTACT_COLOR, ALARM_COLOR, IDLE_ARTIFACT = (0, 200, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)


def render_live(ctx, p, e2e_ms):
    vis = p.work.copy()
    union = ctx.scene_at(p.idx)['union']
    if union.any():
        vis[union] = (vis[union] * 0.7 + np.array(IDLE_ARTIFACT) * 0.3).astype(np.uint8)
    alarms = ctx.board.active(p.idx, int(ctx.rt['alarm_show_sec'] * ctx.fps), persist=ctx.rt.get('alarm_persist', True))
    ctx.last_alarm_n = len(ctx.board.alarms)
    contact = {s['track_id'] for s in (p.states or []) if s['state'] == 'CONTACT_LIKE'}
    for rec in p.tracks or []:
        tid = rec['id']
        col = ALARM_COLOR if tid in alarms else CONTACT_COLOR if tid in contact else IDLE_PERSON
        x1, y1, x2, y2 = map(int, rec['box'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if tid in alarms else 2)
        beh = alarms[tid].get('behavior') if tid in alarms else None
        txt = f'ID {tid}' + (f' | {str(beh).upper()}' if beh else '')
        put_label(vis, txt, (x1, y1 - 4), (0, 0, 0), 0.5, fg=col)
        if rec.get('kp') is not None:
            for hand in WRIST_IDX:
                pt, _ = hand_anchor_point(rec, hand)
                if pt is not None:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    st = ctx.sink_times
    span = st[-1] - st[0] - ctx.pause_overlap(st[0], st[-1]) if len(st) > 1 else 0.0
    live_fps = (len(st) - 1) / span if span > 0 else 0.0
    hook = ctx.rt.get('frame_hook')           # dashboard: nhận frame đã vẽ (không có dòng chữ thông số) + tracks
    if hook is not None:
        try:
            hook(ctx, p, vis, dict(fps=live_fps, e2e_ms=e2e_ms))
        except Exception:
            pass
    if not ctx.rt['render_live']:
        return
    vis = vis.copy()                           # dòng thông số chỉ ghi vào video lưu ra file, không vẽ lên frame của dashboard
    lines = [f"frame {p.idx}  t={p.idx / ctx.fps:.2f}s  [{ctx.mode}{' realtime' if ctx.rt['realtime'] else ''}]",
             f'pipeline {live_fps:.1f} FPS | e2e {e2e_ms:.0f} ms | dropped {len(ctx.dropped)}',
             f'VLM queue {ctx.vlm.pending()} | alarms {len(ctx.board.alarms)} | flagged {len(ctx.board.flagged)}']
    for k, text in enumerate(lines):
        put_label(vis, text, (8, 22 + 22 * k), (0, 0, 0), 0.55, fg=(0, 255, 255))
    if ctx.writer is None:
        ctx.live_tmp = ctx.dir / 'live_raw.mp4'
        ctx.live_size = (vis.shape[1] // 2 * 2, vis.shape[0] // 2 * 2)
        ctx.writer = cv2.VideoWriter(str(ctx.live_tmp), cv2.VideoWriter_fourcc(*'mp4v'), ctx.fps, ctx.live_size)
    frame = vis[:ctx.live_size[1], :ctx.live_size[0]]
    prev = getattr(ctx, 'live_prev', None)
    if prev is not None:                       # frame bị bỏ (realtime): lặp lại frame trước để video giữ đúng thời gian thật
        for _ in range(p.idx - prev[0] - 1):
            ctx.writer.write(prev[1])
    ctx.writer.write(frame)
    ctx.live_prev = (p.idx, frame)


def run_video(exp, key, d, M, vlm, art, mode='parallel', rt=None):
    """art: kết quả prepare_artifacts() của video này (segmentation đã chạy xong trước pipeline)."""
    rt = rt or RT
    ctx = VideoRun(exp, key, d, M, vlm, mode, rt, art)
    M.tracker.reset(ctx.fps)
    from . import prof
    prof.reset()
    for m in (M.tracker, M.pose, M.hand):                # đặt lại bộ đếm thời gian (cả ở tiến trình con)
        if hasattr(m, 'reset_prof'):
            m.reset_prof()
    for i in range(settings.N_GPU):
        reset_peak_mem(i)
    seq = mode == 'sequential'
    thr0 = (torch.get_num_threads(), cv2.getNumThreads())
    if rt.get('limit_cpu_threads'):
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
    fns = _stage_fns(ctx)
    qs = [queue.Queue(maxsize=0 if seq else rt['queue_size']) for _ in range(len(fns))] + [queue.Queue()]
    sync = bool(rt.get('vlm_sync')) and mode == 'parallel'     # tuần tự: VLM chạy sau cùng nên không có gì để đợi
    stages = [Stage(name, fn, qs[k], qs[k + 1], ctx, flush, after=ctx.vlm_sync_wait if sync and name == 'interact' else None)
              for k, (name, fn, flush) in enumerate(fns)]
    reader = threading.Thread(target=reader_loop, args=(ctx, qs[0]), daemon=True, name='reader')
    mon = GpuMonitor(RUN['gpu_monitor_interval']) if RUN['gpu_monitor'] else None
    if mon:
        mon.start()
    t0 = time.perf_counter()
    if seq:
        vlm.gate.clear()      # baseline tuần tự: VLM chỉ chạy sau khi perception xong cả video
        reader.start()
        reader.join()
        for st in stages:
            st.start()
            st.join()
        vlm.gate.set()
    else:
        vlm.gate.set()
        for st in stages:
            st.start()
        reader.start()
        reader.join()
        for st in stages:
            st.join()
    ctx.t_pipeline_done = time.perf_counter()
    if sync and ctx.last_pkt is not None:
        # đoạn cuối gửi VLM sát cuối video: đợi VLM trả lời xong rồi vẽ lại frame cuối với khung đỏ, để dashboard thấy cảnh báo ngay
        # (không đợi tới lúc xem lại)
        vlm.drain()
    if sync and ctx.last_pkt is not None:
        # luôn vẽ lại + mã hoá frame cuối (force): frame thật có thể bị giới hạn jpeg/s bỏ qua nên dashboard còn giữ ảnh cũ
        try:
            ctx.last_pkt.redraw = True
            render_live(ctx, ctx.last_pkt, getattr(ctx, 'last_e2e', 0.0))
        except Exception:
            import traceback
            print('!!! vẽ lại frame cuối lỗi:\n' + traceback.format_exc(), flush=True)
    torch.set_num_threads(thr0[0])
    cv2.setNumThreads(thr0[1])
    if ctx.stop.is_set() and not ctx.errors:
        vlm.cancel_pending()              # dừng theo yêu cầu: không đợi VLM xử lý hết các đoạn còn xếp hàng
    if RUN['wait_vlm_per_video']:
        vlm.drain()
    ctx.t_all_done = time.perf_counter()
    if mon:
        mon.stop()
        ctx.gpu_samples = [s for s in mon.samples if s['t'] >= t0]
    else:
        ctx.gpu_samples = []
    ctx.wall_s = ctx.t_pipeline_done - t0
    ctx.wall_with_vlm_s = ctx.t_all_done - t0
    ctx.stage_busy = {st.sname: st.busy_s for st in stages}
    ctx.gpu_peak_gb = {f'cuda:{i}': peak_mem_gb(i) for i in range(settings.N_GPU)}
    if ctx.writer is not None:
        ctx.writer.release()
        ctx.live_video = encode_browser_mp4(ctx.live_tmp, ctx.dir / 'live.mp4')
    if getattr(ctx.art, 'review', None) is not None:
        cv2.imwrite(str(ctx.dir / 'artifact_review.jpg'), ctx.art.review)
    if ctx.errors:
        for name, tb in ctx.errors:
            print(f'!!! stage {name} lỗi:\n{tb[-2000:]}')
        raise RuntimeError(f'{key}: pipeline lỗi ở stage {[n for n, _ in ctx.errors]}')
    finalize_video(ctx)
    return ctx


def finalize_video(ctx):
    """Sau khi VLM trả lời: gán quyết định cho từng event, tính đoạn báo động (như Phase 7)."""
    for e in ctx.inter.events:
        e.update(decision='CANDIDATE', behavior=None, vlm_conf=None, vlm_evidence=None, task=None, alarm_spans=[],
                 alarm_onset_source=None, t_alarm=None)
    for g in ctx.inter.tasks:
        r = ctx.vlm.results.get(g.get('task_id'))
        decision, behavior, conf = decide(r)
        parsed = (r or {}).get('parsed') or {}
        g.update(decision=decision, behavior=behavior, vlm_conf=conf, vlm_evidence=parsed.get('evidence'),
                 vlm_status=(r or {}).get('status', g.get('vlm_status', 'not_run')), t_alarm=(r or {}).get('t_done'))
        vidx = vlm_contact_indices(parsed, len(g.get('samples', [])))
        for e in g['events']:
            if 'eid' not in e:
                continue
            e.update(decision=decision, behavior=behavior, vlm_conf=conf, vlm_evidence=g['vlm_evidence'], task=g['name'],
                     t_alarm=g['t_alarm'])
            if decision == 'ANOMALY':
                lo, src = alarm_onset(e, g, vidx)
                hi = max(e['end_frame'], g['samples'][vidx[-1] - 1]) if vidx else e['end_frame']
                e['alarm_spans'] = [(lo, hi)] if lo <= hi else []
                e['alarm_onset_source'] = src
    rows = [{k: e.get(k) for k in ('eid', 'track_id', 'artifact_id', 'hand', 'start_frame', 'end_frame', 'decision',
                                   'behavior', 'vlm_conf', 'alarm_spans', 'alarm_onset_source', 'task', 'vlm_evidence')}
            for e in ctx.inter.events]
    pd.DataFrame(rows).to_csv(ctx.dir / 'events.csv', index=False, encoding='utf-8')
