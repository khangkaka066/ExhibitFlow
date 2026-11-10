"""Cầu nối giữa dashboard và pipeline ExhibitFlow.

- `State`: trạng thái dùng chung (frame mới nhất, track, alert, bộ đếm) cho server đọc.
- `PipelineRunner`: chạy `run_pipeline` trong 1 thread, gắn `frame_hook` / `alert_hook` vào engine.
- `DemoRunner`: nguồn giả (không cần GPU / trọng số) để xem thử giao diện.
"""
import json
import math
import os
import re
import shutil
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

SEVERITY = {'stealing': 'high', 'holding': 'medium', 'touching': 'low'}

DEFAULT_SETTINGS = dict(
    # --- model ---
    vlm_backend='llama.cpp',          # 'llama.cpp' | 'dry_run' (không gọi VLM, chỉ thử luồng)
    vlm_n_images=8, vlm_max_side=512, vlm_max_new_tokens=64, vlm_short_json=True,
    vlm_min_conf=None,
    # --- pipeline ---
    pose_gate_norm_dist=1.5, queue_size=8, max_frames=0, alarm_show_sec=4.0,
    limit_cpu_threads=False,            # True: torch / OpenCV chỉ 1 luồng mỗi stage (máy ít nhân như Kaggle 4 vCPU); máy nhiều nhân nên tắt
    hand_batch=True,                    # True: RTMW chạy 1 lượt cho mọi người trong frame (nhanh ~1.8x; vài điểm tin cậy thấp có thể lệch nhẹ so với chạy từng người)
    hand_process=False,                 # True: chạy RTMW hand ở tiến trình riêng (như tracker; +~0.5 GB VRAM)
    tracker_process=False,              # True: chạy tracker ở tiến trình riêng (thoát GIL; +~0.5 GB VRAM, nạp lại model khi đổi)
    realtime=False, drop_frames=True,   # realtime: đọc video theo nhịp fps gốc; drop_frames: stage đầu bận thì bỏ frame mới (giữ độ trễ thấp)
    weights_dir='', work_dir='work', use_cpu=False,
    pipeline_mode='full',             # 'tracking' | 'tracking_seg' (+ segmentation hiện vật, phát hiện chạm) | 'full' (+ VLM, cảnh báo)
    scene_mode='auto',                # 'auto' = tách cảnh nhanh (video có đổi camera) | 'first' = 1 cảnh cố định, bỏ qua tách cảnh
    vlm_prompt='', vlm_system_prompt='',   # trống = dùng prompt mặc định; {n} số ảnh, {dt} giây giữa 2 ảnh, {tid} ID người; { } khác viết {{ }}
    # --- map 2D: 4 điểm ảnh video (0-1) -> 4 điểm trên map (0-1) ---
    make_map=False,                   # True: tự tạo map 2D từ video khi phân tích (mất ~15-25 s lần đầu, cache theo video). Mặc định tắt
    map_image='auto',                 # 'auto' (tạo từ video khi phân tích; trống lúc đầu) | 'none' | tên file đã upload (hiệu chỉnh tay)
    homography=dict(src=[[0, 0], [1, 0], [1, 1], [0, 1]], dst=[[0, 0], [1, 0], [1, 1], [0, 1]]),
)


def overrides_from_settings(s):
    vlm = dict(n_images=int(s['vlm_n_images']), max_side=int(s['vlm_max_side']),
               max_new_tokens=int(s['vlm_max_new_tokens']), short_json=bool(s['vlm_short_json']))
    tracker = dict(backend='bytetrack_dma_ltc_proc' if s.get('tracker_process') else 'bytetrack_dma_ltc')
    from exhibitflow import settings
    hand = {}
    if settings.MODELS['hand'].get('backend') in ('rtmw', 'rtmw_proc'):      # không đè nếu đã đổi sang backend khác (vd 'none')
        hand = dict(backend='rtmw_proc' if s.get('hand_process') else 'rtmw', batch=bool(s.get('hand_batch', True)))
    ov = dict(models=dict(vlm=vlm, tracker=tracker, hand=hand),
              rt=dict(pose_gate_norm_dist=float(s['pose_gate_norm_dist']) or None, queue_size=int(s['queue_size']),
                      alarm_show_sec=float(s['alarm_show_sec']), realtime=bool(s.get('realtime')),
                      limit_cpu_threads=bool(s.get('limit_cpu_threads')),
                      drop_frames=bool(s.get('drop_frames', True))))
    if s.get('vlm_min_conf') not in (None, ''):
        ov['vlm_min_conf'] = float(s['vlm_min_conf'])
    if s.get('weights_dir'):
        ov['WEIGHT_ROOTS'] = [x.strip() for x in str(s['weights_dir']).split(',') if x.strip()]
    return ov


MODEL_KEYS = ('tracker_process', 'hand_process', 'hand_batch', 'vlm_backend', 'vlm_n_images', 'vlm_max_side', 'vlm_max_new_tokens', 'vlm_short_json', 'weights_dir',
              'use_cpu', 'work_dir')


def need_components(s):
    from exhibitflow.service import components_for
    return components_for(s.get('pipeline_mode', 'full'))


def model_key(s):
    return tuple(str(s.get(k)) for k in MODEL_KEYS)


def auto_map(s):
    return bool(s.get('make_map')) and s.get('map_image', 'auto') in ('auto', 'default', '')


def _settings_prompt(name):
    from exhibitflow import settings
    return getattr(settings, name)


def validate_prompt(text):
    """Thử format với số liệu mẫu để báo lỗi sớm (ngoặc { } sai, biến lạ) thay vì để VLM lỗi giữa lần chạy."""
    from exhibitflow import settings
    try:
        text.format(n=8, dt=0.5, tid=1)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f'Prompt không hợp lệ ({type(e).__name__}: {e}). Chỉ dùng biến {{n}} {{dt}} {{tid}}; dấu {{ }} khác phải viết {{{{ }}}}')
    return text


def apply_prompts(s):
    """Áp prompt từ cài đặt vào engine mỗi lần chạy (không cần nạp lại model); trống -> khôi phục prompt gốc."""
    from exhibitflow import settings
    pt = (s.get('vlm_prompt') or '').strip() or settings.DEFAULT_PROMPT_TEMPLATE
    sp = (s.get('vlm_system_prompt') or '').strip() or settings.DEFAULT_SYSTEM_PROMPT
    settings.apply_overrides(dict(PROMPT_TEMPLATE=validate_prompt(pt), SYSTEM_PROMPT=sp))


def models_info():
    """Tên model đang cấu hình (đọc từ settings.MODELS) để hiển thị trên dashboard / ghi vào run.json."""
    from exhibitflow import settings
    M = settings.MODELS
    d, t, p, h, sg, v = (M.get(k) or {} for k in ('detector', 'tracker', 'pose', 'hand', 'segmenter', 'vlm'))
    vname = (f"{str(v.get('repo_id', '')).split('/')[-1].replace('-GGUF', '')} {v.get('quant_file', '')}".strip()
             if v.get('backend') == 'llama_cpp' else v.get('backend'))
    return [dict(role='Detector', name='YOLOX', detail=f"{d.get('ckpt')} · {d.get('test_size')} · fp16={d.get('fp16')}"),
            dict(role='Tracker', name='ByteTrack-DMA-LTC',
                 detail=f"ReID {t.get('reid_model') if t.get('with_reid') else 'off'} · LTC {t.get('ltc_ckpt')} · DMA {t.get('dma_ckpt')}"),
            dict(role='Pose', name=str(p.get('weights') or p.get('backend')), detail=p.get('backend')),
            dict(role='Hand', name=f"RTMW {h.get('mode', '')}".strip() if h.get('backend') == 'rtmw' else str(h.get('backend')),
                 detail=h.get('backend')),
            dict(role='Segmenter · DINO', name=str(sg.get('dino')), detail='Grounding-DINO · fp16=%s' % sg.get('fp16')),
            dict(role='Segmenter · SAM', name=str(sg.get('sam')), detail='SAM 2.1'),
            dict(role='VLM', name=vname, detail=f"{v.get('repo_id', '')} · mmproj {v.get('mmproj', '')} · ctx {v.get('ctx')}")]


def homography(h):
    src = np.array(h['src'], np.float32)
    dst = np.array(h['dst'], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


class State:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'snapshots').mkdir(exist_ok=True)
        (self.root / 'uploads').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.settings_path = self.root / 'settings.json'
        self.settings = dict(DEFAULT_SETTINGS)
        if os.environ.get('EF_WEIGHTS_DIR'):
            self.settings['weights_dir'] = os.environ['EF_WEIGHTS_DIR']
        if self.settings_path.exists():
            try:
                self.settings.update(json.loads(self.settings_path.read_text(encoding='utf-8')))
            except Exception:
                pass
        if self.settings.get('map_image') in ('default', ''):    # cài đặt cũ (map mặc định CITIC) -> map tự tạo
            self.settings['map_image'] = 'auto'
        self.models_stage = lambda: ''                      # runner gắn hàm trả tiến độ nạp
        self.models_state, self.models_msg = 'idle', ''     # idle | loading | ready | error (độc lập với lần chạy video)
        self.map_version = 0                                # tăng mỗi khi map tự tạo đổi / bị xoá -> giao diện tải lại ảnh
        self.reset()

    def save_settings(self, new):
        for k in ('vlm_prompt',):
            if (new.get(k) or '').strip():
                validate_prompt(new[k])                        # ValueError -> server trả 400, không lưu
        with self.lock:
            for k, v in new.items():
                if k in DEFAULT_SETTINGS:
                    self.settings[k] = v
            self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding='utf-8')
        return self.settings

    def reset(self):
        with self.lock:
            self.status = 'idle'            # idle | loading | running | finishing | done | error
            self.phase = ''
            self.message = ''
            self.source = None
            self.frame_jpeg = None
            self.frame_seq = 0
            self.frame_size = (0, 0)
            self.people = []                # [{id, x, y, state}] toạ độ trên map (0-1)
            self.artifacts = []             # [{id, x, y}]
            self.alerts = []
            self.detections_total = 0
            self.frame_idx = 0
            self.n_frames = 0
            self.fps = 0.0
            self.e2e_ms = 0.0
            self.started_at = None
            self.cam_name = 'CAM 1'
            self.logs = deque(maxlen=200)
            self.last_bgr = None
            self.last_tracks = []
            self.stage_ms = {}
            self._t_jpeg = 0.0
            self.map_path, self.map_H, self.map_info = None, None, None   # map 2D tự tạo từ video + homography chuẩn hoá video -> map
            self.map_version += 1
            self.timeline = {}               # frame -> [[box...], [vị trí trên map...]] để xem lại + tua video
            self.scene_contours = []         # đường viền hiện vật theo cảnh (vẽ lại khi xem lại)
            self.run_id = None               # id lần chạy đã lưu (runs/<id>/)

    def log(self, msg):
        with self.lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def set_frame(self, bgr, tracks, size, idx, n, fps, e2e, force=False):
        now = time.time()
        with self.lock:
            self.frame_idx, self.n_frames, self.fps, self.e2e_ms = idx, n, fps, e2e
            self.last_bgr, self.last_tracks = bgr, tracks
        if not force and now - self._t_jpeg < 0.04:      # giới hạn ~25 jpeg/s; frame vẽ lại cuối (force) luôn được mã hoá
            return
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.frame_jpeg = buf.tobytes()
                self.frame_seq += 1
                self.frame_size = size
                self._t_jpeg = now

    def add_alert(self, a):
        with self.lock:
            a['id'] = len(self.alerts) + 1
            self.alerts.append(a)

    def snapshot(self):
        with self.lock:
            sev = {'high': 0, 'medium': 0, 'low': 0}
            for a in self.alerts:
                sev[a['severity']] += 1
            return dict(models=dict(state=self.models_state,
                                    msg=self.models_msg or (self.models_stage() if self.models_state == 'loading' else '')),
                        map=self.map_version, run=self.run_id, status=self.status, phase=self.phase, message=self.message, source=self.source,
                        cam=self.cam_name, frame=self.frame_idx, n_frames=self.n_frames, fps=round(self.fps, 1), stage_ms=self.stage_ms,
                        e2e_ms=round(self.e2e_ms), people=self.people, artifacts=self.artifacts,
                        alerts=list(reversed(self.alerts))[:200], n_alerts=len(self.alerts), severity=sev,
                        detections=self.detections_total, logs=list(self.logs)[-40:],
                        elapsed=round(time.time() - self.started_at) if self.started_at else 0)


# ------------------------------------------------------------------ pipeline thật
class PipelineRunner:
    def __init__(self, state: State, local=False):
        self.state, self.thread, self.ctx, self.cancel = state, None, None, threading.Event()
        self.local = local         # True: segmentation rồi mới tracking, không nạp mọi model lên GPU cùng lúc
        self.engine, self.loader, self.load_lock = None, None, threading.Lock()
        state.models_stage = lambda: getattr(self.engine, 'stage', '') if self.engine is not None else ''

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def loading(self):
        return self.loader is not None and self.loader.is_alive()

    # ---- nạp model (nền, ngay khi server lên) ----
    def preload(self):
        """Nạp model nếu chưa có / cấu hình model đã đổi. Không làm gì khi đang chạy hoặc đang nạp."""
        if self.running() or self.loading():
            return
        eng = self.engine
        if eng is not None and eng.key == model_key(self.state.settings) and eng.covers(need_components(self.state.settings)):
            return
        self.loader = threading.Thread(target=self._load, daemon=True, name='model-loader')
        self.loader.start()

    def _load(self):
        st = self.state
        with self.load_lock:
            s = dict(st.settings)
            key = model_key(s)
            need = need_components(s)
            if self.engine is not None and self.engine.key == key and self.engine.covers(need):
                return
            with st.lock:
                st.models_state, st.models_msg = 'loading', ''
            st.log('Nạp model (lần đầu có thể tải trọng số, mất vài phút)...')
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                if s.get('use_cpu'):
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                from exhibitflow.service import Engine
                if self.engine is None:
                    self.engine = Engine()
                self.engine.load(overrides=overrides_from_settings(s), work_dir=str(Path(s.get('work_dir') or 'work').resolve()),
                                 out_dir=str(st.root / 'results'), dry_run_vlm=(s.get('vlm_backend') == 'dry_run'), key=key,
                                 components=need, local=self.local)
                with st.lock:
                    st.models_state = 'ready'
                st.log('Model đã sẵn sàng')
            except BaseException as e:   # noqa: BLE001
                tb = traceback.format_exc()
                with st.lock:
                    st.models_state, st.models_msg = 'error', f'{type(e).__name__}: {e}'
                st.log('LỖI nạp model: ' + tb[-600:])
                print(tb, flush=True)

    def start(self, video_path):
        if self.running():
            raise RuntimeError('Pipeline đang chạy')
        if self.loading():
            raise RuntimeError('Model đang nạp, đợi vài phút rồi thử lại')
        self.state.reset()
        self.cancel.clear()
        self.state.source = Path(video_path).name
        self.state.status, self.state.phase, self.state.started_at = 'loading', 'Chuẩn bị', time.time()
        self.thread = threading.Thread(target=self._run, args=(str(video_path),), daemon=True, name='pipeline')
        self.thread.start()

    def stop(self):
        self.cancel.set()
        ctx = self.ctx
        if ctx is not None:
            ctx.stop.set()
        st = self.state
        with st.lock:
            if st.status == 'running':
                st.status, st.phase = 'finishing', 'Đang dừng...'
        st.log('Yêu cầu dừng...')

    # ---- hook trong engine ----
    def _frame_hook(self, ctx, p, vis, perf):
        st = self.state
        self.ctx = ctx
        if self.cancel.is_set():
            ctx.stop.set()
        if st.status == 'loading':
            st.status, st.phase = 'running', 'Đang phân tích'
            st.cam_name = ctx.key
            st.log(f'Bắt đầu video {ctx.key}: {ctx.n} frame @ {ctx.fps:.1f}fps')
        H = homography(st.settings['homography'])
        if auto_map(st.settings):                  # map tự tạo: dùng homography ước lượng từ video (chưa có -> đồng nhất)
            H = np.array(st.map_H, np.float64) if st.map_H is not None else np.eye(3)
        W, Hh = vis.shape[1], vis.shape[0]
        people = []
        pts = []
        for r in p.tracks or []:
            x1, y1, x2, y2 = r['box']
            pts.append([[((x1 + x2) / 2) / W, y2 / Hh]])
        contact = {s['track_id'] for s in (p.states or []) if s['state'] == 'CONTACT_LIKE'}
        if pts:
            proj = cv2.perspectiveTransform(np.array(pts, np.float32), H)
            alarms = ctx.board.active(p.idx, int(ctx.rt['alarm_show_sec'] * ctx.fps))
            for r, q in zip(p.tracks, proj):
                tid = r['id']
                state = 'alert' if tid in alarms else 'contact' if tid in contact else 'idle'
                people.append(dict(id=int(tid), x=float(q[0][0]), y=float(q[0][1]), state=state))
        # ghi lại để xem lại / tua: box (0-1 theo khung hình) + trạng thái chạm + vị trí trên map. Khung đỏ tính lúc xem lại từ
        # đoạn báo động của event (VLM trả lời trễ hơn lúc chạm, nên không ghi màu đỏ tại thời điểm chạy)
        st.timeline[int(p.idx)] = [
            [[int(r['id']), *(round(float(v), 4) for v in (r['box'][0] / W, r['box'][1] / Hh, r['box'][2] / W, r['box'][3] / Hh)),
              1 if r['id'] in contact else 0] for r in p.tracks or []],
            [[q['id'], round(q['x'], 4), round(q['y'], 4)] for q in people]]
        if not st.scene_contours and ctx.scenes:
            for sc in ctx.scenes:
                arts = []
                for aid, mk in sc['masks'].items():
                    cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polys = [[[round(float(x) / W, 4), round(float(y) / Hh, 4)] for x, y in cv2.approxPolyDP(c, 2.0, True)[:, 0]]
                             for c in sorted(cs, key=cv2.contourArea, reverse=True)[:3] if len(c) >= 3]
                    arts.append(dict(id=int(aid), c=polys))
                st.scene_contours.append(dict(start=int(sc['start']), arts=arts))
        if not st.artifacts and ctx.scenes:
            arts = []
            for aid, mk in ctx.scenes[0]['masks'].items():
                ys, xs = np.nonzero(mk)
                if len(xs):
                    q = cv2.perspectiveTransform(np.array([[[xs.mean() / W, ys.max() / Hh]]], np.float32), H)[0][0]
                    arts.append(dict(id=int(aid), x=float(q[0]), y=float(q[1])))
            st.artifacts = arts
        if p.idx % 20 == 0:                       # ms trung bình mỗi stage trong ~40 frame gần nhất -> thấy nút thắt
            acc = {}
            for stage, _, _, ms in list(ctx.rec.rows)[-300:]:
                acc.setdefault(stage, []).append(ms)
            st.stage_ms = {k: round(sum(v) / len(v)) for k, v in acc.items() if k != 'read'}
        st.people = people
        st.detections_total += len(p.tracks or [])
        st.set_frame(vis, [dict(id=r['id'], box=list(map(float, r['box']))) for r in p.tracks or []],
                     (W, Hh), p.idx, ctx.n, perf['fps'], perf['e2e_ms'],
                     force=bool(getattr(p, 'redraw', False)))

    def _alert_hook(self, ctx, task, rec, behavior, conf):
        st = self.state
        behavior = str(behavior or 'touching').lower()
        tid = int(task['tid'])
        art_ids = [int(a) for a in (task.get('artifact_ids') or [])]
        with st.lock:
            bgr, tracks = (st.last_bgr.copy() if st.last_bgr is not None else None), list(st.last_tracks)
        person_jpg = art_jpg = None
        if bgr is not None:
            h, w = bgr.shape[:2]
            tr = next((t for t in tracks if t['id'] == tid), None)
            if tr:
                x1, y1, x2, y2 = [int(v) for v in tr['box']]
                person_jpg = self._crop(bgr, x1, y1, x2, y2)
            for aid in art_ids[:1]:
                mk = None
                for sc in ctx.scenes:
                    mk = sc['masks'].get(aid)
                    if mk is not None:
                        break
                if mk is not None and mk.any():
                    ys, xs = np.nonzero(mk)
                    art_jpg = self._crop(bgr, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        aid_ = len(st.alerts) + 1
        files = {}
        for name, jpg in (('person', person_jpg), ('artifact', art_jpg)):
            if jpg is not None:
                fn = f'alert{aid_:04d}_{name}.jpg'
                (st.root / 'snapshots' / fn).write_bytes(jpg)
                files[name] = f'/snapshots/{fn}'
        parsed = rec.get('parsed') or {}
        st.add_alert(dict(time=time.strftime('%H:%M:%S'), video_time=round(st.frame_idx / max(ctx.fps, 1), 1),
                          camera=ctx.key, track_id=tid, artifact_ids=art_ids, behavior=behavior,
                          confidence=round(float(conf or 0), 2), severity=SEVERITY.get(behavior, 'low'),
                          evidence=parsed.get('evidence') or '', person_img=files.get('person'),
                          artifact_img=files.get('artifact')))
        st.log(f'ALERT {behavior.upper()} — người ID {tid}, hiện vật {art_ids} ({conf})')

    @staticmethod
    def _crop(img, x1, y1, x2, y2, pad=0.12):
        h, w = img.shape[:2]
        pw, ph = int((x2 - x1) * pad) + 4, int((y2 - y1) * pad) + 4
        x1, y1, x2, y2 = max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        c = img[y1:y2, x1:x2]
        s = 160 / max(c.shape[:2])
        if s < 1:
            c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def _save_run(self, video, res, s):
        """Lưu mỗi lần phân tích: runs/<id>/run.json (thông tin video, latency từng module + VLM), timeline.json (xem lại / tua), map.png."""
        st = self.state
        v, perf, cfg = res['video'], res.get('performance') or {}, res.get('config') or {}
        rid = time.strftime('%Y%m%d_%H%M%S') + '_' + re.sub(r'[^\w.-]+', '_', Path(video).stem)[:40]
        d = st.root / 'runs' / rid
        d.mkdir(parents=True, exist_ok=True)
        def span(x):                                           # result.json: dict(start_frame, end_frame); engine: (lo, hi)
            return [int(x['start_frame']), int(x['end_frame'])] if isinstance(x, dict) else [int(x[0]), int(x[1])]

        alerts = []
        for e in res.get('events', []):
            if e.get('decision') != 'ANOMALY':
                continue
            try:
                spans = [span(x) for x in (e.get('alarm_spans') or [])]
            except Exception:
                spans = []
            alerts.append(dict(tid=int(e['track_id']), behavior=e.get('behavior'), conf=e.get('vlm_confidence'),
                               artifact_id=e.get('artifact_id'), hand=e.get('hand'), start=int(e['start_frame']),
                               end=int(e['end_frame']), evidence=e.get('vlm_evidence'), spans=spans))
        has_map = bool(st.map_path and Path(st.map_path).is_file())
        if has_map:
            shutil.copy(st.map_path, d / 'map.png')
        try:
            size_mb = round(Path(video).stat().st_size / 1e6, 1)
        except OSError:
            size_mb = None
        with st.lock:
            frames, artifacts, scenes = dict(st.timeline), list(st.artifacts), list(st.scene_contours)
            alerts_ui = list(st.alerts)
        keep = ('pipeline_mode', 'scene_mode', 'max_frames', 'queue_size', 'pose_gate_norm_dist', 'vlm_backend', 'vlm_n_images',
                'vlm_max_side', 'vlm_max_new_tokens', 'map_image')
        info = dict(id=rid, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    video=dict(name=v.get('name'), path=str(Path(video).resolve()), size_mb=size_mb, fps=v.get('fps'),
                               n_frames=v.get('n_frames'), width=v.get('width'), height=v.get('height'),
                               duration_s=round(v['n_frames'] / v['fps'], 2) if v.get('fps') else None),
                    mode=res.get('mode'), settings={k: s.get(k) for k in keep}, local=bool(self.local),
                    models=models_info(), prompt=dict(vlm=_settings_prompt('PROMPT_TEMPLATE'), system=_settings_prompt('SYSTEM_PROMPT')),
                    prepare=dict(res.get('prepare') or {}, map=st.map_info or dict(source='off')), load_times=cfg.get('load_times'),
                    environment=cfg.get('environment'), performance=perf, summary=res.get('summary'),
                    vlm_tasks=[{k: t.get(k) for k in ('task', 'track_id', 'status', 'decision', 'label', 'confidence', 'n_images',
                                                      'n_input_tokens', 'latency_s', 'infer_s', 'prefill_s', 'queue_wait_s', 'error')}
                               for t in res.get('vlm_tasks', [])],
                    alerts=alerts, alerts_ui=alerts_ui, has_map=has_map)
        (d / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        (d / 'timeline.json').write_text(json.dumps(dict(fps=v.get('fps'), n_frames=v.get('n_frames'), width=v.get('width'),
                                                         height=v.get('height'), frames=frames, alerts=alerts, artifacts=artifacts,
                                                         scenes=scenes), separators=(',', ':'), default=str), encoding='utf-8')
        with st.lock:
            st.run_id = rid
        st.log(f'Đã lưu kết quả lần chạy: {d}')

    def _make_map(self, video):
        """Tạo map 2D từ video bằng multicam-2d-floorplan-tracker (tiến trình riêng, CPU). Lỗi -> map trống, pipeline vẫn chạy."""
        st = self.state
        try:
            from dashboard import mapgen
            t0 = time.time()
            vp = Path(video)
            from exhibitflow.common import video_fingerprint   # khoá theo nội dung video: upload lại cùng video thì dùng lại map
            out = st.root / 'maps' / f'v{video_fingerprint(vp)}'
            if (out / 'map.png').is_file() and (out / 'map_info.json').is_file():
                info, src = json.loads((out / 'map_info.json').read_text()), 'cache'
                st.log('Dùng lại map 2D đã tạo cho video này')
            else:
                st.log('Tạo map 2D từ video...')
                info, src = mapgen.generate(video, out, log=st.log), 'computed'
            with st.lock:
                st.map_path, st.map_H = out / 'map.png', info['homography']
                st.map_info = dict(method=info['method'], source=src, seconds=round(time.time() - t0, 2))
                st.map_version += 1
            st.log(f"Map 2D sẵn sàng ({info['method']}, {src})")
        except BaseException as e:   # noqa: BLE001
            st.log(f'Không tạo được map 2D: {type(e).__name__}: {e}')

    def _run(self, video):
        st = self.state
        map_thread = None
        try:
            s = st.settings
            apply_prompts(s)                           # prompt VLM theo cài đặt (đọc lúc chạy, không cần nạp lại model)
            if auto_map(s):                            # chạy song song với nạp model
                map_thread = threading.Thread(target=self._make_map, args=(video,), daemon=True, name='map-gen')
                map_thread.start()
            eng = self.engine
            if eng is None or eng.key != model_key(s) or not eng.covers(need_components(s)):   # cấu hình đổi / thiếu thành phần -> nạp lại
                st.phase = 'Nạp model'
                self._load()
                eng = self.engine
                if eng is None or not eng.ready:
                    raise RuntimeError(st.models_msg or 'Model nạp lỗi')
            from exhibitflow import settings as efs
            efs.apply_overrides(overrides_from_settings(s))                # tham số chạy (pose gate, queue, ngưỡng...)
            efs.RT['scene_step_sec'] = 0.2                                 # tách cảnh thưa: so sánh mỗi 0.2s (cảnh luôn dài > 1s)
            efs.RT['seg_ref_default'] = 'first' if s.get('scene_mode') == 'first' else 'auto'
            eng.on_stage = lambda m: setattr(st, 'phase', m)
            efs.RT['frame_hook'] = self._frame_hook
            efs.RT['alert_hook'] = self._alert_hook
            if map_thread is not None:                 # cần homography trước khi chiếu người lên map
                st.phase = 'Tạo map 2D'
                map_thread.join()
            res = eng.run(video, max_frames=int(s.get('max_frames') or 0) or None, mode=s.get('pipeline_mode', 'full'))
            try:
                self._save_run(video, res, s)
            except Exception:                                  # lưu lịch sử lỗi không được làm hỏng lần chạy
                st.log('Không lưu được kết quả lần chạy: ' + traceback.format_exc()[-300:])
            with st.lock:
                st.status, st.phase = 'done', 'Hoàn tất'
            st.log(f"Xong: {res['summary']['n_alerts']} cảnh báo. Kết quả: {st.root / 'results'}")
        except BaseException as e:   # noqa: BLE001
            tb = traceback.format_exc()
            with st.lock:
                st.status, st.message = 'error', f'{type(e).__name__}: {e}'
            st.log('LỖI: ' + tb[-600:])
            print(tb, flush=True)
        finally:
            try:
                from exhibitflow import settings as efs
                efs.RT.pop('frame_hook', None)
                efs.RT.pop('alert_hook', None)
            except Exception:
                pass


# ------------------------------------------------------------------ demo (không cần GPU)
class DemoRunner(PipelineRunner):
    """Sinh frame giả: vài người đi quanh 3 hiện vật; thỉnh thoảng có alert. Chỉ để xem giao diện."""

    def preload(self):
        self.state.models_state = 'ready'

    def start(self, video_path):
        if self.running():
            raise RuntimeError('Pipeline đang chạy')
        self.state.reset()
        self.cancel.clear()
        self.state.source = Path(video_path).name if video_path else 'demo.mp4'
        self.state.status, self.state.phase, self.state.started_at = 'running', 'Demo (dữ liệu giả)', time.time()
        self.thread = threading.Thread(target=self._demo, daemon=True, name='demo')
        self.thread.start()

    def _demo(self):
        st = self.state
        W, H, n, fps = 960, 540, 900, 25.0
        arts = [(0.25, 0.55, 'Tượng đồng'), (0.55, 0.50, 'Bình gốm'), (0.8, 0.6, 'Tranh sơn dầu')]
        rng = np.random.default_rng(0)
        people = [dict(id=i + 1, ph=rng.uniform(0, 6.28), sp=rng.uniform(0.01, 0.025), cx=rng.uniform(.25, .75),
                       rx=rng.uniform(.15, .3)) for i in range(3)]
        st.cam_name = 'CAM 1'
        st.artifacts = [dict(id=i + 1, x=a[0], y=a[1]) for i, a in enumerate(arts)]
        st.log('Chế độ DEMO: dữ liệu giả lập, chưa chạy model thật')
        fake_ctx_fps = fps
        for idx in range(n):
            if self.cancel.is_set():
                break
            t0 = time.time()
            img = np.full((H, W, 3), (46, 40, 36), np.uint8)
            cv2.rectangle(img, (0, int(H * .62)), (W, H), (66, 58, 52), -1)
            for x, y, nm in arts:
                cv2.rectangle(img, (int(x * W) - 40, int(y * H) - 70), (int(x * W) + 40, int(y * H) + 10), (90, 130, 200), -1)
                cv2.putText(img, nm, (int(x * W) - 45, int(y * H) + 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 200), 1)
            ppl, tr = [], []
            for p in people:
                p['ph'] += p['sp']
                px = p['cx'] + p['rx'] * math.sin(p['ph'])
                py = 0.72 + 0.1 * math.sin(p['ph'] * 1.7 + p['id'])
                bw, bh = 60, 170
                x1, y1 = int(px * W - bw / 2), int(py * H - bh)
                near = any(abs(px - a[0]) < .08 and abs(py - a[1] - .15) < .12 for a in arts)
                col = (0, 0, 255) if near and idx % 300 > 200 else (0, 255, 255) if near else (0, 200, 0)
                cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), col, 2)
                cv2.putText(img, f"ID {p['id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
                ppl.append(dict(id=p['id'], x=px, y=py, state='alert' if col == (0, 0, 255) else 'contact' if near else 'idle'))
                tr.append(dict(id=p['id'], box=[x1, y1, x1 + bw, y1 + bh]))
            st.people = ppl
            st.detections_total += len(tr)
            st.set_frame(img, tr, (W, H), idx, n, 25.0, 40.0)
            if idx and idx % 300 == 250:
                beh = ['touching', 'holding', 'stealing'][(idx // 300) % 3]
                p = people[(idx // 300) % 3]
                x1 = int((p['cx'] + p['rx'] * math.sin(p['ph'])) * W) - 30
                crop = self._crop(img, x1, 120, x1 + 60, 400)
                fn = f'demo{len(st.alerts) + 1}.jpg'
                if crop:
                    (st.root / 'snapshots' / fn).write_bytes(crop)
                a = (idx // 300) % 3
                st.add_alert(dict(time=time.strftime('%H:%M:%S'), video_time=round(idx / fps, 1), camera='CAM 1',
                                  track_id=p['id'], artifact_ids=[a + 1], behavior=beh, confidence=0.8,
                                  severity=SEVERITY[beh], evidence='(demo) người đưa tay vào hiện vật',
                                  person_img=f'/snapshots/{fn}' if crop else None, artifact_img=None))
                st.log(f"ALERT {beh.upper()} — người ID {p['id']}, hiện vật {a + 1} (demo)")
            time.sleep(max(0, 1 / fake_ctx_fps - (time.time() - t0)))
        with st.lock:
            st.status, st.phase = 'done', 'Hoàn tất (demo)'
