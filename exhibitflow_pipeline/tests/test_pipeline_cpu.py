"""Kiểm thử end-to-end trên CPU, không cần trọng số / GPU / model thật.

Video tổng hợp: một "người" (hình chữ nhật xanh) đi tới sát hiện vật và đứng đó. Detector / pose / segmenter giả,
tracker là IoUTracker thật, VLM là dry_run hoặc llama-server giả. Kiểm tra toàn bộ đường đi thật của engine:
reader -> detect -> track -> pose -> interact (state machine, gom event, cắt ảnh) -> VLM -> quyết định -> result.json.

Chạy:  CUDA_VISIBLE_DEVICES= python tests/test_pipeline_cpu.py      (hoặc pytest tests/)
"""
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from exhibitflow import settings  # noqa: E402
from exhibitflow.common import register  # noqa: E402

W, H, FPS, N = 960, 540, 25, 150
ART = (560, 300, 640, 420)                         # hiện vật (x1, y1, x2, y2) trên ảnh 960x540


def person_box(i):
    """Người đi từ x=100 tới x=430 (frame 20-60) rồi đứng yên tới frame 130."""
    if i < 20 or i > 130:
        return None
    x1 = 100 + min(i - 20, 40) * (330 / 40)
    return [x1, 200, x1 + 80, 420]


def make_video(path):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    for i in range(N):
        f = np.full((H, W, 3), 90, np.uint8)
        cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (200, 200, 200), -1)
        b = person_box(i)
        if b:
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), -1)
        wr.write(f)
    wr.release()


# ---------------- backend giả ----------------
@register('detector', 'fake')
class FakeDetector:
    def __init__(self, **_):
        pass

    def detect(self, frame):
        m = cv2.inRange(frame, (0, 150, 0), (80, 255, 80))
        ys, xs = np.nonzero(m)
        if len(xs) < 200:
            return []
        return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9]]


@register('pose', 'fake')
class FakePose:
    def __init__(self, **_):
        pass

    def __call__(self, native, recs, S):
        for r in recs:
            x1, y1, x2, y2 = r['box']
            cx, h = (x1 + x2) / 2, y2 - y1
            kp = [[cx, y1 + 10]] * 5 + [[cx - 25, y1 + .2 * h], [cx + 25, y1 + .2 * h], [cx - 30, y1 + .4 * h], [x2 + 10, y1 + .4 * h],
                                          [cx - 30, y1 + .6 * h], [x2 + 15, y1 + .5 * h], [cx - 15, y1 + .55 * h], [cx + 15, y1 + .55 * h],
                                          [cx - 15, y1 + .75 * h], [cx + 15, y1 + .75 * h], [cx - 15, y2], [cx + 15, y2]]
            r['kp'], r['kc'] = kp, [1.0] * 17


@register('segmenter', 'fake')
class FakeSegmenter:
    def __init__(self, **_):
        pass

    def warmup(self):
        pass

    def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
        mask = np.zeros(frame.shape[:2], bool)
        mask[ART[1]:ART[3], ART[0]:ART[2]] = True
        return [dict(id=0, box=[float(v) for v in ART], label='statue', score=0.9, pedestal=False, fallback=False)], {0: mask}, []

    def manual(self, *a):
        raise NotImplementedError

    def release(self):
        pass


FAKE_MODELS = dict(detector=dict(backend='fake'), pose=dict(backend='fake'), hand=dict(backend='none', device='cpu'),
                   segmenter=dict(backend='fake'), tracker=dict(backend='iou', iou_thr=0.3))


def _run(tmp, vlm_spec):
    video = Path(tmp) / 'museum_test.avi'
    make_video(video)
    from exhibitflow import run_pipeline
    out = Path(tmp) / 'out'
    summary = run_pipeline([str(video)], out_dir=out, work_dir=Path(tmp) / 'work',
                           overrides=dict(models=dict(FAKE_MODELS, vlm=vlm_spec), rt=dict(queue_size=4, pose_gate_norm_dist=None)))
    return summary, out


def _check_result(summary, out):
    assert not summary['errors'], summary['errors']
    assert summary['n_videos'] == 1 and len(summary['videos']) == 1
    res = json.loads((out / 'museum_test' / 'result.json').read_text(encoding='utf-8'))
    assert (out / 'summary.json').exists() and (out / 'museum_test' / 'events.csv').exists()
    assert res['schema_version'] == '1.0' and res['video']['n_frames'] == N and res['video']['fps'] == FPS
    assert len(res['artifacts']) == 1 and res['events'], 'phải có event chạm'
    ev = res['events'][0]
    assert ev['artifact_id'] == 0 and ev['end_frame'] - ev['start_frame'] >= settings.CFG['event_min_frames']
    assert ev['decision'] == 'ANOMALY' and ev['behavior'] == 'touching', ev
    assert res['alerts'] and res['alerts'][0]['behavior'] == 'touching' and res['summary']['has_anomaly']
    assert res['alerts'][0]['alarm_start_time_s'] is not None
    t = res['vlm_tasks'][0]
    assert t['status'] == 'ok' and t['label'] == 'touching' and t['n_input_tokens'] is not None
    assert res['performance']['frames_processed'] == N and res['performance']['stages']['detect']['mean_ms'] >= 0
    assert (out / 'museum_test' / 'vlm_inputs').is_dir()
    return res


def test_end_to_end_dry_run_vlm():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        summary, out = _run(tmp, dict(backend='dry_run', delay_s=0.0))
        _check_result(summary, out)


def test_end_to_end_fake_llama_server():
    """VLM thật của sản phẩm (backend llama_cpp, n_images=8, max_side=512, short_json) nói chuyện với llama-server giả."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        shim = Path(tmp) / 'bin'
        shim.mkdir()
        if os.name == 'nt':
            (shim / 'llama-server.bat').write_text(f'@echo off\r\n"{sys.executable}" "{HERE / "fake_llama_server.py"}" %*\r\n')
        else:
            p = shim / 'llama-server'
            p.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_llama_server.py"}" "$@"\n')
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        os.environ['PATH'] = str(shim) + os.pathsep + os.environ['PATH']
        vlm = dict(backend='llama_cpp', model_path='fake.gguf', mmproj_path='fake-mmproj.gguf', device='cpu', port=8097,
                   startup_timeout=60, n_images=8, max_side=512, short_json=True, max_new_tokens=64)
        summary, out = _run(tmp, vlm)
        res = _check_result(summary, out)
        assert res['vlm_tasks'][0]['evidence'] == '9 images', res['vlm_tasks'][0]       # 1 BEFORE + 8 trong 12 ảnh mẫu
        seg_dir = next((out / 'museum_test' / 'vlm_inputs').iterdir())
        assert any(p.name.startswith('00_before_') for p in (seg_dir / 'marked_512').glob('*.jpg'))
        sizes = {cv2.imread(str(p)).shape[:2] for p in (seg_dir / 'marked_512').glob('*.jpg')}
        assert sizes and all(max(s) <= 512 for s in sizes), sizes


def test_camera_stream_source():
    """Nguồn camera (.efstream -> LiveCapture): phát file như camera. Kiểm tra: mở được, segment trên khung đầu, ra event + cảnh báo,
    và số khung trong kết quả là số khung THẬT đã đọc (không phải giới hạn thời lượng tối đa)."""
    from exhibitflow import run_pipeline, source
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / 'cam_src.avi'
        make_video(video)
        stub = source.write_stream_file(Path(tmp) / 'cam_test.efstream', url=str(video))
        out = Path(tmp) / 'out'
        summary = run_pipeline([str(stub)], out_dir=out, work_dir=Path(tmp) / 'work',
                               overrides=dict(models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0)),
                                              rt=dict(queue_size=4, pose_gate_norm_dist=None)))
        assert not summary['errors'], summary['errors']
        res = json.loads((out / 'cam_test' / 'result.json').read_text(encoding='utf-8'))
        n = res['video']['n_frames']
        assert 0.8 * N <= n <= N, f'n_frames phải là số khung thật đã đọc (~{N}), nhận {n}'
        assert res['video']['width'] == W and res['video']['fps'] == FPS
        assert res['events'] and res['alerts'], 'camera: phải có event + cảnh báo như video file'
        assert source.INFO[str(stub)]['width'] == W


def test_multi_camera_sequence():
    """Multi-camera (MultiRun, chạy tuần tự) với engine thật: 2 camera = 2 video tổng hợp, thêm 1 camera lỗi.
    Kiểm tra: 2 camera đầu đều có kết quả, camera lỗi không làm hỏng các camera khác, cảnh báo gộp mang đúng nhãn camera."""
    from exhibitflow import run_pipeline
    from exhibitflow.multicam import MultiRun, from_result_alerts
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / 'out'
        ov = dict(models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0)), rt=dict(queue_size=4, pose_gate_norm_dist=None))
        paths = {}
        for name in ('cam_a', 'cam_b'):
            paths[name] = Path(tmp) / f'{name}.avi'
            make_video(paths[name])

        def launch(spec):
            summary = run_pipeline([str(spec['path'])], out_dir=out, work_dir=Path(tmp) / 'work', overrides=ov)
            if summary['errors']:
                raise RuntimeError(str(summary['errors'][0])[:200])
            return Path(spec['path']).stem

        def alerts_of(jid, final):
            if not final:
                return []
            p = out / jid / 'result.json'
            return from_result_alerts(json.loads(p.read_text(encoding='utf-8'))) if p.exists() else None

        specs = [dict(name='Cửa A', path=paths['cam_a']), dict(name='Cửa B', path=paths['cam_b']),
                 dict(name='Hỏng', path=Path(tmp) / 'khong_co.avi')]
        mr = MultiRun(specs, launch=launch, poll=lambda jid: dict(status='done', key=jid, error=None), stop=lambda jid: None,
                      alerts_of=alerts_of, interval=0.01).start()
        mr.thread.join(600)
        snap = mr.snapshot()
        assert snap['status'] == 'done', snap
        assert [c['status'] for c in snap['cams']] == ['done', 'done', 'error'], snap['cams']
        assert snap['cams'][2]['error'], snap['cams'][2]
        assert {a['camera'] for a in snap['alerts']} == {'Cửa A', 'Cửa B'}, snap['alerts']
        assert all(a['behavior'] == 'touching' and a['cam'] in (0, 1) for a in snap['alerts']), snap['alerts']
        assert [c['n_alerts'] for c in snap['cams']][:2] == [sum(1 for a in snap['alerts'] if a['cam'] == i) for i in (0, 1)]


def _parallel_cfg(tmp, name_dir, video):
    """Cấu hình một tiến trình camera (cam_worker): model giả, VLM dry_run, thư mục gốc riêng."""
    from dashboard.runner import DEFAULT_SETTINGS
    return dict(root=str(tmp / 'multi' / name_dir), video=str(video), demo=False, local=False,
                settings=dict(DEFAULT_SETTINGS, vlm_backend='dry_run', work_dir=str(tmp / 'work')),
                extra=dict(models=FAKE_MODELS, rt=dict(pose_gate_norm_dist=None, queue_size=4)),
                preload=['test_pipeline_cpu'], flock=str(tmp / 'load.lock'), stream_max_sec=None)


def _parallel_result_alerts(root, final, p_live):
    """Cảnh báo của một camera song song: đang chạy (từ tiến trình con) hoặc đã chốt (result.json trong thư mục gốc)."""
    from exhibitflow.multicam import from_live_alerts, from_result_alerts
    if not final:
        return from_live_alerts(p_live.snap()['alerts'])
    files = list((root / 'results').rglob('result.json'))
    if not files:
        return None
    return from_result_alerts(json.loads(max(files, key=lambda f: f.stat().st_mtime).read_text(encoding='utf-8')))


def test_parallel_cameras():
    """Multi-camera SONG SONG (exhibitflow/cam_worker.py): 3 camera tổng hợp, mỗi camera một tiến trình con, thêm 1 camera lỗi (không có file).
    Kiểm tra: (1) có lúc nhiều camera cùng đang chạy và cả 3 camera chạy chồng thời gian; (2) cảnh báo gộp mang đúng nhãn camera;
    (3) camera lỗi không làm hỏng các camera khác; (4) xong thì mọi tiến trình con đã thoát (không mồ côi)."""
    import time
    from exhibitflow import cam_worker
    from exhibitflow.multicam import MultiRun
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), os.environ.get('PYTHONPATH', '')])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        names = ['Cửa A', 'Cửa B', 'Cửa C']
        specs = []
        for i, nm in enumerate(names):
            v = tmp / f'cam{i}.avi'
            make_video(v)
            specs.append(dict(name=nm, dir=f'cam{i}', path=v))
        specs.append(dict(name='Hỏng', dir='bad', path=tmp / 'khong_co.avi'))
        procs, roots = [], {}

        def launch(spec):
            p = cam_worker.CamProc(_parallel_cfg(tmp, spec['dir'], spec['path']))
            procs.append(p)
            roots[p] = tmp / 'multi' / spec['dir']
            return p

        def poll(p):
            s = p.snap()
            v = cam_worker.view_of(s)
            return dict(status=v['status'], key=s['cam'], error=v['error'])

        mr = MultiRun(specs, launch=launch, poll=poll, stop=lambda p: p.stop(),
                      alerts_of=lambda p, final: _parallel_result_alerts(roots[p], final, p), interval=0.05,
                      parallel=3, release=lambda p: p.close())
        mr.start()
        peak, deadline = 0, time.time() + 900
        while mr.thread.is_alive() and time.time() < deadline:
            peak = max(peak, len(mr.snapshot()['running']))
            time.sleep(0.05)
        assert not mr.thread.is_alive(), 'multi-camera song song không kết thúc'
        snap = mr.snapshot()
        assert snap['status'] == 'done', snap['status']
        assert [c['status'] for c in snap['cams']] == ['done', 'done', 'done', 'error'], [c['status'] for c in snap['cams']]
        assert snap['cams'][3]['error'], snap['cams'][3]
        assert peak >= 2, f'không có camera nào chạy song song (peak={peak})'
        good = snap['cams'][:3]
        assert max(c['t0'] for c in good) < min(c['t1'] for c in good), 'ba camera không chạy chồng thời gian'
        assert {a['camera'] for a in snap['alerts']} == set(names), snap['alerts']
        assert all(a['camera'] == names[a['cam']] for a in snap['alerts']), snap['alerts']
        assert all(p.proc.poll() is not None for p in procs), 'còn tiến trình con chưa thoát'
        print(f'    peak={peak} camera chạy cùng lúc, cảnh báo: {len(snap["alerts"])}')


def test_parallel_stop_and_cleanup():
    """Dừng giữa chừng multi song song: camera chưa tới lượt bị bỏ qua, camera đang chạy xử lý nốt; không còn tiến trình con
    (kể cả khi server tắt: close_all). Tiến trình con bị diệt đột ngột thì báo lỗi ngay, không treo."""
    import signal
    import time
    from exhibitflow import cam_worker
    from exhibitflow.multicam import MultiRun
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), os.environ.get('PYTHONPATH', '')])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        specs, procs, roots = [], [], {}
        for i in range(3):
            v = tmp / f'cam{i}.avi'
            make_video(v)
            specs.append(dict(name=f'C{i}', dir=f'cam{i}', path=v))

        def launch(spec):
            p = cam_worker.CamProc(_parallel_cfg(tmp, spec['dir'], spec['path']))
            procs.append(p)
            roots[p] = tmp / 'multi' / spec['dir']
            return p

        def poll(p):
            s = p.snap()
            v = cam_worker.view_of(s)
            return dict(status=v['status'], key=s['cam'], error=v['error'])

        mr = MultiRun(specs, launch=launch, poll=poll, stop=lambda p: p.stop(),
                      alerts_of=lambda p, final: _parallel_result_alerts(roots[p], final, p), interval=0.05,
                      parallel=2, release=lambda p: p.close())
        mr.start()
        deadline = time.time() + 300
        while time.time() < deadline and len(mr.snapshot()['running']) < 2 and mr.thread.is_alive():
            time.sleep(0.05)
        mr.finish()
        mr.thread.join(900)
        assert not mr.thread.is_alive(), 'dừng giữa chừng mà multi không kết thúc'
        snap = mr.snapshot()
        assert snap['stop'] is True
        assert [c['status'] for c in snap['cams']] == ['done', 'done', 'skipped'], [c['status'] for c in snap['cams']]
        assert len(procs) == 2, 'camera chưa tới lượt không được khởi chạy'
        assert all(p.proc.poll() is not None for p in procs), 'còn tiến trình con sau khi dừng'

        # server tắt giữa chừng: close_all phải dọn mọi tiến trình con còn sống
        alive = [cam_worker.CamProc(_parallel_cfg(tmp, f'shut{i}', tmp / 'cam0.avi')) for i in range(2)]
        cam_worker.close_all()
        assert all(p.proc.poll() is not None for p in alive), 'close_all không dọn hết tiến trình con'

        # tiến trình con bị diệt đột ngột: lệnh phải báo lỗi ngay, không treo
        p = cam_worker.CamProc(_parallel_cfg(tmp, 'kill', tmp / 'cam0.avi'))
        os.kill(p.pid, signal.SIGKILL)
        t0, raised = time.time(), False
        try:
            p.snap()
        except Exception:  # noqa: BLE001 - kỳ vọng: lỗi kết nối / tiến trình đã thoát
            raised = True
        assert raised and time.time() - t0 < 30, 'tiến trình con đã chết mà lệnh không báo lỗi'
        p.close()


def test_alarm_board_persist_and_flagged():
    """AlarmBoard giữ cờ đỏ suốt video khi alarm_persist=True, vẫn giữ alarm_show_sec khi tắt, và lưu flagged."""
    from exhibitflow.engine import AlarmBoard

    assert settings.RT['alarm_persist'] is True
    board = AlarmBoard(persist=True)
    board.add(dict(tid=7, behavior='touching', conf=0.8, segment='s1', shown_from=20, t_alarm=1.0, artifact_ids=[0]))
    assert board.flagged == {7: {'behavior': 'touching', 'first_frame': 20}}
    assert 7 not in board.active(19, hold_frames=10)
    assert 7 in board.active(20, hold_frames=10)
    assert 7 in board.active(30, hold_frames=10)
    assert 7 in board.active(500, hold_frames=10)

    # Báo động lặp lại cùng track: giữ first_frame sớm nhất, cập nhật hành vi mới nhất
    board.add(dict(tid=7, behavior='stealing', conf=0.9, segment='s2', shown_from=45, t_alarm=2.0, artifact_ids=[0]))
    assert board.flagged[7] == {'behavior': 'stealing', 'first_frame': 20}
    assert board.active(500, hold_frames=10)[7]['behavior'] == 'stealing'

    # Khi tắt alarm_persist: hết hiệu lực sau hold_frames (= alarm_show_sec * fps)
    board_off = AlarmBoard(persist=False)
    board_off.add(dict(tid=3, behavior='holding', conf=0.7, segment='s1', shown_from=20, t_alarm=1.0, artifact_ids=[0]))
    assert 3 in board_off.active(25, hold_frames=10)
    assert 3 in board_off.active(30, hold_frames=10)
    assert 3 not in board_off.active(31, hold_frames=10)


def test_alarm_persist_render_live():
    """on_vlm_result + render_live: track bị gắn cờ giữ bbox đỏ sau alarm_show_sec khi alarm_persist=True và trở lại xanh khi False."""
    from types import SimpleNamespace
    from exhibitflow.engine import ALARM_COLOR, IDLE_PERSON, Packet, VideoRun, render_live

    old_paths = (settings.WORK, settings.OUT, settings.CACHE)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            settings.set_paths(work=tmp / 'work', out=tmp / 'out')
            d = dict(fps=25.0, n=150, S=1.0, NW=W, NH=H, W=W, H=H)
            art = SimpleNamespace(error=None, artifacts=[], scenes=[dict(start=0, ref_idx=0, artifacts=[], masks={})])
            vlm = SimpleNamespace(pending=lambda: 0)
            box = [100.0, 200.0, 180.0, 420.0]

            for persist, expected_late_color in ((True, ALARM_COLOR), (False, IDLE_PERSON)):
                frames = {}
                rt = dict(settings.RT, alarm_show_sec=0.2, alarm_persist=persist, render_live=False,
                          frame_hook=lambda _ctx, pkt, vis, _perf, out=frames: out.__setitem__(pkt.idx, vis))
                ctx = VideoRun('exp', f'v_{persist}', d, SimpleNamespace(), vlm, 'parallel', rt, art)
                ctx.last_frame = 10
                ctx.on_vlm_result(
                    dict(task_id='t1', tid=5, segment='seg1'),
                    dict(label='stealing', parsed=dict(label='stealing', confidence=0.95), t_done=1.2),
                )
                assert ctx.flagged[5] == {'behavior': 'stealing', 'first_frame': 10}
                for fidx in (10, 100):  # 100 > 10 + 0.2 * 25 (= 15)
                    pkt = Packet(idx=fidx, work=np.full((H, W, 3), 90, np.uint8),
                                 tracks=[dict(id=5, box=box)], states=[])
                    render_live(ctx, pkt, 1.0)
                # Kiểm tra màu đường viền đáy bbox (không bị đè bởi nhãn phía trên)
                assert tuple(frames[10][420, 140]) == ALARM_COLOR
                assert tuple(frames[100][420, 140]) == expected_late_color
    finally:
        settings.WORK, settings.OUT, settings.CACHE = old_paths


def test_alarm_persist_end_to_end():
    """End-to-end với run_pipeline: khi vlm_early_sec báo sớm, bbox vẫn đỏ tới cuối track nếu alarm_persist=True."""
    from exhibitflow import run_pipeline
    from exhibitflow.engine import ALARM_COLOR

    old_rt = dict(settings.RT)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'alarm_e2e.avi'
            make_video(video)

            for persist in (True, False):
                rendered = {}
                alarm_frames = []

                def on_alert(ctx, task, rec, behavior, conf):
                    alarm_frames.append(ctx.last_frame)

                def on_frame(ctx, pkt, vis, perf):
                    if pkt.tracks:
                        x1, y1, x2, y2 = map(int, pkt.tracks[0]['box'])
                        rendered[pkt.idx] = tuple(int(c) for c in vis[min(y2, H - 1), (x1 + x2) // 2])

                out = tmp / f'out_{persist}'
                summary = run_pipeline(
                    [str(video)], out_dir=out, work_dir=tmp / f'work_{persist}',
                    overrides=dict(
                        models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0, label='stealing')),
                        rt=dict(queue_size=4, pose_gate_norm_dist=None, vlm_early_sec=0.4,
                                alarm_show_sec=0.2, alarm_persist=persist,
                                alert_hook=on_alert, frame_hook=on_frame),
                    ),
                )
                assert not summary['errors'], summary['errors']
                assert alarm_frames, 'phải kích hoạt ít nhất 1 báo động sớm'
                hold_frames = int(0.2 * FPS)  # 5 frames
                late = [col for fidx, col in rendered.items() if fidx > alarm_frames[0] + hold_frames + 5]
                assert late, f'phải còn frame sau alarm_show_sec (alarm={alarm_frames[0]}, rendered_max={max(rendered)})'
                if persist:
                    assert all(c == ALARM_COLOR for c in late), f'alarm_persist=True nhưng bbox mất màu đỏ: {set(late)}'
                else:
                    assert all(c != ALARM_COLOR for c in late), f'alarm_persist=False nhưng bbox vẫn đỏ: {set(late)}'
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_gate_rejects_reasons_and_scene_tracking():
    """OnlineInteraction ghi nhận đầy đủ lý do loại ở contact_gate/pairing/close, theo track & cảnh, không đổi quyết định."""
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    class _DummyVlm:
        def submit(self, *_a, **_kw):
            raise AssertionError('không được gửi VLM khi mọi ứng viên đều bị loại')

    old_cfg = dict(settings.CFG)
    try:
        settings.CFG['relaxed_gate'] = False
        settings.CFG['ambiguous_policy'] = 'reject'
        ctx = SimpleNamespace(
            fps=25.0,
            n=120,
            last_frame=0,
            vlm=_DummyVlm(),
            cur_scene=0,
            scene_index=lambda fi: 0 if fi < 50 else 1,
        )
        inter = OnlineInteraction(ctx)

        def mk_rec(tid, box, rw, lw=None, r_conf=1.0, l_conf=0.0, head=None):
            x1, y1, x2, y2 = box
            cx, bh = (x1 + x2) / 2, y2 - y1
            hx, hy = head if head is not None else (cx, y1 + 0.08 * bh)
            kp = [[hx, hy]] * 5 + [
                [cx - 20, y1 + 0.2 * bh], [cx + 20, y1 + 0.2 * bh],
                [cx - 22, y1 + 0.4 * bh], [cx + 22, y1 + 0.4 * bh],
                lw or [cx - 20, y1 + 0.5 * bh], rw,
                [cx - 15, y1 + 0.55 * bh], [cx + 15, y1 + 0.55 * bh],
                [cx - 15, y1 + 0.75 * bh], [cx + 15, y1 + 0.75 * bh],
                [cx - 15, y2], [cx + 15, y2],
            ]
            kc = [1.0] * 17
            kc[9], kc[10] = l_conf, r_conf
            return dict(id=tid, box=list(box), kp=kp, kc=kc)

        def mk_mask(x1, y1, x2, y2):
            m = np.zeros((H, W), bool)
            m[y1:y2, x1:x2] = True
            return m

        # Cảnh 0, chưa có mask -> no_artifacts
        inter.step(0, [dict(id=1, box=[440.0, 200.0, 520.0, 420.0], kp=None, kc=None)])

        # Cảnh 0, có mask pedestal -> no_pose, no_hand_anchor, no_artifact_in_range, too_small, depth_mismatch, self_touch, arm_at_rest
        art0 = [dict(id=0, box=[560.0, 150.0, 640.0, 250.0], pedestal=True)]
        inter.set_artifacts(art0, {0: mk_mask(560, 150, 640, 250)})
        inter.step(1, [
            dict(id=2, box=[440.0, 200.0, 520.0, 420.0], kp=None, kc=None),                          # no_pose
            mk_rec(3, [10.0, 200.0, 90.0, 420.0], [50.0, 300.0], r_conf=0.0, l_conf=0.0),             # no_hand_anchor (x2)
            mk_rec(4, [10.0, 200.0, 70.0, 320.0], [40.0, 260.0], r_conf=1.0, l_conf=0.0),             # no_artifact_in_range (dist/120 > 3.0)
            mk_rec(5, [450.0, 170.0, 500.0, 230.0], [565.0, 200.0], r_conf=1.0, l_conf=0.0),          # too_small (h=60 < 80)
            mk_rec(6, [440.0, 200.0, 520.0, 420.0], [565.0, 200.0], r_conf=1.0, l_conf=0.0),          # depth_mismatch (foot 420 vs 250)
            mk_rec(7, [500.0, 30.0, 580.0, 250.0], [562.0, 152.0], r_conf=1.0, l_conf=0.0,           # self_touch (foot=250 khớp pedestal)
                   head=(560.0, 150.0)),
        ])
        # arm_at_rest trên mask không phải pedestal
        inter.set_artifacts([dict(id=0, box=[540.0, 300.0, 620.0, 400.0], pedestal=False)], {0: mk_mask(540, 300, 620, 400)})
        inter.step(2, [mk_rec(8, [510.0, 200.0, 590.0, 420.0], [560.0, 330.0], r_conf=1.0, l_conf=0.0)])

        # Cảnh 1: ambiguous (2 mask sát nhau) + event_too_short (chạm 9 frame < 12)
        art1 = [dict(id=10, box=[560.0, 300.0, 600.0, 420.0], pedestal=False),
                dict(id=11, box=[605.0, 300.0, 645.0, 420.0], pedestal=False)]
        inter.set_artifacts(art1, {10: mk_mask(560, 300, 600, 420), 11: mk_mask(605, 300, 645, 420)})
        inter.step(55, [mk_rec(9, [440.0, 200.0, 520.0, 420.0], [602.0, 350.0], r_conf=1.0, l_conf=0.0)])

        # Track 10: chạm thật nhưng ngắn (9 frame < event_min_frames=12) -> event_too_short
        inter.set_artifacts([dict(id=12, box=[560.0, 300.0, 640.0, 420.0], pedestal=False)], {12: mk_mask(560, 300, 640, 420)})
        for f in range(60, 69):
            inter.step(f, [mk_rec(10, [440.0, 200.0, 520.0, 420.0], [565.0, 350.0], r_conf=1.0, l_conf=0.0)])
        inter.flush(90)

        assert inter.events == [], f'không được tạo event khi đều bị loại/quá ngắn: {inter.events}'
        diag = inter.diagnostics()
        assert diag['gate_rejects']['1']['no_artifacts'] == 1
        assert diag['gate_rejects']['2']['no_pose'] == 1
        assert diag['gate_rejects']['3']['no_hand_anchor'] == 2
        assert diag['gate_rejects']['4']['no_artifact_in_range'] == 1
        assert diag['gate_rejects']['5']['too_small'] == 1
        assert diag['gate_rejects']['6']['depth_mismatch'] == 1
        assert diag['gate_rejects']['7']['self_touch'] == 1
        assert diag['gate_rejects']['8']['arm_at_rest'] == 1
        assert diag['gate_rejects']['9']['ambiguous'] == 1
        assert diag['gate_rejects']['10']['event_too_short'] == 1
        assert set(diag['gate_rejects_by_scene']) == {'0', '1'}
        assert 'too_small' in diag['gate_rejects_by_scene']['0']
        assert 'ambiguous' in diag['gate_rejects_by_scene']['1']
        assert 'event_too_short' in diag['gate_rejects_by_scene']['1']

        assert diag['tracks']['4']['near_frames'] == 0 and diag['tracks']['4']['min_norm_dist'] > 3.0
        assert diag['tracks']['5']['near_frames'] == 1 and diag['tracks']['5']['min_norm_dist'] == 0.0
        assert diag['tracks']['10']['near_frames'] == 9 and diag['tracks']['10']['contact_frames'] >= 1
        rows = inter.diagnostic_rows()
        assert any(r['scene_id'] == 1 and r['track_id'] == 10 and r['reason'] == 'event_too_short' for r in rows)
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_diagnostics_end_to_end_result_json_and_csv():
    """End-to-end ghi result.json['diagnostics'] và diagnostics.csv, kể cả khi gate chặn toàn bộ event."""
    import pandas as pd
    from exhibitflow import run_pipeline

    old_cfg = dict(settings.CFG)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'diag_e2e.avi'
            make_video(video)

            # Lần 1: chạy bình thường -> có event, diagnostics ghi nhận near_frames/contact_frames/min_norm_dist
            out_ok = tmp / 'out_ok'
            summary_ok = run_pipeline(
                [str(video)], out_dir=out_ok, work_dir=tmp / 'work_ok',
                overrides=dict(models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0)),
                               rt=dict(queue_size=4, pose_gate_norm_dist=None)),
            )
            assert not summary_ok['errors'], summary_ok['errors']
            res_ok = json.loads((out_ok / 'diag_e2e' / 'result.json').read_text(encoding='utf-8'))
            diag_csv_ok = out_ok / 'diag_e2e' / 'diagnostics.csv'
            assert diag_csv_ok.is_file()
            df_ok = pd.read_csv(diag_csv_ok)
            assert {'scene_id', 'track_id', 'reason', 'count', 'frames_seen', 'near_frames', 'contact_frames', 'min_norm_dist'} <= set(df_ok.columns)
            d_ok = res_ok['diagnostics']
            assert d_ok['tracks'], d_ok
            t1 = next(iter(d_ok['tracks'].values()))
            assert t1['frames_seen'] > 0 and t1['near_frames'] > 0 and t1['contact_frames'] > 0
            assert t1['min_norm_dist'] is not None and t1['min_norm_dist'] <= settings.CFG['state_contact_thr']

            # Lần 2: nâng state_min_person_h_px để chặn too_small -> 0 event, diagnostics giải thích rõ lý do
            out_rej = tmp / 'out_rej'
            summary_rej = run_pipeline(
                [str(video)], out_dir=out_rej, work_dir=tmp / 'work_rej',
                overrides=dict(models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0)),
                               rt=dict(queue_size=4, pose_gate_norm_dist=None),
                               cfg=dict(state_min_person_h_px=500)),
            )
            assert not summary_rej['errors'], summary_rej['errors']
            res_rej = json.loads((out_rej / 'diag_e2e' / 'result.json').read_text(encoding='utf-8'))
            assert res_rej['summary']['n_events'] == 0
            d_rej = res_rej['diagnostics']
            assert d_rej['gate_rejects_total'].get('too_small', 0) > 0, d_rej
            assert d_rej['gate_rejects_by_scene']['0'].get('too_small', 0) > 0, d_rej
            t_rej = next(iter(d_rej['tracks'].values()))
            assert t_rej['near_frames'] > 0 and t_rej['contact_frames'] == 0
            df_rej = pd.read_csv(out_rej / 'diag_e2e' / 'diagnostics.csv')
            assert (df_rej['reason'] == 'too_small').any()
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_pose_smooth_one_euro_and_conf_gate():
    """OneEuroFilter/PoseSmoother làm mượt theo track_id khi kc >= pose_conf_thr, bỏ qua kc thấp, và tắt được qua CFG['pose_smooth']."""
    from exhibitflow.perception import PoseSmoother

    old_cfg = dict(settings.CFG)
    try:
        settings.CFG['pose_smooth'] = True
        smoother = PoseSmoother()
        raw_xs, smooth_xs = [], []
        for f in range(25):
            jitter = 12.0 if (f % 2 == 0) else -12.0
            rx = 200.0 + jitter
            raw_xs.append(rx)
            kp = [[rx, 150.0]] * 17
            kc = [0.9] * 17
            # Frame 10: keypoint 0 có conf thấp và toạ độ nhiễu cực lớn -> không được đưa vào bộ lọc
            if f == 10:
                kp[0] = [999.0, 999.0]
                kc[0] = 0.05
            # Track 2 ở xa không được làm ảnh hưởng Track 1
            rec1 = dict(id=1, box=[160.0, 100.0, 240.0, 320.0], kp=kp, kc=kc)
            rec2 = dict(id=2, box=[560.0, 100.0, 640.0, 320.0], kp=[[600.0, 300.0]] * 17, kc=[0.9] * 17)
            smoother.update([rec1, rec2], frame_idx=f, fps=25.0)
            if f != 10:
                smooth_xs.append(rec1['kp'][0][0])
            else:
                # kc[0] < pose_conf_thr -> giữ nguyên toạ độ gốc không lọc
                assert rec1['kp'][0] == [999.0, 999.0]
            assert rec2['kp'][0] == [600.0, 300.0]

        # Dao động sau khi làm mượt phải giảm mạnh so với raw (±12) và không bị kéo về 999 ở frame 11
        raw_std = float(np.std(raw_xs[5:]))
        sm_std = float(np.std(smooth_xs[5:]))
        assert sm_std < 0.45 * raw_std, f'One-Euro phải giảm jitter rõ rệt: raw_std={raw_std:.2f}, sm_std={sm_std:.2f}'
        assert max(abs(x - 200.0) for x in smooth_xs[5:]) < 10.0, smooth_xs[5:]

        # Khi tắt CFG['pose_smooth']=False: không sửa kp, không đặt _pose_smoothed
        settings.CFG['pose_smooth'] = False
        smoother_off = PoseSmoother()
        rec_off = dict(id=3, box=[100.0, 100.0, 180.0, 320.0], kp=[[123.0, 234.0]] * 17, kc=[0.9] * 17)
        smoother_off.update([rec_off], frame_idx=0, fps=25.0)
        assert rec_off['kp'][0] == [123.0, 234.0] and '_pose_smoothed' not in rec_off
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_wrist_interp_sequence_and_online():
    """Nội suy tuyến tính cổ tay cho khoảng mất <= 8 frame (không nội suy > 8 frame) và nối cầu online trong OnlineInteraction."""
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction, hand_anchor_point
    from exhibitflow.perception import interpolate_wrist_keypoints

    old_cfg = dict(settings.CFG)
    try:
        settings.CFG['pose_smooth'] = True
        # 1) Chuỗi có khoảng mất 8 frame (1..8) -> nội suy tuyến tính; và khoảng mất 9 frame (10..18) -> KHÔNG nội suy
        seq = []
        for f in range(20):
            kp = [[100.0, 200.0]] * 17
            kc = [0.9] * 17
            if f == 0:
                kp[9] = [100.0, 200.0]
            elif 1 <= f <= 8:
                kp[9], kc[9] = [0.0, 0.0], 0.0
            elif f == 9:
                kp[9] = [190.0, 290.0]
            elif 10 <= f <= 18:
                kp[9], kc[9] = [0.0, 0.0], 0.0
            elif f == 19:
                kp[9] = [290.0, 390.0]
            seq.append((f, dict(id=7, box=[50.0, 100.0, 150.0, 320.0], kp=kp, kc=kc)))

        interpolate_wrist_keypoints(seq)
        # Kiểm tra nội suy tuyến tính tại f=3 (1/3 đoạn) và f=8 (8/9 đoạn)
        assert np.allclose(seq[3][1]['kp'][9], [130.0, 230.0], atol=1e-5)
        assert np.allclose(seq[8][1]['kp'][9], [180.0, 280.0], atol=1e-5)
        assert seq[4][1].get('wrist_interp', {}).get('left') is True
        pt, src = hand_anchor_point(seq[4][1], 'left')
        assert src == 'wrist_interp' and np.allclose(pt, [140.0, 240.0], atol=1e-5)

        # Khoảng mất 9 frame (> 8) không được nội suy
        for f in range(10, 19):
            assert not (seq[f][1].get('wrist_interp') or {}).get('left'), f'gap=9 không được nội suy ở frame {f}'
            assert hand_anchor_point(seq[f][1], 'left') == (None, None)

        # 2) OnlineInteraction: cổ tay phải mất 6 frame (<= 8) giữa lúc chạm -> không đứt chuỗi CONTACT_LIKE
        class _CapVlm:
            def submit(self, *_a, **_kw):
                pass

        ctx = SimpleNamespace(fps=25.0, n=60, last_frame=0, vlm=_CapVlm(), cur_scene=0, scene_index=lambda _i: 0)
        inter = OnlineInteraction(ctx)
        inter._submit = lambda g: None
        mask = np.zeros((H, W), bool)
        mask[300:420, 560:640] = True
        inter.set_artifacts([dict(id=0, box=[560.0, 300.0, 640.0, 420.0], pedestal=False)], {0: mask})

        for f in range(30):
            kp = [[480.0, 210.0]] * 17
            kp[5], kp[6] = [460.0, 244.0], [500.0, 244.0]
            kp[11], kp[12] = [465.0, 320.0], [495.0, 320.0]
            kp[15], kp[16] = [465.0, 420.0], [495.0, 420.0]
            kp[10] = [565.0, 350.0]
            kc = [0.9] * 17
            kc[9] = 0.0
            if 15 <= f <= 20:  # mất 6 frame cổ tay phải (<= 8)
                kp[10], kc[10] = [0.0, 0.0], 0.0
            fs = inter.step(f, [dict(id=1, box=[440.0, 200.0, 520.0, 420.0], kp=kp, kc=kc)])
            if 15 <= f <= 20:
                right_states = [s for s in fs if s['hand'] == 'right']
                assert right_states and right_states[0]['anchor_source'] == 'wrist_interp'
                assert right_states[0]['state'] == 'CONTACT_LIKE'
        inter.flush(35)
        assert len(inter.events) == 1, f'phải gom thành 1 event liên tục: {inter.events}'
        assert inter.events[0]['start_frame'] == 0 and inter.events[0]['end_frame'] == 29
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_pair_state_hysteresis_and_grace():
    """PairState dùng hysteresis (vào near/contact, ra ×1.3) và state_lost_grace_frames=12; pose_gate_norm_dist=2.0."""
    from exhibitflow.interaction import PairState

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    assert default_ns['CFG']['state_lost_grace_frames'] == 12
    assert default_ns['RT']['pose_gate_norm_dist'] == 2.0
    assert default_ns['CFG']['pose_smooth'] is True
    assert settings.CFG['state_lost_grace_frames'] == 12
    assert settings.CFG['pose_smooth'] is True

    old_cfg = dict(settings.CFG)
    try:
        settings.CFG['state_hysteresis'] = True
        settings.CFG['state_hysteresis_mult'] = 1.3
        settings.CFG['state_near_thr'] = 0.9
        settings.CFG['state_contact_thr'] = 0.35
        settings.CFG['state_hold_seconds'] = 0.25
        fps = 25.0  # need = ceil(0.25 * 25) = 7

        ps = PairState()
        # Ở ngoài FAR: 1.0 > 0.9 -> vẫn FAR (vào dùng state_near_thr)
        st, _, _ = ps.update(1.0, 0, fps)
        assert st == 'FAR'
        # Vào vùng near (0.85 <= 0.9) -> rời FAR; lùi nhẹ ra 1.05 (<= 0.9*1.3=1.17) -> chưa về FAR
        st, _, _ = ps.update(0.85, 1, fps)
        assert st == 'APPROACHING'
        st, _, _ = ps.update(1.05, 2, fps)
        assert st != 'FAR'
        # Vượt quá 1.17 -> về FAR
        st, _, _ = ps.update(1.20, 3, fps)
        assert st == 'FAR'

        # Ở APPROACHING: 0.40 > 0.35 -> không được vào NEAR (vào dùng state_contact_thr)
        ps2 = PairState()
        ps2.update(0.80, 0, fps)
        st, _, _ = ps2.update(0.40, 1, fps)
        assert st in ('APPROACHING', 'WITHDRAWING')

        # Đủ 7 frame <= 0.35 -> vào CONTACT_LIKE; dao động ra 0.42 (<= 0.35*1.3=0.455) -> vẫn CONTACT_LIKE
        ps3 = PairState()
        for f in range(7):
            st, _, _ = ps3.update(0.30, f, fps)
        assert st == 'CONTACT_LIKE'
        st, _, _ = ps3.update(0.42, 7, fps)
        assert st == 'CONTACT_LIKE', f'hysteresis phải giữ CONTACT_LIKE ở d=0.42 <= 0.455, nhận {st}'

        # Mất tín hiệu 12 frame -> vẫn giữ state nhờ grace=12; frame 13 -> OCCLUDED
        for f in range(8, 20):  # 12 frames of None
            st, _, _ = ps3.update(None, f, fps)
            assert st == 'CONTACT_LIKE', f'grace=12 nhưng đổi state ở mất frame thứ {f - 7}: {st}'
        st, _, _ = ps3.update(None, 20, fps)
        assert st == 'OCCLUDED'

        # Ra hẳn (> 0.455) -> thoát CONTACT_LIKE; muốn vào lại phải <= 0.35
        ps4 = PairState()
        for f in range(7):
            ps4.update(0.30, f, fps)
        st, _, _ = ps4.update(0.50, 7, fps)
        assert st == 'WITHDRAWING'
        st, _, _ = ps4.update(0.40, 8, fps)
        assert st != 'CONTACT_LIKE' and st != 'NEAR'

        # Tắt hysteresis -> mất CONTACT_LIKE ngay ở 0.42
        settings.CFG['state_hysteresis'] = False
        ps_no_h = PairState()
        for f in range(7):
            ps_no_h.update(0.30, f, fps)
        st, _, _ = ps_no_h.update(0.42, 7, fps)
        assert st != 'CONTACT_LIKE'
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_pose_smooth_end_to_end_pipeline():
    """End-to-end: pose giả có rung nhiễu + mất cổ tay 6 frame giữa lúc chạm -> pipeline vẫn ra 1 event liên tục & alert."""
    from exhibitflow import run_pipeline

    @register('pose', 'fake_jitter_drop')
    class FakeJitterDropPose:
        def __init__(self, **_):
            self._step = -1

        def __call__(self, native, recs, S):
            self._step += 1
            for r in recs:
                x1, y1, x2, y2 = r['box']
                cx, h = (x1 + x2) / 2, y2 - y1
                jit = 3.0 if (self._step % 2 == 0) else -3.0
                kp = [[cx, y1 + 10]] * 5 + [
                    [cx - 25, y1 + .2 * h], [cx + 25, y1 + .2 * h],
                    [cx - 30, y1 + .4 * h], [x2 + 10, y1 + .4 * h],
                    [cx - 30, y1 + .6 * h], [x2 + 15 + jit, y1 + .5 * h],
                    [cx - 15, y1 + .55 * h], [cx + 15, y1 + .55 * h],
                    [cx - 15, y1 + .75 * h], [cx + 15, y1 + .75 * h],
                    [cx - 15, y2], [cx + 15, y2],
                ]
                kc = [1.0] * 17
                if 75 <= self._step <= 80:  # mất cổ tay phải 6 frame giữa đoạn chạm (60..130)
                    kp[10] = [0.0, 0.0]
                    kc[10] = 0.0
                r['kp'], r['kc'] = kp, kc

    import copy
    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'smooth_e2e.avi'
            make_video(video)
            out = tmp / 'out'
            models = dict(FAKE_MODELS, pose=dict(backend='fake_jitter_drop'),
                          vlm=dict(backend='dry_run', delay_s=0.0, label='touching'))
            summary = run_pipeline(
                [str(video)], out_dir=out, work_dir=tmp / 'work',
                overrides=dict(models=models, rt=dict(queue_size=4, pose_gate_norm_dist=None)),
            )
            assert not summary['errors'], summary['errors']
            res = json.loads((out / 'smooth_e2e' / 'result.json').read_text(encoding='utf-8'))
            assert len(res['events']) == 1, f'phải giữ thành 1 event liên tục: {res["events"]}'
            assert res['alerts'] and res['alerts'][0]['behavior'] == 'touching'
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_person_filter_settings_and_box_rules():
    """Kiểm tra mặc định settings cho person_filter/artifact_person_min_score và lọc score, area (960px), aspect, min_hits."""
    from exhibitflow.perception import PersonTrackFilter, check_person_box

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    assert default_ns['MODELS']['detector']['conf'] == 0.01
    assert default_ns['RT']['artifact_person_min_score'] == 0.45
    pf_def = default_ns['RT']['person_filter']
    assert pf_def['enabled'] is True
    assert pf_def['display_min_score'] == 0.5
    assert pf_def['min_hits'] == 3
    assert pf_def['min_box_area'] == 600
    assert tuple(pf_def['aspect_range']) == (0.2, 1.0)
    assert pf_def['min_valid_kpts'] == 5

    # check_person_box trên ảnh 960px
    assert check_person_box(dict(id=1, box=[100.0, 100.0, 140.0, 220.0], score=0.5), work_width=960)  # 40x120=4800, w/h=0.33
    assert not check_person_box(dict(id=1, box=[100.0, 100.0, 140.0, 220.0], score=0.49), work_width=960)  # score < 0.5
    assert not check_person_box(dict(id=1, box=[100.0, 100.0, 120.0, 125.0], score=0.9), work_width=960)  # 20x25=500 < 600
    assert not check_person_box(dict(id=1, box=[100.0, 100.0, 110.0, 200.0], score=0.9), work_width=960)  # w/h=0.1 < 0.2
    assert not check_person_box(dict(id=1, box=[100.0, 100.0, 250.0, 200.0], score=0.9), work_width=960)  # w/h=1.5 > 1.0

    # min_box_area quy đổi theo ảnh 960px khi work_width khác 960 (480px -> hệ số 2x -> area x4)
    assert check_person_box(dict(id=1, box=[10.0, 10.0, 25.0, 30.0], score=0.9), work_width=480)      # 15x20 -> 30x40=1200 >= 600
    assert not check_person_box(dict(id=1, box=[10.0, 10.0, 20.0, 24.0], score=0.9), work_width=480)  # 10x14 -> 20x28=560 < 600

    # min_hits=3: 2 frame đầu bị giữ lại, từ frame 3 mới qua lọc; score thấp không cộng dồn hits
    flt = PersonTrackFilter(work_width=960)
    good = dict(id=1, box=[100.0, 100.0, 160.0, 260.0], score=0.8)
    low = dict(id=2, box=[200.0, 100.0, 260.0, 260.0], score=0.2)
    assert flt.filter_pre_pose([good, low], frame_idx=0) == []
    assert flt.filter_pre_pose([good, low], frame_idx=1) == []
    kept = flt.filter_pre_pose([good, low], frame_idx=2)
    assert [r['id'] for r in kept] == [1]

    # Tắt cờ enabled=False -> giữ nguyên mọi box ngay từ frame đầu
    flt_off = PersonTrackFilter(cfg=dict(enabled=False), work_width=960)
    assert [r['id'] for r in flt_off.filter_pre_pose([low], frame_idx=0)] == [2]


def test_person_filter_pose_kpts_window():
    """Kiểm tra filter_post_pose: track chưa chạy pose được giữ; khi đã có pose cần >= 5 keypoint hợp lệ trong >= 50% frame gần nhất."""
    from exhibitflow.perception import PersonTrackFilter

    flt = PersonTrackFilter(cfg=dict(min_valid_kpts=5, pose_min_valid_frac=0.5, pose_window=4))
    kp = [[100.0, 100.0]] * 17
    good_kc = [0.9] * 5 + [0.0] * 12   # đúng 5 keypoint >= pose_conf_thr (0.30)
    bad_kc = [0.9] * 4 + [0.0] * 13    # chỉ 4 keypoint < 5

    # Track 10 chưa từng chạy pose (xa hiện vật) -> vẫn giữ
    far_rec = dict(id=10, box=[10.0, 100.0, 70.0, 260.0], score=0.9, kp=None, kc=None)
    assert [r['id'] for r in flt.filter_post_pose([far_rec], posed_ids=set(), frame_idx=0)] == [10]

    # Track 20 (vật thể giả) đã chạy pose nhưng chỉ có 4 keypoint -> bị loại ngay (0/1 < 0.5)
    statue_bad = dict(id=20, box=[300.0, 100.0, 360.0, 260.0], score=0.9, kp=kp, kc=bad_kc)
    assert flt.filter_post_pose([statue_bad], posed_ids={20}, frame_idx=0) == []

    # Track 30 có pose xen kẽ: [T, F] -> 50% (giữ); [T, F, F] -> 33% (loại); [T, F, F, T] -> 50% (giữ lại)
    p_good = dict(id=30, box=[300.0, 100.0, 360.0, 260.0], score=0.9, kp=kp, kc=good_kc)
    p_bad = dict(id=30, box=[300.0, 100.0, 360.0, 260.0], score=0.9, kp=kp, kc=bad_kc)
    assert [r['id'] for r in flt.filter_post_pose([p_good], posed_ids={30}, frame_idx=0)] == [30]
    assert [r['id'] for r in flt.filter_post_pose([p_bad], posed_ids={30}, frame_idx=1)] == [30]
    assert flt.filter_post_pose([p_bad], posed_ids={30}, frame_idx=2) == []
    assert [r['id'] for r in flt.filter_post_pose([p_good], posed_ids={30}, frame_idx=3)] == [30]

    # Track 40 được đưa vào M.pose (posed_ids={40}) nhưng pose không ra kp (kp=None) -> tính là 0 kpts (loại)
    no_kp = dict(id=40, box=[300.0, 100.0, 360.0, 260.0], score=0.9, kp=None, kc=None)
    assert flt.filter_post_pose([no_kp], posed_ids={40}, frame_idx=0) == []


def test_person_filter_end_to_end_pipeline():
    """End-to-end: giữ conf=0.01 cho tracker, lọc box giả/chưa đủ min_hits/thiếu keypoint: không vẽ, không pose, không event."""
    import copy
    from collections import defaultdict
    from exhibitflow import run_pipeline

    tracker_det_counts = []
    pose_called_frames = defaultdict(list)
    run_state = {'started': False}

    @register('detector', 'fake_with_fp')
    class FakeDetectorWithFP:
        def __init__(self, **_):
            self._idx = -1

        def detect(self, frame):
            if not run_state['started']:
                return []
            self._idx += 1
            dets = []
            # 1) Box giả score thấp (0.20) nhưng >= 0.01 (từ frame 0..59): vẫn phải tới được tracker
            dets.append([20.0, 200.0, 90.0, 400.0, 0.20])
            # 2) Tượng giả sát hiện vật, score cao (0.85) nhưng pose trả < 5 keypoint (từ frame 10..59)
            if 10 <= self._idx < 60:
                dets.append([500.0, 240.0, 555.0, 420.0, 0.85])
            # 3) Người thật đi tới chạm hiện vật ở frame 10..45 (36 frame)
            if 10 <= self._idx <= 45:
                dets.append([440.0, 200.0, 520.0, 420.0, 0.90])
            return dets

    @register('tracker', 'fake_iou_spy')
    class FakeIoUSpyTracker:
        def __init__(self, **_):
            from exhibitflow.perception import IoUTracker
            self._inner = IoUTracker(iou_thr=0.3, max_age=5, min_score=0.01)

        def reset(self, fps):
            run_state['started'] = True
            self._inner.reset(fps)

        def update(self, dets, frame, **_):
            tracker_det_counts.append((len(dets), [round(d[4], 2) for d in dets]))
            return self._inner.update(dets, frame)

    @register('pose', 'fake_fp_pose')
    class FakeFPPose:
        def __init__(self, **_):
            self._frame = -1

        def __call__(self, native, recs, S):
            self._frame += 1
            for r in recs:
                x1, y1, x2, y2 = r['box']
                cx, h = (x1 + x2) / 2, y2 - y1
                pose_called_frames[round(x1, 0)].append(len(recs))
                if x1 >= 490.0:  # tượng giả sát hiện vật: pose chỉ có 2 keypoint (không phải người)
                    r['kp'] = [[cx, y1 + 10]] * 17
                    r['kc'] = [0.9, 0.9] + [0.0] * 15
                    continue
                kp = [[cx, y1 + 10]] * 5 + [
                    [cx - 25, y1 + .2 * h], [cx + 25, y1 + .2 * h],
                    [cx - 30, y1 + .4 * h], [x2 + 10, y1 + .4 * h],
                    [cx - 30, y1 + .6 * h], [x2 + 35, y1 + .5 * h],
                    [cx - 15, y1 + .55 * h], [cx + 15, y1 + .55 * h],
                    [cx - 15, y1 + .75 * h], [cx + 15, y1 + .75 * h],
                    [cx - 15, y2], [cx + 15, y2],
                ]
                r['kp'], r['kc'] = kp, [1.0] * 17

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'pf_e2e.avi'
            make_video(video)
            drawn_by_frame = {}

            def on_frame(ctx, pkt, vis, perf):
                drawn_by_frame[pkt.idx] = [round(r['box'][0], 1) for r in (pkt.tracks or [])]

            out = tmp / 'out'
            models = dict(
                FAKE_MODELS,
                detector=dict(backend='fake_with_fp'),
                tracker=dict(backend='fake_iou_spy'),
                pose=dict(backend='fake_fp_pose'),
                vlm=dict(backend='dry_run', delay_s=0.0, label='touching'),
            )
            summary = run_pipeline(
                [str(video)],
                out_dir=out,
                work_dir=tmp / 'work',
                max_frames=65,
                overrides=dict(
                    models=models,
                    rt=dict(queue_size=4, pose_gate_norm_dist=None, frame_hook=on_frame),
                ),
            )
            assert not summary['errors'], summary['errors']
            # Tracker vẫn nhận box score=0.20 (không phá recall đầu vào của tracker)
            assert any(0.2 in scores for _, scores in tracker_det_counts), tracker_det_counts[:5]
            # Box score=0.20 (x1=20) không bao giờ được chạy pose hay vẽ
            assert 20.0 not in pose_called_frames
            assert all(20.0 not in xs for xs in drawn_by_frame.values())
            # 2 frame đầu của người thật (frame 10, 11) chưa qua min_hits=3 -> không vẽ, không chạy pose
            assert drawn_by_frame[10] == [] and drawn_by_frame[11] == []
            assert 440.0 in drawn_by_frame[12]
            # Người thật chỉ được gọi pose từ hit thứ 3 (36 - 2 = 34 lần, không phải 36)
            assert len(pose_called_frames[440.0]) == 34, len(pose_called_frames[440.0])
            # Tượng giả (x1=500) bị loại sau pose -> không vẽ và không tạo event
            assert all(500.0 not in xs for xs in drawn_by_frame.values())

            res = json.loads((out / 'pf_e2e' / 'result.json').read_text(encoding='utf-8'))
            assert len(res['events']) == 1, res['events']
            assert res['summary']['n_people_tracks'] == 1, res['summary']
            assert res['alerts'] and res['alerts'][0]['behavior'] == 'touching'
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_flag_inherit_settings_and_reid_embedding():
    """Kiểm tra mặc định RT['flag_inherit'], trích xuất OSNet ReID từ tracker, và kế thừa cờ qua ReID + vị trí."""
    from types import SimpleNamespace
    from exhibitflow.engine import ALARM_COLOR, AlarmBoard, FlagInheritor
    from exhibitflow.perception import ByteTrackDMALTC, extract_track_embedding

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    fi_def = default_ns['RT']['flag_inherit']
    assert fi_def == dict(enabled=True, reid_cos=0.7, max_gap_sec=3.0, max_center_dist_norm=2.0), fi_def

    # 1) ByteTrackDMALTC.update trích xuất smooth_feat/curr_feat từ track nếu có
    wrapper = object.__new__(ByteTrackDMALTC)
    wrapper.min_box_area = 100
    wrapper.max_aspect = 1.6
    fake_strack = SimpleNamespace(
        track_id=11,
        tlwh=np.array([100.0, 200.0, 80.0, 200.0], np.float32),
        tlbr=np.array([100.0, 200.0, 180.0, 400.0], np.float32),
        smooth_feat=np.array([3.0, 4.0, 0.0, 0.0], np.float32),
        curr_feat=np.array([0.0, 1.0, 0.0, 0.0], np.float32),
    )
    wrapper.tracker = SimpleNamespace(update=lambda *a, **kw: [fake_strack])
    out_tracks = wrapper.update([[100.0, 200.0, 180.0, 400.0, 0.9]], np.zeros((540, 960, 3), np.uint8))
    assert len(out_tracks) == 1 and out_tracks[0]['id'] == 11
    assert np.allclose(out_tracks[0]['reid_feat'], [0.6, 0.8, 0.0, 0.0], atol=1e-5)
    assert extract_track_embedding(SimpleNamespace(smooth_feat=None, curr_feat=np.array([0.0, 5.0]))) == [0.0, 1.0]

    # 2) FlagInheritor với ReID embedding: A biến mất ở frame 100, B xuất hiện cùng chỗ ở frame 105 -> B đỏ, C ở xa -> không đỏ
    board = AlarmBoard(persist=True)
    inheritor = FlagInheritor(fps=25.0, cfg=dict(enabled=True, reid_cos=0.7, max_gap_sec=3.0, max_center_dist_norm=2.0))
    feat_a = [1.0, 0.0, 0.0, 0.0]
    feat_other = [0.0, 1.0, 0.0, 0.0]

    inheritor.step(99, [dict(id=1, box=[600.0, 200.0, 680.0, 420.0], reid_feat=feat_a)], None, board)
    board.add(dict(tid=1, behavior='stealing', conf=0.95, segment='s1', shown_from=99, t_alarm=1.0, artifact_ids=[0]))
    inheritor.step(100, [dict(id=1, box=[602.0, 200.0, 682.0, 420.0], reid_feat=feat_a)], None, board)

    # Gap 101..104: A biến mất
    for f in range(101, 105):
        inheritor.step(f, [], None, board)

    # Frame 105:
    # - Track 2 (B): cùng chỗ, ReID giống A (cos ~0.99 >= 0.7) -> kế thừa cờ đỏ
    # - Track 3 (C): ReID giống nhưng ở rất xa (dist_norm = 592/220 = 2.69 > 2.0) -> KHÔNG đỏ
    # - Track 4 (D): cùng chỗ nhưng ReID khác (cos = 0.0 < 0.7) -> KHÔNG đỏ
    # - Track 5 (E): cùng chỗ, ReID hơi thấp hơn B (cos ~0.85) -> KHÔNG đỏ vì mỗi cờ chỉ kế thừa tối đa 1 track cùng lúc
    res_105 = inheritor.step(
        105,
        [
            dict(id=2, box=[605.0, 200.0, 685.0, 420.0], reid_feat=[0.99, 0.14, 0.0, 0.0]),
            dict(id=3, box=[10.0, 200.0, 90.0, 420.0], reid_feat=feat_a),
            dict(id=4, box=[606.0, 200.0, 686.0, 420.0], reid_feat=feat_other),
            dict(id=5, box=[608.0, 200.0, 688.0, 420.0], reid_feat=[0.85, 0.52, 0.0, 0.0]),
        ],
        None,
        board,
    )
    assert set(res_105.keys()) == {2}, res_105
    assert res_105[2]['method'] == 'reid' and res_105[2]['src_tid'] == 1
    active = board.active(105, hold_frames=10)
    assert 2 in active and active[2]['behavior'] == 'stealing'
    assert 3 not in active and 4 not in active and 5 not in active
    assert ALARM_COLOR == (0, 0, 255)


def test_flag_inherit_hsv_fallback_and_gap_limit():
    """Fallback histogram HSV vùng thân khi không có ReID, giới hạn max_gap_sec, chuỗi A->B->C và cờ tắt."""
    from exhibitflow.engine import AlarmBoard, FlagInheritor

    def mk_frame(boxes_colors):
        img = np.full((H, W, 3), 90, np.uint8)
        for box, bgr in boxes_colors:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
        return img

    green = (0, 200, 0)
    blue = (200, 0, 0)

    board = AlarmBoard(persist=True)
    inheritor = FlagInheritor(fps=25.0, cfg=dict(enabled=True, reid_cos=0.7, max_gap_sec=1.0, max_center_dist_norm=2.0))

    box_a = [600.0, 200.0, 680.0, 420.0]
    box_b = [604.0, 202.0, 684.0, 422.0]
    box_blue = [420.0, 200.0, 500.0, 420.0]  # gần (dist_norm ~0.82 <= 2.0) nhưng khác màu áo
    box_far = [10.0, 200.0, 90.0, 420.0]     # cùng màu nhưng xa (dist_norm ~2.68 > 2.0)

    # Track 10 mặc áo xanh + Track 99 (bystander) cùng xuất hiện ở frame 100
    f100 = mk_frame([(box_a, green), (box_far, green)])
    inheritor.step(100, [dict(id=10, box=box_a), dict(id=99, box=box_far)], f100, board)
    board.add(dict(tid=10, behavior='touching', conf=0.9, segment='s1', shown_from=100, t_alarm=1.0, artifact_ids=[0]))

    # Frame 105: Track 10 mất, Track 20 (áo xanh, cùng chỗ) & Track 21 (áo xanh dương, gần) & Track 22 (áo xanh, xa)
    f105 = mk_frame([(box_b, green), (box_blue, blue), (box_far, green)])
    res_105 = inheritor.step(
        105,
        [
            dict(id=20, box=box_b),
            dict(id=21, box=box_blue),
            dict(id=22, box=box_far),
            dict(id=99, box=box_far),
        ],
        f105,
        board,
    )
    assert set(res_105.keys()) == {20}, res_105
    assert res_105[20]['method'] == 'hsv_fallback'
    assert 20 in board.flagged and 21 not in board.flagged and 22 not in board.flagged and 99 not in board.flagged

    # Khi Track 20 ĐANG hoạt động, Track 23 xuất hiện gần chỗ cũ ở frame 106 -> KHÔNG được kế thừa (tối đa 1 track / thời điểm)
    f106 = mk_frame([(box_b, green), ([500.0, 200.0, 580.0, 420.0], green)])
    res_106 = inheritor.step(106, [dict(id=20, box=box_b), dict(id=23, box=[500.0, 200.0, 580.0, 420.0])], f106, board)
    assert res_106 == {} and 23 not in board.flagged

    # Track 20 mất sau frame 106; quá max_gap_sec=1.0s (25 frames) ở frame 132 -> KHÔNG kế thừa
    f132 = mk_frame([(box_b, green)])
    res_132 = inheritor.step(132, [dict(id=24, box=box_b)], f132, board)
    assert res_132 == {} and 24 not in board.flagged

    # Chuỗi kế thừa hợp lệ A(10)->B(20)->C(25) trong giới hạn gap
    board2 = AlarmBoard(persist=True)
    inh2 = FlagInheritor(fps=25.0, cfg=dict(enabled=True, reid_cos=0.7, max_gap_sec=3.0, max_center_dist_norm=2.0))
    inh2.step(10, [dict(id=1, box=box_a)], f100, board2)
    board2.add(dict(tid=1, behavior='stealing', conf=0.9, segment='s1', shown_from=10, t_alarm=0.4, artifact_ids=[0]))
    inh2.step(15, [dict(id=2, box=box_a)], f100, board2)
    inh2.step(20, [dict(id=3, box=box_a)], f100, board2)
    assert 2 in board2.flagged and 3 in board2.flagged

    # Tắt enabled=False -> không kế thừa
    board_off = AlarmBoard(persist=True)
    inh_off = FlagInheritor(fps=25.0, cfg=dict(enabled=False))
    inh_off.step(10, [dict(id=1, box=box_a)], f100, board_off)
    board_off.add(dict(tid=1, behavior='stealing', conf=0.9, segment='s1', shown_from=10, t_alarm=0.4, artifact_ids=[0]))
    assert inh_off.step(15, [dict(id=2, box=box_a)], f100, board_off) == {}
    assert 2 not in board_off.flagged


def test_flag_inherit_end_to_end_pipeline():
    """End-to-end: track A bị cờ biến mất ở frame 100, track B xuất hiện cùng chỗ frame 105 -> B đỏ; track C ở xa -> xanh."""
    import copy
    from exhibitflow import run_pipeline
    from exhibitflow.engine import ALARM_COLOR, IDLE_PERSON

    art_box = (660, 300, 740, 420)
    box_ab = (540, 200, 620, 420)
    box_c = (20, 200, 100, 420)

    @register('detector', 'fake_inherit_det')
    class FakeInheritDetector:
        def __init__(self, **_):
            pass

        def detect(self, frame):
            m = cv2.inRange(frame, (0, 150, 0), (80, 255, 80))
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(m)
            dets = []
            for k in range(1, n_lbl):
                x, y, w, h, area = stats[k]
                if area >= 200:
                    dets.append([float(x), float(y), float(x + w), float(y + h), 0.9])
            return dets

    @register('segmenter', 'fake_inherit_seg')
    class FakeInheritSegmenter:
        def __init__(self, **_):
            pass

        def warmup(self):
            pass

        def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
            mask = np.zeros(frame.shape[:2], bool)
            mask[art_box[1]:art_box[3], art_box[0]:art_box[2]] = True
            return [dict(id=0, box=[float(v) for v in art_box], label='statue', score=0.9, pedestal=False, fallback=False)], {0: mask}, []

        def release(self):
            pass

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'inherit_e2e.avi'
            wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for i in range(N):
                f = np.full((H, W, 3), 90, np.uint8)
                cv2.rectangle(f, (art_box[0], art_box[1]), (art_box[2], art_box[3]), (200, 200, 200), -1)
                if 10 <= i <= 100:
                    cv2.rectangle(f, (box_ab[0], box_ab[1]), (box_ab[2], box_ab[3]), (0, 200, 0), -1)
                elif 105 <= i <= 140:
                    cv2.rectangle(f, (box_ab[0], box_ab[1]), (box_ab[2], box_ab[3]), (0, 200, 0), -1)
                    cv2.rectangle(f, (box_c[0], box_c[1]), (box_c[2], box_c[3]), (0, 200, 0), -1)
                wr.write(f)
            wr.release()

            seen_ab, seen_c, flagged_snapshots = {}, {}, {}

            def on_frame(ctx, pkt, vis, perf):
                for r in pkt.tracks or []:
                    x1, y1, x2, y2 = map(int, r['box'])
                    col = tuple(int(c) for c in vis[min(y2, H - 1), (x1 + x2) // 2])
                    if x1 >= 500:
                        seen_ab[pkt.idx] = (r['id'], col)
                    elif x1 <= 150:
                        seen_c[pkt.idx] = (r['id'], col)
                if pkt.idx == 135:
                    flagged_snapshots.update(ctx.flagged)

            out = tmp / 'out'
            models = dict(
                FAKE_MODELS,
                detector=dict(backend='fake_inherit_det'),
                segmenter=dict(backend='fake_inherit_seg'),
                tracker=dict(backend='iou', iou_thr=0.3, max_age=2),
                vlm=dict(backend='dry_run', delay_s=0.0, label='stealing'),
            )
            summary = run_pipeline(
                [str(video)],
                out_dir=out,
                work_dir=tmp / 'work',
                overrides=dict(
                    models=models,
                    rt=dict(queue_size=4, pose_gate_norm_dist=None, vlm_early_sec=0.4, frame_hook=on_frame),
                ),
            )
            assert not summary['errors'], summary['errors']
            # Track A (f12..100) và Track B (f107..140) có ID khác nhau do đứt 4 frame (> max_age=2)
            id_a = seen_ab[90][0]
            id_b = seen_ab[115][0]
            id_c = seen_c[115][0]
            assert id_a != id_b and id_b != id_c
            # Track A bị gắn cờ đỏ và Track B xuất hiện cùng chỗ ở f105 kế thừa màu đỏ
            assert seen_ab[90][1] == ALARM_COLOR
            late_ab = [seen_ab[f][1] for f in range(110, 140) if f in seen_ab]
            assert late_ab and all(col == ALARM_COLOR for col in late_ab), late_ab
            # Track C ở xa không được kế thừa cờ đỏ -> giữ màu xanh
            late_c = [seen_c[f][1] for f in range(110, 140) if f in seen_c]
            assert late_c and all(col == IDLE_PERSON for col in late_c), late_c
            assert id_a in flagged_snapshots and id_b in flagged_snapshots and id_c not in flagged_snapshots
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_relaxed_gate_settings_and_thresholds():
    """Kiểm tra mặc định CFG['relaxed_gate'] và 4 ngưỡng nới lỏng (8 frame, 60px@960, 0.15s, 0.45 pedestal) vs khi tắt."""
    from types import SimpleNamespace
    from exhibitflow.interaction import (
        OnlineInteraction,
        PairState,
        contact_gate,
        effective_contact_thr,
        effective_event_min_frames,
        effective_hold_seconds,
        effective_min_person_h_px,
    )

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    cfg_def = default_ns['CFG']
    rt_def = default_ns['RT']
    assert cfg_def['relaxed_gate'] is True
    assert cfg_def['relaxed_event_min_frames'] == 8
    assert cfg_def['relaxed_min_person_h_px'] == 60
    assert cfg_def['relaxed_hold_seconds'] == 0.15
    assert cfg_def['relaxed_pedestal_contact_thr'] == 0.45
    assert cfg_def['max_tasks_per_video'] == 40
    assert cfg_def['ambiguous_policy'] == 'nearest'
    assert cfg_def['ambiguous_diff_thr'] == 0.15
    assert rt_def['periodic_vlm']['enabled'] is True
    assert rt_def['periodic_vlm']['min_near_sec'] == 1.0
    assert rt_def['periodic_vlm']['flush_on_end'] is True
    assert rt_def['periodic_vlm']['flush_min_sec'] == 1.0
    assert rt_def['periodic_vlm']['flush_max_per_track'] == 2

    old_cfg = dict(settings.CFG)
    try:
        # 1) Bật relaxed_gate=True
        settings.CFG['relaxed_gate'] = True
        assert effective_event_min_frames() == 8
        assert effective_min_person_h_px(960) == 60.0
        assert effective_min_person_h_px(480) == 30.0
        assert effective_hold_seconds() == 0.15
        assert effective_contact_thr(is_pedestal=False) == 0.35
        assert effective_contact_thr(is_pedestal=True) == 0.45

        # contact_gate: h=60 (work 960) và h=35 (work 480 >= 30) qua cổng; h=55 (work 960) vẫn bị too_small
        def mk_rec(tid, box, rw):
            x1, y1, x2, y2 = box
            cx, bh = (x1 + x2) / 2, y2 - y1
            kp = [[cx, y1 + 0.08 * bh]] * 5 + [
                [cx - 10, y1 + 0.2 * bh], [cx + 10, y1 + 0.2 * bh],
                [cx - 12, y1 + 0.4 * bh], [cx + 12, y1 + 0.4 * bh],
                [cx - 10, y1 + 0.5 * bh], rw,
                [cx - 8, y1 + 0.55 * bh], [cx + 8, y1 + 0.55 * bh],
                [cx - 8, y1 + 0.75 * bh], [cx + 8, y1 + 0.75 * bh],
                [cx - 8, y2], [cx + 8, y2],
            ]
            kc = [1.0] * 17
            kc[9] = 0.0
            return dict(id=tid, box=list(box), kp=kp, kc=kc)

        rec60 = mk_rec(1, [450.0, 170.0, 480.0, 230.0], [565.0, 200.0])  # h=60
        rec55 = mk_rec(2, [450.0, 175.0, 480.0, 230.0], [565.0, 200.0])  # h=55
        rec35 = mk_rec(3, [200.0, 100.0, 220.0, 135.0], [260.0, 115.0])  # h=35 @ 480px
        assert contact_gate(rec60, 'right', [565.0, 200.0], 0, {}, work_width=960) is None
        assert contact_gate(rec55, 'right', [565.0, 200.0], 0, {}, work_width=960) == 'too_small'
        assert contact_gate(rec35, 'right', [260.0, 115.0], 0, {}, work_width=480) is None

        # PairState: 0.15s @ 25fps -> ceil(3.75) = 4 frame vào CONTACT_LIKE; pedestal cho phép d=0.40 <= 0.45
        fps = 25.0
        ps_ped = PairState()
        for f in range(3):
            st, _, _ = ps_ped.update(0.40, f, fps, is_pedestal=True)
            assert st == 'NEAR'
        st, _, _ = ps_ped.update(0.40, 3, fps, is_pedestal=True)
        assert st == 'CONTACT_LIKE', f'pedestal d=0.40 phải vào CONTACT_LIKE sau 4 frame, nhận {st}'

        ps_non_ped = PairState()
        for f in range(6):
            st, _, _ = ps_non_ped.update(0.40, f, fps, is_pedestal=False)
            assert st in ('APPROACHING', 'WITHDRAWING'), f'non-pedestal d=0.40 > 0.35 không được vào CONTACT_LIKE: {st}'

        # OnlineInteraction: chuỗi 8 frame (f=0..7) tạo event khi relaxed_gate=True; 7 frame vẫn bị event_too_short
        def run_short_touch(n_frames, relaxed):
            settings.CFG['relaxed_gate'] = relaxed
            ctx = SimpleNamespace(fps=25.0, n=60, last_frame=0, vlm=SimpleNamespace(submit=lambda *a, **kw: None),
                                  cur_scene=0, scene_index=lambda _i: 0, d=dict(W=960, H=540))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: None
            mask = np.zeros((H, W), bool)
            mask[300:420, 560:640] = True
            inter.set_artifacts([dict(id=0, box=[560.0, 300.0, 640.0, 420.0], pedestal=True)], {0: mask})
            rec = mk_rec(1, [480.0, 360.0, 510.0, 420.0], [540.0, 380.0])  # h=60, d=(560-540)/60=0.333
            for f in range(n_frames):
                inter.step(f, [rec])
            inter.flush(50)
            return inter

        inter8_on = run_short_touch(8, True)
        assert len(inter8_on.events) == 1 and inter8_on.events[0]['end_frame'] - inter8_on.events[0]['start_frame'] + 1 == 8
        inter7_on = run_short_touch(7, True)
        assert inter7_on.events == [] and inter7_on.diagnostics()['gate_rejects']['1']['event_too_short'] == 1

        # 2) Tắt relaxed_gate=False -> giữ nguyên hành vi cũ (12 frame, 80px, 0.25s, 0.35)
        settings.CFG['relaxed_gate'] = False
        assert effective_event_min_frames() == 12
        assert effective_min_person_h_px(960) == 80.0
        assert effective_hold_seconds() == 0.25
        assert effective_contact_thr(is_pedestal=True) == 0.35
        assert contact_gate(rec60, 'right', [565.0, 200.0], 0, {}, work_width=960) == 'too_small'
        assert contact_gate(rec35, 'right', [260.0, 115.0], 0, {}, work_width=480) == 'too_small'
        ps_off = PairState()
        for f in range(4):
            st, _, _ = ps_off.update(0.40, f, fps, is_pedestal=True)
            assert st != 'CONTACT_LIKE'
        inter8_off = run_short_touch(8, False)
        assert inter8_off.events == [] and inter8_off.diagnostics()['gate_rejects']['1']['too_small'] == 8
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_periodic_vlm_near_fallback_and_filter():
    """RT['periodic_vlm'] gửi VLM khi ở vùng NEAR liên tục >= 3s mà chưa có event, tôn trọng cooldown/max_tasks và tắt được."""
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    old_cfg, old_rt = dict(settings.CFG), dict(settings.RT)
    try:
        settings.CFG['relaxed_gate'] = True
        settings.RT['periodic_vlm'] = dict(enabled=True, min_near_sec=3.0, cooldown_sec=3.0, max_near_gap_frames=5)

        def mk_rec(tid, box, rw):
            x1, y1, x2, y2 = box
            cx, bh = (x1 + x2) / 2, y2 - y1
            kp = [[cx, y1 + 0.08 * bh]] * 5 + [
                [cx - 20, y1 + 0.2 * bh], [cx + 20, y1 + 0.2 * bh],
                [cx - 22, y1 + 0.4 * bh], [cx + 22, y1 + 0.4 * bh],
                [cx - 20, y1 + 0.5 * bh], rw,
                [cx - 15, y1 + 0.55 * bh], [cx + 15, y1 + 0.55 * bh],
                [cx - 15, y1 + 0.75 * bh], [cx + 15, y1 + 0.75 * bh],
                [cx - 15, y2], [cx + 15, y2],
            ]
            kc = [1.0] * 17
            kc[9] = 0.0
            return dict(id=tid, box=list(box), kp=kp, kc=kc)

        def setup_inter():
            submitted = []
            ctx = SimpleNamespace(fps=25.0, n=300, last_frame=0, vlm=SimpleNamespace(submit=lambda *a, **kw: None),
                                  cur_scene=0, scene_index=lambda _i: 0, d=dict(W=960, H=540))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (g.update(name=f'p_{len(inter.tasks)}'), inter.tasks.append(g), submitted.append(g))
            mask = np.zeros((H, W), bool)
            mask[300:420, 560:640] = True
            inter.set_artifacts([dict(id=0, box=[560.0, 300.0, 640.0, 420.0], pedestal=False)], {0: mask})
            return inter, submitted

        # 1) Ở vùng NEAR (d0 = (560-450)/220 = 0.50 <= 0.9, > 0.35) liên tục 155 frame @ 25fps:
        #    - Frame 0..73 (74 frame < 75): chưa gửi
        #    - Frame 74 (đủ 75 frame = 3.0s): gửi lần 1
        #    - Frame 149 (đủ thêm 75 frame cooldown = 3.0s): gửi lần 2
        inter, submitted = setup_inter()
        near_rec = mk_rec(1, [330.0, 200.0, 410.0, 420.0], [450.0, 350.0])  # h=220, d=110/220=0.50
        for f in range(74):
            inter.step(f, [near_rec])
        assert submitted == [], f'chưa đủ 3s (75 frame) không được gửi: {len(submitted)}'
        inter.step(74, [near_rec])
        assert len(submitted) == 1 and submitted[0]['periodic'] is True
        assert submitted[0]['window'][1] == 74 and submitted[0]['start'] == 0
        for f in range(75, 149):
            inter.step(f, [near_rec])
        assert len(submitted) == 1, 'không được gửi lặp lại trong cooldown 3s'
        inter.step(149, [near_rec])
        assert len(submitted) == 2 and submitted[1]['window'][1] == 149

        # 2) Đã có contact event trong chuỗi NEAR -> không gửi periodic thừa
        inter_ev, sub_ev = setup_inter()
        touch_rec = mk_rec(2, [440.0, 200.0, 520.0, 420.0], [565.0, 350.0])  # d=0.0 -> CONTACT_LIKE
        for f in range(90):
            inter_ev.step(f, [touch_rec])
        inter_ev.flush(110)
        assert len(sub_ev) == 1 and not sub_ev[0].get('periodic', False)

        # 3) Người quá nhỏ (too_small) ở NEAR >= 3s -> không được gửi periodic VLM
        inter_sm, sub_sm = setup_inter()
        small_rec = mk_rec(3, [500.0, 320.0, 520.0, 370.0], [540.0, 345.0])  # h=50 < 60, d=0.40
        for f in range(90):
            inter_sm.step(f, [small_rec])
        assert sub_sm == []

        # 4) Tắt RT['periodic_vlm'] -> không gửi
        settings.RT['periodic_vlm'] = dict(enabled=False, min_near_sec=3.0, cooldown_sec=3.0, max_near_gap_frames=5)
        inter_off, sub_off = setup_inter()
        for f in range(100):
            inter_off.step(f, [near_rec])
        assert sub_off == []
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_relaxed_gate_and_periodic_vlm_end_to_end():
    """End-to-end: (1) chạm ngắn 8 frame của người nhỏ 64px lên pedestal bắt được khi relaxed_gate=True;
    (2) ở vùng NEAR >= 3s không chạm kích hoạt periodic_vlm và VLM là bộ lọc cuối (normal->0 alert, touching->1 alert)."""
    import copy
    from exhibitflow import run_pipeline

    @register('detector', 'fake_t7_det')
    class FakeT7Det:
        def __init__(self, **_):
            pass

        def detect(self, frame):
            m = cv2.inRange(frame, (0, 150, 0), (80, 255, 80))
            ys, xs = np.nonzero(m)
            if len(xs) < 100:
                return []
            return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9]]

    @register('pose', 'fake_t7_pose')
    class FakeT7Pose:
        def __init__(self, **_):
            pass

        def __call__(self, native, recs, S):
            for r in recs:
                x1, y1, x2, y2 = r['box']
                cx, h = (x1 + x2) / 2, y2 - y1
                # Nếu người nhỏ (h<=70) -> chạm pedestal ở d ~ 0.38; nếu người cao -> lơ lửng ở NEAR d ~ 0.50 (không chạm)
                wx = (ART[0] - 0.38 * h) if h <= 70 else (ART[0] - 0.50 * h)
                kp = [[cx, y1 + 0.1 * h]] * 5 + [
                    [cx - 10, y1 + .2 * h], [cx + 10, y1 + .2 * h],
                    [cx - 12, y1 + .4 * h], [wx - 5, y1 + .4 * h],
                    [cx - 12, y1 + .6 * h], [wx, y1 + .5 * h],
                    [cx - 8, y1 + .55 * h], [cx + 8, y1 + .55 * h],
                    [cx - 8, y1 + .75 * h], [cx + 8, y1 + .75 * h],
                    [cx - 8, y2], [cx + 8, y2],
                ]
                r['kp'], r['kc'] = kp, [1.0] * 17

    @register('segmenter', 'fake_t7_seg')
    class FakeT7Seg:
        def __init__(self, **_):
            pass

        def warmup(self):
            pass

        def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
            mask = np.zeros(frame.shape[:2], bool)
            mask[ART[1]:ART[3], ART[0]:ART[2]] = True
            return [dict(id=0, box=[float(v) for v in ART], label='pedestal', score=0.9, pedestal=True, fallback=False)], {0: mask}, []

        def release(self):
            pass

    def make_custom_video(path, box, f_start, f_end, n_frames):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
        for i in range(n_frames):
            f = np.full((H, W, 3), 90, np.uint8)
            cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (200, 200, 200), -1)
            if f_start <= i <= f_end:
                cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), -1)
            wr.write(f)
        wr.release()

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            # Kịch bản A: người nhỏ h=64px chạm pedestal trong 8 frame (f=10..17, cộng 2 frame min_hits -> f=10..19 = 10 frame để track qua filter từ f=12..19 = 8 frame)
            vid_short = tmp / 'short_small.avi'
            make_custom_video(vid_short, (480, 356, 512, 420), f_start=10, f_end=19, n_frames=50)
            base_models = dict(FAKE_MODELS, detector=dict(backend='fake_t7_det'), pose=dict(backend='fake_t7_pose'),
                               segmenter=dict(backend='fake_t7_seg'), vlm=dict(backend='dry_run', delay_s=0.0, label='touching'))

            sum_on = run_pipeline(
                [str(vid_short)], out_dir=tmp / 'out_on', work_dir=tmp / 'work_on',
                overrides=dict(models=base_models, rt=dict(queue_size=4, pose_gate_norm_dist=None), cfg=dict(relaxed_gate=True)),
            )
            assert not sum_on['errors'], sum_on['errors']
            res_on = json.loads((tmp / 'out_on' / 'short_small' / 'result.json').read_text(encoding='utf-8'))
            assert res_on['summary']['n_events'] == 1 and res_on['summary']['n_alerts'] == 1, res_on['summary']

            sum_off = run_pipeline(
                [str(vid_short)], out_dir=tmp / 'out_off', work_dir=tmp / 'work_off',
                overrides=dict(models=base_models, rt=dict(queue_size=4, pose_gate_norm_dist=None), cfg=dict(relaxed_gate=False)),
            )
            assert not sum_off['errors'], sum_off['errors']
            res_off = json.loads((tmp / 'out_off' / 'short_small' / 'result.json').read_text(encoding='utf-8'))
            assert res_off['summary']['n_events'] == 0 and res_off['summary']['n_alerts'] == 0, res_off['summary']

            # Kịch bản B: người cao đứng ở vùng NEAR (d=0.50) trong 90 frame (3.6s >= 3s), không bao giờ vào CONTACT_LIKE
            vid_near = tmp / 'near_3s.avi'
            make_custom_video(vid_near, (370, 200, 450, 420), f_start=10, f_end=99, n_frames=120)

            # VLM lọc trả 'normal' -> vẫn gửi VLM nhưng không tạo alert giả
            sum_norm = run_pipeline(
                [str(vid_near)], out_dir=tmp / 'out_near_norm', work_dir=tmp / 'work_near_norm',
                overrides=dict(models=dict(base_models, vlm=dict(backend='dry_run', delay_s=0.0, label='normal')),
                               rt=dict(queue_size=4, pose_gate_norm_dist=None)),
            )
            assert not sum_norm['errors'], sum_norm['errors']
            res_norm = json.loads((tmp / 'out_near_norm' / 'near_3s' / 'result.json').read_text(encoding='utf-8'))
            assert res_norm['summary']['n_vlm_tasks'] >= 1 and res_norm['summary']['n_alerts'] == 0, res_norm['summary']
            assert res_norm['summary']['has_anomaly'] is False

            # VLM trả 'touching' -> bắt được alert từ đường dự phòng periodic_vlm
            sum_anom = run_pipeline(
                [str(vid_near)], out_dir=tmp / 'out_near_anom', work_dir=tmp / 'work_near_anom',
                overrides=dict(models=base_models, rt=dict(queue_size=4, pose_gate_norm_dist=None)),
            )
            assert not sum_anom['errors'], sum_anom['errors']
            res_anom = json.loads((tmp / 'out_near_anom' / 'near_3s' / 'result.json').read_text(encoding='utf-8'))
            assert res_anom['summary']['n_vlm_tasks'] >= 1 and res_anom['summary']['n_alerts'] >= 1, res_anom['summary']
            assert res_anom['summary']['has_anomaly'] is True
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_seg_settings_phrases_and_fallback_empty_thr():
    """Kiểm tra mặc định RT['seg_multi_frame'], phrases mới, hạ fallback_thr 0.20->0.15 khi cảnh có 0 mask, và tăng cache key."""
    import copy
    import threading
    from exhibitflow.common import spec_tag
    from exhibitflow.segmenter import GroundingDinoSAM, artifact_cache_dir

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    rt_def = default_ns['RT']
    seg_def = default_ns['SEGMENTATION_CFG']

    assert rt_def['seg_multi_frame'] == dict(enabled=True, step_sec=2.0, max_frames_per_scene=4)
    assert rt_def['seg_merge_iou'] == 0.5
    assert rt_def['seg_merge_min_votes'] == 1
    assert rt_def['seg_merge_keep_ref'] is True
    assert rt_def['seg_merge_min_votes_extra'] == 1
    assert rt_def['scene_color_cut'] is True
    assert rt_def['scene_color_hist_thr'] == 0.35

    for p in ('picture frame', 'artwork on wall', 'wall art', 'mounted painting'):
        assert p in seg_def['painting_phrases'], p
    for p in ('display case', 'glass cabinet', 'shelf'):
        assert p in seg_def['fallback_object_phrases'], p
    assert seg_def['fallback_thr'] == 0.20
    assert seg_def['fallback_empty_thr'] == 0.15
    assert seg_def['fallback_lower_when_empty'] is True

    old_seg_cfg, old_rt = copy.deepcopy(settings.SEGMENTATION_CFG), copy.deepcopy(settings.RT)
    try:
        # 1) GroundingDinoSAM._segment với detect/sam_box giả
        seg_obj = object.__new__(GroundingDinoSAM)
        seg_obj.lock = threading.Lock()
        detect_calls = []

        def mk_sam(rgb, box):
            m = np.zeros(rgb.shape[:2], bool)
            x1, y1, x2, y2 = map(int, box)
            m[y1:y2, x1:x2] = True
            return m

        seg_obj.sam_box = mk_sam
        frame = np.zeros((200, 300, 3), np.uint8)

        # Trường hợp A: pass 1 ra 0 mask -> fallback dùng thr=0.15
        def detect_empty_primary(rgb, phrases, thr, text_thr):
            detect_calls.append((list(phrases), float(thr)))
            if len(detect_calls) % 2 == 1:
                return [], [], np.array([], np.float32)
            return [[20.0, 20.0, 100.0, 120.0]], ['glass cabinet'], np.array([0.17], np.float32)

        seg_obj.detect = detect_empty_primary
        arts, masks, _ = seg_obj.segment(frame, 0, [], {0: frame}, {0: []})
        assert detect_calls[1][1] == 0.15, detect_calls
        assert len(arts) == 1 and arts[0]['label'] == 'glass cabinet' and arts[0]['fallback'] is True

        # Trường hợp B: pass 1 có 1 painting (không phải 3D) -> fallback giữ thr=0.20
        detect_calls.clear()

        def detect_has_painting(rgb, phrases, thr, text_thr):
            detect_calls.append((list(phrases), float(thr)))
            if len(detect_calls) == 1:
                return [[10.0, 10.0, 60.0, 60.0]], ['mounted painting'], np.array([0.6], np.float32)
            return [[120.0, 20.0, 200.0, 120.0]], ['display case'], np.array([0.22], np.float32)

        seg_obj.detect = detect_has_painting
        arts2, _, _ = seg_obj.segment(frame, 0, [], {0: frame}, {0: []})
        assert detect_calls[1][1] == 0.20, detect_calls
        assert {a['label'] for a in arts2} == {'mounted painting', 'display case'}

        # Trường hợp C: tắt fallback_lower_when_empty=False -> 0 mask vẫn giữ fallback_thr=0.20
        detect_calls.clear()
        settings.SEGMENTATION_CFG['fallback_lower_when_empty'] = False
        seg_obj.detect = detect_empty_primary
        seg_obj.segment(frame, 0, [], {0: frame}, {0: []})
        assert detect_calls[1][1] == 0.20, detect_calls

        # 2) Cache key mới khác cache key cũ và đổi khi sample_idxs hoặc seg_multi_frame thay đổi
        spec = dict(backend='gdino_sam')
        legacy_tag = spec_tag(dict(seg=spec, cfg=settings.SEGMENTATION_CFG, ref=0, w=settings.CFG['width']))
        c1 = artifact_cache_dir('vid', spec, 0)
        c2 = artifact_cache_dir('vid', spec, 0, sample_idxs=[0, 50])
        assert c1.name != f'vid_artifacts_{legacy_tag}'
        assert c1 != c2
        settings.RT['seg_multi_frame']['step_sec'] = 1.0
        c3 = artifact_cache_dir('vid', spec, 0)
        assert c3 != c1
    finally:
        settings.SEGMENTATION_CFG.clear()
        settings.SEGMENTATION_CFG.update(old_seg_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_merge_masks_multi_frame_iou_and_voting():
    """Hợp nhất mask đa frame theo IoU >= 0.5: chế độ union giữ đủ mask, chế độ min_votes=2 & keep_ref=False lọc theo số phiếu."""
    from exhibitflow.segmenter import merge_masks

    def mk_box_mask(x1, y1, x2, y2):
        m = np.zeros((200, 300), bool)
        m[y1:y2, x1:x2] = True
        return m

    # Frame 0: A (score 0.85) + stone (score 0.30)
    # Frame 1: A hơi lệch (IoU = 36*40 / (44*40) = 0.82 >= 0.5, score 0.80) + B (score 0.60)
    # Frame 2: A (score 0.90) + B hơi lệch (score 0.65)
    m_a0 = mk_box_mask(20, 20, 60, 60)
    m_a1 = mk_box_mask(24, 20, 64, 60)
    m_a2 = mk_box_mask(20, 20, 60, 60)
    m_b1 = mk_box_mask(150, 30, 210, 90)
    m_b2 = mk_box_mask(152, 30, 212, 90)
    m_fp = mk_box_mask(230, 120, 270, 160)

    res = [
        (0, [dict(id=0, box=[20, 20, 60, 60], label='painting', score=0.85, pedestal=False, fallback=False),
             dict(id=1, box=[230, 120, 270, 160], label='stone', score=0.30, pedestal=False, fallback=True)],
         {0: m_a0, 1: m_fp}),
        (50, [dict(id=0, box=[24, 20, 64, 60], label='painting', score=0.80, pedestal=False, fallback=False),
              dict(id=1, box=[150, 30, 210, 90], label='display case', score=0.60, pedestal=False, fallback=True)],
         {0: m_a1, 1: m_b1}),
        (100, [dict(id=0, box=[20, 20, 60, 60], label='painting', score=0.90, pedestal=False, fallback=False),
               dict(id=1, box=[152, 30, 212, 90], label='display case', score=0.65, pedestal=False, fallback=True)],
         {0: m_a2, 1: m_b2}),
    ]
    # Chế độ bỏ phiếu chặt (keep_ref=False, min_votes=2): chỉ giữ >= 2 phiếu
    arts, masks = merge_masks(res, iou_thr=0.5, min_votes=2, keep_ref=False)
    assert len(arts) == 2, arts
    labels = {a['label']: a for a in arts}
    assert 'painting' in labels and labels['painting']['votes'] == 3
    assert 'display case' in labels and labels['display case']['votes'] == 2
    assert 'stone' not in labels
    # Majority vote ở n=3 giữ đúng vùng [20:60, 20:60] (xuất hiện ở 2/3 frame)
    assert np.array_equal(masks[labels['painting']['id']], m_a0)

    # Mặc định TASK-012 (union: min_votes=1, keep_ref=True): giữ đủ cả 3 mask (không mất stone ở frame 0)
    arts_u, _ = merge_masks(res)
    assert {a['label'] for a in arts_u} == {'painting', 'display case', 'stone'}

    # Hai detection trùng nhau trên CÙNG 1 frame không được tính là 2 frame và chỉ giữ 1
    same_frame_only = [
        (0, [dict(id=0, box=[20, 20, 60, 60], label='painting', score=0.80, pedestal=False, fallback=False),
             dict(id=1, box=[22, 20, 62, 60], label='painting', score=0.50, pedestal=False, fallback=False)],
         {0: m_a0, 1: m_a1}),
    ]
    arts_sf, _ = merge_masks(same_frame_only, iou_thr=0.5)
    assert len(arts_sf) == 1 and arts_sf[0]['votes'] == 1 and arts_sf[0]['score'] == 0.80

    # Không có mask nào >= 2 lần:
    # - Mặc định (min_votes=1, keep_ref=True): giữ cả 3 (union)
    # - keep_ref=True, min_votes_extra=2: chỉ giữ mask ở frame tham chiếu (frame 0)
    # - keep_ref=False, min_votes=2: bỏ điều kiện cl is clusters[0] -> 0 mask
    no_repeat = [
        (0, [dict(id=0, box=[20, 20, 60, 60], label='statue', score=0.45, pedestal=False, fallback=False)], {0: m_a0}),
        (50, [dict(id=0, box=[150, 30, 210, 90], label='painting', score=0.72, pedestal=False, fallback=False)], {0: m_b1}),
        (100, [dict(id=0, box=[230, 120, 270, 160], label='vase', score=0.55, pedestal=False, fallback=True)], {0: m_fp}),
    ]
    arts_nr_def, _ = merge_masks(no_repeat, iou_thr=0.5)
    assert {a['label'] for a in arts_nr_def} == {'statue', 'painting', 'vase'}
    arts_nr_ref, _ = merge_masks(no_repeat, iou_thr=0.5, min_votes=1, keep_ref=True, min_votes_extra=2)
    assert len(arts_nr_ref) == 1 and arts_nr_ref[0]['label'] == 'statue'
    arts_nr_strict, _ = merge_masks(no_repeat, iou_thr=0.5, min_votes=2, keep_ref=False)
    assert len(arts_nr_strict) == 0


def test_seg_union_three_frames_and_ref_superset():
    """TASK-012: 3 frame [A,B,C], [A,D], [B,E] -> đủ {A,B,C,D,E}; mask frame tham chiếu không mất; mask trùng IoU>=0.5 chỉ giữ 1."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.common import video_meta
    from exhibitflow.segmenter import SegmenterSession, artifact_cache_dir, merge_masks, prepare_artifacts

    def mk_box(x1, y1, x2, y2):
        m = np.zeros((240, 400), bool)
        m[y1:y2, x1:x2] = True
        return m

    # 5 vùng rời nhau (IoU đôi một = 0); A', B' lệch nhẹ so với A, B nhưng IoU >= 0.5
    m_A = mk_box(10, 10, 60, 60)
    m_A2 = mk_box(14, 10, 64, 60)   # IoU(A, A2) = 46/54 = 0.85 >= 0.5
    m_B = mk_box(90, 10, 140, 60)
    m_B2 = mk_box(92, 12, 142, 62)  # IoU(B, B2) = 48*48 / (50*50*2 - 48*48) = 2304/2696 = 0.85 >= 0.5
    m_C = mk_box(170, 10, 220, 60)
    m_D = mk_box(250, 10, 300, 60)
    m_E = mk_box(330, 10, 380, 60)

    def art(aid, label, score, box):
        return dict(id=aid, box=box, label=label, score=score, pedestal=False, fallback=False)

    frames_res = [
        (0, [art(0, 'A', 0.55, [10, 10, 60, 60]),
             art(1, 'B', 0.50, [90, 10, 140, 60]),
             art(2, 'C', 0.35, [170, 10, 220, 60])],
         {0: m_A, 1: m_B, 2: m_C}),
        (50, [art(0, 'A', 0.90, [14, 10, 64, 60]),
              art(1, 'D', 0.45, [250, 10, 300, 60])],
         {0: m_A2, 1: m_D}),
        (100, [art(0, 'B', 0.88, [92, 12, 142, 62]),
               art(1, 'E', 0.42, [330, 10, 380, 60])],
         {0: m_B2, 1: m_E}),
    ]

    # 1) Mặc định union: đủ {A, B, C, D, E}, không trùng mask IoU>=0.5
    arts, masks = merge_masks(frames_res)
    by_lbl = {a['label']: a for a in arts}
    assert set(by_lbl) == {'A', 'B', 'C', 'D', 'E'}, arts
    assert len(arts) == 5 and len(masks) == 5
    assert by_lbl['A']['votes'] == 2 and by_lbl['B']['votes'] == 2
    assert by_lbl['C']['votes'] == 1 and by_lbl['D']['votes'] == 1 and by_lbl['E']['votes'] == 1
    # Mask của frame tham chiếu (A, B, C) giữ nguyên pixel-for-pixel
    assert np.array_equal(masks[by_lbl['A']['id']], m_A)
    assert np.array_equal(masks[by_lbl['B']['id']], m_B)
    assert np.array_equal(masks[by_lbl['C']['id']], m_C)

    # 2) Bật lọc nhiễu cho frame phụ (min_votes_extra=2): C (ở ref frame) vẫn giữ, D & E (1 phiếu ngoài ref) bị lọc
    arts_ext, _ = merge_masks(frames_res, min_votes=1, keep_ref=True, min_votes_extra=2)
    assert {a['label'] for a in arts_ext} == {'A', 'B', 'C'}

    # 3) Kể cả min_votes=2 khi keep_ref=True: mọi mask ở frame tham chiếu (A, B, C) vẫn không bao giờ bị mất
    arts_kr, _ = merge_masks(frames_res, min_votes=2, keep_ref=True, min_votes_extra=1)
    assert {a['label'] for a in arts_kr} == {'A', 'B', 'C'}

    # 4) Cache key đổi theo seg_merge_keep_ref / seg_merge_min_votes_extra và chuẩn hoá superset trong prepare_artifacts
    old_rt = copy.deepcopy(settings.RT)
    old_paths = (settings.WORK, settings.OUT, settings.CACHE)
    try:
        spec = dict(backend='gdino_sam')
        c0 = artifact_cache_dir('v_t12', spec, 0, sample_idxs=[0, 50, 100])
        settings.RT['seg_merge_min_votes_extra'] = 2
        c1 = artifact_cache_dir('v_t12', spec, 0, sample_idxs=[0, 50, 100])
        assert c1 != c0
        settings.RT['seg_merge_keep_ref'] = False
        c2 = artifact_cache_dir('v_t12', spec, 0, sample_idxs=[0, 50, 100])
        assert c2 != c1 and c2 != c0

        # End-to-end prepare_artifacts: frame đầu thiếu D (bị người che), frame sau thấy D -> kết quả là superset {A, B, D}
        settings.RT['seg_merge_min_votes'] = 1
        settings.RT['seg_merge_keep_ref'] = True
        settings.RT['seg_merge_min_votes_extra'] = 1
        call_n = [0]

        @register('segmenter', 'fake_t12_seg')
        class FakeT12Seg:
            def __init__(self, **_):
                pass
            def warmup(self):
                pass
            def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
                idx = call_n[0]
                call_n[0] += 1
                h, w = frame.shape[:2]
                m1 = np.zeros((h, w), bool); m1[20:80, 20:80] = True
                m2 = np.zeros((h, w), bool); m2[20:80, 140:200] = True
                m3 = np.zeros((h, w), bool); m3[20:80, 260:320] = True
                if idx == 0:
                    return [art(0, 'A', 0.8, [20, 20, 80, 80]), art(1, 'B', 0.4, [140, 20, 200, 80])], {0: m1, 1: m2}, []
                return [art(0, 'A', 0.85, [20, 20, 80, 80]), art(1, 'D', 0.7, [260, 20, 320, 80])], {0: m1, 1: m3}, []
            def release(self):
                pass

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            settings.set_paths(work=tmp / 'work', out=tmp / 'out')
            vid = tmp / 't12.avi'
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for _ in range(100):
                wr.write(np.full((H, W, 3), 100, np.uint8))
            wr.release()
            M = SimpleNamespace(detector=SimpleNamespace(detect=lambda _img: []))
            d = dict(path=vid, **video_meta(vid))
            res = prepare_artifacts('t12', d, M, SegmenterSession(dict(backend='fake_t12_seg')))
            assert not res.error, res.error
            assert {a['label'] for a in res.artifacts} == {'A', 'B', 'D'}
            assert len(res.artifacts) >= 2  # tối thiểu >= bản 1 frame (frame 0 có 2 mask)
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.WORK, settings.OUT, settings.CACHE = old_paths


def test_scene_cut_color_histogram_gray_to_color():
    """detect_scenes bắt chuyển đổi xám->màu (chênh lệch xám nhỏ < 25 nhưng histogram màu đổi mạnh), và tắt được."""
    from exhibitflow.segmenter import detect_scenes

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        vid = Path(tmp) / 'gray_to_color.avi'
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'MJPG'), 25, (320, 180))
        for i in range(150):
            if i < 50:
                f = np.full((180, 320, 3), (120, 120, 120), np.uint8)     # cảnh 0: xám (gray ~ 120)
            elif i < 100:
                f = np.full((180, 320, 3), (40, 120, 200), np.uint8)      # cảnh 1: cam ấm (gray ~ 135, |diff| ~ 15 < 25)
            else:
                f = np.full((180, 320, 3), (180, 125, 80), np.uint8)      # cảnh 2: xanh lam (gray ~ 118, |diff| ~ 17 < 25)
            # Thêm người nhỏ (< 5% diện tích) đi qua trong cảnh 0: không được gây cắt cảnh giả
            if 10 <= i < 30:
                cv2.rectangle(f, (10 + i, 40), (35 + i, 120), (0, 180, 0), -1)
            wr.write(f)
        wr.release()

        scenes_color = detect_scenes(vid, diff_thr=25.0, min_scene_sec=1.0, ref_offset_sec=0.4, color_hist_thr=0.35)
        assert [s for s, _ in scenes_color] == [0, 50, 100], scenes_color

        # Khi tắt histogram màu (color_hist_thr=0.0): chỉ còn 1 cảnh vì chênh lệch xám < 25
        scenes_gray_only = detect_scenes(vid, diff_thr=25.0, min_scene_sec=1.0, ref_offset_sec=0.4, color_hist_thr=0.0)
        assert [s for s, _ in scenes_gray_only] == [0], scenes_gray_only


def test_seg_multi_frame_prepare_artifacts_and_compat():
    """prepare_artifacts chạy segment đa frame, hợp nhất, dùng cache đúng, và tương thích MANUAL_ARTIFACTS / SEG_REF_FRAMES."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.common import video_meta
    from exhibitflow.segmenter import SegmenterSession, prepare_artifacts

    seg_calls = []

    @register('segmenter', 'fake_t8_seg')
    class FakeT8Seg:
        def __init__(self, **_):
            pass

        def warmup(self):
            pass

        def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
            idx = len(seg_calls)
            seg_calls.append(tuple(int(v) for v in frame[10, 10]))
            # Hiện vật chính xuất hiện ở mọi frame mẫu
            mk_main = np.zeros(frame.shape[:2], bool)
            mk_main[120:260, 500:640] = True
            arts = [dict(id=0, box=[500.0, 120.0, 640.0, 260.0], label='mounted painting', score=0.85,
                         pedestal=False, fallback=False)]
            masks = {0: mk_main}
            # Nhiễu 1 frame chỉ xuất hiện ở lần gọi đầu tiên của cảnh (score thấp 0.35) -> phải bị lọc bỏ
            if idx == 0:
                mk_fp = np.zeros(frame.shape[:2], bool)
                mk_fp[20:80, 20:80] = True
                arts.append(dict(id=1, box=[20.0, 20.0, 80.0, 80.0], label='stone', score=0.35,
                                 pedestal=False, fallback=True))
                masks[1] = mk_fp
            return arts, masks, []

        def manual(self, frame, box, aid):
            mk = np.zeros(frame.shape[:2], bool)
            x1, y1, x2, y2 = map(int, box)
            mk[y1:y2, x1:x2] = True
            return dict(id=aid, box=[float(v) for v in box], label='manual', score=1.0, pedestal=False), mk

        def release(self):
            pass

    old_rt = copy.deepcopy(settings.RT)
    old_manual = copy.deepcopy(settings.MANUAL_ARTIFACTS)
    old_refs = copy.deepcopy(settings.SEG_REF_FRAMES)
    old_paths = (settings.WORK, settings.OUT, settings.CACHE)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            settings.set_paths(work=tmp / 'work', out=tmp / 'out')
            vid = tmp / 'multi_scene_t8.avi'
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for i in range(250):
                # Cảnh 0 (0..124): xám; Cảnh 1 (125..249): màu ấm (chênh xám < 25, cắt bằng histogram màu)
                col = (120, 120, 120) if i < 125 else (40, 120, 200)
                f = np.full((H, W, 3), col, np.uint8)
                wr.write(f)
            wr.release()

            M = SimpleNamespace(detector=SimpleNamespace(detect=lambda _img: []))
            d = dict(path=vid, **video_meta(vid))

            # 1) Bật seg_multi_frame (step=2s=50f, max=4): cảnh 0 (12,62,112 -> 3f), cảnh 1 (137,187,237 -> 3f) => 6 lần gọi
            seg_calls.clear()
            seg = SegmenterSession(dict(backend='fake_t8_seg'))
            res = prepare_artifacts('multi_scene_t8', d, M, seg)
            assert not res.error, res.error
            assert len(res.scenes) == 2
            assert res.scenes[0]['start'] == 0 and res.scenes[1]['start'] == 125
            assert len(seg_calls) == 6, f'kỳ vọng 3 frame x 2 cảnh = 6 lần segment, nhận {len(seg_calls)}'
            # TASK-012: mặc định union (keep_ref=True, min_votes=1) giữ đủ cả mask frame tham chiếu (stone) lẫn mounted painting
            assert len(res.scenes[0]['artifacts']) == 2
            s0_by_lbl = {a['label']: a for a in res.scenes[0]['artifacts']}
            assert s0_by_lbl['mounted painting']['votes'] == 3
            assert s0_by_lbl['stone']['votes'] == 1
            assert res.timing['source'] == 'computed'

            # 2) Chạy lần 2 -> cache hit (0 lần gọi segmenter)
            seg_calls.clear()
            seg2 = SegmenterSession(dict(backend='fake_t8_seg'))
            res_cached = prepare_artifacts('multi_scene_t8', d, M, seg2)
            assert not res_cached.error, res_cached.error
            assert len(seg_calls) == 0 and res_cached.timing['source'] == 'cache'
            assert len(res_cached.artifacts) == 3

            # 3) Tắt seg_multi_frame -> chỉ gọi 1 lần mỗi cảnh (= 2 lần)
            settings.RT['seg_multi_frame'] = dict(enabled=False, step_sec=2.0, max_frames_per_scene=4)
            seg_calls.clear()
            res_single = prepare_artifacts('multi_scene_t8', d, M, SegmenterSession(dict(backend='fake_t8_seg')))
            assert not res_single.error, res_single.error
            assert len(seg_calls) == 2, f'khi tắt seg_multi_frame phải gọi 1 frame/cảnh, nhận {len(seg_calls)}'

            # 4) Tương thích SEG_REF_FRAMES + MANUAL_ARTIFACTS
            settings.RT['seg_multi_frame'] = dict(enabled=True, step_sec=2.0, max_frames_per_scene=4)
            settings.SEG_REF_FRAMES['multi_scene_t8'] = [(0, 10), (125, 140)]
            settings.MANUAL_ARTIFACTS['multi_scene_t8'] = [[40.0, 50.0, 140.0, 150.0]]
            seg_calls.clear()
            res_compat = prepare_artifacts('multi_scene_t8', d, M, SegmenterSession(dict(backend='fake_t8_seg')))
            assert not res_compat.error, res_compat.error
            assert [(s['start'], s['ref_idx']) for s in res_compat.scenes] == [(0, 10), (125, 140)]
            assert len(seg_calls) == 2  # SEG_REF_FRAMES chỉ định frame chính xác -> giữ 1 frame/cảnh
            assert any(a['id'] == 1000 and a['label'] == 'manual' for a in res_compat.scenes[0]['artifacts'])
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MANUAL_ARTIFACTS.clear()
        settings.MANUAL_ARTIFACTS.update(old_manual)
        settings.SEG_REF_FRAMES.clear()
        settings.SEG_REF_FRAMES.update(old_refs)
        settings.WORK, settings.OUT, settings.CACHE = old_paths


def test_missing_detect_settings_and_toggle():
    """Kiểm tra mặc định RT['missing_detect'] và khả năng tắt/bật hoàn toàn đường dự phòng suspect_missing."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    md_def = default_ns['RT']['missing_detect']
    assert md_def['enabled'] is True
    assert md_def['check_sec'] == 1.0
    assert md_def['sustain_sec'] == 1.5
    assert md_def['max_side'] == 160
    assert md_def['person_range_norm'] == 2.0
    assert md_def['global_luma_diff'] == 25.0
    assert md_def['global_hist_thr'] == 0.35

    old_rt = copy.deepcopy(settings.RT)
    try:
        def make_frame(present=True):
            f = np.full((H, W, 3), 90, np.uint8)
            if present:
                cv2.rectangle(f, (560, 180), (660, 280), (210, 160, 40), -1)
                cv2.rectangle(f, (575, 195), (645, 265), (40, 40, 200), 3)
            return f

        mask = np.zeros((H, W), bool)
        mask[180:280, 560:660] = True
        near_rec = dict(id=1, box=[420.0, 180.0, 500.0, 400.0])  # h=220, dist=60/220=0.27 <= 2.0

        for off_val in (False, dict(enabled=False)):
            settings.RT['missing_detect'] = off_val
            submitted = []
            ctx = SimpleNamespace(fps=25.0, n=150, last_frame=0, artifact_by_id={0: dict(box=[560.0, 180.0, 660.0, 280.0])},
                                  vlm=SimpleNamespace(submit=lambda *a, **kw: None), cur_scene=0,
                                  scene_index=lambda _i: 0, d=dict(W=W, H=H))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (inter.tasks.append(g), submitted.append(g))
            inter.set_artifacts([dict(id=0, box=[560.0, 180.0, 660.0, 280.0], pedestal=False)], {0: mask})
            for fi in range(120):
                inter.step(fi, [near_rec], work=make_frame(present=(fi < 25)))
            assert submitted == [], f'khi tắt missing_detect ({off_val}) không được phát suspect_missing: {submitted}'
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_missing_detect_disappear_and_proximity():
    """Phát hiện hiện vật biến mất >= 1.5s khi có người trong vòng 2x chiều cao; bỏ qua khi chớp ngắn, người xa, hoặc bị che."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    old_rt = copy.deepcopy(settings.RT)
    try:
        settings.RT['periodic_vlm'] = dict(enabled=False)
        settings.RT['missing_detect'] = dict(
            enabled=True, check_sec=1.0, sustain_sec=1.5, max_side=160, change_thr=0.35,
            edge_weight=0.5, edge_diff_scale=64.0, color_diff_scale=48.0,
            max_person_occlude=0.25, person_range_norm=2.0, global_luma_diff=25.0,
            global_hist_thr=0.35, cooldown_sec=4.0, carry_detect=False,
        )

        mask = np.zeros((H, W), bool)
        mask[160:260, 560:660] = True
        art = [dict(id=0, box=[560.0, 160.0, 660.0, 260.0], pedestal=False)]

        def make_frame(present=True):
            f = np.full((H, W, 3), 90, np.uint8)
            if present:
                cv2.rectangle(f, (560, 160), (660, 260), (200, 190, 60), -1)
                cv2.line(f, (570, 170), (650, 250), (30, 30, 180), 4)
            return f

        def setup():
            submitted = []
            ctx = SimpleNamespace(fps=25.0, n=200, last_frame=0, artifact_by_id={0: dict(box=[560.0, 160.0, 660.0, 260.0])},
                                  vlm=SimpleNamespace(submit=lambda *a, **kw: None), cur_scene=0,
                                  scene_index=lambda _i: 0, d=dict(W=W, H=H))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (g.update(name=f'm_{len(inter.tasks)}'), inter.tasks.append(g), submitted.append(g))
            inter.set_artifacts(art, {0: mask}, ref_idx=0)
            return inter, submitted

        # 1) Hiện vật biến mất từ frame 30 đến 120, người đứng gần (dist = 100/200 = 0.5 <= 2.0) -> tạo event suspect_missing
        inter1, sub1 = setup()
        near_rec = dict(id=7, box=[380.0, 180.0, 460.0, 380.0])
        for fi in range(120):
            inter1.step(fi, [near_rec], work=make_frame(present=(fi < 30)))
        assert len(sub1) == 1, f'kỳ vọng đúng 1 suspect_missing, nhận {len(sub1)}'
        ev1 = inter1.events[0]
        assert ev1['event_type'] == 'suspect_missing' and ev1['track_id'] == 7 and ev1['artifact_id'] == 0
        assert ev1['end_frame'] - ev1['start_frame'] >= int(1.5 * 25)
        assert inter1.missing_detector.small_shape[0] <= 160 and inter1.missing_detector.small_shape[1] <= 160

        # 2) Biến mất ngắn (< 1.5s: frame 25..42 = 0.68s) rồi hiện lại -> KHÔNG báo
        inter2, sub2 = setup()
        for fi in range(120):
            inter2.step(fi, [near_rec], work=make_frame(present=not (25 <= fi <= 42)))
        assert sub2 == [], f'biến mất < 1.5s không được báo: {sub2}'

        # 3) Biến mất lâu nhưng người ở XA (> 2x chiều cao: x=0..80, h=200 -> dist=480/200=2.4 > 2.0) -> KHÔNG báo
        inter3, sub3 = setup()
        far_rec = dict(id=8, box=[0.0, 180.0, 80.0, 380.0])
        for fi in range(120):
            inter3.step(fi, [far_rec], work=make_frame(present=(fi < 30)))
        assert sub3 == [], f'người ngoài 2x chiều cao không được báo: {sub3}'

        # 4) Hiện vật bị bbox người CHE phủ (> 25% mask) -> bỏ qua khi bị che, không báo giả
        inter4, sub4 = setup()
        occlude_rec = dict(id=9, box=[540.0, 150.0, 680.0, 390.0])
        for fi in range(120):
            inter4.step(fi, [occlude_rec], work=make_frame(present=(fi < 30)))
        assert sub4 == [], f'bị người che khuất không được báo giả: {sub4}'
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_missing_detect_lighting_and_scene_cut_guard():
    """Đổi sáng toàn cục / đổi xám->màu / scene_cut phải bỏ qua và reset streak, không gây FP."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    old_rt = copy.deepcopy(settings.RT)
    try:
        settings.RT['periodic_vlm'] = dict(enabled=False)
        settings.RT['missing_detect'] = dict(
            enabled=True, check_sec=1.0, sustain_sec=1.5, max_side=160, change_thr=0.35,
            edge_weight=0.5, edge_diff_scale=64.0, color_diff_scale=48.0,
            max_person_occlude=0.25, person_range_norm=2.0, global_luma_diff=25.0,
            global_hist_thr=0.35, cooldown_sec=4.0, carry_detect=False,
        )

        mask = np.zeros((H, W), bool)
        mask[160:260, 560:660] = True
        art = [dict(id=0, box=[560.0, 160.0, 660.0, 260.0], pedestal=False)]
        near_rec = dict(id=3, box=[400.0, 180.0, 480.0, 380.0])

        def setup():
            submitted = []
            ctx = SimpleNamespace(fps=25.0, n=250, last_frame=0, artifact_by_id={0: dict(box=[560.0, 160.0, 660.0, 260.0])},
                                  vlm=SimpleNamespace(submit=lambda *a, **kw: None), cur_scene=0,
                                  scene_index=lambda _i: 0, d=dict(W=W, H=H))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (inter.tasks.append(g), submitted.append(g))
            inter.set_artifacts(art, {0: mask}, ref_idx=0)
            return inter, submitted

        # 1) Đổi độ sáng toàn cục (+80 luma ở frame 30..120) -> không báo giả
        inter_luma, sub_luma = setup()
        for fi in range(120):
            bg_val = 70 if fi < 30 else 155
            f = np.full((H, W, 3), bg_val, np.uint8)
            cv2.rectangle(f, (560, 160), (660, 260), (min(255, bg_val + 60), bg_val, bg_val), -1)
            inter_luma.step(fi, [near_rec], work=f)
        assert sub_luma == [], f'đổi độ sáng toàn cục không được tạo FP: {sub_luma}'

        # 2) Đổi xám -> màu (video 11) không tách cảnh -> hist_diff >= 0.35 bỏ qua -> không báo giả
        inter_col, sub_col = setup()
        for fi in range(120):
            col = (120, 120, 120) if fi < 30 else (40, 120, 200)
            f = np.full((H, W, 3), col, np.uint8)
            cv2.rectangle(f, (560, 160), (660, 260), (160, 160, 160) if fi < 30 else (220, 160, 80), -1)
            inter_col.step(fi, [near_rec], work=f)
        assert sub_col == [], f'đổi xám->màu không được tạo FP: {sub_col}'

        # 3) Scene cut qua set_artifacts: streak từ cảnh 0 phải bị xoá, cảnh 1 dùng frame tham chiếu mới
        inter_sc, sub_sc = setup()
        for fi in range(55):
            # Cảnh 0: hiện vật biến mất ở f25..54 (mới ~1.16s < 1.5s)
            f = np.full((H, W, 3), 90, np.uint8)
            if fi < 25:
                cv2.rectangle(f, (560, 160), (660, 260), (200, 180, 50), -1)
            inter_sc.step(fi, [near_rec], work=f)
        assert sub_sc == []
        # Sang cảnh 1 tại frame 55 (ref_idx=55) với nền mới và hiện vật mới còn nguyên
        inter_sc.set_artifacts([dict(id=10000, box=[560.0, 160.0, 660.0, 260.0], pedestal=False)],
                               {10000: mask}, ref_idx=55, scene_start=55)
        for fi in range(55, 130):
            f = np.full((H, W, 3), (40, 110, 180), np.uint8)
            cv2.rectangle(f, (560, 160), (660, 260), (220, 220, 220), -1)
            inter_sc.step(fi, [near_rec], work=f)
        assert sub_sc == [], f'scene_cut phải reset streak cũ và so với ref cảnh mới: {sub_sc}'
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_missing_detect_carry_out_signal():
    """Người đi tay không tới gần/khuất rồi mang vật đi ra >= 1.5s -> tạo suspect_missing; khách đi tay không bình thường -> không báo."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    old_rt = copy.deepcopy(settings.RT)
    try:
        settings.RT['periodic_vlm'] = dict(enabled=False)
        settings.RT['missing_detect'] = dict(
            enabled=True, check_sec=1.0, sustain_sec=1.5, max_side=160, change_thr=0.35,
            edge_weight=0.5, edge_diff_scale=64.0, color_diff_scale=48.0,
            max_person_occlude=0.25, person_range_norm=2.0, global_luma_diff=25.0,
            global_hist_thr=0.35, cooldown_sec=4.0, carry_detect=True,
            carry_change_thr=0.40, carry_sustain_sec=1.5,
        )

        # Cảnh không có mask hiện vật (mô phỏng mask thiếu), người dừng lại lấy vật rồi mang đi ra
        def run_walk(carrying_after):
            submitted = []
            ctx = SimpleNamespace(fps=25.0, n=180, last_frame=0, artifact_by_id={},
                                  vlm=SimpleNamespace(submit=lambda *a, **kw: None), cur_scene=0,
                                  scene_index=lambda _i: 0, d=dict(W=W, H=H))
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (g.update(name=f'c_{len(inter.tasks)}'), inter.tasks.append(g), submitted.append(g))
            inter.set_artifacts([], {}, ref_idx=0)

            for fi in range(150):
                if fi < 25:
                    x1 = 100.0 + fi * 12.0       # đi vào (x=100..400)
                elif fi < 50:
                    x1 = 400.0                   # dừng lại ở khu vực trưng bày (25 frame = 1.0s)
                else:
                    x1 = 400.0 + (fi - 50) * 4.0 # đi ra (x=400..800)
                box = [x1, 180.0, x1 + 80.0, 400.0]
                f = np.full((H, W, 3), 90, np.uint8)
                cv2.rectangle(f, (int(x1), 180), (int(x1 + 80), 400), (0, 180, 0), -1)
                if carrying_after and fi >= 45:
                    # Vật mang theo trước ngực/tay (hộp sáng có viền)
                    cv2.rectangle(f, (int(x1 + 10), 230), (int(x1 + 75), 340), (240, 240, 240), -1)
                    cv2.rectangle(f, (int(x1 + 10), 230), (int(x1 + 75), 340), (20, 20, 220), 4)
                inter.step(fi, [dict(id=5, box=box)], work=f)
            return inter, submitted

        inter_c, sub_c = run_walk(carrying_after=True)
        assert len(sub_c) == 1, f'người mang vật đi ra phải kích hoạt suspect_missing: {sub_c}'
        assert inter_c.events[0]['event_type'] == 'suspect_missing' and inter_c.events[0]['carry'] is True

        _, sub_norm = run_walk(carrying_after=False)
        assert sub_norm == [], f'khách đi tay không bình thường không được báo giả: {sub_norm}'
    finally:
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_missing_detect_end_to_end_pipeline():
    """End-to-end: không có chạm tay (pose xa hơn state_contact_thr), hiện vật biến mất khi có người trong 2x chiều cao -> VLM trả stealing -> alert đỏ."""
    import copy
    from exhibitflow import run_pipeline

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            vid = tmp / 'steal_missing.avi'
            n_frames = 140
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for fi in range(n_frames):
                f = np.full((H, W, 3), 90, np.uint8)
                # Hiện vật tồn tại frame 0..34, biến mất từ frame 35 trở đi
                if fi < 35:
                    cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (220, 200, 60), -1)
                    cv2.rectangle(f, (ART[0] + 10, ART[1] + 10), (ART[2] - 10, ART[3] - 10), (30, 30, 200), 4)
                # Người màu xanh đứng ở x=390..470, y=200..420 (h=220, cách hiện vật 90px -> norm_dist ~0.41..0.55, không vào CONTACT_LIKE)
                if 5 <= fi <= 125:
                    cv2.rectangle(f, (390, 200), (470, 420), (0, 200, 0), -1)
                wr.write(f)
            wr.release()

            @register('pose', 'fake_t9_far_hand_pose')
            class FakeT9FarHandPose:
                def __init__(self, **_):
                    pass

                def __call__(self, native, recs, S):
                    for r in recs:
                        x1, y1, x2, y2 = r['box']
                        cx, h = (x1 + x2) / 2, y2 - y1
                        # Cổ tay giữ sát thân (cách hiện vật ~ 130/220 = 0.59 > state_contact_thr=0.35 -> không có contact event)
                        kp = [[cx, y1 + 10]] * 5 + [
                            [cx - 25, y1 + .2 * h], [cx + 25, y1 + .2 * h],
                            [cx - 25, y1 + .4 * h], [cx + 25, y1 + .4 * h],
                            [cx - 20, y1 + .5 * h], [cx + 20, y1 + .5 * h],
                            [cx - 15, y1 + .55 * h], [cx + 15, y1 + .55 * h],
                            [cx - 15, y1 + .75 * h], [cx + 15, y1 + .75 * h],
                            [cx - 15, y2], [cx + 15, y2],
                        ]
                        r['kp'], r['kc'] = kp, [1.0] * 17

            models = dict(
                FAKE_MODELS,
                pose=dict(backend='fake_t9_far_hand_pose'),
                vlm=dict(backend='dry_run', delay_s=0.0, label='stealing'),
            )
            out = tmp / 'out'
            summary = run_pipeline(
                [str(vid)],
                out_dir=out,
                work_dir=tmp / 'work',
                overrides=dict(
                    models=models,
                    rt=dict(queue_size=4, pose_gate_norm_dist=None, periodic_vlm=dict(enabled=False)),
                ),
            )
            assert not summary['errors'], summary['errors']
            res = json.loads((out / 'steal_missing' / 'result.json').read_text(encoding='utf-8'))
            assert res['summary']['n_events'] >= 1, res['summary']
            assert any(ev.get('hand') == 'suspect_missing' for ev in res['events']), res['events']
            assert res['summary']['n_alerts'] >= 1 and res['summary']['has_anomaly'] is True, res['summary']
            assert res['alerts'][0]['behavior'] == 'stealing', res['alerts']
            # Prompt gửi VLM vẫn giữ nguyên nhãn 'stealing'
            prompt_files = list((out / 'steal_missing' / 'vlm_inputs').glob('*/prompt.txt'))
            assert prompt_files and '"stealing"' in prompt_files[0].read_text(encoding='utf-8')
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_tracker_cfg_defaults_and_set_overrides():
    """Kiểm tra mặc định MODELS['tracker'], MODELS['detector'].low_light_boost và ghi đè mọi giá trị qua --set."""
    import copy
    from main import parse_set

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)
    tr = default_ns['MODELS']['tracker']
    assert tr['track_buffer'] == 75, tr
    assert tr['track_thresh'] == 0.5, tr
    assert tr['reid_weight'] == 0.5, tr
    assert tr['reid_thresh'] == 0.6, tr
    assert tr['ltc_min_history'] == 10, tr
    assert tr['keep_predicted_frames'] == 8, tr
    assert tr['attach_min_iou'] == 0.30, tr
    assert tr['attach_stable_min_iou'] == 0.15, tr
    assert tr['attach_stable_frames'] == 10, tr
    assert default_ns['MODELS']['detector']['low_light_boost']['enabled'] is True
    assert default_ns['CFG']['pose_predicted_expand'] is True

    old_models, old_cfg = copy.deepcopy(settings.MODELS), copy.deepcopy(settings.CFG)
    try:
        ov = parse_set([
            'models.tracker.track_buffer=90',
            'models.tracker.track_thresh=0.45',
            'models.tracker.reid_weight=0.6',
            'models.tracker.reid_thresh=0.55',
            'models.tracker.ltc_min_history=12',
            'models.tracker.keep_predicted_frames=6',
            'models.detector.low_light_boost.enabled=false',
            'cfg.pose_predicted_expand=false',
        ])
        settings.apply_overrides(ov)
        assert settings.MODELS['tracker']['track_buffer'] == 90
        assert settings.MODELS['tracker']['track_thresh'] == 0.45
        assert settings.MODELS['tracker']['reid_weight'] == 0.6
        assert settings.MODELS['tracker']['reid_thresh'] == 0.55
        assert settings.MODELS['tracker']['ltc_min_history'] == 12
        assert settings.MODELS['tracker']['keep_predicted_frames'] == 6
        assert settings.MODELS['detector']['low_light_boost']['enabled'] is False
        assert settings.CFG['pose_predicted_expand'] is False

        # Bật/tắt low_light_boost dạng bool trực tiếp qua --set
        settings.apply_overrides(parse_set(['models.detector.low_light_boost=false']))
        assert settings.MODELS['detector']['low_light_boost'] is False
    finally:
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.CFG.clear()
        settings.CFG.update(old_cfg)


def test_attach_scores_stable_min_iou_and_predicted_decay():
    """Kiểm tra _attach_scores: min_iou 0.30->0.15 cho track >=10 frame, keep_predicted_frames=8 và suy giảm score."""
    from exhibitflow.common import box_iou
    from exhibitflow.perception import _attach_scores

    tb = [100.0, 100.0, 200.0, 300.0]
    # Box lệch có IoU ~0.20 (nằm giữa 0.15 và 0.30)
    shifted_det = [166.0, 100.0, 266.0, 300.0, 0.88]
    iou = box_iou(tb, shifted_det[:4])
    assert 0.15 <= iou < 0.30, iou

    # 1) Track mới (< 10 frame) không khớp shifted_det với min_iou=0.30 -> khi keep_predicted_frames=0 bị bỏ (legacy)
    assert _attach_scores([(1, tb)], [shifted_det], keep_predicted_frames=0) == []

    # 2) Track mới giữ được tối đa keep_predicted_frames=8 với predicted=True và score suy giảm, frame 9 bị loại
    st_new = {}
    matched = _attach_scores([(1, tb)], [[100.0, 100.0, 200.0, 300.0, 0.90]], keep_predicted_frames=8, track_state=st_new)
    assert len(matched) == 1 and matched[0]['predicted'] is False and matched[0]['score'] == 0.90

    prev_score = 0.90
    for k in range(1, 9):
        res = _attach_scores([(1, tb)], [], keep_predicted_frames=8, track_state=st_new)
        assert len(res) == 1, f'frame dự đoán {k} phải còn'
        assert res[0]['predicted'] is True and res[0]['predicted_frames'] == k
        assert res[0]['score'] < prev_score, (res[0]['score'], prev_score)
        prev_score = res[0]['score']
    # Quá 8 frame dự đoán -> loại để không tạo event mới trên dự đoán dài
    assert _attach_scores([(1, tb)], [], keep_predicted_frames=8, track_state=st_new) == []

    # 3) Track đã ổn định (>= 10 frame) dùng min_iou=0.15 -> khớp shifted_det (IoU ~0.20)
    st_stable = {}
    for _ in range(10):
        _attach_scores([(2, tb)], [[100.0, 100.0, 200.0, 300.0, 0.85]], track_state=st_stable)
    res_stable = _attach_scores([(2, tb)], [shifted_det], track_state=st_stable)
    assert len(res_stable) == 1 and res_stable[0]['predicted'] is False and res_stable[0]['score'] == 0.88
    # Cũng hỗ trợ truyền hits trực tiếp qua metadata không cần track_state
    res_meta = _attach_scores([(3, tb, {'hits': 10})], [shifted_det])
    assert len(res_meta) == 1 and res_meta[0]['predicted'] is False and res_meta[0]['score'] == 0.88


def test_predicted_pose_crop_and_low_light_boost():
    """Kiểm tra mở rộng crop pose cho box dự đoán, low_light_boost (CLAHE kênh L) và ByteTrackDMALTC wrapper với mock tracker."""
    import copy
    from types import SimpleNamespace
    from exhibitflow.perception import ByteTrackDMALTC, _crop_box, apply_low_light_boost

    old_cfg = copy.deepcopy(settings.CFG)
    try:
        # 1) _crop_box mở rộng thêm pose_crop_pad_frac khi predicted=True
        rec_det = dict(id=1, box=[300.0, 150.0, 400.0, 350.0], predicted=False)
        rec_pred = dict(id=1, box=[300.0, 150.0, 400.0, 350.0], predicted=True)
        _, c_det = _crop_box(rec_det, 1.0, 960, 540)
        _, c_pred = _crop_box(rec_pred, 1.0, 960, 540)
        assert (c_pred[2] - c_pred[0]) > (c_det[2] - c_det[0])
        assert (c_pred[3] - c_pred[1]) > (c_det[3] - c_det[1])
        settings.CFG['pose_predicted_expand'] = False
        _, c_off = _crop_box(rec_pred, 1.0, 960, 540)
        assert c_off == c_det
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)

    # 2) low_light_boost: ảnh sáng giữ nguyên, ảnh tối tăng tương phản kênh L, tắt thì giữ nguyên
    bright = np.full((120, 160, 3), 160, np.uint8)
    assert np.array_equal(apply_low_light_boost(bright), bright)

    dark = np.zeros((120, 160, 3), np.uint8)
    dark[:, :80] = (15, 15, 15)
    dark[:, 80:] = (35, 35, 35)
    boosted = apply_low_light_boost(dark, cfg=dict(enabled=True, luma_thr=75.0, clip_limit=2.0, tile_grid=4))
    assert boosted.std() > dark.std()
    assert np.array_equal(apply_low_light_boost(dark, cfg=False), dark)

    # 3) ByteTrackDMALTC.update dùng lost_stracks (box dự đoán Kalman/LTC) khi không có detection khớp
    wrapper = object.__new__(ByteTrackDMALTC)
    wrapper.min_box_area = 100
    wrapper.max_aspect = 1.6
    wrapper.keep_predicted_frames = 8
    wrapper.attach_min_iou = 0.30
    wrapper.attach_stable_min_iou = 0.15
    wrapper.attach_stable_frames = 10
    wrapper.predicted_score_decay = 0.9
    wrapper._track_state = {}

    strack = SimpleNamespace(
        track_id=7,
        tlwh=np.array([200.0, 150.0, 80.0, 200.0], np.float32),
        tlbr=np.array([200.0, 150.0, 280.0, 350.0], np.float32),
        score=0.85,
        is_activated=True,
        tracklet_len=12,
        missing_count=0,
        end_frame=1,
        smooth_feat=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        curr_feat=None,
    )
    frame_img = np.zeros((540, 960, 3), np.uint8)
    wrapper.tracker = SimpleNamespace(frame_id=1, lost_stracks=[], update=lambda *a, **kw: [strack])
    out1 = wrapper.update([[200.0, 150.0, 280.0, 350.0, 0.85]], frame_img)
    assert len(out1) == 1 and out1[0]['id'] == 7 and out1[0]['predicted'] is False

    # Frame 2..9: detector hụt -> mock tracker chuyển sang lost_stracks với box dự đoán dịch nhẹ
    for miss in range(1, 10):
        strack.missing_count = miss
        strack.tlbr = np.array([200.0 + miss * 2.0, 150.0, 280.0 + miss * 2.0, 350.0], np.float32)
        strack.tlwh = np.array([200.0 + miss * 2.0, 150.0, 80.0, 200.0], np.float32)
        wrapper.tracker = SimpleNamespace(frame_id=1 + miss, lost_stracks=[strack], update=lambda *a, **kw: [])
        out_m = wrapper.update([], frame_img)
        if miss <= 8:
            assert len(out_m) == 1 and out_m[0]['id'] == 7
            assert out_m[0]['predicted'] is True and out_m[0]['predicted_frames'] == miss
            assert out_m[0]['score'] < 0.85
            assert 'reid_feat' in out_m[0]
        else:
            assert out_m == []


def test_predicted_track_100frames_gap_end_to_end():
    """Giả lập 100 frame, detection hụt hoàn toàn frame 30-60: track vẫn sống predicted=True (<=8f), pose được gọi, ID không đổi."""
    import copy
    from types import SimpleNamespace
    from exhibitflow import run_pipeline
    from exhibitflow.common import box_iou

    n_frames = 100
    pose_frames_called = []
    pose_crop_widths = {'normal': [], 'predicted': []}

    @register('detector', 'fake_gap_100')
    class FakeGapDetector:
        def __init__(self, **_):
            self._idx = -1

        def detect(self, frame):
            # segment_all_videos không gọi detector nếu artifact_person_filter=False, nhưng giữ an toàn
            self._idx += 1
            i = self._idx
            if 30 <= i <= 60:
                return []  # Hụt detection hoàn toàn ở frame 30..60 (31 frame)
            # Người di chuyển từ x=250 (f0) tới x=430 (f61..f99 sát hiện vật ART x=560)
            x1 = 250.0 + min(i, 61) * ((430.0 - 250.0) / 61.0)
            return [[x1, 200.0, x1 + 80.0, 420.0, 0.88]]

    class MockByteTrackerWithGap:
        """Mock BYTETracker giữ lost_stracks tới track_buffer=75 và dự đoán vị trí tuyến tính."""

        def __init__(self, track_buffer=75):
            self.frame_id = 0
            self.track_buffer = track_buffer
            self.lost_stracks = []
            self._active = None
            self._vel_x = (430.0 - 250.0) / 61.0

        def update(self, arr, img_info, img_size, frame=None):
            self.frame_id += 1
            if len(arr) > 0:
                d = arr[0]
                if self._active is None:
                    self._active = SimpleNamespace(
                        track_id=1,
                        tlwh=np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]], np.float32),
                        tlbr=np.array(d[:4], np.float32),
                        score=float(d[4]),
                        is_activated=True,
                        tracklet_len=1,
                        missing_count=0,
                        end_frame=self.frame_id,
                    )
                else:
                    # Kiểm tra match lại với box dự đoán trong track_buffer=75
                    pred_box = self._active.tlbr.tolist()
                    if self._active.missing_count <= self.track_buffer and box_iou(pred_box, d[:4].tolist()) >= 0.05:
                        self._active.tlbr = np.array(d[:4], np.float32)
                        self._active.tlwh = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]], np.float32)
                        self._active.score = float(d[4])
                        self._active.tracklet_len += 1
                        self._active.missing_count = 0
                        self._active.end_frame = self.frame_id
                self.lost_stracks = []
                return [self._active]

            if self._active is not None:
                self._active.missing_count += 1
                if self._active.missing_count <= self.track_buffer:
                    b = self._active.tlbr
                    nb = np.array([b[0] + self._vel_x, b[1], b[2] + self._vel_x, b[3]], np.float32)
                    self._active.tlbr = nb
                    self._active.tlwh = np.array([nb[0], nb[1], nb[2] - nb[0], nb[3] - nb[1]], np.float32)
                    self.lost_stracks = [self._active]
                else:
                    self.lost_stracks = []
            return []

    @register('tracker', 'fake_bytetrack_gap')
    class FakeByteTrackGapWrapper:
        def __init__(self, keep_predicted_frames=8, track_buffer=75, **_):
            from exhibitflow.perception import ByteTrackDMALTC
            self._inner = object.__new__(ByteTrackDMALTC)
            self._inner.min_box_area = 100
            self._inner.max_aspect = 1.6
            self._inner.keep_predicted_frames = int(keep_predicted_frames)
            self._inner.attach_min_iou = 0.30
            self._inner.attach_stable_min_iou = 0.15
            self._inner.attach_stable_frames = 10
            self._inner.predicted_score_decay = 0.9
            self._inner._track_state = {}
            self._track_buffer = int(track_buffer)

        def reset(self, fps):
            self._inner._track_state = {}
            self._inner.tracker = MockByteTrackerWithGap(track_buffer=self._track_buffer)

        def update(self, dets, frame, native=None, S=1.0):
            return self._inner.update(dets, frame, native=native, S=S)

    @register('pose', 'fake_gap_pose')
    class FakeGapPose:
        def __init__(self, **_):
            self._call_idx = -1

        def __call__(self, native, recs, S):
            from exhibitflow.perception import _crop_box
            nh, nw = native.shape[:2]
            for r in recs:
                _, crop = _crop_box(r, S, nw, nh)
                cw = crop[2] - crop[0]
                pose_crop_widths['predicted' if r.get('predicted') else 'normal'].append(cw)
                pose_frames_called.append((bool(r.get('predicted')), r['id']))
                x1, y1, x2, y2 = r['box']
                cx, h = (x1 + x2) / 2, y2 - y1
                kp = [[cx, y1 + 10]] * 5 + [
                    [cx - 25, y1 + .2 * h], [cx + 25, y1 + .2 * h],
                    [cx - 30, y1 + .4 * h], [x2 + 10, y1 + .4 * h],
                    [cx - 30, y1 + .6 * h], [x2 + 15, y1 + .5 * h],
                    [cx - 15, y1 + .55 * h], [cx + 15, y1 + .55 * h],
                    [cx - 15, y1 + .75 * h], [cx + 15, y1 + .75 * h],
                    [cx - 15, y2], [cx + 15, y2],
                ]
                r['kp'], r['kc'] = kp, [1.0] * 17

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            video = tmp / 'gap_100.avi'
            wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for _ in range(n_frames):
                f = np.full((H, W, 3), 90, np.uint8)
                cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (200, 200, 200), -1)
                wr.write(f)
            wr.release()

            tracks_by_frame = {}

            def on_frame(ctx, pkt, vis, perf):
                tracks_by_frame[pkt.idx] = [dict(r) for r in (pkt.tracks or [])]

            out = tmp / 'out'
            models = dict(
                FAKE_MODELS,
                detector=dict(backend='fake_gap_100'),
                tracker=dict(backend='fake_bytetrack_gap', track_buffer=75, keep_predicted_frames=8),
                pose=dict(backend='fake_gap_pose'),
                vlm=dict(backend='dry_run', delay_s=0.0, label='touching'),
            )
            summary = run_pipeline(
                [str(video)],
                out_dir=out,
                work_dir=tmp / 'work',
                max_frames=n_frames,
                overrides=dict(
                    models=models,
                    rt=dict(queue_size=4, pose_gate_norm_dist=None, artifact_person_filter=False, frame_hook=on_frame),
                ),
            )
            assert not summary['errors'], summary['errors']

            # Frame 30..37 (8 frame đầu của khoảng hụt): track vẫn sống, predicted=True, score suy giảm
            for fi in range(30, 38):
                assert len(tracks_by_frame[fi]) == 1, (fi, tracks_by_frame[fi])
                rec = tracks_by_frame[fi][0]
                assert rec['id'] == 1 and rec['predicted'] is True and rec['predicted_frames'] == fi - 29
                assert rec['score'] < 0.88

            # Frame 38..60 (> 8 frame dự đoán): không giữ predicted track -> không tạo event mới ở frame dự đoán dài > 8
            for fi in range(38, 61):
                assert tracks_by_frame[fi] == [], (fi, tracks_by_frame[fi])

            # Pose được gọi trên các frame dự đoán với crop mở rộng lớn hơn bình thường
            assert any(is_pred and tid == 1 for is_pred, tid in pose_frames_called)
            assert sum(1 for is_pred, _ in pose_frames_called if is_pred) == 8
            assert min(pose_crop_widths['predicted']) > max(pose_crop_widths['normal'])

            # Frame 61..99: detection quay lại -> tiếp tục đúng ID 1, không đổi ID
            for fi in range(61, n_frames):
                assert len(tracks_by_frame[fi]) == 1 and tracks_by_frame[fi][0]['id'] == 1
                assert tracks_by_frame[fi][0]['predicted'] is False

            res = json.loads((out / 'gap_100' / 'result.json').read_text(encoding='utf-8'))
            assert res['summary']['n_people_tracks'] == 1, res['summary']
            assert res['alerts'] and res['alerts'][0]['track_id'] == 1, res['alerts']
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_helpers():
    from exhibitflow.vlm import adapt_task, parse_response
    from main import parse_set
    assert parse_set(['models.vlm.max_side=768', 'ALARM_ONSET=event', 'rt.queue_size=4']) == \
        {'models': {'vlm': {'max_side': 768}}, 'ALARM_ONSET': 'event', 'rt': {'queue_size': 4}}
    assert parse_response('```json\n{"label": "Stealing", "confidence": 0.7}\n```')['label'] == 'stealing'
    assert parse_response('no json')  is None
    assert parse_response('{"label": "dancing"}')['label'] == 'invalid:dancing'
    task = dict(images=[f'x{i}.jpg' for i in range(12)], times=[i * .5 for i in range(12)], tid=3,
                prompt=settings.PROMPT_TEMPLATE.format(n=12, dt=0.5, tid=3, before_note=''))
    t = adapt_task(task, dict(n_images=8, short_json=True))
    assert len(t['images']) == 8 and 'contact_frames' not in t['prompt'].split('Answer JSON only:')[1] and 'The 8 images' in t['prompt']
    assert adapt_task(task, {}) is task
    before = settings.RT['queue_size']
    settings.apply_overrides({'rt': {'queue_size': before + 1}})
    assert settings.RT['queue_size'] == before + 1
    settings.RT['queue_size'] = before
    try:
        settings.apply_overrides({'nonsense': 1})
        raise AssertionError('phải báo lỗi khoá lạ')
    except KeyError:
        pass


def test_tuning_registry_paths_and_mandatory_keys():
    """(1) Mọi path trong tuning.PARAMS tồn tại thật trong settings và đủ toàn bộ nhóm/khoá bắt buộc."""
    from exhibitflow import tuning

    default_ns = {'__file__': settings.__file__}
    exec(compile(Path(settings.__file__).read_text(encoding='utf-8'), settings.__file__, 'exec'), default_ns)

    keys = [p.key for p in tuning.PARAMS]
    paths = [p.path for p in tuning.PARAMS]
    assert len(keys) == len(set(keys)), 'key bị trùng trong tuning.PARAMS'
    assert len(paths) == len(set(paths)), 'path bị trùng trong tuning.PARAMS'

    group_ids = {g['id'] for g in tuning.GROUPS}
    for p in tuning.PARAMS:
        assert p.group in group_ids, f'group lạ: {p.group}'
        assert p.type in ('bool', 'int', 'float'), p.type
        assert p.label_vi and p.help_vi
        # Tồn tại thật trong settings và default khớp settings.py gốc
        parts = p.path.split('.')
        cur = default_ns[parts[0]]
        for part in parts[1:]:
            assert isinstance(cur, dict) and part in cur, f'Thiếu path {p.path} trong settings.py'
            cur = cur[part]
        assert p.default == cur, f'Default của {p.path} không khớp settings.py: {p.default!r} vs {cur!r}'
        if p.path.startswith('MODELS.tracker.') or p.path.startswith('MODELS.detector.low_light_boost.'):
            assert p.needs_reload is True, f'{p.path} phải có needs_reload=True'
        else:
            assert p.needs_reload is False, f'{p.path} không được đặt needs_reload=True'

    required_paths = {
        'RT.alarm_persist',
        'RT.flag_inherit.enabled', 'RT.flag_inherit.reid_cos', 'RT.flag_inherit.max_gap_sec', 'RT.flag_inherit.max_center_dist_norm',
        'MODELS.tracker.track_thresh', 'MODELS.tracker.track_buffer', 'MODELS.tracker.match_thresh',
        'MODELS.tracker.reid_weight', 'MODELS.tracker.reid_thresh', 'MODELS.tracker.ltc_min_history',
        'MODELS.tracker.keep_predicted_frames', 'MODELS.tracker.attach_min_iou',
        'MODELS.tracker.attach_stable_min_iou', 'MODELS.tracker.attach_stable_frames', 'MODELS.tracker.predicted_score_decay',
        'MODELS.detector.low_light_boost.enabled', 'MODELS.detector.low_light_boost.luma_thr', 'MODELS.detector.low_light_boost.clip_limit',
        'RT.person_filter.enabled', 'RT.person_filter.display_min_score', 'RT.person_filter.min_hits',
        'RT.person_filter.min_box_area', 'RT.person_filter.min_valid_kpts', 'RT.person_filter.pose_min_valid_frac',
        'RT.artifact_person_min_score',
        'CFG.pose_smooth', 'CFG.pose_smooth_min_cutoff', 'CFG.pose_smooth_beta', 'CFG.pose_wrist_interp_max_gap',
        'CFG.pose_conf_thr', 'CFG.hand_conf_thr', 'CFG.state_lost_grace_frames',
        'CFG.state_hysteresis', 'CFG.state_hysteresis_mult',
        'CFG.relaxed_gate', 'CFG.relaxed_event_min_frames', 'CFG.relaxed_min_person_h_px',
        'CFG.relaxed_hold_seconds', 'CFG.relaxed_pedestal_contact_thr',
        'CFG.state_near_thr', 'CFG.state_contact_thr', 'CFG.state_min_person_h_px', 'CFG.event_min_frames',
        'RT.periodic_vlm.enabled', 'RT.periodic_vlm.min_near_sec', 'RT.periodic_vlm.cooldown_sec',
        'RT.missing_detect.enabled', 'RT.missing_detect.check_sec', 'RT.missing_detect.sustain_sec',
        'RT.missing_detect.change_thr', 'RT.missing_detect.carry_detect', 'RT.missing_detect.carry_change_thr',
        'RT.missing_detect.carry_sustain_sec', 'RT.missing_detect.max_person_occlude',
        'RT.seg_multi_frame.enabled', 'RT.seg_multi_frame.step_sec', 'RT.seg_multi_frame.max_frames_per_scene',
        'RT.scene_color_cut', 'RT.scene_color_hist_thr', 'RT.scene_cut_diff',
        'SEGMENTATION_CFG.full_thr', 'SEGMENTATION_CFG.fallback_thr',
    }
    missing = required_paths - set(paths)
    assert not missing, f'Thiếu các khoá bắt buộc trong tuning.PARAMS: {sorted(missing)}'


def test_tuning_overrides_and_apply_overrides():
    """(2) overrides_from_settings sinh đúng override lồng, apply_overrides đổi được giá trị và giữ nguyên khoá cũ."""
    import copy
    from dashboard.runner import DEFAULT_SETTINGS, overrides_from_settings

    old_cfg, old_rt, old_models, old_seg = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.SEGMENTATION_CFG),
    )
    try:
        s = copy.deepcopy(DEFAULT_SETTINGS)
        s['alarm_show_sec'] = 9.5
        s['pose_gate_norm_dist'] = 2.7
        s['tuning'] = {
            'alarm_persist': False,
            'flag_inherit_reid_cos': 0.85,
            'tracker_track_buffer': 95,
            'low_light_boost_enabled': False,
            'person_filter_min_hits': 5,
            'state_contact_thr': 0.42,
            'seg_full_thr': 0.36,
            'missing_detect_check_sec': None,  # rỗng/None -> dùng mặc định settings.py
        }
        ov = overrides_from_settings(s)
        assert ov['rt']['alarm_persist'] is False
        assert ov['rt']['flag_inherit']['reid_cos'] == 0.85
        assert ov['rt']['person_filter']['min_hits'] == 5
        assert ov['rt']['missing_detect']['check_sec'] == 1.0
        assert ov['rt']['alarm_show_sec'] == 9.5
        assert ov['rt']['pose_gate_norm_dist'] == 2.7
        assert ov['cfg']['state_contact_thr'] == 0.42
        assert ov['models']['tracker']['track_buffer'] == 95
        assert ov['models']['tracker']['backend'] == 'bytetrack_dma_ltc'
        assert ov['models']['detector']['low_light_boost']['enabled'] is False
        assert ov['segmentation_cfg']['full_thr'] == 0.36

        settings.apply_overrides(ov)
        assert settings.RT['alarm_persist'] is False
        assert settings.RT['flag_inherit']['reid_cos'] == 0.85
        assert settings.RT['person_filter']['min_hits'] == 5
        assert settings.RT['alarm_show_sec'] == 9.5
        assert settings.RT['pose_gate_norm_dist'] == 2.7
        assert settings.CFG['state_contact_thr'] == 0.42
        assert settings.MODELS['tracker']['track_buffer'] == 95
        assert settings.MODELS['detector']['low_light_boost']['enabled'] is False
        assert settings.SEGMENTATION_CFG['full_thr'] == 0.36

        # Khôi phục mặc định (tuning rỗng) -> lần áp dụng kế tiếp trả về giá trị mặc định
        s['tuning'] = {}
        settings.apply_overrides(overrides_from_settings(s))
        assert settings.RT['alarm_persist'] is True
        assert settings.RT['person_filter']['min_hits'] == 3
        assert settings.CFG['state_contact_thr'] == 0.35
        assert settings.MODELS['tracker']['track_buffer'] == 75
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.SEGMENTATION_CFG.clear()
        settings.SEGMENTATION_CFG.update(old_seg)


def test_tuning_validation_and_model_key():
    """(3) Giá trị ngoài khoảng/sai kiểu bị từ chối kèm tên khoá; (4) needs_reload đổi model_key còn RT/CFG không đổi."""
    import copy
    from exhibitflow import tuning
    from dashboard.runner import DEFAULT_SETTINGS, model_key

    # (3) Kiểm tra kiểu & min/max
    bad_cases = [
        ('person_filter_min_hits', 0),        # < min (1)
        ('person_filter_min_hits', 99),       # > max (30)
        ('person_filter_min_hits', 2.5),      # float cho int
        ('person_filter_min_hits', True),     # bool cho int
        ('state_contact_thr', -0.1),          # < min (0.05)
        ('state_contact_thr', 5.0),           # > max (1.5)
        ('state_contact_thr', 'abc'),         # chuỗi
        ('alarm_persist', 'true'),            # chuỗi cho bool
        ('alarm_persist', 1),                 # int cho bool
    ]
    for k, v in bad_cases:
        try:
            tuning.validate_tuning({k: v})
            raise AssertionError(f'Phải từ chối {k}={v!r}')
        except ValueError as e:
            assert k in str(e), f'Thông báo lỗi phải kèm tên khoá {k}: {e}'

    # (4) model_key chỉ đổi khi khoá needs_reload=True đổi
    base = copy.deepcopy(DEFAULT_SETTINGS)
    k0 = model_key(base)

    # Đặt rõ giá trị bằng mặc định -> model_key không đổi
    same_def = copy.deepcopy(base)
    same_def['tuning'] = {'tracker_track_buffer': 75, 'low_light_boost_enabled': True}
    assert model_key(same_def) == k0

    # Đổi các khoá RT / CFG / SEGMENTATION_CFG -> model_key KHÔNG đổi
    rt_cfg_changed = copy.deepcopy(base)
    rt_cfg_changed['tuning'] = {
        'alarm_persist': False,
        'person_filter_min_hits': 6,
        'state_contact_thr': 0.5,
        'missing_detect_enabled': False,
        'seg_full_thr': 0.4,
    }
    assert model_key(rt_cfg_changed) == k0

    # Đổi khoá MODELS.tracker.* -> model_key ĐỔI
    tr_changed = copy.deepcopy(base)
    tr_changed['tuning'] = {'tracker_track_buffer': 90}
    assert model_key(tr_changed) != k0

    # Đổi khoá MODELS.detector.low_light_boost.* -> model_key ĐỔI
    llb_changed = copy.deepcopy(base)
    llb_changed['tuning'] = {'low_light_boost_enabled': False}
    assert model_key(llb_changed) != k0


def test_tuning_api_schema_and_validation_both_servers():
    """(5) GET /api/tuning/schema ở cả hai server trả đủ nhóm; lưu/áp dụng và từ chối HTTP 400 giá trị sai."""
    from fastapi.testclient import TestClient
    from dashboard.server import create_app as create_dash_app
    from server.web_server import create_app as create_web_app
    from exhibitflow import tuning

    expected_groups = {g['id'] for g in tuning.GROUPS}
    assert len(expected_groups) == 8

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        dash_app = create_dash_app(tmp / 'dash_data', demo=True, preload=False)
        web_app = create_web_app(tmp / 'web_data', demo=True, preload=False)

        with TestClient(dash_app) as dc, TestClient(web_app) as wc:
            # 1) Schema ở cả hai server
            for client, name in ((dc, 'dashboard'), (wc, 'web_server')):
                r = client.get('/api/tuning/schema')
                assert r.status_code == 200, (name, r.text)
                body = r.json()
                assert {g['id'] for g in body['groups']} == expected_groups, name
                assert len(body['params']) == len(tuning.PARAMS), name
                assert all('default' in p and 'label_vi' in p and 'help_vi' in p for p in body['params']), name

            # 2) Dashboard: lưu tuning hợp lệ vào settings.json, từ chối giá trị ngoài khoảng (HTTP 400 + không lưu)
            r_ok = dc.put('/api/settings', json={'tuning': {'person_filter_min_hits': 5, 'tracker_track_buffer': 100}})
            assert r_ok.status_code == 200, r_ok.text
            assert r_ok.json()['tuning'] == {'person_filter_min_hits': 5, 'tracker_track_buffer': 100}

            r_bad = dc.put('/api/settings', json={'tuning': {'person_filter_min_hits': 999}})
            assert r_bad.status_code == 400, r_bad.text
            assert 'person_filter_min_hits' in r_bad.json()['detail']
            # Không ghi đè giá trị trước đó khi lỗi
            assert dc.get('/api/settings').json()['tuning'] == {'person_filter_min_hits': 5, 'tracker_track_buffer': 100}

            # 3) Web server: gửi tuning sai qua POST /api/jobs và POST /api/multi-jobs bị HTTP 400 kèm tên khoá
            r_wbad = wc.post('/api/jobs', data={'source': 'demo.mp4', 'options': json.dumps({'tuning': {'state_contact_thr': -1.0}})})
            assert r_wbad.status_code == 400, r_wbad.text
            assert 'state_contact_thr' in r_wbad.json()['detail']

            r_mbad = wc.post('/api/multi-jobs', data={
                'options': json.dumps({'tuning': {'tracker_track_buffer': 1}}),
                'cameras': json.dumps([{'name': 'c1', 'source': 'demo1.mp4'}, {'name': 'c2', 'source': 'demo2.mp4'}]),
            })
            assert r_mbad.status_code == 400, r_mbad.text
            assert 'tracker_track_buffer' in r_mbad.json()['detail']

            # Web server: gửi tuning hợp lệ qua POST /api/jobs
            r_wok = wc.post('/api/jobs', files={'video': ('sample.mp4', b'00', 'video/mp4')},
                            data={'options': json.dumps({'tuning': {'alarm_persist': False, 'seg_full_thr': 0.25}})})
            assert r_wok.status_code == 200, r_wok.text
            jid = r_wok.json()['job_id']
            wc.post(f'/api/jobs/{jid}/cancel')


def test_ambiguous_policy_nearest_vs_reject():
    """TASK-013 (1): 2 mask cách nhau 0.1 (< 0.15) -> policy 'nearest' chọn mask gần nhất, tạo event + ghi ambiguous_resolved; 'reject' -> không event."""
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    def mk_rec(tid, box, rw):
        x1, y1, x2, y2 = box
        cx, bh = (x1 + x2) / 2, y2 - y1
        kp = [[cx, y1 + 0.08 * bh]] * 5 + [
            [cx - 20, y1 + 0.2 * bh], [cx + 20, y1 + 0.2 * bh],
            [cx - 22, y1 + 0.4 * bh], [cx + 22, y1 + 0.4 * bh],
            [cx - 20, y1 + 0.5 * bh], rw,
            [cx - 15, y1 + 0.55 * bh], [cx + 15, y1 + 0.55 * bh],
            [cx - 15, y1 + 0.75 * bh], [cx + 15, y1 + 0.75 * bh],
            [cx - 15, y2], [cx + 15, y2],
        ]
        kc = [1.0] * 17
        kc[9] = 0.0
        return dict(id=tid, box=list(box), kp=kp, kc=kc)

    def mk_mask(x1, y1, x2, y2):
        m = np.zeros((H, W), bool)
        m[y1:y2, x1:x2] = True
        return m

    old_cfg, old_rt = dict(settings.CFG), dict(settings.RT)
    try:
        settings.RT['periodic_vlm'] = dict(enabled=False)

        def run_case(policy):
            settings.CFG['ambiguous_policy'] = policy
            submitted = []
            ctx = SimpleNamespace(
                fps=25.0, n=120, last_frame=0,
                vlm=SimpleNamespace(submit=lambda *a, **kw: None),
                cur_scene=0, scene_index=lambda _i: 0, d=dict(W=960, H=540),
            )
            inter = OnlineInteraction(ctx)
            inter._submit = lambda g: (g.update(name=f't_{len(inter.tasks)}'), inter.tasks.append(g), submitted.append(g))
            # Chiều cao người = 220px. Mask 10: x=[570..610] -> d=20/220=0.091; Mask 11: x=[592..632] -> d=42/220=0.191 (chênh 0.10 < 0.15)
            arts = [
                dict(id=10, box=[570.0, 300.0, 610.0, 420.0], pedestal=False),
                dict(id=11, box=[592.0, 300.0, 632.0, 420.0], pedestal=False),
            ]
            inter.set_artifacts(arts, {10: mk_mask(570, 300, 610, 420), 11: mk_mask(592, 300, 632, 420)})
            rec = mk_rec(1, [440.0, 200.0, 520.0, 420.0], [550.0, 350.0])
            for f in range(15):
                inter.step(f, [rec])
            inter.flush(30)
            return inter, submitted

        # Policy mặc định 'nearest' -> chọn mask gần hơn (id=10), tạo event
        inter_n, sub_n = run_case('nearest')
        diag_n = inter_n.diagnostics()
        assert len(inter_n.events) == 1, inter_n.events
        assert inter_n.events[0]['artifact_id'] == 10
        assert len(sub_n) == 1 and sub_n[0]['artifact_ids'] == [10]
        assert diag_n['ambiguous_resolved'] == 15
        assert 'ambiguous' not in diag_n['gate_rejects'].get('1', {})

        # Policy 'reject' -> hành vi cũ, loại không tạo event
        inter_r, sub_r = run_case('reject')
        diag_r = inter_r.diagnostics()
        assert inter_r.events == [] and sub_r == []
        assert diag_r['ambiguous_resolved'] == 0
        assert diag_r['gate_rejects']['1']['ambiguous'] == 15
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)


def test_periodic_vlm_flush_on_end_and_track_loss():
    """TASK-013 (2)-(3): flush khi hết video hoặc mất track > max_gap nếu NEAR >= flush_min_sec; tôn trọng max_tasks và tối đa 2 lần/track."""
    from types import SimpleNamespace
    from exhibitflow.interaction import OnlineInteraction

    def mk_rec(tid, box, rw):
        x1, y1, x2, y2 = box
        cx, bh = (x1 + x2) / 2, y2 - y1
        kp = [[cx, y1 + 0.08 * bh]] * 5 + [
            [cx - 20, y1 + 0.2 * bh], [cx + 20, y1 + 0.2 * bh],
            [cx - 22, y1 + 0.4 * bh], [cx + 22, y1 + 0.4 * bh],
            [cx - 20, y1 + 0.5 * bh], rw,
            [cx - 15, y1 + 0.55 * bh], [cx + 15, y1 + 0.55 * bh],
            [cx - 15, y1 + 0.75 * bh], [cx + 15, y1 + 0.75 * bh],
            [cx - 15, y2], [cx + 15, y2],
        ]
        kc = [1.0] * 17
        kc[9] = 0.0
        return dict(id=tid, box=list(box), kp=kp, kc=kc)

    def setup_inter(n_frames=300):
        submitted = []
        ctx = SimpleNamespace(
            fps=25.0, n=n_frames, last_frame=0,
            vlm=SimpleNamespace(submit=lambda *a, **kw: None),
            cur_scene=0, scene_index=lambda _i: 0, d=dict(W=960, H=540),
        )
        inter = OnlineInteraction(ctx)
        inter._submit = lambda g: (g.update(name=f'f_{len(inter.tasks)}'), inter.tasks.append(g), submitted.append(g))
        mask = np.zeros((H, W), bool)
        mask[300:420, 560:640] = True
        inter.set_artifacts([dict(id=0, box=[560.0, 300.0, 640.0, 420.0], pedestal=False)], {0: mask})
        return inter, submitted

    old_cfg, old_rt, old_export = dict(settings.CFG), dict(settings.RT), dict(settings.EXPORT)
    try:
        settings.EXPORT['pre_sec'] = 1.0
        settings.EXPORT['post_sec'] = 1.0
        settings.EXPORT['max_window_sec'] = 8.0
        near_rec = mk_rec(1, [330.0, 200.0, 410.0, 420.0], [450.0, 350.0])  # NEAR d=0.50 (không chạm)

        # (2) Track NEAR 1.5s (38 frame @ 25fps), min_near_sec=3.0 (chưa kích hoạt giữa chừng), hết video -> đúng 1 task flush
        settings.RT['periodic_vlm'] = dict(
            enabled=True, min_near_sec=3.0, cooldown_sec=3.0, max_near_gap_frames=5,
            flush_on_end=True, flush_min_sec=1.0, flush_max_per_track=2,
        )
        inter1, sub1 = setup_inter(n_frames=100)
        for f in range(38):
            inter1.step(f, [near_rec])
        assert sub1 == [], 'chưa hết video / chưa mất track -> không flush sớm'
        inter1.flush(37)
        assert len(sub1) == 1 and sub1[0].get('flush') is True and sub1[0].get('periodic') is True
        assert sub1[0]['start'] == 0 and sub1[0]['end'] == 37
        assert sub1[0]['window'] == [0, 37]

        # (3) Track NEAR 0.5s (13 frame @ 25fps < flush_min_sec=1.0) -> không flush
        inter2, sub2 = setup_inter(n_frames=100)
        for f in range(13):
            inter2.step(f, [near_rec])
        inter2.flush(12)
        assert sub2 == []

        # (4) Track mất > max_gap sau 1.5s NEAR -> flush + post_sec; lặp lại tối đa 2 lần flush mỗi track
        inter3, sub3 = setup_inter(n_frames=400)
        # Đợt 1: f=10..47 (38 frame = 1.52s), rồi mất track từ f=48..80
        for f in range(10, 48):
            inter3.step(f, [near_rec])
        for f in range(48, 80):
            inter3.step(f, [])
        assert len(sub3) == 1 and sub3[0].get('flush') is True
        assert sub3[0]['start'] == 10 and sub3[0]['end'] == 47
        assert sub3[0]['window'] == [0, 72]  # 47 + 25 (post_sec=1.0s)

        # Đợt 2: f=100..137 (38 frame), rồi mất track f=138..170 -> flush lần 2
        for f in range(100, 138):
            inter3.step(f, [near_rec])
        for f in range(138, 170):
            inter3.step(f, [])
        assert len(sub3) == 2

        # Đợt 3: f=200..237 (38 frame), kết thúc video -> bị chặn bởi flush_max_per_track=2
        for f in range(200, 238):
            inter3.step(f, [near_rec])
        inter3.flush(237)
        assert len(sub3) == 2, f'tối đa 2 lần flush mỗi track, nhận {len(sub3)}'

        # (5) Tôn trọng max_tasks_per_video và flush_on_end=False
        settings.CFG['max_tasks_per_video'] = 1
        inter4, sub4 = setup_inter(n_frames=300)
        for f in range(0, 38):
            inter4.step(f, [near_rec])
        for f in range(38, 70):
            inter4.step(f, [])
        for f in range(80, 118):
            inter4.step(f, [near_rec])
        inter4.flush(117)
        assert len(sub4) == 1, 'không được vượt quá max_tasks_per_video=1'

        settings.CFG['max_tasks_per_video'] = 40
        settings.RT['periodic_vlm']['flush_on_end'] = False
        inter5, sub5 = setup_inter(n_frames=100)
        for f in range(38):
            inter5.step(f, [near_rec])
        inter5.flush(37)
        assert sub5 == []
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.EXPORT.clear()
        settings.EXPORT.update(old_export)


def test_ambiguous_and_flush_end_to_end_pipeline():
    """End-to-end TASK-013: (1) 2 mask sát nhau vẫn tạo event/alert với ambiguous_policy='nearest'; (2) NEAR 1.5s hết video -> flush VLM -> alert."""
    import copy
    from exhibitflow import run_pipeline

    @register('detector', 'fake_t13_det')
    class FakeT13Det:
        def __init__(self, **_):
            pass

        def detect(self, frame):
            m = cv2.inRange(frame, (0, 150, 0), (80, 255, 80))
            ys, xs = np.nonzero(m)
            if len(xs) < 100:
                return []
            return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9]]

    @register('pose', 'fake_t13_pose')
    class FakeT13Pose:
        def __init__(self, mode='touch', **_):
            self.mode = mode

        def __call__(self, native, recs, S):
            for r in recs:
                x1, y1, x2, y2 = r['box']
                cx, h = (x1 + x2) / 2, y2 - y1
                # touch: d0 ~ 0.09 (chạm mask 0); near_only: d0 ~ 0.50 (chỉ NEAR)
                wx = (ART[0] - 0.09 * h) if self.mode == 'touch' else (ART[0] - 0.50 * h)
                kp = [[cx, y1 + 0.1 * h]] * 5 + [
                    [cx - 10, y1 + .2 * h], [cx + 10, y1 + .2 * h],
                    [cx - 12, y1 + .4 * h], [wx - 5, y1 + .4 * h],
                    [cx - 12, y1 + .6 * h], [wx, y1 + .5 * h],
                    [cx - 8, y1 + .55 * h], [cx + 8, y1 + .55 * h],
                    [cx - 8, y1 + .75 * h], [cx + 8, y1 + .75 * h],
                    [cx - 8, y2], [cx + 8, y2],
                ]
                r['kp'], r['kc'] = kp, [1.0] * 17

    @register('segmenter', 'fake_t13_seg')
    class FakeT13Seg:
        def __init__(self, two_masks=True, **_):
            self.two_masks = two_masks

        def warmup(self):
            pass

        def segment(self, frame, ref_idx, person_boxes, sample_frames, tracks):
            m0 = np.zeros(frame.shape[:2], bool)
            m0[ART[1]:ART[3], ART[0]:ART[2]] = True
            arts = [dict(id=0, box=[float(v) for v in ART], label='sculpture', score=0.9, pedestal=False, fallback=False)]
            masks = {0: m0}
            if self.two_masks:
                # Mask 1 dịch sang phải 22px -> chênh khoảng cách ~ 22/220 = 0.10 < 0.15 (ambiguous)
                m1 = np.zeros(frame.shape[:2], bool)
                m1[ART[1]:ART[3], ART[0] + 22:ART[2] + 22] = True
                arts.append(dict(id=1, box=[float(ART[0] + 22), float(ART[1]), float(ART[2] + 22), float(ART[3])],
                                 label='sculpture', score=0.85, pedestal=False, fallback=False))
                masks[1] = m1
            return arts, masks, []

        def release(self):
            pass

    def make_custom_video(path, box, f_start, f_end, n_frames):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
        for i in range(n_frames):
            f = np.full((H, W, 3), 90, np.uint8)
            cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (200, 200, 200), -1)
            if f_start <= i <= f_end:
                cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), -1)
            wr.write(f)
        wr.release()

    old_cfg, old_rt, old_models, old_run = (
        copy.deepcopy(settings.CFG), copy.deepcopy(settings.RT),
        copy.deepcopy(settings.MODELS), copy.deepcopy(settings.RUN),
    )
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            vid_amb = tmp / 'amb.avi'
            make_custom_video(vid_amb, (440, 200, 520, 420), f_start=5, f_end=25, n_frames=40)
            models_amb = dict(
                FAKE_MODELS,
                detector=dict(backend='fake_t13_det'),
                pose=dict(backend='fake_t13_pose', mode='touch'),
                segmenter=dict(backend='fake_t13_seg', two_masks=True),
                vlm=dict(backend='dry_run', delay_s=0.0, label='touching'),
            )
            sum_amb = run_pipeline(
                [str(vid_amb)], out_dir=tmp / 'out_amb', work_dir=tmp / 'work_amb',
                overrides=dict(models=models_amb, rt=dict(queue_size=4, pose_gate_norm_dist=None)),
            )
            assert not sum_amb['errors'], sum_amb['errors']
            res_amb = json.loads((tmp / 'out_amb' / 'amb' / 'result.json').read_text(encoding='utf-8'))
            assert res_amb['summary']['n_events'] >= 1 and res_amb['summary']['n_alerts'] >= 1, res_amb['summary']
            assert res_amb['diagnostics']['ambiguous_resolved'] > 0

            # End-to-end flush cuối video khi đứng NEAR 1.5s (40 frame @ 25fps) và min_near_sec=3.0
            vid_flush = tmp / 'flush_near.avi'
            make_custom_video(vid_flush, (330, 200, 410, 420), f_start=0, f_end=39, n_frames=40)
            models_flush = dict(
                FAKE_MODELS,
                detector=dict(backend='fake_t13_det'),
                pose=dict(backend='fake_t13_pose', mode='near_only'),
                segmenter=dict(backend='fake_t13_seg', two_masks=False),
                vlm=dict(backend='dry_run', delay_s=0.0, label='stealing'),
            )
            sum_fl = run_pipeline(
                [str(vid_flush)], out_dir=tmp / 'out_fl', work_dir=tmp / 'work_fl',
                overrides=dict(
                    models=models_flush,
                    rt=dict(
                        queue_size=4,
                        pose_gate_norm_dist=None,
                        periodic_vlm=dict(enabled=True, min_near_sec=3.0, cooldown_sec=3.0,
                                          max_near_gap_frames=5, flush_on_end=True, flush_min_sec=1.0),
                    ),
                ),
            )
            assert not sum_fl['errors'], sum_fl['errors']
            res_fl = json.loads((tmp / 'out_fl' / 'flush_near' / 'result.json').read_text(encoding='utf-8'))
            assert res_fl['summary']['n_vlm_tasks'] == 1, res_fl['summary']
            assert res_fl['summary']['n_alerts'] == 1 and res_fl['summary']['has_anomaly'] is True
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.RUN.clear()
        settings.RUN.update(old_run)


def test_before_ref_settings_prompt_and_adapt_task():
    """TASK-014: kiểm tra EXPORT, PROMPT_FIELDS, adapt_task giữ ảnh BEFORE (tổng n_images+1) và tương thích prompt tuỳ chỉnh."""
    from unittest.mock import patch
    from exhibitflow.vlm import LlamaCppVLM, adapt_task

    assert settings.EXPORT['post_sec'] == 3.0
    assert settings.EXPORT['max_window_sec'] == 10.0
    assert settings.EXPORT['before_ref'] is True
    assert 'before_note' in settings.PROMPT_FIELDS

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        img_paths = []
        for i in range(13):
            p = tmp / (f'00_before_f00000.jpg' if i == 0 else f'{i:02d}_f{i:05d}.jpg')
            im = np.full((64, 64, 3), 80 + i * 5, np.uint8)
            cv2.imwrite(str(p), im)
            img_paths.append(str(p))

        task = dict(
            images=img_paths,
            times=[None] + [i * 0.25 for i in range(12)],
            tid=7,
            has_before=True,
            before_frame=0,
            prompt=settings.PROMPT_TEMPLATE.format(n=13, dt=0.25, tid=7, before_note=settings.BEFORE_NOTE),
        )
        t = adapt_task(task, dict(n_images=8, max_side=512, short_json=True))
        assert len(t['images']) == 9, f'kỳ vọng 8+1=9 ảnh, nhận {len(t["images"])}'
        assert Path(t['images'][0]).name == '00_before_f00000.jpg'
        assert t['times'][0] is None
        assert 'Image 1 is a reference taken BEFORE' in t['prompt']
        assert 'contact_frames' not in t['prompt'].split('Answer JSON only:')[1]

        # Kiểm tra LlamaCppVLM.infer gắn nhãn Image 1 là BEFORE trong payload gửi server
        captured = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def read(self):
                return json.dumps(dict(
                    choices=[dict(message=dict(content='{"label": "stealing", "confidence": 0.9}'))],
                    usage=dict(prompt_tokens=100, completion_tokens=10),
                )).encode()

        def _fake_urlopen(req, timeout=900):
            captured['body'] = json.loads(req.data.decode('utf-8'))
            return _FakeResp()

        dummy = object.__new__(LlamaCppVLM)
        dummy.max_pixels = 768 * 768
        dummy.max_new_tokens = 64
        dummy.port = 8089
        dummy.cache_prompt = False
        dummy.gpus = []
        with patch('urllib.request.urlopen', side_effect=_fake_urlopen):
            text, _, _ = dummy.infer(t)
        assert 'stealing' in text
        msg_content = captured['body']['messages'][-1]['content']
        text_parts = [c['text'] for c in msg_content if c['type'] == 'text']
        assert 'BEFORE' in text_parts[0], text_parts[0]
        assert text_parts[1].startswith('Frame 2 (t='), text_parts[1]

        # Prompt tuỳ chỉnh thiếu {before_note} vẫn chạy bình thường và không chèn BEFORE note
        old_prompt = settings.PROMPT_TEMPLATE
        try:
            settings.PROMPT_TEMPLATE = (
                'Custom {n} frames dt={dt:.2f}s ID={tid}.\n'
                'Answer JSON only:\n{{"label": "...", "confidence": 0.0, "contact_frames": []}}'
            )
            t_custom = adapt_task(task, dict(n_images=8, short_json=True))
            assert len(t_custom['images']) == 9
            assert t_custom['prompt'].startswith('Custom 9 frames')
            assert 'Image 1 is a reference taken BEFORE' not in t_custom['prompt']
        finally:
            settings.PROMPT_TEMPLATE = old_prompt


def test_before_ref_export_window_and_end_to_end():
    """TASK-014: _set_window kéo dài post_sec=3.0s; export_task tạo ảnh đầu là 'before', prompt giải thích BEFORE; tắt before_ref không tạo ảnh before."""
    import types
    from exhibitflow import run_pipeline
    from exhibitflow.interaction import OnlineInteraction

    old_cfg, old_rt, old_models, old_export = (
        dict(settings.CFG), dict(settings.RT), dict(settings.MODELS), dict(settings.EXPORT)
    )
    try:
        # (1) _set_window với EXPORT['post_sec']=3.0 (75 frame @ 25fps) và max_window_sec=10.0
        ctx_win = types.SimpleNamespace(fps=25.0, n=500, d=dict(W=960, H=540))
        inter = OnlineInteraction(ctx_win)
        g = dict(tid=1, artifact_ids=[0], start=100, end=130, peak=115, events=[], _main=None)
        inter._set_window(g)
        assert g['window'] == [75, 205], g['window']  # 100 - 25 = 75, 130 + 75 (3.0s) = 205

        # (2) End-to-end với before_ref=True (mặc định): ring_seconds nhỏ (2.0s) vẫn giữ được ref frame ở f0 nhờ ctx.ref_frames
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            vid = tmp / 'before_e2e.avi'
            n_fr = 160
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
            for i in range(n_fr):
                f = np.full((H, W, 3), 90, np.uint8)
                if i < 100:
                    cv2.rectangle(f, (ART[0], ART[1]), (ART[2], ART[3]), (200, 200, 200), -1)
                if 60 <= i <= 110:
                    cv2.rectangle(f, (430, 200), (510, 420), (0, 200, 0), -1)
                wr.write(f)
            wr.release()

            out_on = tmp / 'out_on'
            sum_on = run_pipeline(
                [str(vid)], out_dir=out_on, work_dir=tmp / 'work_on',
                overrides=dict(
                    models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0, label='stealing')),
                    rt=dict(queue_size=4, pose_gate_norm_dist=None, ring_seconds=3.0,
                            periodic_vlm=dict(enabled=False), missing_detect=dict(enabled=False)),
                ),
            )
            assert not sum_on['errors'], sum_on['errors']
            seg_dirs = sorted((out_on / 'before_e2e' / 'vlm_inputs').iterdir())
            assert seg_dirs
            marked_imgs = sorted((seg_dirs[0] / 'marked').glob('*.jpg'))
            assert len(marked_imgs) == settings.EXPORT['n_sample_frames'] + 1
            assert marked_imgs[0].name.startswith('00_before_f')
            prompt_txt = (seg_dirs[0] / 'prompt.txt').read_text(encoding='utf-8')
            assert 'Image 1 is a reference taken BEFORE' in prompt_txt
            res_on = json.loads((out_on / 'before_e2e' / 'result.json').read_text(encoding='utf-8'))
            assert res_on['track_summary'] == {'1': {'behavior': 'stealing', 'n_tasks': 1}}, res_on['track_summary']

            # (3) Tắt before_ref=False -> không có ảnh before và không chèn ghi chú BEFORE
            out_off = tmp / 'out_off'
            sum_off = run_pipeline(
                [str(vid)], out_dir=out_off, work_dir=tmp / 'work_off',
                overrides=dict(
                    models=dict(FAKE_MODELS, vlm=dict(backend='dry_run', delay_s=0.0, label='touching')),
                    rt=dict(queue_size=4, pose_gate_norm_dist=None,
                            periodic_vlm=dict(enabled=False), missing_detect=dict(enabled=False)),
                    export=dict(before_ref=False),
                ),
            )
            assert not sum_off['errors'], sum_off['errors']
            seg_off = list((out_off / 'before_e2e' / 'vlm_inputs').iterdir())[0]
            marked_off = sorted((seg_off / 'marked').glob('*.jpg'))
            assert len(marked_off) == settings.EXPORT['n_sample_frames']
            assert not any('before' in p.name for p in marked_off)
            prompt_off = (seg_off / 'prompt.txt').read_text(encoding='utf-8')
            assert 'Image 1 is a reference taken BEFORE' not in prompt_off
    finally:
        settings.CFG.clear()
        settings.CFG.update(old_cfg)
        settings.RT.clear()
        settings.RT.update(old_rt)
        settings.MODELS.clear()
        settings.MODELS.update(old_models)
        settings.EXPORT.clear()
        settings.EXPORT.update(old_export)


def test_track_summary_severity_and_result_json():
    """TASK-014: track_summary chọn nhãn nặng nhất (stealing > holding > touching > normal) mà không đổi nhãn từng event."""
    import types
    from exhibitflow.engine import AlarmBoard, finalize_video
    from exhibitflow.pipeline import summarize_tracks, video_result

    assert summarize_tracks([
        dict(track_id=1, label='normal'),
        dict(track_id=1, label='touching'),
        dict(track_id=1, label='stealing'),
        dict(track_id=2, label='touching'),
        dict(track_id=2, label='holding'),
        dict(track_id=3, label='normal'),
        dict(track_id=3, label='touching'),
        dict(track_id=4, label='uncertain'),
        dict(track_id=4, label='normal'),
        dict(track_id=5, label='error'),
        dict(track_id=5, label=None),
    ]) == {
        '1': {'behavior': 'stealing', 'n_tasks': 3},
        '2': {'behavior': 'holding', 'n_tasks': 2},
        '3': {'behavior': 'touching', 'n_tasks': 2},
        '4': {'behavior': 'normal', 'n_tasks': 2},
        '5': {'behavior': None, 'n_tasks': 2},
    }

    # AlarmBoard giữ nhãn nặng hơn nếu có alarm nhẹ hơn đến sau
    board = AlarmBoard(persist=True)
    board.add(dict(tid=1, behavior='stealing', conf=0.9, segment='s1', shown_from=10, t_alarm=1.0, artifact_ids=[0]))
    board.add(dict(tid=1, behavior='touching', conf=0.8, segment='s2', shown_from=20, t_alarm=2.0, artifact_ids=[0]))
    assert board.flagged[1]['behavior'] == 'stealing'

    # Kiểm tra finalize_video + video_result giữ nguyên nhãn từng event nhưng track_summary chọn nhãn nặng nhất
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out_dir = Path(tmp)
        e1 = dict(eid=0, track_id=10, artifact_id=0, hand='right_wrist', start_frame=10, end_frame=20,
                  contact_frames=[12], min_norm_dist=0.1, peak=12, group=None)
        e2 = dict(eid=1, track_id=10, artifact_id=0, hand='right_wrist', start_frame=40, end_frame=50,
                  contact_frames=[42], min_norm_dist=0.1, peak=42, group=None)
        e3 = dict(eid=2, track_id=11, artifact_id=0, hand='left_wrist', start_frame=60, end_frame=70,
                  contact_frames=[62], min_norm_dist=0.1, peak=62, group=None)
        g1 = dict(tid=10, events=[e1], artifact_ids=[0], start=10, end=20, peak=12, window=[0, 30],
                  name='seg_000', task_id='run/v/seg_000', samples=[10, 12, 20])
        g2 = dict(tid=10, events=[e2], artifact_ids=[0], start=40, end=50, peak=42, window=[30, 60],
                  name='seg_001', task_id='run/v/seg_001', samples=[40, 42, 50])
        g3 = dict(tid=11, events=[e3], artifact_ids=[0], start=60, end=70, peak=62, window=[50, 80],
                  name='seg_002', task_id='run/v/seg_002', samples=[60, 62, 70])
        e1['group'], e2['group'], e3['group'] = g1, g2, g3

        vlm_records = [
            dict(task_id='run/v/seg_000', status='ok', label='touching',
                 parsed=dict(label='touching', confidence=0.8, contact_frames=[2]), t_done=1.0, infer_s=0.1, queue_wait_s=0.01),
            dict(task_id='run/v/seg_001', status='ok', label='stealing',
                 parsed=dict(label='stealing', confidence=0.9, contact_frames=[2]), t_done=2.0, infer_s=0.1, queue_wait_s=0.01),
            dict(task_id='run/v/seg_002', status='ok', label='holding',
                 parsed=dict(label='holding', confidence=0.85, contact_frames=[2]), t_done=3.0, infer_s=0.1, queue_wait_s=0.01),
        ]
        ctx = types.SimpleNamespace(
            dir=out_dir, fps=25.0, n=100, key='v', tracks={10: [{'id': 10}], 60: [{'id': 11}]},
            inter=types.SimpleNamespace(events=[e1, e2, e3], tasks=[g1, g2, g3], diagnostics=lambda: {}),
            vlm=types.SimpleNamespace(results={r['task_id']: r for r in vlm_records}),
            art=types.SimpleNamespace(artifacts=[dict(id=0, label='statue', box=[10, 10, 50, 50], pedestal=False)],
                                      scenes=[dict(start=0, ref_idx=0, artifacts=[0])], error=None),
            rec=types.SimpleNamespace(df=lambda: __import__('pandas').DataFrame(
                [('detect', 0, 0.0, 1.0)], columns=['stage', 'frame', 'wait_ms', 'compute_ms'])),
            frame_log=[dict(frame=0, t_cap=0.0, t_out=0.04, e2e_ms=40.0)],
            pauses=[], pause_overlap=lambda a, b: 0.0, wall_s=1.0, wall_with_vlm_s=1.5,
            dropped=[], gpu_peak_gb={},
            M=types.SimpleNamespace(tracker=None, pose=None, hand=None),
        )
        finalize_video(ctx)
        res = video_result(ctx, out_dir / 'v.mp4', dict(NW=960, NH=540), dict(vlm=dict(backend='dry_run')), vlm_records, {})
        assert [ev['behavior'] for ev in res['events']] == ['touching', 'stealing', 'holding']
        assert res['track_summary'] == {
            '10': {'behavior': 'stealing', 'n_tasks': 2},
            '11': {'behavior': 'holding', 'n_tasks': 1},
        }


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(f'\n=== {name}')
            fn()
            print(f'--- {name}: OK')
    print('\nTẤT CẢ TEST ĐẠT')
