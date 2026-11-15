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
        assert res['vlm_tasks'][0]['evidence'] == '8 images', res['vlm_tasks'][0]       # 12 ảnh đã xuất -> gửi 8
        seg_dir = next((out / 'museum_test' / 'vlm_inputs').iterdir())
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


def test_helpers():
    from exhibitflow.vlm import adapt_task, parse_response
    from main import parse_set
    assert parse_set(['models.vlm.max_side=768', 'ALARM_ONSET=event', 'rt.queue_size=4']) == \
        {'models': {'vlm': {'max_side': 768}}, 'ALARM_ONSET': 'event', 'rt': {'queue_size': 4}}
    assert parse_response('```json\n{"label": "Stealing", "confidence": 0.7}\n```')['label'] == 'stealing'
    assert parse_response('no json')  is None
    assert parse_response('{"label": "dancing"}')['label'] == 'invalid:dancing'
    task = dict(images=[f'x{i}.jpg' for i in range(12)], times=[i * .5 for i in range(12)], tid=3,
                prompt=settings.PROMPT_TEMPLATE.format(n=12, dt=0.5, tid=3))
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


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(f'\n=== {name}')
            fn()
            print(f'--- {name}: OK')
    print('\nTẤT CẢ TEST ĐẠT')
