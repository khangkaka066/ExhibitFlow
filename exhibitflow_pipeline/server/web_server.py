"""ExhibitFlow — server nối giao diện web (thư mục `web/`) với engine realtime của dashboard Hưng.

Không sửa file nào của `dashboard/`: dùng lại `State` + `PipelineRunner` (model nạp sẵn một lần, chạy nhiều video)
và trả lời bằng API "job" mà web đang dùng:

    GET  /api/health                      kiểm tra sống + trạng thái nạp model
    GET  /api/state                       trạng thái dashboard (dùng cho launcher chờ nạp model)
    GET  /api/system                      CPU / RAM / GPU
    POST /api/jobs                        multipart: video (upload) | source (đường dẫn trên server / URL) + options -> {job_id}
    GET  /api/jobs/{id}                   {status, stage, stages, progress, key, error}
    POST /api/jobs/{id}/cancel            dừng
    GET  /api/jobs/{id}/live              khung hình mới nhất (JPEG, đã vẽ khung người / hiện vật)
    GET  /api/jobs/{id}/live/events       meta + trạng thái + cảnh báo đang chạy (?after=<seq>)
    GET  /api/jobs/{id}/result            result.json của pipeline
    GET  /api/jobs/{id}/video             live.mp4 (có khung); ?kind=original là video gốc
    GET  /api/jobs/{id}/files/<path>      file trong thư mục kết quả (ảnh bằng chứng)

Chạy:
    python server/web_server.py --port 8000             # pipeline thật (cần GPU + trọng số)
    python server/web_server.py --port 8000 --demo      # dữ liệu giả của dashboard (không cần GPU), để thử giao diện
Server cũng phục vụ giao diện `web/` tại `/` (cùng cổng): mở http://<ip>:<port>/ là dùng được, không cần URL backend riêng.
Biến môi trường: EXHIBITFLOW_WEB_DIR (thư mục giao diện, mặc định <repo>/web), EF_WEIGHTS_DIR (thư mục trọng số), EXHIBITFLOW_SOURCE_ROOTS (thư mục được phép đọc, ngăn bằng dấu ':'
trên Linux), EXHIBITFLOW_MAX_MB (giới hạn upload, mặc định 600), EXHIBITFLOW_SAVE_VIDEO=0 để không ghi live.mp4.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # trước khi torch dùng CUDA

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.runner import DemoRunner, PipelineRunner, State  # noqa: E402

ALLOWED = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_BYTES = int(os.environ.get('EXHIBITFLOW_MAX_MB', '600')) * 1024 * 1024
STAGES = ['Khởi động & nạp model', 'Tách hiện vật', 'Nạp VLM', 'Phân tích video', 'Ghi kết quả']
SOURCE_ROOTS = [Path(p).resolve() for p in (os.environ.get('EXHIBITFLOW_SOURCE_ROOTS')
                                            or os.pathsep.join(['/workspace', '/kaggle/input', '/kaggle/working', '/content', str(PROJ)])).split(os.pathsep) if p]
WEB_DIR = Path(os.environ.get('EXHIBITFLOW_WEB_DIR') or PROJ.parent / 'web')   # giao diện web, phục vụ chung cổng với API
# Cấu hình pipeline web được phép đặt (khoá + kiểu) = các mục tương ứng trong Settings của dashboard cũ
PIPE_OPTS = dict(tracker_process=bool, hand_process=bool, pose_process=bool, pose_gpu_preproc=bool, hand_batch=bool, drop_frames=bool, limit_cpu_threads=bool, alarm_show_sec=float,
                  tracker_threads=int, hand_threads=int, pose_threads=int, queue_size=int, pose_gate_norm_dist=float)
NW = 960                                   # web vẽ hộp hiện vật theo khung rộng 960 px


def create_app(data_dir, demo=False, preload=True, mode=None, local=False):
    state = State(data_dir)
    runner = (DemoRunner if demo else PipelineRunner)(state, local=local)
    if mode:
        state.save_settings(dict(pipeline_mode=mode))
    # mặc định bật khi khởi động (preload nạp model đúng cấu hình ngay, không phải nạp lại ở lần chạy đầu); web gửi lại mỗi lần chạy
    state.save_settings(dict(tracker_process=True, hand_process=True, pose_process=True, hand_batch=True, drop_frames=True))
    if not demo and os.environ.get('EXHIBITFLOW_SAVE_VIDEO', '1') != '0':
        # Engine.run đặt RT['render_live']=False trước mỗi lần chạy, nên bật lại ngay trước run_video (không sửa file của Hưng):
        # engine ghi live.mp4 (video đã vẽ khung người / hiện vật) để web xem lại sau khi chạy xong.
        import exhibitflow.service as svc
        _orig_run_video = svc.run_video

        def _run_video_with_live(*a, **k):
            svc.RT['render_live'] = True
            return _orig_run_video(*a, **k)

        svc.run_video = _run_video_with_live
    if preload:
        runner.preload()                   # nạp model một lần ngay khi server lên

    app = FastAPI(title='ExhibitFlow Web API')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
    jobs, cur = {}, {'id': None}
    uploads = state.root / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tiện ích
    def check_ext(ext):
        if ext not in ALLOWED:
            raise HTTPException(400, f'Định dạng không hỗ trợ ({ext or "không có đuôi"}). Dùng: {", ".join(sorted(ALLOWED))}')

    def resolve_source(src, dst_dir):
        """URL http(s) (tải về) hoặc đường dẫn file / thư mục trên server (chỉ trong SOURCE_ROOTS)."""
        if re.match(r'^(rtsps?|rtmps?)://', src, re.I):
            raise HTTPException(501, 'Chưa hỗ trợ nguồn camera trực tiếp (RTSP/RTMP): pipeline hiện chỉ đọc file video.')
        if re.match(r'^https?://', src, re.I):
            ext = Path(urllib.parse.urlparse(src).path).suffix.lower() or '.mp4'
            check_ext(ext)
            dst, size = dst_dir / f'{uuid.uuid4().hex[:8]}{ext}', 0
            try:
                with urllib.request.urlopen(src, timeout=30) as r, open(dst, 'wb') as f:
                    while chunk := r.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                        f.write(chunk)
            except HTTPException:
                raise
            except Exception as e:             # noqa: BLE001
                raise HTTPException(400, f'Không tải được URL: {e}')
            return dst, Path(urllib.parse.urlparse(src).path).name or dst.name
        p = Path(src).expanduser().resolve()
        if not any(r == p or r in p.parents for r in SOURCE_ROOTS):
            raise HTTPException(403, 'Đường dẫn nằm ngoài thư mục cho phép: ' + ', '.join(map(str, SOURCE_ROOTS)))
        if p.is_dir():
            vids = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED)
            if not vids:
                raise HTTPException(404, f'Không có video trong thư mục {p}')
            p = vids[0]
        if not p.is_file():
            raise HTTPException(404, f'Không thấy file {p}')
        check_ext(p.suffix.lower())
        return p, p.name

    def video_info(path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            info = dict(fps=fps if fps > 1 else 25.0, n_frames=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)),
                        width=max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)), height=max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
            cap.release()
            return info
        except Exception:                      # noqa: BLE001
            return dict(fps=25.0, n_frames=0, width=0, height=0)

    def get_job(jid):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, 'Không có job này')
        return j

    def result_dir(j):
        """Thư mục kết quả (chứa result.json) của job: tìm result.json mới nhất sinh ra sau khi job bắt đầu."""
        if j.get('rdir') is not None:
            return j['rdir']
        found = [p for p in (state.root / 'results').rglob('result.json') if p.stat().st_mtime >= j['t0'] - 2]
        if not found:
            return None
        j['rdir'] = max(found, key=lambda p: p.stat().st_mtime).parent
        return j['rdir']

    def finished(j):
        """Job đã có kết quả cuối (trạng thái chốt, không đổi nữa)."""
        return j.get('final') is not None

    def view(j):
        """Chuyển trạng thái `State` của dashboard sang trạng thái job mà web hiểu."""
        if finished(j):
            return j['final']
        if cur['id'] != j['id']:                # job cũ bị job mới thay: coi như đã dừng
            j['final'] = dict(status='cancelled', stage=0, progress=0, error=None)
            return j['final']
        with state.lock:
            st, phase, msg, fi, nf = state.status, state.phase, state.message, state.frame_idx, state.n_frames
        alive = runner.running()
        if j['cancelled'] and not alive:
            j['final'] = dict(status='cancelled', stage=4, progress=0, error=None)
        elif st == 'error':
            j['final'] = dict(status='error', stage=0, progress=0, error=msg or 'Pipeline lỗi')
        elif st == 'done' and not alive:
            j['final'] = dict(status='done', stage=4, progress=1.0, error=None)
        elif st in ('idle', 'loading'):
            stage = 1 if 'map' in (phase or '').lower() else 0
            return dict(status='running', stage=stage, progress=0.0, error=None)
        elif st == 'running':
            return dict(status='running', stage=3, progress=round(fi / nf, 3) if nf else 0.0, error=None)
        else:                                   # finishing / done nhưng luồng chưa tắt
            return dict(status='running', stage=4, progress=0.5, error=None)
        return j['final']

    # ------------------------------------------------------------------ cơ bản
    @app.get('/api/health')
    def health():
        with state.lock:
            ms = state.models_state
        return {'ok': True, 'version': 'ef-web-1', 'demo': demo, 'models': ms, 'jobs': len(jobs)}

    @app.get('/api/state')
    def get_state():
        return state.snapshot()

    def cpu_cores():
        """Số nhân mà tiến trình thực sự dùng được: min(affinity, hạn mức cgroup). Container Vast: os.cpu_count() thường báo cả máy chủ."""
        info = {'logical': os.cpu_count() or 1}
        try:
            info['usable'] = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            info['usable'] = info['logical']
        quota = None
        try:
            q, per = open('/sys/fs/cgroup/cpu.max').read().split()[:2]          # cgroup v2: "<quota> <period>" hoặc "max <period>"
            quota = None if q == 'max' else int(q) / int(per)
        except Exception:                      # noqa: BLE001
            try:                               # cgroup v1
                q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
                per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
                quota = q / per if q > 0 else None
            except Exception:                  # noqa: BLE001
                pass
        if quota:
            info['quota'] = round(quota, 1)
            info['usable'] = max(1, min(info['usable'], int(quota)))
        return info

    @app.get('/api/system')
    def system():
        out = {'cpu': None, 'ram': None, 'gpus': [], 'cores': cpu_cores()}
        try:
            la, n = os.getloadavg(), os.cpu_count() or 1
            out['cpu'] = {'load': round(la[0], 2), 'pct': round(min(100.0, la[0] / n * 100), 1)}
        except (AttributeError, OSError):
            try:
                import psutil
                out['cpu'] = {'load': 0, 'pct': psutil.cpu_percent()}
            except Exception:                  # noqa: BLE001
                pass
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    info[k] = int(v.split()[0])
            tot = info['MemTotal'] / 1048576
            out['ram'] = {'used_gb': round(tot - info['MemAvailable'] / 1048576, 1), 'total_gb': round(tot, 1)}
        except Exception:                      # noqa: BLE001
            try:
                import psutil
                vm = psutil.virtual_memory()
                out['ram'] = {'used_gb': round(vm.used / 1e9, 1), 'total_gb': round(vm.total / 1e9, 1)}
            except Exception:                  # noqa: BLE001
                pass
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
            for line in r.stdout.strip().splitlines():
                name, util, mu, mt = [x.strip() for x in line.split(',')]
                out['gpus'].append({'name': name, 'util': float(util), 'mem_used_gb': round(float(mu) / 1024, 1),
                                    'mem_total_gb': round(float(mt) / 1024, 1)})
        except Exception:                      # noqa: BLE001
            pass
        return out

    # ------------------------------------------------------------------ job
    @app.post('/api/jobs')
    async def create_job(video: UploadFile | None = File(None), options: str = Form('{}'), source: str = Form('')):
        try:
            opts = json.loads(options or '{}')
        except json.JSONDecodeError:
            opts = {}
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy video khác, đợi xong hoặc bấm Dừng.')
        if runner.loading():
            raise HTTPException(409, 'Model đang nạp (lần đầu mất vài phút), thử lại sau.')
        if video is not None and video.filename:
            ext = Path(video.filename).suffix.lower()
            check_ext(ext)
            vpath, size = uploads / f'{uuid.uuid4().hex[:8]}{ext}', 0
            with open(vpath, 'wb') as f:
                while chunk := await video.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        f.close()
                        vpath.unlink(missing_ok=True)
                        raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                    f.write(chunk)
            name = video.filename
        elif source.strip():
            vpath, name = resolve_source(source.strip(), uploads)
        else:
            raise HTTPException(400, 'Cần gửi video hoặc đường dẫn / URL')
        # realtime: đọc đúng nhịp fps như camera, bỏ frame nếu không kịp; theo lô (Video): xử lý mọi frame
        pipe = {k: t(opts[k]) for k, t in PIPE_OPTS.items() if opts.get(k) is not None}
        state.save_settings(dict(pipe, realtime=bool(opts.get('realtime'))))
        try:
            runner.start(str(vpath))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = dict(id=jid, path=Path(vpath), name=name, t0=time.time(), cancelled=False, final=None, rdir=None,
                         realtime=bool(opts.get('realtime')), vinfo=video_info(vpath))
        cur['id'] = jid
        return {'job_id': jid}

    @app.get('/api/jobs/{jid}')
    def job_status(jid: str):
        j = get_job(jid)
        v = view(j)
        key = None
        if v['status'] == 'done':
            rd = result_dir(j)
            key = rd.name if rd else (state.cam_name or j['path'].stem)
        return dict(status=v['status'], stage=v['stage'], stages=STAGES, progress=v['progress'], log=[], key=key,
                    error=v['error'], name=j['name'], realtime=j['realtime'])

    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid: str):
        j = get_job(jid)
        v = view(j)
        if v['status'] in ('done', 'error', 'cancelled'):
            return {'status': v['status']}
        j['cancelled'] = True
        runner.stop()
        return {'status': 'cancelling'}

    # ------------------------------------------------------------------ phát trực tiếp
    @app.get('/api/jobs/{jid}/live')
    def live_frame(jid: str):
        get_job(jid)
        with state.lock:
            jpg = state.frame_jpeg
        if jpg is None:
            raise HTTPException(404, 'Chưa có khung hình')
        return Response(jpg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    def art_boxes(j):
        """Hộp hiện vật (toạ độ khung rộng 960 px) lấy từ đường viền cảnh đầu tiên của dashboard."""
        vi = j['vinfo']
        ratio = (vi['height'] / vi['width']) if vi['width'] else 0.5625
        with state.lock:
            scenes = list(state.scene_contours)
        out = []
        for a in (scenes[0]['arts'] if scenes else []):
            pts = [pt for poly in a['c'] for pt in poly]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(dict(id=a['id'], label=f'Hiện vật {a["id"]}', pedestal=False,
                            box=[round(min(xs) * NW, 1), round(min(ys) * NW * ratio, 1), round(max(xs) * NW, 1), round(max(ys) * NW * ratio, 1)]))
        return out

    @app.get('/api/jobs/{jid}/live/events')
    def live_events(jid: str, after: int = 0):
        j = get_job(jid)
        vi = j['vinfo']
        with state.lock:
            fi, nf, fps_now, alerts = state.frame_idx, state.n_frames, state.fps, list(state.alerts)
            running = state.status in ('running', 'finishing', 'done')
        nf = nf or vi['n_frames']
        vfps = vi['fps'] or 25.0
        meta = dict(artifacts=art_boxes(j), n_frames=nf, fps=vfps, width=vi['width'], height=vi['height']) if (running and nf) else None
        status = dict(frame=fi, n_frames=nf, t_s=round(fi / vfps, 2), live_fps=round(fps_now, 1), dropped=0, vlm_pending=None) if running else None
        events = []
        for a in alerts:
            if a['id'] <= after:
                continue
            vt = float(a.get('video_time') or 0)
            events.append(dict(seq=a['id'], kind='vlm', track=a['track_id'], artifacts=a.get('artifact_ids') or [],
                               decision='ANOMALY', behavior=a['behavior'], confidence=a['confidence'], evidence=a.get('evidence') or '',
                               label=a['behavior'], start_s=max(0.0, vt - 3), end_s=vt + 1))
        return {'meta': meta, 'status': status, 'events': events, 'job': view(j)['status']}

    # ------------------------------------------------------------------ kết quả
    def synth_result(j):
        """Không có result.json (chế độ demo): dựng từ danh sách cảnh báo để web vẫn hiện được."""
        vi = j['vinfo']
        with state.lock:
            alerts = list(state.alerts)
        fps = vi['fps'] or 25.0
        events = []
        for a in alerts:
            vt = float(a.get('video_time') or 0)
            events.append(dict(event_id=f'e{a["id"]}', track_id=a['track_id'], artifact_id=(a.get('artifact_ids') or [None])[0],
                               start_time_s=max(0.0, vt - 3), end_time_s=vt + 1, decision='ANOMALY', behavior=a['behavior'],
                               vlm_confidence=a['confidence'], vlm_evidence=a.get('evidence') or ''))
        return dict(video=dict(key=j['path'].stem, name=j['name'], fps=fps, n_frames=vi['n_frames'] or int(fps * 60),
                               width=vi['width'] or 960, height=vi['height'] or 540),
                    artifacts=[], alerts=[], events=events)

    @app.get('/api/jobs/{jid}/result')
    def get_result(jid: str):
        j = get_job(jid)
        if view(j)['status'] != 'done':
            raise HTTPException(409, 'Job chưa xong')
        rd = result_dir(j)
        if rd is not None and (rd / 'result.json').is_file():
            return FileResponse(rd / 'result.json', media_type='application/json')
        return synth_result(j)

    @app.get('/api/jobs/{jid}/video')
    def get_video(jid: str, kind: str = 'live'):
        j = get_job(jid)
        rd = result_dir(j) if finished(j) else None
        if kind != 'original' and rd is not None and (rd / 'live.mp4').is_file():
            return FileResponse(rd / 'live.mp4', media_type='video/mp4')
        if not j['path'].is_file():
            raise HTTPException(404, 'Không còn video')
        return FileResponse(j['path'])

    @app.get('/api/jobs/{jid}/files/{path:path}')
    def get_file(jid: str, path: str):
        j = get_job(jid)
        rd = result_dir(j)
        if rd is None:
            raise HTTPException(404, 'Chưa có kết quả')
        base = rd.parent.resolve()
        target = (base / path).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(404, 'Không có file')
        return FileResponse(target)

    # Giao diện web cùng origin với API (chạy thẳng trên server GPU, không cần dán URL backend). Phải mount SAU các route /api.
    if WEB_DIR.is_dir():
        app.mount('/', StaticFiles(directory=str(WEB_DIR), html=True), name='web')
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--data-dir', default=str(PROJ / 'dashboard_data'))
    ap.add_argument('--demo', action='store_true', help='dữ liệu giả của dashboard, không cần GPU / trọng số')
    ap.add_argument('--mode', choices=('tracking', 'tracking_seg', 'full'), default=None)
    ap.add_argument('--no-preload', action='store_true', help='không nạp model lúc khởi động')
    ap.add_argument('--local', action='store_true', help='máy ít VRAM (xem dashboard.server --local)')
    a = ap.parse_args()
    uvicorn.run(create_app(a.data_dir, a.demo, preload=not a.no_preload, mode=a.mode, local=a.local),
                host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
