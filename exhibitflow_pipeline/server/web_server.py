"""ExhibitFlow — server nối giao diện web (thư mục `web/`) với engine realtime của dashboard Hưng.

Không sửa file nào của `dashboard/`: dùng lại `State` + `PipelineRunner` (model nạp sẵn một lần, chạy nhiều video)
và trả lời bằng API "job" mà web đang dùng:

    GET  /api/health                      kiểm tra sống + trạng thái nạp model
    GET  /api/state                       trạng thái dashboard (dùng cho launcher chờ nạp model)
    GET  /api/system                      CPU / RAM / GPU
    POST /api/jobs                        multipart: video (upload) | source (đường dẫn trên server / URL) + options -> {job_id}
    GET  /api/jobs/{id}                   {status, stage, stages, progress, key, error}
    POST /api/jobs/{id}/cancel            dừng (bỏ kết quả)
    POST /api/jobs/{id}/finish            camera: dừng thu và xử lý nốt để có kết quả
    POST /api/multi-jobs                  nhiều camera (2-8): song song (options.multi_parallel, tối đa multi_max_parallel camera cùng lúc,
                                          mỗi camera một tiến trình riêng) hoặc tuần tự. Form gồm options (JSON), cameras (JSON [{name, source}]),
                                          và video_<i> (file upload của camera i) -> {multi_id}
    GET  /api/multi-jobs/{id}             trạng thái từng camera (job_id, status), cảnh báo gộp (có nhãn camera), camera đang chạy (cur)
    POST /api/multi-jobs/{id}/finish      dừng: job con đang chạy xử lý nốt, các camera còn lại bỏ qua
    POST /api/camera/session              tạo phiên webcam của trình duyệt -> {sid}; rồi mở WebSocket /api/camera/{sid}/push gửi từng khung JPEG
                                          và tạo job với source=push://<sid>?fps=15. source=rtsp://... để đọc camera IP
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
import asyncio
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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.runner import DemoRunner, PipelineRunner, State  # noqa: E402
from exhibitflow import cam_worker  # noqa: E402
from exhibitflow import multicam  # noqa: E402
from exhibitflow import source as efsource  # noqa: E402
from exhibitflow import tuning as _tuning  # noqa: E402

ALLOWED = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_BYTES = int(os.environ.get('EXHIBITFLOW_MAX_MB', '600')) * 1024 * 1024
STAGES = ['Khởi động & nạp model', 'Tách hiện vật', 'Nạp VLM', 'Phân tích video', 'Ghi kết quả']
SOURCE_ROOTS = [Path(p).resolve() for p in (os.environ.get('EXHIBITFLOW_SOURCE_ROOTS')
                                            or os.pathsep.join(['/workspace', '/kaggle/input', '/kaggle/working', '/content', str(PROJ)])).split(os.pathsep) if p]
WEB_DIR = Path(os.environ.get('EXHIBITFLOW_WEB_DIR') or PROJ.parent / 'web')   # giao diện web, phục vụ chung cổng với API
# Cấu hình pipeline web được phép đặt (khoá + kiểu) = các mục tương ứng trong Settings của dashboard cũ
PIPE_MODES = ('tracking', 'tracking_seg', 'full')
PIPE_OPTS = dict(pipeline_mode=lambda v: v if v in PIPE_MODES else 'full', tracker_process=bool, hand_process=bool, pose_process=bool, pose_gpu_preproc=bool, hand_batch=bool, drop_frames=bool, limit_cpu_threads=bool, alarm_show_sec=float,
                  tracker_threads=int, hand_threads=int, pose_threads=int, queue_size=int, pose_gate_norm_dist=float,
                  multi_parallel=bool, multi_max_parallel=int, tuning=_tuning.validate_tuning)
NW = 960


def pipe_of(opts):
    """Cấu hình pipeline lấy từ options của request (chỉ các khoá trong PIPE_OPTS)."""
    return {k: t(opts[k]) for k, t in PIPE_OPTS.items() if opts.get(k) is not None}                                   # web vẽ hộp hiện vật theo khung rộng 960 px


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
    jobs, cur, multis = {}, {'id': None}, {}
    uploads = state.root / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tiện ích
    def check_ext(ext):
        if ext not in ALLOWED:
            raise HTTPException(400, f'Định dạng không hỗ trợ ({ext or "không có đuôi"}). Dùng: {", ".join(sorted(ALLOWED))}')

    async def save_upload(video):
        """Ghi file upload vào thư mục uploads -> (đường dẫn, tên gốc)."""
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
        return vpath, video.filename

    def resolve_source(src, dst_dir):
        """URL http(s) (tải về) hoặc đường dẫn file / thư mục trên server (chỉ trong SOURCE_ROOTS)."""
        if re.match(r'^(rtsps?|rtmps?)://', src, re.I):          # camera IP: file mô tả .efstream thay cho đường dẫn video
            return efsource.write_stream_file(dst_dir / f'cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}', url=src), 'Camera ' + efsource._mask(src)
        m = re.match(r'^push://([\w-]+)(?:\?fps=(\d+))?$', src)
        if m:                                                      # webcam của trình duyệt (khung đẩy qua WebSocket)
            if m.group(1) not in efsource.PUSH:
                raise HTTPException(400, 'Phiên webcam không tồn tại hoặc đã đóng: mở lại camera rồi thử lại.')
            return efsource.write_stream_file(dst_dir / f'cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}', push=m.group(1),
                                              fps=int(m.group(2) or 15)), 'Camera máy'
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
        if efsource.is_stream(path):             # camera: chưa mở (mở ở luồng pipeline); kích thước lấy từ efsource.INFO khi có
            return dict(fps=25.0, n_frames=0, width=0, height=0)
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

    def vinfo_of(j):
        """Thông tin video của job; với camera thì cập nhật từ kích thước / fps pipeline đã đo ở lần mở đầu."""
        if efsource.is_stream(j['path']) and (m := efsource.INFO.get(str(j['path']))):
            j['vinfo'] = dict(fps=m['fps'], n_frames=0, width=m['width'], height=m['height'])
        return j['vinfo']

    def get_job(jid):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, 'Không có job này')
        return j

    def result_dir(j):
        """Thư mục kết quả (chứa result.json) của job: tìm result.json mới nhất sinh ra sau khi job bắt đầu."""
        if j.get('rdir') is not None:
            return j['rdir']
        found = [p for p in ((j.get('root') or state.root) / 'results').rglob('result.json') if p.stat().st_mtime >= j['t0'] - 2]
        if not found:
            return None
        j['rdir'] = max(found, key=lambda p: p.stat().st_mtime).parent
        return j['rdir']

    def finished(j):
        """Job đã có kết quả cuối (trạng thái chốt, không đổi nữa)."""
        return j.get('final') is not None

    def remote_view(j):
        """Job con chạy trong tiến trình riêng (multi-camera song song): trạng thái đọc từ tiến trình đó."""
        if finished(j):
            return j['final']
        try:
            v = cam_worker.view_of(j['remote'].snap(), cancelled=j['cancelled'])
        except Exception as e:  # noqa: BLE001 - tiến trình con chết / lỗi IPC: chỉ camera này lỗi
            v = dict(status='error', stage=0, progress=0, error=f'Tiến trình camera lỗi: {e}'[:300])
        if v['status'] in ('done', 'error', 'cancelled'):
            j['final'] = v
        return v

    def remote_frame(j, last):
        try:
            return j['remote'].frame(last)
        except Exception:  # noqa: BLE001
            return None

    def stop_job(j):
        """Dừng một job: job con (multi song song) hoặc pipeline chung của server."""
        if j.get('remote') is not None:
            j['remote'].stop()
        else:
            runner.stop()

    def view(j):
        """Chuyển trạng thái `State` của dashboard sang trạng thái job mà web hiểu."""
        if j.get('remote') is not None:
            return remote_view(j)
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

    @app.get('/api/tuning/schema')
    def get_tuning_schema():
        return _tuning.schema()

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
        try:
            pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy video khác, đợi xong hoặc bấm Dừng.')
        if runner.loading():
            raise HTTPException(409, 'Model đang nạp (lần đầu mất vài phút), thử lại sau.')
        if video is not None and video.filename:
            vpath, name = await save_upload(video)
        elif source.strip():
            vpath, name = resolve_source(source.strip(), uploads)
        else:
            raise HTTPException(400, 'Cần gửi video hoặc đường dẫn / URL')
        return {'job_id': start_job(vpath, name, opts, realtime=bool(opts.get('realtime')))}

    def start_job(vpath, name, opts, realtime):
        """Khởi chạy một lần pipeline cho một nguồn; dùng chung cho job đơn và job con của multi-camera."""
        # realtime: đọc đúng nhịp fps như camera, bỏ frame nếu không kịp; theo lô (Video): xử lý mọi frame
        try:
            pipe = pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if efsource.is_stream(vpath):             # camera: thời lượng tối đa (phút) + không giả lập realtime
            from exhibitflow import settings as efs
            efs.RT['stream_max_sec'] = max(10.0, float(opts.get('camera_max_min') or 30) * 60)
        state.save_settings(dict(pipe, realtime=realtime))
        try:
            runner.start(str(vpath))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = dict(id=jid, path=Path(vpath), name=name, t0=time.time(), cancelled=False, final=None, rdir=None,
                         realtime=realtime, vinfo=video_info(vpath))
        cur['id'] = jid
        return jid

    # ------------------------------------------------------------------ multi-camera (chạy tuần tự, xem exhibitflow/multicam.py)
    def get_multi(mid):
        mr = multis.get(mid)
        if mr is None:
            raise HTTPException(404, 'Không có lần multi-camera này')
        return mr

    @app.post('/api/multi-jobs')
    async def create_multi(request: Request):
        form = await request.form()
        try:
            opts = json.loads(form.get('options') or '{}')
            specs = json.loads(form.get('cameras') or '[]')
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(400, 'Danh sách camera không hợp lệ')
        try:
            pipe_opts = pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if not isinstance(specs, list) or not (multicam.MIN_CAMS <= len(specs) <= multicam.MAX_CAMS):
            raise HTTPException(400, f'Cần từ {multicam.MIN_CAMS} đến {multicam.MAX_CAMS} camera')
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy, đợi xong hoặc bấm Dừng.')
        if runner.loading():
            raise HTTPException(409, 'Model đang nạp (lần đầu mất vài phút), thử lại sau.')
        cams = []
        for i, sp in enumerate(specs):
            sp = sp if isinstance(sp, dict) else {}
            up = form.get(f'video_{i}')                # UploadFile của Starlette (không phải lớp con của fastapi.UploadFile)
            if up is not None and getattr(up, 'filename', None):
                vpath, label = await save_upload(up)
            else:
                src = str(sp.get('source') or '').strip()
                if not src:
                    raise HTTPException(400, f'Camera {i + 1} chưa có nguồn (URL RTSP, đường dẫn hoặc file)')
                if src.lower().startswith('push://'):
                    raise HTTPException(400, 'Multi-camera không nhận webcam của trình duyệt: dùng RTSP hoặc file video.')
                vpath, label = resolve_source(src, uploads)
            cams.append(dict(name=str(sp.get('name') or '').strip()[:40] or label, path=vpath))

        maxp = min(max(1, int(opts.get('multi_max_parallel') or 2)), len(cams))
        par = bool(opts.get('multi_parallel', True)) and maxp > 1
        run_root = state.root / 'multi' / uuid.uuid4().hex[:12]     # thư mục của lần multi này (mỗi camera một thư mục con)
        if par:
            for i, c in enumerate(cams):
                c['root'] = run_root / f'cam{i + 1}'
            if runner.engine is not None:               # model của tiến trình chính nhường VRAM cho các tiến trình camera
                runner.engine.close()
                runner.engine = None
                with state.lock:
                    state.models_state = 'idle'

        def launch_seq(c):
            while runner.loading() or runner.running():     # luồng pipeline của camera trước phải hết hẳn mới mở camera kế
                time.sleep(0.3)
            return start_job(c['path'], c['name'], opts, realtime=efsource.is_stream(c['path']))

        def launch_par(c):
            """Camera chạy trong tiến trình riêng (cam_worker): cấu hình = cài đặt hiện tại + tuỳ chọn của lần này."""
            realtime = efsource.is_stream(c['path'])
            cfg = dict(root=str(c['root']), video=str(c['path']), demo=demo, local=local,
                       settings=dict(state.settings, **pipe_opts, realtime=realtime),
                       extra=dict(models=dict(vlm=dict(port=cam_worker.free_port()))),   # llama-server riêng cho từng camera
                       flock=str(run_root / 'load.lock'),
                       stream_max_sec=max(10.0, float(opts.get('camera_max_min') or 30) * 60) if realtime else None)
            proc = cam_worker.CamProc(cfg)
            jid = uuid.uuid4().hex[:12]
            jobs[jid] = dict(id=jid, path=Path(c['path']), name=c['name'], t0=time.time(), cancelled=False, final=None, rdir=None,
                             realtime=realtime, vinfo=video_info(c['path']), root=c['root'], remote=proc)
            return jid

        def release_par(jid):
            j = jobs.get(jid)
            if j is not None and j.get('remote') is not None:
                j['remote'].close()

        def poll(jid):
            j = get_job(jid)
            v = view(j)
            key = None
            if v['status'] == 'done':
                rd = result_dir(j)
                key = rd.name if rd else (state.cam_name if j.get('remote') is None else j['path'].stem) or j['path'].stem
            return dict(status=v['status'], key=key, error=v['error'])

        def alerts_of(jid, final):
            j = get_job(jid)
            if not final:
                if j.get('remote') is not None:
                    return multicam.from_live_alerts(j['remote'].snap()['alerts'])
                with state.lock:
                    return multicam.from_live_alerts(list(state.alerts))
            rd = result_dir(j)
            if rd is None or not (rd / 'result.json').is_file():
                return None
            return multicam.from_result_alerts(json.loads((rd / 'result.json').read_text(encoding='utf-8')))

        def stop_cam(jid):
            stop_job(get_job(jid))

        mr = multicam.MultiRun(cams, launch=launch_par if par else launch_seq, poll=poll, stop=stop_cam, alerts_of=alerts_of,
                               parallel=maxp if par else 1, release=release_par if par else None).start()
        multis[mr.id] = mr
        return {'multi_id': mr.id, 'cameras': [c['name'] for c in mr.cams], 'parallel': maxp if par else 1}

    @app.get('/api/multi-jobs/{mid}')
    def multi_status(mid: str):
        return get_multi(mid).snapshot()

    @app.get('/api/multi-jobs/{mid}/results')
    def multi_results(mid: str):
        """result.json của từng camera đã xong. Đọc thẳng từ thư mục kết quả (không qua get_result), vì job cũ đã bị coi là 'cancelled'
        khi job con sau bắt đầu. Camera chưa có kết quả (lỗi / bỏ qua / demo) không có trong danh sách."""
        out = []
        for c in get_multi(mid).snapshot()['cams']:
            j = jobs.get(c['job_id']) if c['job_id'] else None
            rd = result_dir(j) if (j is not None and c['status'] == 'done') else None
            if rd is not None and (rd / 'result.json').is_file():
                out.append(dict(idx=c['idx'], name=c['name'], job_id=c['job_id'], key=rd.name,
                                result=json.loads((rd / 'result.json').read_text(encoding='utf-8'))))
        return out

    @app.post('/api/multi-jobs/{mid}/finish')
    def multi_finish(mid: str):
        get_multi(mid).finish()
        return {'status': 'finishing'}

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

    @app.post('/api/camera/session')
    def camera_session():
        """Phiên webcam: trình duyệt mở WebSocket /api/camera/{sid}/push và gửi từng khung JPEG (binary)."""
        sid = uuid.uuid4().hex[:12]
        efsource.PUSH[sid] = efsource.PushBuffer()
        return {'sid': sid, 'ws': f'/api/camera/{sid}/push'}

    @app.websocket('/api/camera/{sid}/push')
    async def camera_push(ws: WebSocket, sid: str):
        buf = efsource.PUSH.get(sid)
        if buf is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        try:
            while True:
                buf.put(await ws.receive_bytes())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            buf.close()                           # đóng luồng: pipeline đọc hết khung còn lại rồi kết thúc và ghi kết quả
            efsource.PUSH.pop(sid, None)

    @app.post('/api/jobs/{jid}/finish')
    def finish(jid: str):
        """Camera: dừng thu rồi xử lý nốt (VLM các đoạn còn lại) để ra kết quả. Khác /cancel: không bỏ kết quả."""
        j = get_job(jid)
        v = view(j)
        if v['status'] in ('done', 'error', 'cancelled'):
            return {'status': v['status']}
        stop_job(j)
        return {'status': 'finishing'}

    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid: str):
        j = get_job(jid)
        v = view(j)
        if v['status'] in ('done', 'error', 'cancelled'):
            return {'status': v['status']}
        j['cancelled'] = True
        stop_job(j)
        return {'status': 'cancelling'}

    # ------------------------------------------------------------------ phát trực tiếp
    @app.get('/api/jobs/{jid}/live')
    def live_frame(jid: str):
        j = get_job(jid)
        if j.get('remote') is not None:
            r = remote_frame(j, -1)
            jpg = r['jpg'] if r else None
        else:
            with state.lock:
                jpg = state.frame_jpeg
        if jpg is None:
            raise HTTPException(404, 'Chưa có khung hình')
        return Response(jpg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/jobs/{jid}/live/stream')
    async def live_stream(jid: str):
        """MJPEG (multipart/x-mixed-replace) như /api/stream của dashboard cũ: server đẩy mỗi khung mới ngay khi có, không bị trễ do web poll.
        Kết thúc khi pipeline xong và không còn khung mới (để <img> dừng)."""
        j = get_job(jid)
        if j.get('remote') is not None:                 # camera chạy trong tiến trình riêng: lấy khung từ tiến trình đó
            async def gen_remote():
                last, idle = -1, 0
                while True:
                    r = await asyncio.to_thread(remote_frame, j, last)
                    if r is None:
                        return
                    if r['jpg'] is not None:
                        last, idle = r['seq'], 0
                        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(r['jpg'])).encode() + b'\r\n\r\n' + r['jpg'] + b'\r\n'
                    else:
                        idle += 1
                        if r['st'] in ('done', 'error') and not r['alive'] and idle > 60:
                            return
                        await asyncio.sleep(0.03)
            return StreamingResponse(gen_remote(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

        async def gen():
            last, idle = -1, 0
            while True:
                with state.lock:
                    seq, jpg, status = state.frame_seq, state.frame_jpeg, state.status
                if jpg is not None and seq != last:
                    last, idle = seq, 0
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
                else:
                    idle += 1
                    if (finished(j) or status in ('done', 'error', 'idle')) and idle > 60:      # ~2 s không có khung mới sau khi xong
                        return
                await asyncio.sleep(0.03)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

    def art_boxes(j):
        """Hộp hiện vật (toạ độ khung rộng 960 px) lấy từ đường viền cảnh đầu tiên của dashboard."""
        vi = vinfo_of(j)
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
        if j.get('remote') is not None:                 # job con song song: không có sự kiện chi tiết theo khung (dùng trạng thái multi)
            return {'meta': None, 'status': None, 'events': [], 'job': view(j)['status']}
        vi = vinfo_of(j)
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
        vi = vinfo_of(j)
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
        if efsource.is_stream(j['path']) or not j['path'].is_file():
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
