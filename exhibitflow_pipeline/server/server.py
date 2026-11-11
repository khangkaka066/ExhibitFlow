#!/usr/bin/env python
"""ExhibitFlow — API server cho web (FastAPI). Nhận video, chạy `main.py`, trả kết quả.

    python server/server.py --port 8000

Endpoint (web/js/app.js dùng đúng các đường dẫn này):
    GET  /api/health                      -> {ok, version}
    POST /api/jobs                        (multipart: video HOẶC source=<đường dẫn trên server | URL>, options=JSON) -> {job_id}
    POST /api/jobs/{id}/cancel            dừng job
    GET  /api/system                      CPU / RAM / GPU / VRAM
    GET  /api/jobs/{id}/live              khung hình mới nhất (chế độ realtime: options.realtime=true)
    GET  /api/jobs/{id}/live/events       meta + trạng thái + sự kiện đang chạy (?after=<seq>)
    GET  /api/jobs/{id}                   -> {status, stage, stages, progress, log, key, error}
    GET  /api/jobs/{id}/result            -> result.json
    GET  /api/jobs/{id}/video             -> live.mp4 (video có khung); ?kind=original -> video gốc
    GET  /api/jobs/{id}/files/{path}      -> file trong thư mục kết quả (ảnh bằng chứng VLM)

Biến môi trường: EXHIBITFLOW_PROJ (thư mục exhibitflow_pipeline), EXHIBITFLOW_JOBS (nơi lưu job),
EXHIBITFLOW_WEIGHTS (thư mục trọng số, ngăn cách bằng ':' hoặc ';'), EXHIBITFLOW_MAX_MB (mặc định 600), EXHIBITFLOW_SOURCE_ROOTS (thư mục được phép dùng làm `source`),
EXHIBITFLOW_FAKE=<thư mục kết quả mẫu> (chạy giả để thử giao diện, không cần GPU).
Mỗi lúc chỉ chạy 1 job (GPU); job tới sau xếp hàng.
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJ = Path(os.environ.get('EXHIBITFLOW_PROJ') or Path(__file__).resolve().parent.parent)
JOBS_DIR = Path(os.environ.get('EXHIBITFLOW_JOBS') or PROJ / 'work' / 'jobs')
FAKE = os.environ.get('EXHIBITFLOW_FAKE')
MAX_BYTES = int(os.environ.get('EXHIBITFLOW_MAX_MB', '600')) * 1024 * 1024
ALLOWED = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
STAGES = ['Khởi động & nạp model', 'Tách hiện vật', 'Nạp VLM', 'Phân tích video', 'Ghi kết quả']
MARKS = (('①', 0), ('②', 1), ('③', 2), ('④', 3))          # dòng log của pipeline.py đánh dấu từng pha
ID_RE = re.compile(r'^[0-9a-f]{12}$')
SOURCE_ROOTS = [Path(p).resolve() for p in (os.environ.get('EXHIBITFLOW_SOURCE_ROOTS')
                                            or os.pathsep.join(['/kaggle/input', '/kaggle/working', '/content', str(PROJ)])).split(os.pathsep) if p]

JOBS = {}
RUN_LOCK = threading.Lock()
app = FastAPI(title='ExhibitFlow API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


def _version():
    try:
        sys.path.insert(0, str(PROJ))
        from exhibitflow import settings
        return settings.VERSION
    except Exception:
        return 'unknown'


def _job(job_id):
    if not ID_RE.match(job_id) or job_id not in JOBS:
        raise HTTPException(404, 'Không có job này')
    return JOBS[job_id]


def _set_stage(job, stage):
    job['stage'] = stage
    job['stage_t0'] = time.time()


def _log(job, line):
    job['log'].append(line.rstrip())
    del job['log'][:-200]


def _run_real(job):
    video, out = job['video'], job['out']
    cmd = [sys.executable, str(PROJ / 'main.py'), '--videos', str(video), '--out', str(out),
           '--work-dir', str(JOBS_DIR.parent), '--save-video']
    o = job['options']
    cmd += ['--set', 'rt.hud=false']                             # thông số FPS/frame không vẽ lên video (web hiển thị riêng)
    if o.get('max_frames'):
        cmd += ['--max-frames', str(int(o['max_frames']))]
    if o.get('conf') is not None:
        cmd += ['--set', f"vlm_min_conf={float(o['conf'])}"]
    if job.get('live'):                                          # realtime: đọc đúng nhịp fps + phát kết quả từng phần
        cmd.remove('--save-video')
        cmd += ['--set', 'rt.realtime=true', '--set', f"rt.publish_dir={job['live']}"]
        if o.get('early_sec') is not None:
            cmd += ['--set', f"rt.vlm_early_sec={float(o['early_sec'])}"]
    if o.get('dry_run_vlm'):
        cmd += ['--dry-run-vlm']
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(cmd, cwd=str(PROJ), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', start_new_session=(os.name == 'posix'))
    job['proc'] = proc
    for line in proc.stdout:
        _log(job, line)
        for mark, stage in MARKS:
            if mark in line:
                _set_stage(job, stage)
    return proc.wait()


def _atomic_copy(src, dst):
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    shutil.copyfile(src, tmp)
    try:
        os.replace(tmp, dst)
    except OSError:
        pass


def _run_fake_live(job, src):
    """Giả lập realtime: phát khung hình + sự kiện theo thời gian để thử giao diện."""
    live = job['live']
    live.mkdir(parents=True, exist_ok=True)
    res = json.loads((src / 'result.json').read_text(encoding='utf-8'))
    v = res['video']
    fps = v.get('fps') or 24
    nw = min(960, v['width'])
    arts = [dict(id=a['id'], label=a.get('label'), box=a['box'], pedestal=bool(a.get('pedestal'))) for a in res['artifacts']]
    for i, msg in enumerate(['① Nạp model (giả lập)', '② Tách hiện vật (giả lập)', '③ Nạp VLM (giả lập)']):
        _set_stage(job, i)
        _log(job, msg)
        time.sleep(1.0)
        if job.get('cancelled'):
            return 1
    _set_stage(job, 3)
    _log(job, '④ Chạy realtime (giả lập)')
    total = 14.0
    (live / 'meta.json').write_text(json.dumps(dict(video=src.name, fps=fps, n_frames=int(total * fps), width=nw,
                                                    height=round(v['height'] * nw / v['width']), artifacts=arts)), encoding='utf-8')
    (live / 'events.jsonl').write_text('', encoding='utf-8')
    frame_src = src / 'artifact_review.jpg'
    plan = [
        (2.0, dict(kind='contact', track=7, artifact=3, hand='left')),
        (6.0, dict(kind='vlm', task='seg_demo_1', track=7, artifacts=[3], start_s=1.0, end_s=5.0, decision='ANOMALY', behavior='touching',
                   label='touching', confidence=0.91, evidence='Bàn tay chạm vào hiện vật khoảng 3 giây (giả lập).', latency_s=2.1, status='ok')),
        (8.0, dict(kind='contact', track=4, artifact=0, hand='right')),
        (11.0, dict(kind='vlm', task='seg_demo_2', track=4, artifacts=[0], start_s=7.0, end_s=10.0, decision='ANOMALY', behavior='holding',
                    label='holding', confidence=0.87, evidence='Khách nhấc hiện vật lên (giả lập).', latency_s=2.4, status='ok')),
        (12.0, dict(kind='contact', track=2, artifact=502, hand='left')),
        (13.5, dict(kind='vlm', task='seg_demo_3', track=2, artifacts=[502], start_s=11.0, end_s=13.0, decision='NORMAL', behavior='normal',
                    label='normal', confidence=0.95, evidence='Chỉ đứng gần, không chạm (giả lập).', latency_s=1.8, status='ok')),
    ]
    t0, seq, k = time.time(), 0, 0
    while True:
        el = time.time() - t0
        if job.get('cancelled'):
            return 1
        while k < len(plan) and plan[k][0] <= el:
            seq += 1
            t_s, ev = plan[k]
            if ev['kind'] == 'contact':
                ev.update(frame=int(t_s * fps), t_s=t_s)
            ev.update(start_frame=int(ev['start_s'] * fps) if 'start_s' in ev else None,
                      end_frame=int(ev['end_s'] * fps) if 'end_s' in ev else None) if ev['kind'] == 'vlm' else None
            with open(live / 'events.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(seq=seq, **ev), ensure_ascii=False) + '\n')
            k += 1
        if frame_src.exists():
            _atomic_copy(frame_src, live / 'live.jpg')
        done = el >= total
        (live / 'status.json').write_text(json.dumps(dict(
            state='done' if done else 'running', frame=int(min(el, total) * fps), n_frames=int(total * fps), fps=fps, t_s=round(min(el, total), 2),
            live_fps=round(fps, 1), dropped=0, vlm_pending=0, n_alerts=sum(1 for _, e in plan[:k] if e.get('decision') == 'ANOMALY'),
            n_events=seq, wall_s=round(el, 1))), encoding='utf-8')
        if done:
            break
        time.sleep(0.15)
    dst = job['out'] / job['video'].stem
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.copy(os.environ.get('EXHIBITFLOW_FAKE_VIDEO') or job['video'], dst / 'live.mp4')
    return 0


def _run_fake(job):
    """Giả lập: chép kết quả mẫu để thử giao diện không cần GPU."""
    src = Path(FAKE)
    if job.get('live'):
        return _run_fake_live(job, src)
    for i, msg in enumerate(['① Nạp model (giả lập)', '② Tách hiện vật (giả lập)', '③ Nạp VLM (giả lập)', '④ Chạy pipeline (giả lập)']):
        _set_stage(job, i)
        _log(job, msg)
        time.sleep(1.5)
        if job.get('cancelled'):
            return 1
    dst = job['out'] / job['video'].stem
    shutil.copytree(src, dst, dirs_exist_ok=True)
    fake_video = os.environ.get('EXHIBITFLOW_FAKE_VIDEO')
    shutil.copy(fake_video or job['video'], dst / 'live.mp4')
    return 0


def _worker(job):
    _log(job, 'Đang chờ tới lượt chạy…')
    with RUN_LOCK:
        if job.get('cancelled'):
            job['status'] = 'cancelled'
            return
        job['status'] = 'running'
        _set_stage(job, 0)
        t0 = time.time()
        try:
            rc = _run_fake(job) if FAKE else _run_real(job)
            if job.get('cancelled'):
                job['status'] = 'cancelled'
                _log(job, 'Đã dừng theo yêu cầu')
                return
            _set_stage(job, 4)
            found = sorted(job['out'].glob('*/result.json'))
            if rc == 0 and found:
                job['key'] = found[0].parent.name
                job['status'] = 'done'
                _log(job, f'Xong sau {time.time() - t0:.0f}s')
            else:
                job['status'] = 'error'
                job['error'] = ('pipeline kết thúc mã %s, không có result.json. ' % rc) + ' | '.join(job['log'][-6:])
        except Exception as e:                                   # noqa: BLE001
            job['status'] = 'error'
            job['error'] = f'{type(e).__name__}: {e}'


@app.get('/api/health')
def health():
    return {'ok': True, 'version': _version(), 'fake': bool(FAKE), 'jobs': len(JOBS)}


def _check_ext(ext):
    if ext not in ALLOWED:
        raise HTTPException(400, f'Định dạng không hỗ trợ ({ext or "không có đuôi"}). Dùng: {", ".join(sorted(ALLOWED))}')


def _resolve_source(src, jdir):
    """`source` = URL http(s) (tải về) hoặc đường dẫn file / thư mục trên server (chỉ trong SOURCE_ROOTS)."""
    if re.match(r'^(rtsps?|rtmps?)://', src, re.I):
        raise HTTPException(501, 'Chưa hỗ trợ nguồn camera trực tiếp (RTSP/RTMP): pipeline hiện chỉ đọc file video, '
                                 'cần mở rộng bộ đọc khung hình của pipeline. Tạm thời dùng Realtime với File video.')
    if re.match(r'^https?://', src, re.I):
        path = urllib.parse.urlparse(src).path
        ext = Path(path).suffix.lower() or '.mp4'
        _check_ext(ext)
        dst, size = jdir / f'input{ext}', 0
        try:
            with urllib.request.urlopen(src, timeout=30) as r, open(dst, 'wb') as f:
                while chunk := r.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                    f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, f'Không tải được URL: {e}')
        return dst, (Path(path).name or f'video{ext}')
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
    _check_ext(p.suffix.lower())
    return p, p.name


@app.post('/api/jobs')
async def create_job(video: UploadFile | None = File(None), options: str = Form('{}'), source: str = Form('')):
    try:
        opts = json.loads(options or '{}')
    except json.JSONDecodeError:
        opts = {}
    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    (jdir / 'out').mkdir(parents=True)
    try:
        if video is not None and video.filename:
            ext = Path(video.filename).suffix.lower()
            _check_ext(ext)
            vpath, size = jdir / f'input{ext}', 0
            with open(vpath, 'wb') as f:
                while chunk := await video.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                    f.write(chunk)
            name = video.filename
        elif source.strip():
            vpath, name = _resolve_source(source.strip(), jdir)
        else:
            raise HTTPException(400, 'Cần gửi video hoặc đường dẫn / URL')
    except Exception:
        shutil.rmtree(jdir, ignore_errors=True)
        raise
    job = dict(id=job_id, status='queued', stage=0, stage_t0=time.time(), log=[], key=None, error=None, options=opts,
               video=vpath, out=jdir / 'out', name=name, proc=None, cancelled=False,
               live=(jdir / 'live') if opts.get('realtime') else None)
    JOBS[job_id] = job
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    j = _job(job_id)
    prog = 1.0 if j['status'] == 'done' else (1 - math.exp(-(time.time() - j['stage_t0']) / 45)) * .95 if j['status'] == 'running' else 0
    return dict(status=j['status'], stage=j['stage'], stages=STAGES, progress=round(prog, 3), log=j['log'][-40:],
                key=j['key'], error=j['error'], name=j['name'], realtime=bool(j.get('live')))


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    j = _job(job_id)
    if j['status'] in ('done', 'error', 'cancelled'):
        return {'status': j['status']}
    j['cancelled'] = True
    p = j.get('proc')
    if p and p.poll() is None:
        try:
            if os.name == 'posix':
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)       # dừng cả llama-server con
            else:
                p.terminate()
        except OSError:
            pass
    return {'status': 'cancelling'}


@app.get('/api/system')
def system():
    out = {'cpu': None, 'ram': None, 'gpus': []}
    try:
        la, n = os.getloadavg(), os.cpu_count() or 1
        out['cpu'] = {'load': round(la[0], 2), 'pct': round(min(100.0, la[0] / n * 100), 1)}
    except (AttributeError, OSError):
        pass
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.split()[0])
        tot = info['MemTotal'] / 1048576
        out['ram'] = {'used_gb': round(tot - info['MemAvailable'] / 1048576, 1), 'total_gb': round(tot, 1)}
    except Exception:                                            # noqa: BLE001
        pass
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        for line in r.stdout.strip().splitlines():
            name, util, mu, mt = [x.strip() for x in line.split(',')]
            out['gpus'].append({'name': name, 'util': float(util), 'mem_used_gb': round(float(mu) / 1024, 1),
                                'mem_total_gb': round(float(mt) / 1024, 1)})
    except Exception:                                            # noqa: BLE001
        pass
    return out


@app.get('/api/jobs/{job_id}/live')
def live_frame(job_id: str):
    j = _job(job_id)
    f = j['live'] / 'live.jpg' if j.get('live') else None
    if not f or not f.is_file():
        raise HTTPException(404, 'Chưa có khung hình')
    return FileResponse(f, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


@app.get('/api/jobs/{job_id}/live/events')
def live_events(job_id: str, after: int = 0):
    j = _job(job_id)
    d = j.get('live')
    out = {'meta': None, 'status': None, 'events': [], 'job': j['status']}
    if not d or not d.exists():
        return out
    out['meta'] = _read_json(d / 'meta.json')
    out['status'] = _read_json(d / 'status.json')
    try:
        lines = (d / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:                    # dòng cuối có thể đang được ghi dở
            continue
        if e.get('seq', 0) > after:
            out['events'].append(e)
    return out


@app.get('/api/jobs/{job_id}/result')
def get_result(job_id: str):
    j = _job(job_id)
    if j['status'] != 'done':
        raise HTTPException(409, 'Job chưa xong')
    return FileResponse(j['out'] / j['key'] / 'result.json', media_type='application/json')


@app.get('/api/jobs/{job_id}/video')
def get_video(job_id: str, kind: str = 'live'):
    j = _job(job_id)
    live = j['out'] / (j['key'] or '') / 'live.mp4'
    if kind != 'original' and j['key'] and live.exists():
        return FileResponse(live, media_type='video/mp4')
    return FileResponse(j['video'])


@app.get('/api/jobs/{job_id}/files/{path:path}')
def get_file(job_id: str, path: str):
    j = _job(job_id)
    base = j['out'].resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():          # chặn ../ và đường dẫn ngoài thư mục kết quả
        raise HTTPException(404, 'Không có file')
    return FileResponse(target)


if __name__ == '__main__':
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='info')
