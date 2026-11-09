"""Dashboard ExhibitFlow (FastAPI).

    python -m dashboard.server --port 8000            # chạy pipeline thật (cần GPU + trọng số)
    python -m dashboard.server --port 8000 --demo     # dữ liệu giả, xem thử giao diện
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import psutil  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.runner import DemoRunner, PipelineRunner, State  # noqa: E402

VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
FLOORPLAN_CANDIDATES = [
    HERE.parent.parent / 'external' / 'multicam-2d-floorplan-tracker' / 'assets' / 'global_2d_map.png',
    Path(os.environ.get('FLOORPLAN_DIR', '/nonexistent')) / 'assets' / 'global_2d_map.png',
    HERE / 'static' / 'floorplan_default.png',
]


def create_app(data_dir, demo=False):
    state = State(data_dir)
    runner = (DemoRunner if demo else PipelineRunner)(state)
    app = FastAPI(title='ExhibitFlow Dashboard')
    app.state.demo = demo

    app.mount('/snapshots', StaticFiles(directory=str(state.root / 'snapshots')), name='snapshots')
    app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')

    @app.get('/')
    def index():
        return FileResponse(HERE / 'static' / 'index.html', headers={'Cache-Control': 'no-store'})

    # ---------- nguồn video ----------
    @app.post('/api/upload')
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or 'video.mp4').name
        if not name.lower().endswith(VIDEO_EXT):
            raise HTTPException(400, f'Chỉ nhận {", ".join(VIDEO_EXT)}')
        dst = state.root / 'uploads' / name
        with open(dst, 'wb') as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        return dict(path=str(dst), name=name, size_mb=round(dst.stat().st_size / 1e6, 1))

    @app.get('/api/uploads')
    def uploads():
        return [dict(name=p.name, path=str(p), size_mb=round(p.stat().st_size / 1e6, 1))
                for p in sorted((state.root / 'uploads').iterdir()) if p.suffix.lower() in VIDEO_EXT]

    def resolve_source(src: str) -> str:
        src = (src or '').strip()
        if src.startswith(('http://', 'https://')):
            name = Path(src.split('?')[0]).name or 'download.mp4'
            if not name.lower().endswith(VIDEO_EXT):
                raise HTTPException(400, 'URL phải trỏ tới file video (mp4/avi/mov/mkv/webm). RTSP chưa được hỗ trợ.')
            dst = state.root / 'uploads' / name
            try:
                urllib.request.urlretrieve(src, dst)
            except Exception as e:
                raise HTTPException(400, f'Không tải được URL: {e}')
            return str(dst)
        p = Path(src)
        if p.is_dir():
            vids = sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT)
            if not vids:
                raise HTTPException(400, 'Thư mục không có video')
            return str(vids[0])
        if not p.is_file():
            raise HTTPException(400, f'Không thấy file: {src}')
        return str(p)

    @app.post('/api/start')
    def start(body: dict):
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy')
        path = resolve_source(body.get('source', '')) if not demo else (body.get('source') or 'demo.mp4')
        runner.start(path)
        return dict(ok=True, source=path)

    @app.post('/api/stop')
    def stop():
        runner.stop()
        return dict(ok=True)

    # ---------- trạng thái ----------
    @app.get('/api/state')
    def get_state():
        return state.snapshot()

    @app.get('/api/system')
    def system():
        vm, du = psutil.virtual_memory(), shutil.disk_usage(str(state.root))
        out = dict(cpu=psutil.cpu_percent(), load=round(os.getloadavg()[0], 1) if hasattr(os, 'getloadavg') else None,
                   mem_used=round(vm.used / 1e9, 1), mem_total=round(vm.total / 1e9, 1),
                   disk_used=round(du.used / 1e9, 1), disk_total=round(du.total / 1e9, 1), gpu=None)
        exe = shutil.which('nvidia-smi')
        if exe:
            try:
                o = subprocess.run([exe, '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3).stdout
                out['gpu'] = []
                for line in o.strip().splitlines():
                    n, u, mu, mt = [x.strip() for x in line.split(',')]
                    out['gpu'].append(dict(name=n, util=float(u), mem_used=round(float(mu) / 1024, 1), mem_total=round(float(mt) / 1024, 1)))
            except Exception:
                pass
        return out

    # ---------- video MJPEG ----------
    @app.get('/api/stream')
    async def stream():
        async def gen():
            last = -1
            while True:
                with state.lock:
                    seq, jpg = state.frame_seq, state.frame_jpeg
                if jpg is not None and seq != last:
                    last = seq
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
                await asyncio.sleep(0.03)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/api/frame')
    def frame():
        if state.frame_jpeg is None:
            raise HTTPException(404)
        from fastapi.responses import Response
        return Response(state.frame_jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    # ---------- settings ----------
    @app.get('/api/settings')
    def get_settings():
        return state.settings

    @app.put('/api/settings')
    def put_settings(body: dict):
        return state.save_settings(body)

    # ---------- map ----------
    @app.get('/api/map-image')
    def map_image():
        mi = state.settings.get('map_image', 'default')
        if mi == 'none':
            raise HTTPException(404)
        if mi not in ('default', ''):
            p = state.root / 'uploads' / mi
            if p.is_file():
                return FileResponse(p)
        for p in FLOORPLAN_CANDIDATES:
            if p.is_file():
                return FileResponse(p)
        raise HTTPException(404)

    @app.post('/api/map-image')
    async def upload_map(file: UploadFile = File(...)):
        name = 'floorplan_' + Path(file.filename or 'map.png').name
        (state.root / 'uploads' / name).write_bytes(await file.read())
        state.save_settings(dict(map_image=name))
        return dict(name=name)

    # ---------- xuất ----------
    @app.get('/api/alerts.csv')
    def alerts_csv():
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['id', 'time', 'video_time_s', 'camera', 'track_id', 'artifact_ids', 'behavior', 'confidence', 'severity', 'evidence'])
        for a in state.alerts:
            w.writerow([a['id'], a['time'], a['video_time'], a['camera'], a['track_id'], ' '.join(map(str, a['artifact_ids'])),
                        a['behavior'], a['confidence'], a['severity'], a['evidence']])
        return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv',
                                 headers={'Content-Disposition': 'attachment; filename=alerts.csv'})

    @app.exception_handler(Exception)
    async def err(_, e):
        return JSONResponse(status_code=500, content=dict(detail=f'{type(e).__name__}: {e}'))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--data-dir', default=str(HERE.parent / 'dashboard_data'))
    ap.add_argument('--demo', action='store_true', help='dữ liệu giả, không cần GPU / trọng số')
    a = ap.parse_args()
    uvicorn.run(create_app(a.data_dir, a.demo), host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
