"""Chạy server nối web (server/web_server.py) + mở tunnel công khai (Cloudflare) -> in URL để dán vào web (Settings -> URL backend).

    python server/web_launch.py --port 8000 [--demo] [--data-dir DIR] [--no-tunnel]

Dùng lại các hàm tunnel / chờ nạp model của dashboard.launch (không sửa file đó).
"""
import argparse
import atexit
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

from dashboard.launch import (free_port, port_in_use, start_cloudflared, start_localhost_run,  # noqa: E402
                              wait_models, wait_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--no-tunnel', action='store_true')
    ap.add_argument('--no-wait-models', action='store_true', help='in URL ngay, không đợi model nạp xong')
    ap.add_argument('--no-preload', action='store_true')
    ap.add_argument('--mode', choices=('tracking', 'tracking_seg', 'full'), default=None)
    ap.add_argument('--local', action='store_true')
    a = ap.parse_args()
    procs = []
    atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])
    if port_in_use(a.port):
        new = free_port()
        print(f'Cổng {a.port} đang bị chiếm -> dùng cổng {new}', flush=True)
        a.port = new
    cmd = [sys.executable, '-u', str(HERE / 'web_server.py'), '--port', str(a.port)]
    for flag, on in (('--demo', a.demo), ('--no-preload', a.no_preload), ('--local', a.local)):
        if on:
            cmd.append(flag)
    if a.mode:
        cmd += ['--mode', a.mode]
    if a.data_dir:
        cmd += ['--data-dir', a.data_dir]
    env = dict(os.environ)
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(k, '1')
    env.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    srv = subprocess.Popen(cmd, cwd=PROJ, env=env)
    procs.append(srv)
    for _ in range(120):
        if srv.poll() is not None:
            print(f'!!! Server thoát sớm (mã {srv.returncode}), xem lỗi ở trên.', flush=True)
            sys.exit(1)
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/health', timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    print(f'Server nội bộ: http://127.0.0.1:{a.port}', flush=True)
    if not a.no_wait_models and not a.no_preload:
        m = wait_models(a.port)
        print('Model đã sẵn sàng.' if m['state'] == 'ready' else f"!!! Model chưa sẵn sàng ({m['state']}): {m['msg']}", flush=True)
    tun = None
    if not a.no_tunnel:
        url = None
        for name, starter in (('cloudflared', lambda: start_cloudflared(a.port, PROJ / 'work' / 'bin')),
                              ('localhost.run (ssh)', lambda: start_localhost_run(a.port))):
            print(f'Mở tunnel bằng {name} ...', flush=True)
            try:
                tun = starter()
                if tun is not None:
                    procs.append(tun)
            except Exception as e:                      # noqa: BLE001
                print(f'  lỗi: {e}', flush=True)
                tun = None
            if tun is None:
                continue
            url = wait_url(tun)
            if url:
                break
            print(f'  {name} không trả về link, thử cách khác', flush=True)
            tun.terminate()
            tun = None
        if url:
            print(flush=True)
            print('=' * 64, flush=True)
            print(f'  URL BACKEND (dán vào web: Settings -> URL backend):\n  {url}', flush=True)
            print('=' * 64, flush=True)
        else:
            print(f'!!! Không mở được tunnel. Server vẫn chạy ở http://127.0.0.1:{a.port}', flush=True)
            print('    Kaggle: Settings -> Internet = On, rồi chạy lại ô này.', flush=True)
    try:
        srv.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (tun, srv):
            if p and p.poll() is None:
                p.terminate()


if __name__ == '__main__':
    main()
