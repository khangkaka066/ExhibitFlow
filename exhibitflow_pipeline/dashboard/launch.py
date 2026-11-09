"""Chạy dashboard + mở tunnel công khai (Cloudflare quick tunnel, không cần tài khoản) -> in ra link cho mọi người.

    python -m dashboard.launch --port 8000 [--demo]
"""
import argparse
import atexit
import json
import socket
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _download(urls, dst, tries=3):
    last = None
    for url in urls:
        for k in range(tries):
            try:
                print(f'Tải cloudflared ({k + 1}/{tries}): {url}', flush=True)
                req = urllib.request.Request(url, headers={'User-Agent': 'ExhibitFlow-dashboard'})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst, 'wb') as f:
                    f.write(r.read())
                return True
            except Exception as e:                      # 504 / timeout / proxy chặn
                last = e
                print(f'  lỗi: {e}', flush=True)
                time.sleep(3 * (k + 1))
    print(f'Không tải được cloudflared: {last}', flush=True)
    return False


def get_cloudflared(dst_dir):
    found = shutil.which('cloudflared')
    if found:
        return found
    exe = dst_dir / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    if exe.exists() and exe.stat().st_size > 1_000_000:
        return str(exe)
    arch = 'arm64' if platform.machine().lower() in ('aarch64', 'arm64') else 'amd64'
    name = 'windows-amd64.exe' if os.name == 'nt' else f'linux-{arch}'
    base = 'https://github.com/cloudflare/cloudflared/releases'
    urls = [f'{base}/latest/download/cloudflared-{name}', f'{base}/download/2024.12.2/cloudflared-{name}']
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not _download(urls, exe):
        return None
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def start_cloudflared(port, bin_dir):
    exe = get_cloudflared(bin_dir)
    if not exe:
        return None
    return subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{port}'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def start_localhost_run(port):
    """Dự phòng: SSH reverse tunnel qua localhost.run (cần `ssh`, không cần tài khoản)."""
    ssh = shutil.which('ssh')
    if not ssh:
        return None
    return subprocess.Popen([ssh, '-o', 'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30', '-R', f'80:127.0.0.1:{port}',
                             'nokey@localhost.run'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


URL_RE = re.compile(r'https://[a-z0-9-]+\.(?:trycloudflare\.com|lhr\.life|localhost\.run)')


def wait_url(proc, timeout=60):
    """Đọc output của tunnel tới khi thấy URL. Trả (url, thread tiếp tục rút output)."""
    found, box = threading.Event(), {}

    def pump():
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m and 'url' not in box and 'admin.' not in m.group(0):
                box['url'] = m.group(0)
                found.set()
    threading.Thread(target=pump, daemon=True).start()
    found.wait(timeout)
    return box.get('url')


def wait_models(port, timeout=3600):
    """Đợi server nạp xong model (ready) hoặc lỗi, để khi in link thì model đã sẵn sàng."""
    t0, last = time.time(), 0
    print('Đang nạp model trước khi mở link công khai (lần đầu có thể tải ~6 GB, mất vài phút)...', flush=True)
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state', timeout=3) as r:
                m = json.loads(r.read())['models']
        except Exception:
            m = dict(state='loading', msg='')
        if m['state'] in ('ready', 'error'):
            return m
        if time.time() - last > 30:
            last = time.time()
            print(f'  ... vẫn đang nạp model ({int(last - t0)}s)', flush=True)
        time.sleep(2)
    return dict(state='timeout', msg='quá thời gian chờ')


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--no-tunnel', action='store_true')
    ap.add_argument('--no-wait-models', action='store_true', help='mở link ngay, không đợi model nạp xong')
    ap.add_argument('--no-preload', action='store_true')
    a = ap.parse_args()
    procs = []
    atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])   # luôn dọn server + tunnel, kể cả khi lỗi
    if port_in_use(a.port):                   # vd server của lần chạy trước còn sống -> dùng cổng trống khác
        new = free_port()
        print(f'Cổng {a.port} đang bị chiếm (có thể là dashboard của lần chạy trước) -> dùng cổng {new}', flush=True)
        a.port = new
    cmd = [sys.executable, '-m', 'dashboard.server', '--port', str(a.port)] + (['--demo'] if a.demo else []) + (['--no-preload'] if a.no_preload else [])
    if a.data_dir:
        cmd += ['--data-dir', a.data_dir]
    env = dict(os.environ)
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(k, '1')                # mỗi stage là 1 thread Python: thư viện số không được tự mở thêm thread
    env.setdefault('OMP_WAIT_POLICY', 'PASSIVE')   # không spin-wait chiếm CPU khi chờ
    srv = subprocess.Popen(cmd, cwd=HERE.parent, env=env)
    procs.append(srv)
    for _ in range(120):                      # đợi server lên (và phải là server của mình)
        if srv.poll() is not None:
            print(f'!!! Server dashboard thoát sớm (mã {srv.returncode}), xem lỗi ở trên.', flush=True)
            sys.exit(1)
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/state', timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    print(f'Dashboard nội bộ: http://127.0.0.1:{a.port}')
    if not a.no_wait_models and not a.no_preload:
        m = wait_models(a.port)
        if m['state'] == 'ready':
            print('Model đã sẵn sàng.', flush=True)
        else:
            print(f"!!! Model chưa sẵn sàng ({m['state']}): {m['msg']}", flush=True)
            print('    Dashboard vẫn mở; sửa Settings (vd. thư mục trọng số) rồi lưu để nạp lại.', flush=True)
    tun = None
    if not a.no_tunnel:
        url = None
        for name, starter in (('cloudflared', lambda: start_cloudflared(a.port, HERE.parent / 'work' / 'bin')),
                              ('localhost.run (ssh)', lambda: start_localhost_run(a.port))):
            print(f'Mở tunnel bằng {name} ...', flush=True)
            try:
                tun = starter()
                if tun is not None:
                    procs.append(tun)
            except Exception as e:
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
            print('=' * 60, flush=True)
            print(f'  LINK CÔNG KHAI: {url}', flush=True)
            print('=' * 60, flush=True)
        else:
            print(f'!!! Không mở được tunnel công khai. Server vẫn chạy ở http://127.0.0.1:{a.port}', flush=True)
            print('    Kaggle: kiểm tra Settings -> Internet = On. Hoặc chạy lại cell sau ít phút.', flush=True)
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
