"""Chạy dashboard + mở tunnel công khai (Cloudflare quick tunnel, không cần tài khoản) -> in ra link cho mọi người.

    python -m dashboard.launch --port 8000 [--demo]
"""
import argparse
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get_cloudflared(dst_dir):
    exe = dst_dir / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    if exe.exists():
        return str(exe)
    arch = 'arm64' if platform.machine().lower() in ('aarch64', 'arm64') else 'amd64'
    name = {'nt': 'windows-amd64.exe', 'posix': f'linux-{arch}'}[os.name]
    url = f'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-{name}'
    print('Tải cloudflared:', url)
    dst_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, exe)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--no-tunnel', action='store_true')
    a = ap.parse_args()
    cmd = [sys.executable, '-m', 'dashboard.server', '--port', str(a.port)] + (['--demo'] if a.demo else [])
    if a.data_dir:
        cmd += ['--data-dir', a.data_dir]
    srv = subprocess.Popen(cmd, cwd=HERE.parent)
    for _ in range(60):                       # đợi server lên
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/state', timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    print(f'Dashboard nội bộ: http://127.0.0.1:{a.port}')
    tun = None
    if not a.no_tunnel:
        exe = get_cloudflared(HERE.parent / 'work' / 'bin')
        tun = subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{a.port}'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def pump():
            shown = False
            for line in tun.stdout:
                m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                if m and not shown:
                    shown = True
                    print('\n' + '=' * 60 + f'\n  LINK CÔNG KHAI: {m.group(0)}\n' + '=' * 60 + '\n', flush=True)
        threading.Thread(target=pump, daemon=True).start()
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
