"""Chạy web ExhibitFlow và in sẵn link để bấm vào.

    python serve.py            (cổng mặc định 8765)
    python serve.py 9000       (đổi cổng)
"""
import functools
import http.server
import os
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
root = os.path.dirname(os.path.abspath(__file__))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)

with http.server.ThreadingHTTPServer(('127.0.0.1', port), handler) as srv:
    print(f'\n  ExhibitFlow web đang chạy. Mở link này:\n\n      http://127.0.0.1:{port}/index.html\n\n  (Ctrl+C để tắt)\n', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('Đã tắt.')
