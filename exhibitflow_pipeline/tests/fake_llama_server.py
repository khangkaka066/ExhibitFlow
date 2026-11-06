"""Server giả lập llama-server (chỉ /health + /v1/chat/completions) để kiểm thử mà không cần GPU / model thật.

Chạy như llama-server: python fake_llama_server.py -m a.gguf --mmproj b.gguf --port 8097 ...
Mỗi câu trả lời ghi số ảnh nhận được vào `evidence` để test kiểm tra n_images / max_side.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index('--port') + 1])
T0 = time.time()
LAST = [T0]
IDLE_EXIT_S = 20      # tự thoát khi không ai gọi: trên Windows terminate() chỉ giết .bat, không giết tiến trình con
assert '--mmproj' in args and '-m' in args, args


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        LAST[0] = time.time()
        self.send_response(200 if time.time() - T0 > 1.0 else 503)      # giả lập đang nạp model
        self.end_headers()
        self.wfile.write(b'{}')

    def do_POST(self):
        LAST[0] = time.time()
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        content = body['messages'][-1]['content']
        images = [c for c in content if isinstance(c, dict) and c.get('type') == 'image_url'] if isinstance(content, list) else []
        assert body['temperature'] == 0.0 and body.get('cache_prompt') is False
        if images:
            assert all(c['image_url']['url'].startswith('data:image/jpeg;base64,') for c in images)
            out = json.dumps(dict(label='touching', confidence=0.8, evidence=f'{len(images)} images'))
        else:
            out = 'OK'
        resp = dict(choices=[dict(message=dict(role='assistant', content=out))],
                    usage=dict(prompt_tokens=1234, completion_tokens=30),
                    timings=dict(prompt_ms=850.0, predicted_ms=600.0, predicted_per_second=50.0))
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(data)


def _watchdog():
    while True:
        time.sleep(2)
        if time.time() - LAST[0] > IDLE_EXIT_S:
            os._exit(0)


threading.Thread(target=_watchdog, daemon=True).start()
print('fake llama-server on', port, flush=True)
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
