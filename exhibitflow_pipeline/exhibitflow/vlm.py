"""VLM: llama.cpp llama-server (GGUF, đa phương thức) + worker bất đồng bộ + xử lý đầu vào / đọc kết quả."""
from .common import *  # noqa: F401,F403

LABELS = ('normal', 'touching', 'holding', 'stealing', 'uncertain')


def load_image(path, max_pixels):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BICUBIC)
    return img




def parse_response(text):
    cleaned = re.sub(r'```(?:json)?', '', text)
    m = re.search(r'\{.*\}', cleaned, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    label = str(obj.get('label', '')).strip().lower()
    try:
        conf = float(obj.get('confidence'))
    except (TypeError, ValueError):
        conf = None
    return dict(label=label if label in LABELS else f'invalid:{label}', confidence=conf,
                contact_frames=obj.get('contact_frames'), evidence=obj.get('evidence'))




LLAMA_RELEASE = None                        # tag bản prebuilt (vd 'b12125'); None = bản mới nhất có file Ubuntu CUDA
LLAMA_RELEASE_FALLBACK = 'b12125'           # dùng khi không gọi được GitHub API (đã kiểm tra có file Ubuntu CUDA 12.8)
_LLAMA = {}


def _driver_cuda_version():
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


def _download(url, dest):
    tmp = Path(str(dest) + '.part')
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 22)
    tmp.replace(dest)


def _llama_release_assets():
    """(tag, [url llama-*-bin-ubuntu-cuda-12.8-x64.tar.gz, url cudart-llama-*-...]) lấy từ danh sách file của release.
    Không ghép tên từ tag: tag 'v0.9.0' chứa file 'llama-b12119-...'; một số release không có bản CUDA."""
    def pick(assets):
        names = {a['name']: a['browser_download_url'] for a in assets}
        main = [n for n in names if n.startswith('llama-') and n.endswith('-bin-ubuntu-cuda-12.8-x64.tar.gz')]
        rt = [n for n in names if n.startswith('cudart-llama-') and n.endswith('-bin-ubuntu-cuda-12.8-x64.tar.gz')]
        return [names[main[0]], names[rt[0]]] if main and rt else None

    api = 'https://api.github.com/repos/ggml-org/llama.cpp/releases'
    try:
        url = f'{api}/tags/{LLAMA_RELEASE}' if LLAMA_RELEASE else f'{api}?per_page=20'
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'exhibitflow'}), timeout=60) as r:
            rels = json.loads(r.read())
        for rel in ([rels] if isinstance(rels, dict) else rels):
            urls = pick(rel.get('assets', []))
            if urls:
                return rel['tag_name'], urls
        print(f'  release {LLAMA_RELEASE or "(20 bản mới nhất)"} không có file Ubuntu CUDA 12.8 -> dùng tag dự phòng {LLAMA_RELEASE_FALLBACK}')
    except Exception as e:
        print(f'  GitHub API lỗi ({type(e).__name__}: {e}) -> dùng tag dự phòng')
    t = LLAMA_RELEASE_FALLBACK
    base = f'https://github.com/ggml-org/llama.cpp/releases/download/{t}'
    return t, [f'{base}/llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz', f'{base}/cudart-llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz']


def _llama_prebuilt():
    """Bản prebuilt Ubuntu x64 CUDA 12.8 của llama.cpp + thư viện CUDA đi kèm (cudart / cuBLAS).
    Bản này có PTX cho sm_75 (T4): driver tự biên dịch lúc chạy lần đầu -> cần driver hỗ trợ CUDA >= 12.8."""
    import tarfile
    drv = _driver_cuda_version()
    if drv is None or drv < (12, 8):
        print(f'  driver CUDA {drv}: bản prebuilt (CUDA 12.8) cần driver >= 12.8 -> build từ source')
        return None
    tag, urls = _llama_release_assets()
    dirs = []
    for url in urls:
        name = url.rsplit('/', 1)[1][:-len('.tar.gz')]
        out = (settings.WORK / 'llama_bin') / name
        if not (out / '.done').exists():
            out.mkdir(parents=True, exist_ok=True)
            tgz = (settings.WORK / 'llama_bin') / f'{name}.tar.gz'
            print(f'  tải {name}.tar.gz ...')
            _download(url, tgz)
            with tarfile.open(tgz) as tf:
                try:
                    tf.extractall(out, filter='data')
                except TypeError:
                    tf.extractall(out)
            tgz.unlink()
            (out / '.done').touch()
        dirs.append(out)
    exe = next(dirs[0].rglob('llama-server'))
    exe.chmod(0o755)
    libs = sorted({str(p.parent) for d in dirs for p in d.rglob('*.so*')})
    env = {'LD_LIBRARY_PATH': ':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]),
           'CUDA_CACHE_MAXSIZE': str(4 << 30), 'CUDA_CACHE_PATH': str(settings.WORK / '.nv_cache')}
    test = subprocess.run([str(exe), '--list-devices'], env={**os.environ, **env}, capture_output=True, text=True, timeout=300)
    log = (test.stdout or '') + (test.stderr or '')
    if 'CUDA' not in log:
        print(f'  prebuilt {tag} không thấy GPU CUDA (mã {test.returncode}):\n{log[-2000:]}')
        return None
    print(f'  llama.cpp prebuilt {tag}: {exe}')
    return str(exe), env


def _llama_build():
    """Build llama-server từ source (CUDA, đúng kiến trúc GPU). Lỗi -> in đuôi log build."""
    exe = (settings.WORK / 'llama.cpp') / 'build' / 'bin' / 'llama-server'
    if exe.exists():
        return str(exe), {}
    env = dict(os.environ)
    nvcc = shutil.which('nvcc') or next((p for p in ('/usr/local/cuda/bin/nvcc',) if Path(p).exists()), None)
    if settings.N_GPU and not nvcc:
        raise RuntimeError('Không dùng được bản prebuilt (xem log ở trên) và máy không có nvcc (CUDA toolkit) để build llama.cpp.')
    if nvcc:
        env['PATH'] = str(Path(nvcc).parent) + ':' + env.get('PATH', '')
    if not ((settings.WORK / 'llama.cpp') / 'CMakeLists.txt').exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', str((settings.WORK / 'llama.cpp'))], check=True)
    cfg = ['cmake', '-S', str((settings.WORK / 'llama.cpp')), '-B', str((settings.WORK / 'llama.cpp') / 'build'), '-DCMAKE_BUILD_TYPE=Release',
           '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF']
    if settings.N_GPU:
        arch = ';'.join(sorted({''.join(map(str, torch.cuda.get_device_capability(i))) for i in range(settings.N_GPU)}))
        cfg += ['-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF', f'-DCMAKE_CUDA_ARCHITECTURES={arch}', f'-DCMAKE_CUDA_COMPILER={nvcc}']
    build = ['cmake', '--build', str((settings.WORK / 'llama.cpp') / 'build'), '--config', 'Release', '--target', 'llama-server',
             '-j', str(os.cpu_count() or 4)]
    log = settings.WORK / 'llama_build.log'
    log.write_text('')
    print(f'  build llama.cpp từ source — lần đầu ~10-20 phút (log: {log})')
    t0 = time.perf_counter()
    for cmd in (cfg, build):
        with open(log, 'a') as f:
            f.write('\n$ ' + ' '.join(cmd) + '\n')
            f.flush()
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'build llama.cpp lỗi ở bước {cmd[1]} (mã {r.returncode}). Đuôi log {log}:\n'
                               + log.read_text(errors='replace')[-4000:])
    print(f'  build xong sau {time.perf_counter() - t0:.0f}s: {exe}')
    return str(exe), {}


def ensure_llama_server():
    """(đường dẫn llama-server, env bổ sung). Thứ tự: có sẵn trong PATH -> prebuilt -> build từ source. Lỗi được nhớ
    để các cấu hình GGUF sau không thử lại (báo lỗi ngay)."""
    if 'err' in _LLAMA:
        raise RuntimeError(_LLAMA['err'])
    if 'exe' not in _LLAMA:
        try:
            exe = shutil.which('llama-server')
            found = (exe, {}) if exe else None
            if not found and settings.N_GPU:
                try:
                    found = _llama_prebuilt()
                except Exception as e:      # tải / giải nén lỗi -> build từ source
                    print(f'  prebuilt lỗi: {type(e).__name__}: {e} -> build từ source')
            _LLAMA['exe'] = found or _llama_build()
        except Exception as e:
            _LLAMA['err'] = f'{type(e).__name__}: {e}'
            raise
    return _LLAMA['exe']


def fetch_gguf(repo_id, quant_file, mmproj):
    """Tải file GGUF có tên chứa quant_file (vd 'Q4_K_M') + file mmproj (ưu tiên tên chứa `mmproj`, vd 'F16')."""
    from huggingface_hub import list_repo_files, hf_hub_download
    files = [f for f in list_repo_files(repo_id) if f.lower().endswith('.gguf')]
    norm = lambda x: x.lower().replace('-', '_')
    main = [f for f in files if 'mmproj' not in f.lower() and norm(quant_file) in norm(Path(f).name)]
    proj = [f for f in files if 'mmproj' in f.lower()]
    if not main:
        raise RuntimeError(f'{repo_id}: không có file GGUF chứa "{quant_file}". Có: {[f for f in files if "mmproj" not in f.lower()]}')
    if not proj:
        raise RuntimeError(f'{repo_id}: không có file mmproj (cần cho ảnh)')
    first = min(main, key=len)
    m = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', first)
    parts = [f for f in main if m and f.startswith(first[:m.start()])] if m else [first]   # GGUF chia nhiều phần
    paths = [hf_hub_download(repo_id, f) for f in sorted(parts)]
    # khớp nguyên cụm (không để 'F16' khớp nhầm 'mmproj-BF16': T4 không có BF16 -> đọc ảnh chậm hơn ~50%)
    tok = lambda f: re.split(r'[_.]', norm(Path(f).name))
    proj_f = next((f for f in proj if norm(mmproj) in tok(f)), None) if mmproj else None
    proj_f = proj_f or next((f for f in proj if norm(mmproj or '') in norm(Path(f).name)), proj[0])
    return paths[0], hf_hub_download(repo_id, proj_f), first, proj_f


@register('vlm', 'llama_cpp')
class LlamaCppVLM:
    """GGUF qua llama.cpp llama-server (tiến trình riêng, API /v1/chat/completions).
    Ảnh được resize giống hf_vlm (max_pixels) rồi gửi dạng base64; latency tách prefill (đọc ảnh + prompt) / generate."""

    def __init__(self, repo_id=None, quant_file='Q4_K_M', mmproj='F16', model_path=None, mmproj_path=None, device=GPU_B,
                 n_gpu_layers=99, ctx=16384, max_new_tokens=256, max_pixels=768 * 768, port=8089, startup_timeout=900,
                 extra_args=(), cache_prompt=False, flash_attn='on', **_):
        self.cache_prompt = bool(cache_prompt)
        exe, xenv = ensure_llama_server()
        if model_path:
            model, proj, mname, pname = str(model_path), str(mmproj_path), Path(model_path).name, Path(str(mmproj_path)).name
        else:
            model, proj, mname, pname = fetch_gguf(repo_id, quant_file, mmproj)
        self.model_id = f'{repo_id or Path(model).parent}/{mname} + {pname}'
        self.max_new_tokens, self.max_pixels, self.port = max_new_tokens, max_pixels, port
        env = {**os.environ, **xenv}
        vis = (settings.CUDA_VISIBLE_DEVICES_0 or ','.join(map(str, range(settings.N_GPU)))).split(',')
        dev = str(device)
        if dev.startswith('cuda:'):
            self.gpus = [vis[int(dev.split(':')[1])]]
        elif dev == 'auto' and settings.N_GPU:
            self.gpus = vis[:settings.N_GPU]
        else:
            self.gpus, n_gpu_layers = [], 0
        env['CUDA_VISIBLE_DEVICES'] = ','.join(self.gpus)
        self.devices = []            # VRAM đo bằng nvidia-smi (tiến trình riêng, torch không thấy)
        self.log_path = settings.WORK / f'llama_server_{port}.log'
        cmd = [exe, '-m', model, '--mmproj', proj, '-ngl', str(n_gpu_layers), '-c', str(ctx), '-np', '1',
               '--host', '127.0.0.1', '--port', str(port), *map(str, extra_args)]
        if flash_attn and not {'-fa', '--flash-attn'} & set(map(str, extra_args)):
            cmd += ['-fa', str(flash_attn)]
        self.log = open(self.log_path, 'w')
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f'llama-server dừng (mã {self.proc.returncode}):\n{self._log_tail()}')
            if time.perf_counter() - t0 > startup_timeout:
                self.close()
                raise RuntimeError(f'llama-server không sẵn sàng sau {startup_timeout}s:\n{self._log_tail()}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        self.server_info = self._server_info()
        print(f'    GGUF {mname} + {pname} | llama-server GPU {self.gpus or "cpu"} | ngl={n_gpu_layers} ctx={ctx} | '
              f'{self.server_info} | args: {" ".join(cmd[cmd.index("-np"):])}')

    def warmup(self):
        try:
            body = json.dumps(dict(messages=[{'role': 'user', 'content': 'Reply OK.'}], max_tokens=2, temperature=0.0,
                                   cache_prompt=False)).encode()
            req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions', data=body,
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=600).read()
        except Exception as e:
            print(f'    warmup VLM bỏ qua: {type(e).__name__}: {e}')

    def _server_info(self):
        """Trạng thái thật đọc từ log llama-server: flash attention (LLM / phần ảnh), n_batch, n_ubatch."""
        log = self._log_tail(10 ** 7)
        last = lambda pat: (re.findall(pat, log) or ['?'])[-1]
        llm_fa = last(r'flash_attn\s*=\s*(\w+)')
        if re.search(r'Flash Attention not supported', log):
            llm_fa = 'disabled (không hỗ trợ)'
        elif re.search(r': Flash Attention enabled', log):
            llm_fa = 'enabled'
        img_fa = last(r'flash attention is (enabled|disabled)')
        nb_, nub = last(r'n_batch\s*=\s*(\d+)'), last(r'n_ubatch\s*=\s*(\d+)')
        return f'FA LLM={llm_fa} ảnh={img_fa} | n_batch={nb_} n_ubatch={nub}'

    def _log_tail(self, n=3000):
        try:
            self.log.flush()
            return self.log_path.read_text(errors='replace')[-n:]
        except Exception:
            return ''

    def _gpu_mem_gb(self):
        if not self.gpus:
            return None
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                  '-i', ','.join(self.gpus)], capture_output=True, text=True, timeout=10).stdout
            return round(sum(float(x) for x in out.split()) / 1024, 2)
        except Exception:
            return None

    def infer(self, task):
        import io as _io
        t_prep = time.perf_counter()
        content = []
        for i, (path, t) in enumerate(zip(task['images'], task['times'])):
            buf = _io.BytesIO()
            load_image(path, self.max_pixels).save(buf, 'JPEG', quality=92)
            content.append({'type': 'text', 'text': f'Frame {i + 1} (t={t:.2f}s):'})
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()}})
        content.append({'type': 'text', 'text': task['prompt']})
        msgs = ([{'role': 'system', 'content': task['system']}] if task.get('system') else []) + [{'role': 'user', 'content': content}]
        # cache_prompt=False: không dùng lại KV cache của request trước (đoạn warmup trùng đoạn đầu -> prefill ~0 s, latency đo sai)
        body = json.dumps(dict(messages=msgs, max_tokens=self.max_new_tokens, temperature=0.0, seed=0,
                               cache_prompt=self.cache_prompt)).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'llama-server HTTP {e.code}: {e.read()[:500]!r}') from e
        gen_s = time.perf_counter() - t0
        usage, tm = resp.get('usage') or {}, resp.get('timings') or {}
        text = resp['choices'][0]['message'].get('content') or ''
        return text, int(usage.get('prompt_tokens') or 0), dict(
            n_output_tokens=int(usage.get('completion_tokens') or 0), prep_s=round(t0 - t_prep, 3), generate_s=round(gen_s, 3),
            prefill_s=round(tm['prompt_ms'] / 1000, 3) if 'prompt_ms' in tm else None,
            output_tok_per_s=round(tm['predicted_per_second'], 2) if tm.get('predicted_per_second') else None,
            max_pixels_used=self.max_pixels, oom_retries=0, peak_mem_gb=self._gpu_mem_gb())

    def close(self):
        p = getattr(self, 'proc', None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
        self.proc = None
        if getattr(self, 'log', None) is not None and not self.log.closed:
            self.log.close()

    def __del__(self):
        self.close()


@register('vlm', 'dry_run')
class DryRunVLM:
    def __init__(self, delay_s=0.0, label='touching', **_):
        self.delay_s, self.label, self.model_id, self.devices = delay_s, label, 'dry_run', []

    def infer(self, task):
        missing = [p for p in task['images'] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        time.sleep(self.delay_s)
        return (json.dumps(dict(label=self.label, confidence=0.5, contact_frames=[3], evidence='dry run')), 0,
                dict(n_output_tokens=0, generate_s=self.delay_s))


class VLMWorker(threading.Thread):
    """Thread riêng: nạp VLM (trong lúc các model khác đang nạp), rồi xử lý task theo hàng đợi.
    Kết quả gọi callback ngay -> AlarmBoard cập nhật báo động trong lúc video vẫn đang chạy."""

    def __init__(self, spec):
        super().__init__(daemon=True, name='vlm-worker')
        self.spec = spec
        self.q = queue.Queue()
        self.loaded = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.model = self.load_error = None
        self.load_s = None
        self.records = []
        self.results = {}

    def run(self):
        t0 = time.perf_counter()
        try:
            if settings.CUDA_VISIBLE_DEVICES_0 is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = settings.CUDA_VISIBLE_DEVICES_0
            dev = str(self.spec.get('device', ''))
            if dev.startswith('cuda:'):
                torch.cuda.set_device(dev)          # thread nạp VLM dùng đúng GPU đích
            self.model = build('vlm', self.spec)
        except Exception:
            self.load_error = traceback.format_exc()
            print('!!! VLM nạp lỗi:\n', self.load_error)
        finally:
            self.load_s = round(time.perf_counter() - t0, 2)
            self.loaded.set()
        dev_idx = sorted({int(str(d).split(':')[-1]) for d in getattr(self.model, 'devices', [])
                          if str(d).split(':')[-1].isdigit()})
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            self.gate.wait()
            task, callback = item
            t_start = time.perf_counter()
            rec = dict(task_id=task['task_id'], video_key=task['video_key'], segment=task['segment'],
                       n_images=len(task['images']), t_emit=task['t_emit'], t_start=t_start,
                       queue_wait_s=round(t_start - task['t_emit'], 3))
            try:
                if self.load_error:
                    raise RuntimeError('VLM không nạp được')
                for i in dev_idx:
                    reset_peak_mem(i)
                text, n_in, extra = self.model.infer(adapt_task(task, self.spec))
                parsed = parse_response(text)
                rec.update(status='ok', raw=text, n_input_tokens=n_in, parsed=parsed,
                           label=(parsed or {}).get('label', 'parse_error'), **extra)
                if dev_idx:
                    rec['peak_mem_gb'] = [peak_mem_gb(i) for i in dev_idx]
            except Exception as e:
                rec.update(status='error', error=f'{type(e).__name__}: {e}', label='error')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rec['t_done'] = time.perf_counter()
            rec['infer_s'] = round(rec['t_done'] - t_start, 3)
            rec['latency_s'] = round(rec['t_done'] - task['t_emit'], 3)
            self.records.append(rec)
            self.results[task['task_id']] = rec
            try:
                callback(task, rec)
            except Exception:
                traceback.print_exc()
            self.q.task_done()

    def submit(self, task, callback):
        self.q.put((task, callback))

    def pending(self):
        return self.q.unfinished_tasks

    def cancel_pending(self):
        """Bỏ các task còn xếp hàng (task đang chạy dở vẫn chạy nốt). Dùng khi người dùng dừng giữa chừng."""
        n = 0
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if item is None:                 # tín hiệu dừng worker: trả lại
                self.q.put(None)
                self.q.task_done()
                break
            self.q.task_done()
            n += 1
        return n

    def drain(self):
        self.q.join()

    def stop(self):
        self.q.put(None)
        self.join(timeout=60)
        if hasattr(self.model, 'close'):
            self.model.close()           # backend chạy tiến trình riêng (llama_cpp)
        self.model = None
        free_gpu()

# ---------------- Xử lý đầu vào theo cấu hình VLM: n_images / max_side / short_json ----------------
# Giống hệt benchmark (cùng cách lấy mẫu, resize INTER_AREA + JPEG q92, cùng prompt) để số đo benchmark còn đúng.
def _short_template():
    return settings.PROMPT_TEMPLATE.split('Answer JSON only:')[0] + 'Answer JSON only:\n{{"label": "...", "confidence": 0.0}}'


def _resized(path, side):
    p = Path(path)
    dst = p.parent.with_name(f'{p.parent.name}_{side}') / p.name
    if not dst.exists():
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        if max(h, w) > side:
            r = side / max(h, w)
            img = cv2.resize(img, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(dst)


def adapt_task(task, spec):
    """n_images: lấy đều N ảnh trong các ảnh đã xuất | max_side: thu nhỏ cạnh dài | short_json: prompt chỉ hỏi label + confidence."""
    n, side, short = spec.get('n_images'), spec.get('max_side'), spec.get('short_json')
    if not (n or side or short):
        return task
    imgs, times = list(task['images']), list(task['times'])
    idx = list(range(len(imgs)))
    if n and len(imgs) > n:
        idx = sorted(set(np.linspace(0, len(imgs) - 1, n).round().astype(int).tolist()))
        imgs, times = [imgs[i] for i in idx], [times[i] for i in idx]
    if side:
        imgs = [_resized(p, side) for p in imgs]
    prompt = task['prompt']
    if n or short:
        m = re.search(r'about ([\d.]+)s apart', prompt)
        step = float(np.mean(np.diff(idx))) if len(idx) > 1 else 1.0
        dt = float(m.group(1)) * step if m else (float(np.mean(np.diff(times))) if len(times) > 1 else 0.0)
        prompt = (_short_template() if short else settings.PROMPT_TEMPLATE).format(n=len(imgs), dt=dt, tid=task['tid'])
    return {**task, 'images': imgs, 'times': times, 'prompt': prompt}
