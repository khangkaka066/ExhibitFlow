"""Import chuẩn, tiện ích ảnh / GPU, registry model và tìm trọng số. Các module khác `from .common import *`."""
import base64
import copy
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from . import settings
from .settings import (CFG, EXPORT, GPU_A, GPU_B, MODELS, MASK_REGIONS, RT, RUN, SEGMENTATION_CFG,  # noqa: F401
                       EXCLUDE_ARTIFACTS, MANUAL_ARTIFACTS, SEG_REF_FRAMES)

try:
    import torch
except ImportError:          # cho phép import khi chỉ kiểm tra cấu hình
    torch = None

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ---------------- Hình học ----------------
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def iou_matrix(A, B):
    A, B = np.asarray(A, np.float64).reshape(-1, 4), np.asarray(B, np.float64).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def video_meta(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nw, nh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if max_frames:
        n = min(n, max_frames)
    w = CFG['width']
    return dict(fps=fps, n=n, NW=nw, NH=nh, W=w, H=int(round(nh * w / nw)), S=nw / w)


def mask_regions(img):
    if not MASK_REGIONS:
        return img
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in MASK_REGIONS:
        m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 255
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)


def put_label(img, text, org, color, scale=0.6, thick=1, fg=(255, 255, 255)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), max(int(org[1]), th + bl + 2)
    cv2.rectangle(img, (x, y - th - bl - 2), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - bl), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def mask_contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.float32) for c in cs if len(c) >= 3]


def contact_sheet(tiles, cols=4, tile_w=384, pad=4):
    if not tiles:
        return None
    rs = [cv2.resize(t, (tile_w, int(t.shape[0] * tile_w / t.shape[1]))) for t in tiles]
    th = max(r.shape[0] for r in rs)
    sheet = np.full((math.ceil(len(rs) / cols) * (th + pad) + pad, cols * (tile_w + pad) + pad, 3), 30, np.uint8)
    for i, r in enumerate(rs):
        y, x = pad + (i // cols) * (th + pad), pad + (i % cols) * (tile_w + pad)
        sheet[y:y + r.shape[0], x:x + tile_w] = r
    return sheet


# ---------------- GPU ----------------
def reset_peak_mem(i):
    try:
        torch.cuda.reset_peak_memory_stats(i)
    except Exception:        # device chưa khởi tạo CUDA context / không có CUDA
        pass


def peak_mem_gb(i):
    try:
        return round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
    except Exception:
        return 0.0


def free_gpu():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:      # CUDA context đã hỏng: không che mất lỗi gốc
            print(f'  (free_gpu bỏ qua: {str(e).splitlines()[0]})')


def _find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_browser_mp4(tmp, final):
    """mp4v (OpenCV) không phát được trong trình duyệt -> H.264; lỗi thì giữ mp4v."""
    ffmpeg, ok = _find_ffmpeg(), False
    if ffmpeg:
        r = subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-preset', 'veryfast', '-crf', '26', '-movflags', '+faststart', str(final)],
                           capture_output=True, text=True)
        ok = r.returncode == 0 and final.exists()
    if ok:
        tmp.unlink()
    else:
        tmp.replace(final)
    return final


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def video_fingerprint(path):
    """Dấu vân tay theo NỘI DUNG video (kích thước + hash đầu / giữa / cuối file), không phụ thuộc tên hay thời gian sửa.
    Cache segmentation và map khoá theo đây: tải lên lại cùng video (thời gian sửa đổi) vẫn dùng lại cache, không sinh thêm bản mới."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.md5(str(size).encode())
    with open(p, 'rb') as f:
        for off in (0, max(0, size // 2 - 524288), max(0, size - 1048576)):
            f.seek(off)
            h.update(f.read(1048576))
    return f'{size:x}{h.hexdigest()[:10]}'


def spec_tag(spec):
    return hashlib.md5(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]


# ---------------- Registry: REGISTRY[loại][backend] = class / hàm tạo model ----------------
REGISTRY = defaultdict(dict)


def register(kind, name):
    def deco(obj):
        REGISTRY[kind][name] = obj
        return obj
    return deco


def build(kind, spec):
    """Tạo model theo spec; ghi lại thời gian nạp + warmup."""
    spec = dict(spec)
    backend = spec.pop('backend')
    if backend not in REGISTRY[kind]:
        raise KeyError(f"{kind} backend '{backend}' chưa đăng ký. Có: {sorted(REGISTRY[kind])}")
    t0 = time.perf_counter()
    obj = REGISTRY[kind][backend](**spec)
    t1 = time.perf_counter()
    if hasattr(obj, 'warmup'):
        obj.warmup()
    obj.kind, obj.backend = kind, backend
    obj.load_s, obj.warmup_s = round(t1 - t0, 2), round(time.perf_counter() - t1, 2)
    print(f'  [{kind}] {backend}: nạp {obj.load_s}s, warmup {obj.warmup_s}s')
    return obj


def describe_models(models):
    def short(s):
        s = dict(s)
        b = s.pop('backend')
        main = s.get('weights') or s.get('model_id') or s.get('dino') or s.get('mode') or s.get('exp') or s.get('quant_file') or ''
        return f'{b}({Path(str(main)).name})' if main else b
    return ' | '.join(f'{k}={short(v)}' for k, v in models.items())


# ---------------- Tìm trọng số ----------------
WALK_MAX_DEPTH = 6
WALK_MAX_ENTRIES = 2000
WALK_SKIP_DIRS = {'.git', '__MACOSX', 'frames', 'images', 'masks', 'depth', 'node_modules', '__pycache__'}


def walk_dir(root, max_depth=WALK_MAX_DEPTH):
    """Duyệt theo chiều rộng, giới hạn độ sâu / số mục (thư mục Kaggle input có thể rất lớn)."""
    root = Path(root)
    q = deque([(root, 0)])
    while q:
        d, depth = q.popleft()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError:
            continue
        dirs, files = [], []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    dirs.append(e.name)
                elif e.is_file():
                    files.append(e.name)
            except OSError:
                pass
        yield d, dirs, files
        if depth >= max_depth or len(entries) > WALK_MAX_ENTRIES:
            continue
        for name in sorted(dirs):
            if name not in WALK_SKIP_DIRS:
                q.append((d / name, depth + 1))


def find_in(match, root, max_depth=WALK_MAX_DEPTH, dirs_too=False):
    test = (lambda p: p.name == match) if isinstance(match, str) else match
    for d, dirs, files in walk_dir(root, max_depth):
        for name in files + (dirs if dirs_too else []):
            if test(d / name):
                return d / name
    return None


def _safe_extract(zf, dest: Path):
    dest = dest.resolve()
    for member in zf.infolist():
        if not str((dest / member.filename).resolve()).startswith(str(dest)):
            raise RuntimeError(f'Đường dẫn không an toàn trong zip: {member.filename}')
    zf.extractall(dest)


def repack_extracted_torch_checkpoint(folder: Path) -> Path:
    """Đóng gói lại checkpoint PyTorch mà Kaggle bung thành thư mục (data.pkl + data/...)."""
    data_pkls = [p for p in folder.rglob('data.pkl') if p.is_file()]
    if len(data_pkls) != 1:
        raise RuntimeError(f'Checkpoint đã bung phải có đúng một data.pkl; thấy {len(data_pkls)} trong {folder}')
    settings.CACHE.mkdir(parents=True, exist_ok=True)
    destination = settings.CACHE / f'repacked_{folder.name}.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(p for p in folder.rglob('*') if p.is_file()):
            archive.write(path, arcname=path.relative_to(folder).as_posix())
    return destination


def resolve_ckpt_hit(hit: Path) -> Path:
    if hit.is_file():
        return hit
    if hit.is_dir():
        if any(p.is_file() for p in hit.rglob('data.pkl')):
            return repack_extracted_torch_checkpoint(hit)
        files = sorted(p for p in hit.rglob('*') if p.is_file())
        if len(files) == 1:
            return files[0]
    raise RuntimeError(f'Checkpoint không rõ cấu trúc: {hit}')


def locate_weight(name, allow_dir=False):
    """Tìm file theo tên trong settings.WEIGHT_ROOTS (kể cả bên trong file .zip ngay dưới mỗi thư mục gốc)."""
    for root in settings.WEIGHT_ROOTS:
        if not root.exists():
            continue
        subs = sorted(root.glob('*')) if str(root) == '/kaggle/input' else [root]
        for sub in subs:
            if not sub.is_dir():
                continue
            for bundle in sub.glob('*.zip'):
                try:
                    with zipfile.ZipFile(bundle) as zf:
                        if any(Path(n).name == name for n in zf.namelist()):
                            extract_to = settings.CACHE / 'bundle_extracted'
                            if not (extract_to / name).exists():
                                _safe_extract(zf, extract_to)
                            hit = next(extract_to.rglob(name), None)
                            if hit:
                                return resolve_ckpt_hit(hit) if allow_dir else hit
                except zipfile.BadZipFile:
                    continue
            hit = find_in(name, sub, dirs_too=allow_dir)
            if hit:
                return resolve_ckpt_hit(hit) if allow_dir else hit
    return None


def resolve_weight(name, required=False, allow_dir=False):
    """Đường dẫn có sẵn -> tìm theo tên file trong WEIGHT_ROOTS -> (không bắt buộc) trả lại tên để thư viện tự tải."""
    if name is None:
        return None
    p = Path(str(name))
    if p.exists():
        return resolve_ckpt_hit(p) if p.is_dir() else p
    hit = locate_weight(p.name, allow_dir=allow_dir)
    if hit is not None:
        return hit
    if required:
        roots = ', '.join(str(r) for r in settings.WEIGHT_ROOTS if r.exists()) or '(không có thư mục nào tồn tại)'
        raise RuntimeError(f'Không tìm thấy trọng số "{name}". Đã tìm trong: {roots}. '
                           'Đặt file vào thư mục weights/ hoặc dùng --weights-dir <thư mục>.')
    return str(name)


def ensure_tracker_repo():
    """Repo ByteTrack-DMA-LTC-Motion-Tracker (có package `yolox` + tracker). Tìm sẵn, không có thì clone. Thêm vào sys.path."""
    cands = [settings.TRACKER_REPO] if settings.TRACKER_REPO else []
    cands += [settings.PROJECT_DIR.parent / 'external' / 'ByteTrack-DMA-LTC-Motion-Tracker',
              settings.PROJECT_DIR / 'external' / 'ByteTrack-DMA-LTC-Motion-Tracker',
              settings.WORK / 'ByteTrack-DMA-LTC-Motion-Tracker']
    repo = next((c for c in cands if c and (c / 'yolox').is_dir()), None)
    if repo is None:
        repo = settings.WORK / 'ByteTrack-DMA-LTC-Motion-Tracker'
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f'  clone {settings.TRACKER_REPO_URL} -> {repo}')
        subprocess.run(['git', 'clone', '--depth', '1', settings.TRACKER_REPO_URL, str(repo)], check=True)
    settings.TRACKER_REPO = repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo
