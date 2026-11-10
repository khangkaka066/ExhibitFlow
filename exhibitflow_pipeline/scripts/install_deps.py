#!/usr/bin/env python
"""Cài phụ thuộc cho Kaggle / Colab / máy có GPU. Dùng:  python scripts/install_deps.py [--cpu]

Giữ nguyên torch có sẵn. Riêng onnxruntime: rtmlib kéo theo bản CPU `onnxruntime`, bản này che mất `onnxruntime-gpu`
(2 gói dùng chung thư mục) -> RTMW chạy CPU (~700 ms / frame). Script gỡ bản CPU rồi cài lại bản GPU.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cpu = '--cpu' in sys.argv


def pip(*args, check=True):
    """uv (nhanh hơn pip nhiều: tải + cài song song) nếu có / cài được, không thì pip."""
    if _UV:
        return subprocess.run([*_UV, 'pip', *args, '--python', sys.executable], check=check)
    return subprocess.run([sys.executable, '-m', 'pip', *args], check=check)


def _find_uv():
    if '--no-uv' in sys.argv:
        return None
    exe = shutil.which('uv')
    if not exe:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
            exe = shutil.which('uv') or str(Path(sys.executable).parent / 'uv')
        except Exception:
            return None
    return [exe] if Path(exe).exists() else None


_UV = _find_uv()
print('[deps] trình cài:', ' '.join(_UV) + ' pip' if _UV else 'pip', flush=True)
GPU_PKG = 'onnxruntime' if cpu else 'onnxruntime-gpu[cuda,cudnn]'
# 1 lần cài duy nhất: gồm cả onnxruntime-gpu để trình cài không kéo thêm bản CPU (rtmlib phụ thuộc `onnxruntime`)
pip('install', '-q', '-r', str(ROOT / 'requirements.txt'), GPU_PKG)
if not cpu:
    # rtmlib vẫn có thể đã cài kèm bản CPU: 2 gói dùng chung thư mục, bản CPU che mất bản GPU -> gỡ, rồi khôi phục file của bản GPU
    from importlib import metadata
    if any((d.metadata['Name'] or '').lower() == 'onnxruntime' for d in metadata.distributions()):
        if _UV:
            pip('uninstall', '-q', 'onnxruntime', check=False)
            pip('install', '-q', '--reinstall', '--no-deps', 'onnxruntime-gpu')
        else:
            pip('uninstall', '-y', '-q', 'onnxruntime', check=False)
            pip('install', '-q', '--force-reinstall', '--no-deps', 'onnxruntime-gpu')
print('Xong. Kiểm tra: python main.py --check')
