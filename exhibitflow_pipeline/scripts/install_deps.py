#!/usr/bin/env python
"""Cài phụ thuộc cho Kaggle / Colab / máy có GPU. Dùng:  python scripts/install_deps.py [--cpu]

Giữ nguyên torch có sẵn. Riêng onnxruntime: rtmlib kéo theo bản CPU `onnxruntime`, bản này che mất `onnxruntime-gpu`
(2 gói dùng chung thư mục) -> RTMW chạy CPU (~700 ms / frame). Script gỡ bản CPU rồi cài lại bản GPU.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cpu = '--cpu' in sys.argv


def pip(*args, check=True):
    return subprocess.run([sys.executable, '-m', 'pip', *args], check=check)


pip('install', '-q', '-r', str(ROOT / 'requirements.txt'))
pip('uninstall', '-y', '-q', 'onnxruntime', check=False)
if cpu:
    pip('install', '-q', 'onnxruntime')
else:
    pip('install', '-q', 'onnxruntime-gpu[cuda,cudnn]')
    pip('install', '-q', '--force-reinstall', '--no-deps', 'onnxruntime-gpu')
print('Xong. Kiểm tra: python main.py --check')
