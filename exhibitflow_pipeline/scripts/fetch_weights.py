#!/usr/bin/env python
"""Tải trọng số ExhibitFlow từ HuggingFace (bỏ qua nếu đã đủ).   python scripts/fetch_weights.py <thư_mục_đích> <repo_id>

Repo private: đặt biến môi trường HF_TOKEN.
"""
import os
import sys
from pathlib import Path

NEED = ['best_ckpt.pth', 'osnet_x0_25.pth', 'ltc_motion_mot17.pth', 'dma_gbm_mot17.gbm']


def main():
    dst, repo = Path(sys.argv[1]), sys.argv[2]
    dst.mkdir(parents=True, exist_ok=True)
    missing = lambda: [n for n in NEED if not any(dst.rglob(n))]
    if not missing():
        print(f'[weights] đã đủ 4 file trong {dst}')
        return
    from huggingface_hub import snapshot_download
    print(f'[weights] tải {repo} -> {dst}', flush=True)
    try:
        snapshot_download(repo, local_dir=str(dst), token=os.environ.get('HF_TOKEN') or None)
    except Exception as e:
        sys.exit(f'[weights] Không tải được ({type(e).__name__}: {e}). Repo private thì chạy: HF_TOKEN=hf_xxx make')
    if missing():
        sys.exit('[weights] Repo HF thiếu: ' + ', '.join(missing()))
    print('[weights] xong')


if __name__ == '__main__':
    main()
