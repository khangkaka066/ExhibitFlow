#!/usr/bin/env bash
# Khởi động dashboard: tải trọng số từ HuggingFace (lần đầu) rồi chạy server.
set -e
cd /opt/ExhibitFlow/exhibitflow_pipeline
mkdir -p "$DATA_ROOT"/{weights,work,hf,dashboard_data}

# tải 4 file trọng số + file khác trong repo HF vào /workspace/weights (bỏ qua nếu đã đủ)
python - <<PY
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
need = ['best_ckpt.pth', 'osnet_x0_25.pth', 'ltc_motion_mot17.pth', 'dma_gbm_mot17.gbm']
dst = Path(os.environ['DATA_ROOT']) / 'weights'
def have(): return all(any(dst.rglob(n)) for n in need)
if not have():
    print('[entrypoint] tải trọng số từ', os.environ['WEIGHTS_REPO'], flush=True)
    try:
        snapshot_download(os.environ['WEIGHTS_REPO'], local_dir=str(dst), token=os.environ.get('HF_TOKEN') or None)
    except Exception as e:
        sys.exit(f'Không tải được trọng số ({type(e).__name__}: {e}). Repo private thì đặt HF_TOKEN.')
    if not have():
        sys.exit('Repo HF thiếu: ' + ', '.join(n for n in need if not any(dst.rglob(n))))
PY

export EF_WEIGHTS_DIR="$DATA_ROOT/weights"
echo "[entrypoint] mode=$MODE port=$PORT weights=$EF_WEIGHTS_DIR"
exec python -m dashboard.server --host 0.0.0.0 --port "$PORT" --mode "$MODE" --data-dir "$DATA_ROOT/dashboard_data" "$@"
