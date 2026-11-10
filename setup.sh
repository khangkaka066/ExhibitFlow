#!/usr/bin/env bash
# ExhibitFlow - SETUP cho Vast.ai (chỉ chuẩn bị môi trường, KHÔNG chạy dashboard trừ khi START=1).
#
# Dùng làm PROVISIONING_SCRIPT của template vastai/pytorch:
#   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/khangkaka066/ExhibitFlow/Hung/setup.sh
# hoặc chạy tay:  HF_TOKEN=hf_xxx bash setup.sh
#
# Biến (tuỳ chọn): HF_TOKEN  BRANCH=Hung  REPO_DIR=/workspace/ExhibitFlow  MODE=full  PORT=8000
#                  START=1 -> chạy dashboard nền sau khi setup xong (make run)
# Idempotent: chạy lại an toàn, các bước đã xong được bỏ qua (Makefile đánh dấu ở .make/stamp).
set -Eeuo pipefail
trap 'echo "[setup] !!! LỖI ở dòng $LINENO (lệnh: $BASH_COMMAND)" >&2' ERR

REPO_URL="${REPO_URL:-https://github.com/khangkaka066/ExhibitFlow.git}"
BRANCH="${BRANCH:-Hung}"
REPO_DIR="${REPO_DIR:-/workspace/ExhibitFlow}"   # /workspace là nơi lưu bền trên Vast; đường dẫn không được có dấu cách (Makefile)
START="${START:-0}"

# vastai/pytorch dùng venv /venv/main (torch + CUDA nằm trong đó)
if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    . /venv/main/bin/activate
fi
PY="$(command -v python3)"
echo "[setup] python=$PY branch=$BRANCH dir=$REPO_DIR"

command -v git >/dev/null  || { apt-get update -qq && apt-get install -y -qq git; }
command -v make >/dev/null || { apt-get update -qq && apt-get install -y -qq make; }

# 1. code
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" checkout -q "$BRANCH"
    git -C "$REPO_DIR" reset -q --hard "origin/$BRANCH"
else
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 2. deps + externals + weights (HF_TOKEN lấy từ môi trường, không ghi vào file) + llama-server + model bản đồ
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup] cảnh báo: HF_TOKEN chưa đặt - nếu repo trọng số HuggingFace là private thì bước tải weights sẽ lỗi"
fi
export HF_TOKEN
make setup PY="$PY"

echo "[setup] XONG. Chạy dashboard:  cd $REPO_DIR && make run   (hoặc START=1)"

# 3. runtime (tuỳ chọn)
if [ "$START" = "1" ]; then
    make run PY="$PY" MODE="${MODE:-full}" PORT="${PORT:-8000}"
fi
