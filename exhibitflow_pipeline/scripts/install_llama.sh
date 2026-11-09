#!/usr/bin/env bash
# Đảm bảo có llama-server (VLM). Thứ tự:
#   1. đã có trong PATH                         -> xong
#   2. glibc >= 2.38 (Ubuntu 24.04)             -> dùng bản dựng sẵn, code tự tải ở lần chạy đầu
#   3. glibc cũ (Ubuntu 22.04) + có nvcc        -> build từ source (~10-20 phút), cài vào $PREFIX
#   4. còn lại                                  -> báo lỗi (exit 2); chạy MODE=tracking_seg không cần VLM
set -euo pipefail
PREFIX="${PREFIX:-/usr/local}"

if command -v llama-server >/dev/null 2>&1; then
    echo "[llama] llama-server đã có: $(command -v llama-server)"; exit 0
fi

glibc="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
if [ "$(printf '%s\n2.38\n' "$glibc" | sort -V | head -1)" = "2.38" ]; then
    echo "[llama] glibc $glibc >= 2.38: dùng bản llama.cpp dựng sẵn (tự tải ở lần chạy đầu)"; exit 0
fi

NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC" ]; then
    echo "[llama] !!! glibc $glibc < 2.38 và không có nvcc: không có llama-server -> VLM không chạy được."
    echo "[llama]     Chọn image CUDA '-devel' (có nvcc) hoặc Ubuntu 24.04, hoặc chạy: make MODE=tracking_seg"
    exit 2
fi
export PATH="$(dirname "$NVCC"):$PATH"

command -v cmake >/dev/null 2>&1 || { echo "[llama] cài cmake"; python3 -m pip install -q cmake; }
command -v git >/dev/null 2>&1 || { echo "[llama] thiếu git"; exit 2; }

# kiến trúc GPU đang có (vd 8.9 -> 89); không đọc được thì build cho các đời phổ biến
arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u | paste -sd';' || true)"
[ -z "$arch" ] && arch="75;80;86;89;90"
echo "[llama] build llama.cpp (CUDA arch: $arch) - lần đầu ~10-20 phút"

src="$(mktemp -d)/llama.cpp"
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="$arch" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$src/build" --config Release --target llama-server -j"$(nproc)"

mkdir -p "$PREFIX/bin" "$PREFIX/lib"
cp "$src/build/bin/llama-server" "$PREFIX/bin/"
cp -a "$src"/build/bin/*.so* "$PREFIX/lib/" 2>/dev/null || true
ldconfig 2>/dev/null || true
rm -rf "$(dirname "$src")"
echo "[llama] xong: $PREFIX/bin/llama-server"
