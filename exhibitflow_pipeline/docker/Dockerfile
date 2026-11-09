# ExhibitFlow dashboard cho Vast.ai (1 GPU >= 24 GB). Build:  docker build -t <user>/exhibitflow docker/
# Image chỉ chứa code + thư viện + model nhỏ. Trọng số (HF) và model lớn (VLM ~6 GB) tải lần đầu chạy vào /workspace.
# bản devel (có nvcc) để build llama.cpp ngay trong image: bản prebuilt của llama.cpp cần glibc >= 2.38, image này là Ubuntu 22.04 (2.35)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential cmake ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- llama-server (VLM) build từ source cho T4 / A100 / 3090 / 4090 / H100; vlm.py thấy `llama-server` trong PATH là dùng luôn ----
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build /tmp/llama.cpp/build --config Release --target llama-server -j"$(nproc)" \
    && cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/ \
    && (cp -a /tmp/llama.cpp/build/bin/*.so* /usr/local/lib/ 2>/dev/null || true) && ldconfig \
    && rm -rf /tmp/llama.cpp

# ---- code (đổi REPO_REF để buộc clone lại bản mới) ----
ARG BRANCH=Hung
ARG REPO_REF=0
WORKDIR /opt/ExhibitFlow
RUN git clone --depth 1 --branch ${BRANCH} https://github.com/khangkaka066/ExhibitFlow.git . \
    && mkdir -p external \
    && git clone --depth 1 https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git external/ByteTrack-DMA-LTC-Motion-Tracker \
    && git clone --depth 1 https://github.com/khangkaka066/multicam-2d-floorplan-tracker.git external/multicam-2d-floorplan-tracker

WORKDIR /opt/ExhibitFlow/exhibitflow_pipeline
# fastapi/uvicorn/psutil cho dashboard; install_deps.py xử lý xung đột onnxruntime / onnxruntime-gpu
RUN pip install fastapi "uvicorn[standard]" python-multipart psutil
RUN python scripts/install_deps.py
# huggingface_hub / httpx2 mới cần brotli >= 1.2 (bản 1.1 làm tải model HF lỗi 'process() takes no keyword arguments')
RUN pip install -U "brotli>=1.2"

# ---- model nhỏ nấu sẵn vào image (Depth-Anything V2 + SegFormer cho map 2D, ~200 MB) ----
RUN python ../external/multicam-2d-floorplan-tracker/scripts/download_structural_models.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Biến cấu hình (đặt khi tạo instance trên Vast):
#   HF_TOKEN  token HuggingFace (cần nếu repo trọng số private)   MODE  tracking | tracking_seg | full
ENV WEIGHTS_REPO=hungnguyen190204/model_exhibitflow MODE=full PORT=8000 \
    DATA_ROOT=/workspace \
    EXHIBITFLOW_WORK=/workspace/work HF_HOME=/workspace/hf \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
