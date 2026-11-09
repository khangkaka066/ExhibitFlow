# ExhibitFlow dashboard cho Vast.ai (1 GPU >= 24 GB). Build:  docker build -t <user>/exhibitflow docker/
# Image chỉ chứa code + thư viện + model nhỏ. Trọng số (HF) và model lớn (VLM ~6 GB) tải lần đầu chạy vào /workspace.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
