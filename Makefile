# ExhibitFlow dashboard - cài đặt + chạy bằng 1 lệnh (Vast.ai / máy Linux có GPU NVIDIA).
#
#   git clone -b Hung https://github.com/khangkaka066/ExhibitFlow.git && cd ExhibitFlow
#   HF_TOKEN=hf_xxx make            # cài tất cả, chạy nền, in link truy cập
#   HF_TOKEN=hf_xxx make vast       # như trên cho Vast.ai: cache HF ở /workspace, bật tải nhanh, chạy lại nếu đang chạy
#
# Biến (đặt trước lệnh make):  MODE=full|tracking_seg|tracking   PORT=8000   LOCAL=1 (máy ít VRAM)
#                              HF_TOKEN=...   HF_REPO=user/repo   PY=python3
# Kết quả phân tích lưu ở ./dashboard_data/runs/ (đổi bằng DATA=/đường/dẫn).
# Lệnh khác:  make logs | status | stop | restart | setup | clean

SHELL        := /bin/bash
.DEFAULT_GOAL := run

ROOT   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ifneq ($(word 2,$(ROOT)),)
$(error Đường dẫn "$(ROOT)" có dấu cách, make không xử lý được. Clone repo vào thư mục không có dấu cách)
endif
PIPE   := $(ROOT)/exhibitflow_pipeline
EXT    := $(ROOT)/external
WORK   := $(ROOT)/.make
STAMP  := $(WORK)/stamp
PIDF   := $(WORK)/server.pid
LOG    := $(WORK)/server.log
# Kết quả (runs/, uploads/, maps/, snapshots/): thư mục hiện rõ ở gốc repo, mở được bằng trình duyệt file của Vast
DATA   ?= $(ROOT)/dashboard_data

PY      ?= python3
PORT    ?= 8000
MODE    ?= full
LOCAL   ?=
HF_REPO ?= hungnguyen190204/model_exhibitflow

export EF_WEIGHTS_DIR := $(PIPE)/weights
export EXHIBITFLOW_WORK := $(WORK)/work
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

TRACKER_URL := https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git
MAP_URL     := https://github.com/khangkaka066/multicam-2d-floorplan-tracker.git

.PHONY: vast run setup deps externals weights llama mapmodels check url logs status stop restart clean help

help:
	@sed -n "2,8p" Makefile | sed "s/^# \{0,1\}//"

## ---------------- cài đặt (mỗi bước chỉ chạy 1 lần, đánh dấu bằng file trong .make/stamp) ----------------
setup: check deps externals weights llama mapmodels

check:
	@$(PY) -c "import torch,sys; ok=torch.cuda.is_available(); print(\"[check] torch\",torch.__version__,\"| GPU:\",torch.cuda.get_device_name(0) if ok else \"KHÔNG THẤY\"); sys.exit(0 if ok else 1)" \
	  || { echo "[check] !!! cần Python có torch + CUDA (chọn image PyTorch trên Vast)"; exit 1; }

deps: $(STAMP)/deps
$(STAMP)/deps:
	@mkdir -p $(STAMP)
	$(PY) -m pip install fastapi "uvicorn[standard]" python-multipart psutil
	cd $(PIPE) && $(PY) scripts/install_deps.py
	$(PY) -m pip install -U "brotli>=1.2"
	@touch $@

externals: $(EXT)/ByteTrack-DMA-LTC-Motion-Tracker $(EXT)/multicam-2d-floorplan-tracker
$(EXT)/ByteTrack-DMA-LTC-Motion-Tracker:
	git clone --depth 1 $(TRACKER_URL) $@
$(EXT)/multicam-2d-floorplan-tracker:
	git clone --depth 1 $(MAP_URL) $@

weights: deps
	$(PY) $(PIPE)/scripts/fetch_weights.py $(PIPE)/weights $(HF_REPO)

llama: $(STAMP)/llama
$(STAMP)/llama:
	@mkdir -p $(STAMP)
	@bash $(PIPE)/scripts/install_llama.sh && touch $@ || echo "[llama] bỏ qua - chế độ full sẽ không chạy được VLM"

mapmodels: $(STAMP)/mapmodels | externals
$(STAMP)/mapmodels:
	@mkdir -p $(STAMP)
	cd $(EXT)/multicam-2d-floorplan-tracker && $(PY) scripts/download_structural_models.py
	@touch $@

## ---------------- chạy ----------------
# Vast.ai: sau git clone chỉ cần `HF_TOKEN=hf_xxx make vast`. Cache HF đặt ở /workspace (bền, không đầy đĩa container), bật tải nhanh,
# cài (nếu chưa), dừng server cũ nếu có rồi chạy lại. Đặt sẵn HF_HOME / HF_XET_HIGH_PERFORMANCE thì giữ giá trị của bạn.
vast: export HF_HOME ?= /workspace/.hf_home
vast: export HF_XET_HIGH_PERFORMANCE ?= 1
vast: restart
run: setup
	@mkdir -p $(DATA)
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then \
	  echo "[run] dashboard đang chạy (pid $$(cat $(PIDF))). make restart để chạy lại."; \
	else \
	  ( cd $(PIPE) && exec $(PY) server/web_server.py --host 0.0.0.0 --port $(PORT) --mode $(MODE) $(if $(LOCAL),--local) \
	      --data-dir $(DATA) ) > $(LOG) 2>&1 & echo $$! > $(PIDF); \
	  echo "[run] đã chạy nền (pid $$(cat $(PIDF))), mode=$(MODE), port=$(PORT). Log: $(LOG)"; \
	  echo "[run] kết quả lưu ở: $(DATA)/runs/"; \
	fi
	@$(MAKE) --no-print-directory url

url:
	@ip="$${PUBLIC_IPADDR:-}"; ext="$${VAST_TCP_PORT_$(PORT):-}"; \
	echo ""; \
	if [ -n "$$ip" ] && [ -n "$$ext" ]; then echo ">>> Mở dashboard: http://$$ip:$$ext"; \
	else echo ">>> Port $(PORT) chưa được mở ra ngoài. Trên Vast: thêm -p $(PORT):$(PORT) vào Docker options của template, rồi mở IP:port hiện trên trang Instances."; fi; \
	echo "    Lần đầu nạp model mất vài phút (VLM ~6 GB). Theo dõi: make logs"

logs:
	@tail -n 80 -f $(LOG)

status:
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then echo "đang chạy (pid $$(cat $(PIDF)))"; \
	  curl -s localhost:$(PORT)/api/state | $(PY) -c "import sys,json; d=json.load(sys.stdin); print(\"status:\",d[\"status\"],\"| model:\",d[\"models\"][\"state\"],d[\"models\"].get(\"msg\",\"\"))" || echo "(server chưa sẵn sàng)"; \
	else echo "không chạy"; fi

stop:
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then kill $$(cat $(PIDF)) && echo "[stop] đã dừng"; else echo "[stop] không có tiến trình"; fi; rm -f $(PIDF)

restart: stop run


clean:
	rm -rf $(STAMP)
	@echo "[clean] đã xoá đánh dấu cài đặt (lần make sau sẽ cài lại các bước)"
