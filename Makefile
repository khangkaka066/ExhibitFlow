PY := .venv/bin/python
VIDEO := data/MV_2.mp4
CONFIG := configs/bytetrack_dma_caviar.json
OUTPUT := outputs/real_tracker/output.jsonl

#------build-----------
BUILD := build/bin/exhibitflow_detector
BACKEND := onnxruntime
MODEL_ONNXRUNTIME := models/yolox_s_mot17_640.onnx
DEVICE := cuda
#----------------------

# ---- Train Pose ----------------------------------------------------------
STG_NF_DIR := external/PoseLift/STG-NF
LOG_RUNS := ../../../.venv/bin/tensorboard --logdir runs
DATASET_POSE := PoseLift
# Newest checkpoint by mtime (path relative to STG_NF_DIR); override with CHECKPOINT_POSE=...
CHECKPOINT_POSE ?= $(shell cd $(STG_NF_DIR) && ls -1t data/exp_dir/$(DATASET_POSE)/*/*_checkpoint.pth.tar 2>/dev/null | head -n 1)
EPOCH_POSE := 100

# ---- AlphaPose (shares .venv) ----------------------------------------------
ALPHAPOSE_DIR := external/AlphaPose
CUDA_HOME ?= /usr/local/cuda-11.8
TORCH_CUDA_ARCH_LIST ?= 8.9

run-real-tracker:
	$(PY) tools/run_real_tracker.py --video $(VIDEO) --config $(CONFIG) --output $(OUTPUT) --overwrite

exhibitflow-detector:
	$(BUILD) --backend $(BACKEND) --model $(MODEL_ONNXRUNTIME) --video $(VIDEO) --output $(OUTPUT) --device $(DEVICE)

train-pose:
	cd $(STG_NF_DIR) && "$(CURDIR)/$(PY)" train_eval.py --dataset $(DATASET_POSE) --device cuda:0 \
	--num_workers 4 --seed 42 --epochs $(EPOCH_POSE)

eval-pose:
	@test -n "$(CHECKPOINT_POSE)" || { echo "No checkpoint found, run 'make train-pose' first"; exit 1; }
	cd $(STG_NF_DIR) && "$(CURDIR)/$(PY)" train_eval.py --dataset $(DATASET_POSE) --device cuda:0 \
	--num_workers 4 --checkpoint $(CHECKPOINT_POSE)

# setup.py needs pkg_resources (gone in setuptools>=81), so build with an
# ephemeral old setuptools layered on top of .venv instead of downgrading it.
alphapose-setup:
	test -d $(ALPHAPOSE_DIR) || git clone --depth 1 https://github.com/MVIG-SJTU/AlphaPose.git $(ALPHAPOSE_DIR)
	uv pip install --python $(PY) --no-build-isolation \
	halpecocotools munkres natsort tensorboardx terminaltables "timm==0.1.20" "numpy==1.26.4"
	cd $(ALPHAPOSE_DIR) && CUDA_HOME=$(CUDA_HOME) PATH=$(CUDA_HOME)/bin:$$PATH TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	uv run --no-project --python "$(CURDIR)/$(PY)" --with "setuptools<70" --with pytest-runner \
	python setup.py build_ext --inplace
	echo "$(CURDIR)/$(ALPHAPOSE_DIR)" > "$$($(PY) -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/alphapose.pth"
	$(PY) -c "import alphapose, detector.nms, alphapose.models.layers.dcn; print('AlphaPose OK')"

tracking-pose:
	$(PY) tools/run_tracking_pose.py --video $(VIDEO) --tracker-config $(CONFIG)

pose-anomaly:
	$(PY) tools/run_pose_anomaly.py --video $(VIDEO) --tracker-config $(CONFIG)

tensorboard-poselift:
	cd $(STG_NF_DIR) && $(LOG_RUNS)