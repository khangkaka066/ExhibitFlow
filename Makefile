PY := .venv/bin/python
VIDEO := data/MV_1.mp4
CONFIG := configs/bytetrack_dma_caviar.json
OUTPUT := outputs/real_tracker/output.jsonl

#------build-----------
BUILD := build/bin/exhibitflow_detector
BACKEND := onnxruntime
MODEL_ONNXRUNTIME := models/yolox_s_mot17_640.onnx
DEVICE := cuda
#----------------------

run-real-tracker:
	$(PY) tools/run_real_tracker.py --video $(VIDEO) --config $(CONFIG) --output $(OUTPUT) --overwrite

exhibitflow-detector:
	$(BUILD) --backend $(BACKEND) --model $(MODEL_ONNXRUNTIME) --video $(VIDEO) --output $(OUTPUT) --device $(DEVICE)