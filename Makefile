PY := .venv/bin/python
VIDEO := data/output.mp4
CONFIG := configs/bytetrack_dma_caviar.json
OUTPUT := outputs/real_tracker/output.jsonl

run-real-tracker:
	$(PY) tools/run_real_tracker.py --video $(VIDEO) --config $(CONFIG) --output $(OUTPUT) --overwrite