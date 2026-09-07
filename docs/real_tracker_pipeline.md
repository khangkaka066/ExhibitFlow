# Real Video Tracker Pipeline

ExhibitFlow keeps the C++ contract-oriented CLI and adds a Python runner for
the real ByteTrack model repo. The runner lets the project process CAVIAR
videos without changing the downstream `tracks.jsonl` schema.

```text
CAVIAR .mpg video
  -> ByteTrack model repo tools/demo_track.py
  -> MOT-style result txt
  -> tools/convert_bytetrack_results.py
  -> ExhibitFlow tracks JSONL
```

The current runner uses the model repo's `demo_track.py` path. It runs the real
detector and ByteTrack tracker. DMA/LTC-specific checkpoints can be wired in as
the next backend once those weights or demo entry points are available.

## Local Layout

The model repo is expected at:

```text
external/ByteTrack-DMA-LTC-Motion-Tracker
```

The CAVIAR subset is expected at:

```text
data/caviar/videos
data/caviar/annotations
```

Both folders are ignored by Git because they are local datasets and external
model code.

## Install Model Dependencies

From the model repo:

```bash
cd external/ByteTrack-DMA-LTC-Motion-Tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py develop
python tools/download_bytetrack_weights.py --name mot17_x
```

On Apple Silicon, run CPU mode first. GPU/CUDA-specific extensions may require
a Linux CUDA machine.

Check the environment from the ExhibitFlow repo:

```bash
python3 tools/check_real_tracker_env.py
```

## Run A CAVIAR Video

From the ExhibitFlow repo:

```bash
.venv/bin/python tools/make_video_clip.py \
  --input data/caviar/videos/Browse_WhileWaiting1.mpg \
  --output data/caviar/videos/Browse_WhileWaiting1_80f.mp4 \
  --frames 80 \
  --overwrite
```

```bash
.venv/bin/python tools/run_real_tracker.py \
  --video data/caviar/videos/Browse_WhileWaiting1_80f.mp4 \
  --config configs/bytetrack_dma_caviar.json \
  --output outputs/real_tracker/Browse_WhileWaiting1_80f.tracks.jsonl \
  --download-weights \
  --overwrite
```

The output JSONL uses the same frame context, `track_id`, `bbox`,
`point_image`, and `score` fields documented in `docs/data_contract_v0.md`.

## Convert An Existing ByteTrack Result

If `demo_track.py` has already produced a MOT-style result txt:

```bash
python3 tools/convert_bytetrack_results.py \
  --mot-txt path/to/result.txt \
  --video data/caviar/videos/Browse_WhileWaiting1.mpg \
  --output outputs/real_tracker/Browse_WhileWaiting1.tracks.jsonl \
  --sequence-id Browse_WhileWaiting1 \
  --camera-id cam_01 \
  --overwrite
```

The converter writes one JSON object per video frame and keeps empty frames as
`"tracks":[]`, which is useful for dwell-time and journey analytics.
