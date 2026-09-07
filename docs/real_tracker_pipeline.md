# Real Video Tracker Pipeline

ExhibitFlow keeps the C++ contract-oriented CLI and adds a Python runner for
real video tracking with the ByteTrack-DMA-LTC model repo. The runner processes
CAVIAR videos and exports the same downstream `tracks.jsonl` schema used by the
mock C++ CLI.

```text
CAVIAR video
  -> YOLOX detector checkpoint
  -> ByteTrack tracker with LTC motion prediction
  -> FastReID appearance embeddings
  -> DMA GBM motion/appearance fusion
  -> MOT-style result txt
  -> tools/convert_bytetrack_results.py
  -> ExhibitFlow tracks JSONL
```

The default config `configs/bytetrack_dma_caviar.json` uses the MOT17 weights
from the Drive folder:

```text
pretrained/bytetrack_x_mot17.pth.tar   # detector + ByteTrack base checkpoint
pretrained/ltc_motion_mot17.pth        # LTC motion predictor
pretrained/mot17_sbs_S50.pth           # FastReID appearance model
pretrained/dma_gbm_mot17.gbm           # DMA LightGBM fusion weights
```

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

## Check The Environment

From the ExhibitFlow repo:

```bash
.venv/bin/python tools/check_real_tracker_env.py \
  --video data/caviar/videos/Browse_WhileWaiting1_f620_80f.mp4
```

The checker verifies the model repo, video, ByteTrack checkpoint, LTC weight,
FastReID config/weight, DMA GBM weight, and the Python packages needed by the
current DMA/LTC backend.

## Make A Short CAVIAR Clip

Full CAVIAR videos are slow on CPU. For quick local checks, create a short clip:

```bash
.venv/bin/python tools/make_video_clip.py \
  --input data/caviar/videos/Browse_WhileWaiting1.mpg \
  --output data/caviar/videos/Browse_WhileWaiting1_f620_80f.mp4 \
  --start-frame 620 \
  --frames 80 \
  --overwrite
```

## Run DMA/LTC Tracking

```bash
.venv/bin/python tools/run_real_tracker.py \
  --video data/caviar/videos/Browse_WhileWaiting1_f620_80f.mp4 \
  --config configs/bytetrack_dma_caviar.json \
  --output outputs/real_tracker/Browse_WhileWaiting1_f620_80f.dma_ltc.tracks.jsonl \
  --overwrite
```

The runner also saves intermediate artifacts under `outputs/real_tracker/intermediate/`:

```text
<sequence>.mot.txt          # MOT-style tracker rows
<sequence>.annotated.mp4    # rendered video with track boxes and IDs
```

## Run With Realtime Display

Add `--show` to display the annotated video while tracking runs:

```bash
.venv/bin/python tools/run_real_tracker.py \
  --video data/caviar/videos/Browse_WhileWaiting1_f620_80f.mp4 \
  --config configs/bytetrack_dma_caviar.json \
  --output outputs/real_tracker/Browse_WhileWaiting1_f620_80f.dma_ltc.tracks.jsonl \
  --show \
  --overwrite
```

Press `q` or `Esc` in the OpenCV window to stop early. The CAVIAR videos are
384x288, so the default config shows them at 2x scale.

## Convert An Existing MOT Result

If you already have a MOT-style result txt:

```bash
.venv/bin/python tools/convert_bytetrack_results.py \
  --mot-txt path/to/result.txt \
  --video data/caviar/videos/Browse_WhileWaiting1.mpg \
  --output outputs/real_tracker/Browse_WhileWaiting1.tracks.jsonl \
  --sequence-id Browse_WhileWaiting1 \
  --camera-id cam_01 \
  --overwrite
```

The converter writes one JSON object per video frame and keeps empty frames as
`"tracks":[]`, which is useful for dwell-time and journey analytics.
