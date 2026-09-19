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

The default config `configs/bytetrack_dma_caviar.json` uses the YOLOX-S MOT17
detector weights. The required model files are:

```text
pretrained/bytetrack_s_mot17.pth.tar   # YOLOX-S detector + ByteTrack base checkpoint
pretrained/ltc_motion_mot17.pth        # LTC motion predictor
pretrained/mot17_sbs_S50.pth           # FastReID appearance model
pretrained/dma_gbm_mot17.gbm           # DMA LightGBM fusion weights
```

The detector is intentionally kept backend-neutral in the config. The same
YOLOX-S weights can later be exported to ONNX for macOS/portable inference or
TensorRT FP16 for NVIDIA GPUs, while the tracking, LTC, ReID, and DMA stages
remain unchanged.

For the current C++ deployment work, a validated portable detector artifact is
available at:

```text
models/yolox_s_mot17_640.onnx
```

It uses input `1x3x640x640` and produces raw YOLOX output `1x8400x6`; decode,
confidence filtering, and NMS remain in the C++ pipeline. On Windows with an
NVIDIA GPU, build a machine-specific TensorRT FP16 engine from this ONNX file:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_tensorrt_engine.ps1
```

The script requires `trtexec.exe` from NVIDIA TensorRT on `PATH` and creates
`models/yolox_s_mot17_640_fp16.engine`. Build the engine on the target NVIDIA
machine because TensorRT engines depend on the CUDA/TensorRT runtime and GPU
architecture.

Keep ONNX as the portable source artifact. TensorRT engines are tied to the
TensorRT/CUDA/GPU environment, so build one on each target NVIDIA machine and
do not replace the ONNX artifact with a single prebuilt engine.

## C++ Detector CLI

`exhibitflow_detector` is the first native stage of the production path:

```text
OpenCV BGR frame -> C++ YOLOX letterbox/normalization -> ONNX Runtime or TensorRT
               -> C++ decode + confidence filter + NMS -> detections JSONL
```

It writes the existing `DetectionFrame` JSONL contract directly, one row for
every decoded frame. It deliberately does **not** call the tracker yet; that
makes detector accuracy and latency measurable before ByteTrack/LTC/ReID/DMA
is migrated.

The baseline CMake build remains usable without inference dependencies. The
detector target is created only when CMake finds OpenCV and at least one native
inference backend.

For ONNX Runtime, install an ONNX Runtime C++ release together with OpenCV,
then configure its root explicitly when it is not already discoverable:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DEXHIBITFLOW_ONNXRUNTIME_ROOT=C:\sdk\onnxruntime
cmake --build build --config Release --target exhibitflow_detector
```

For TensorRT on an NVIDIA machine, install CUDA, TensorRT, and OpenCV; build
the engine on that machine first; then provide the TensorRT root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_tensorrt_engine.ps1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DEXHIBITFLOW_TENSORRT_ROOT=C:\TensorRT
cmake --build build --config Release --target exhibitflow_detector
```

`CUDA_PATH` must point to the CUDA Toolkit for the TensorRT build. The CMake
diagnostics state which backend is enabled. Copy the matching dynamic libraries
beside the executable or make their `bin` directories available on `PATH`.

Run the detector with one backend at a time:

```powershell
build\bin\Release\exhibitflow_detector `
  --backend onnxruntime `
  --model models\yolox_s_mot17_640.onnx `
  --video data\caviar\videos\Browse_WhileWaiting1_f620_80f.mp4 `
  --output outputs\detector\bww1_onnx.jsonl `
  --conf 0.01 --nms 0.45

build\bin\Release\exhibitflow_detector `
  --backend tensorrt `
  --model models\yolox_s_mot17_640_fp16.engine `
  --video data\caviar\videos\Browse_WhileWaiting1_f620_80f.mp4 `
  --output outputs\detector\bww1_trt.jsonl `
  --conf 0.01 --nms 0.45
```

The TensorRT backend uses the engine's actual input/output tensor names and
requires a TensorRT 8.5+ engine API. It does not assume a particular NVIDIA
GPU; a `.engine` remains specific to its CUDA/TensorRT/GPU environment.

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
