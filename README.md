# ExhibitFlow Tracker Skeleton

This repository contains the C++ foundation for the ExhibitFlow tracking component.
The first prototype exposes a stable tracker interface and a CLI that reads
detection JSONL, applies a deterministic mock tracker, and writes track-history
JSONL that downstream mapping and analytics code can consume.

## Build

Requirements:

- Linux or macOS development machine
- CMake 3.20+
- C++17 compiler

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Run

```bash
./build/bin/exhibitflow_tracker --help
./build/bin/exhibitflow_tracker --version
./build/bin/exhibitflow_tracker \
  --input examples/detections.jsonl \
  --config configs/mock.json \
  --output build/tracks.jsonl
```

The CLI writes logs and errors to stderr. Output JSONL is written only after the
whole input has been parsed and processed successfully.

## Real Video Pipeline

The project can also run the real ByteTrack-DMA-LTC model repo on CAVIAR video.
The default real config enables the MOT17 detector checkpoint, LTC motion
checkpoint, FastReID appearance checkpoint, and DMA GBM fusion weights, then
converts the MOT-style tracker output into ExhibitFlow JSONL:

```bash
.venv/bin/python tools/run_real_tracker.py \
  --video data/caviar/videos/Browse_WhileWaiting1_f620_80f.mp4 \
  --config configs/bytetrack_dma_caviar.json \
  --output outputs/real_tracker/Browse_WhileWaiting1_f620_80f.dma_ltc.tracks.jsonl \
  --overwrite
```

Use `--show` to display the annotated video while tracking runs. The annotated
MP4 and intermediate MOT rows are saved under `outputs/real_tracker/intermediate/`.

See [docs/real_tracker_pipeline.md](docs/real_tracker_pipeline.md) for local
folder layout, required weights, environment checks, and conversion details.

## Test

```bash
ctest --test-dir build --output-on-failure
```

The C++ CLI backend is `mock`. It creates deterministic sample track IDs for
each detection and validates integration, file handling, and the data contract.
The Python real-video wrapper is the bridge to the ByteTrack-DMA-LTC model repo.
