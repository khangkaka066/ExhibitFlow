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

## Test

```bash
ctest --test-dir build --output-on-failure
```

The current backend is `mock`. It creates deterministic sample track IDs for
each detection and is meant to validate integration, file handling, and the
data contract before the real MOT backend is added.
