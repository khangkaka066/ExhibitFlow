#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: tools/run_tensorrt.sh VIDEO OUTPUT.jsonl [detector options]

Builds a TensorRT FP16 engine when needed, builds the C++ detector, then runs
YOLOX inference on VIDEO. Run this script on Linux with an NVIDIA GPU.

Environment variables:
  ONNX_MODEL             Source model (default: models/yolox_s_mot17_640.onnx)
  TENSORRT_ENGINE        Engine output (default: models/yolox_s_mot17_640_fp16.engine)
  TENSORRT_ROOT          TensorRT installation root (auto-detected when possible)
  CUDA_HOME              CUDA Toolkit root (default: /usr/local/cuda)
  TENSORRT_BUILD_DIR     CMake build directory (default: build-tensorrt)
  TENSORRT_WORKSPACE_MIB Engine-builder workspace in MiB (default: 4096)
  REBUILD_ENGINE         Set to 1 to rebuild an existing engine

Example:
  tools/run_tensorrt.sh input.mp4 outputs/input.detections.jsonl \
    --conf 0.25 --nms 0.45
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
video_path=$1
output_path=$2
shift 2

onnx_model=${ONNX_MODEL:-"$repo_dir/models/yolox_s_mot17_640.onnx"}
engine_path=${TENSORRT_ENGINE:-"$repo_dir/models/yolox_s_mot17_640_fp16.engine"}
build_dir=${TENSORRT_BUILD_DIR:-"$repo_dir/build-tensorrt"}
workspace_mib=${TENSORRT_WORKSPACE_MIB:-4096}

if [[ $(uname -s) != "Linux" ]]; then
    echo "error: TensorRT requires a supported NVIDIA Linux/Windows machine" >&2
    exit 1
fi
for command_name in cmake trtexec nvidia-smi; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "error: $command_name was not found in PATH" >&2
        exit 1
    fi
done
if ! nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi cannot access an NVIDIA GPU" >&2
    exit 1
fi
if [[ ! -f $video_path ]]; then
    echo "error: video not found: $video_path" >&2
    exit 1
fi
if [[ ! -f $onnx_model ]]; then
    echo "error: ONNX model not found: $onnx_model" >&2
    exit 1
fi
if [[ -e $output_path ]]; then
    echo "error: output already exists: $output_path" >&2
    exit 1
fi
if [[ ! $workspace_mib =~ ^[1-9][0-9]*$ ]]; then
    echo "error: TENSORRT_WORKSPACE_MIB must be a positive integer" >&2
    exit 1
fi

export CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
export CUDA_PATH=${CUDA_PATH:-$CUDA_HOME}

tensorrt_root=${TENSORRT_ROOT:-}
if [[ -z $tensorrt_root ]]; then
    trtexec_path=$(command -v trtexec)
    if command -v realpath >/dev/null 2>&1; then
        trtexec_path=$(realpath "$trtexec_path")
    fi
    candidate_root=$(cd -- "$(dirname -- "$trtexec_path")/.." && pwd)
    if [[ -f $candidate_root/include/NvInfer.h ]]; then
        tensorrt_root=$candidate_root
    fi
fi

mkdir -p -- "$(dirname -- "$engine_path")" "$(dirname -- "$output_path")"

if [[ ! -f $engine_path || ${REBUILD_ENGINE:-0} == "1" ]]; then
    echo "Building TensorRT engine: $engine_path"
    trtexec \
        --onnx="$onnx_model" \
        --saveEngine="$engine_path" \
        --fp16 \
        --memPoolSize="workspace:${workspace_mib}" \
        --skipInference
else
    echo "Using existing TensorRT engine: $engine_path"
fi

cmake_args=(
    -S "$repo_dir"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DEXHIBITFLOW_BUILD_DETECTOR=ON
    -DEXHIBITFLOW_REQUIRE_TENSORRT=ON
)
if [[ -n $tensorrt_root ]]; then
    cmake_args+=("-DEXHIBITFLOW_TENSORRT_ROOT=$tensorrt_root")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config Release --parallel --target exhibitflow_detector

detector_bin="$build_dir/bin/exhibitflow_detector"
if [[ ! -x $detector_bin ]]; then
    detector_bin="$build_dir/bin/Release/exhibitflow_detector"
fi
if [[ ! -x $detector_bin ]]; then
    echo "error: exhibitflow_detector was not produced by the build" >&2
    exit 1
fi

"$detector_bin" \
    --backend tensorrt \
    --model "$engine_path" \
    --video "$video_path" \
    --output "$output_path" \
    "$@"

echo "TensorRT detections written to: $output_path"
