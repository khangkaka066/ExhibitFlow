#pragma once

#include <cstddef>
#include <vector>

#include "exhibitflow/types.hpp"

namespace exhibitflow {

struct LetterboxTransform {
    double scale = 1.0;
};

// Decodes the raw YOLOX head output exported by tools/export_onnx.py
// (one row per anchor: tx, ty, tw, th, objectness, person_probability).
// The exported MOT17 detector has one class: person.
std::vector<Detection> decode_yolox_person_output(
    const float* output,
    std::size_t prediction_count,
    int input_width,
    int input_height,
    const LetterboxTransform& transform,
    ImageSize original_image_size,
    float confidence_threshold,
    float nms_threshold
);

}  // namespace exhibitflow
