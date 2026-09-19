#pragma once

#include <vector>

#include "exhibitflow/detector.hpp"
#include "exhibitflow/yolox_postprocess.hpp"

namespace exhibitflow {

struct PreparedYoloXInput {
    std::vector<float> chw;
    LetterboxTransform transform;
};

PreparedYoloXInput prepare_yolox_input(const BgrImageView& image, const DetectorOptions& options);

}  // namespace exhibitflow
