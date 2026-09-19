#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "exhibitflow/types.hpp"

namespace exhibitflow {

struct BgrImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int row_stride_bytes = 0;
};

struct DetectorOptions {
    int input_width = 640;
    int input_height = 640;
    float confidence_threshold = 0.01F;
    float nms_threshold = 0.45F;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<Detection> detect(const BgrImageView& image) = 0;
};

std::unique_ptr<IDetector> create_onnxruntime_yolox_detector(
    const std::string& model_path,
    const DetectorOptions& options
);

std::unique_ptr<IDetector> create_tensorrt_yolox_detector(
    const std::string& engine_path,
    const DetectorOptions& options
);

}  // namespace exhibitflow
