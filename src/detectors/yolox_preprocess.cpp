#include "exhibitflow/yolox_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace exhibitflow {

PreparedYoloXInput prepare_yolox_input(const BgrImageView& image, const DetectorOptions& options) {
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 ||
        image.row_stride_bytes < image.width * 3) {
        throw std::invalid_argument("invalid BGR image view");
    }
    if (options.input_width <= 0 || options.input_height <= 0) {
        throw std::invalid_argument("YOLOX input dimensions must be positive");
    }

    const double scale = std::min(
        static_cast<double>(options.input_width) / image.width,
        static_cast<double>(options.input_height) / image.height
    );
    const int resized_width = static_cast<int>(image.width * scale);
    const int resized_height = static_cast<int>(image.height * scale);
    if (resized_width <= 0 || resized_height <= 0) {
        throw std::invalid_argument("YOLOX letterbox dimensions are invalid");
    }

    const cv::Mat source(image.height, image.width, CV_8UC3,
        const_cast<std::uint8_t*>(image.data), static_cast<std::size_t>(image.row_stride_bytes));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    constexpr float kMean[] = {0.485F, 0.456F, 0.406F};
    constexpr float kStd[] = {0.229F, 0.224F, 0.225F};
    constexpr float kPadding = 114.0F / 255.0F;
    const std::size_t plane_size = static_cast<std::size_t>(options.input_width) * options.input_height;
    PreparedYoloXInput prepared;
    prepared.chw.resize(plane_size * 3U);
    prepared.transform.scale = scale;
    for (int channel = 0; channel < 3; ++channel) {
        std::fill(
            prepared.chw.begin() + static_cast<std::size_t>(channel) * plane_size,
            prepared.chw.begin() + static_cast<std::size_t>(channel + 1) * plane_size,
            (kPadding - kMean[channel]) / kStd[channel]
        );
    }

    for (int y = 0; y < resized_height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < resized_width; ++x) {
            const cv::Vec3b pixel = row[x];
            const std::size_t offset = static_cast<std::size_t>(y) * options.input_width + x;
            prepared.chw[offset] = (static_cast<float>(pixel[2]) / 255.0F - kMean[0]) / kStd[0];
            prepared.chw[plane_size + offset] = (static_cast<float>(pixel[1]) / 255.0F - kMean[1]) / kStd[1];
            prepared.chw[plane_size * 2U + offset] =
                (static_cast<float>(pixel[0]) / 255.0F - kMean[2]) / kStd[2];
        }
    }
    return prepared;
}

}  // namespace exhibitflow
