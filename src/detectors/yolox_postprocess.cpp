#include "exhibitflow/yolox_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace exhibitflow {
namespace {

struct Candidate {
    Detection detection;
    double x2 = 0.0;
    double y2 = 0.0;
};

double intersection_over_union(const Candidate& left, const Candidate& right) {
    const double x1 = std::max(left.detection.bbox.x, right.detection.bbox.x);
    const double y1 = std::max(left.detection.bbox.y, right.detection.bbox.y);
    const double x2 = std::min(left.x2, right.x2);
    const double y2 = std::min(left.y2, right.y2);
    const double intersection_width = std::max(0.0, x2 - x1);
    const double intersection_height = std::max(0.0, y2 - y1);
    const double intersection = intersection_width * intersection_height;
    const double left_area = left.detection.bbox.width * left.detection.bbox.height;
    const double right_area = right.detection.bbox.width * right.detection.bbox.height;
    const double union_area = left_area + right_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

}  // namespace

std::vector<Detection> decode_yolox_person_output(
    const float* output,
    std::size_t prediction_count,
    int input_width,
    int input_height,
    const LetterboxTransform& transform,
    ImageSize original_image_size,
    float confidence_threshold,
    float nms_threshold
) {
    if (output == nullptr) {
        throw std::invalid_argument("YOLOX output must not be null");
    }
    if (input_width <= 0 || input_height <= 0 || original_image_size.width <= 0 ||
        original_image_size.height <= 0 || !std::isfinite(transform.scale) || transform.scale <= 0.0) {
        throw std::invalid_argument("invalid YOLOX image dimensions or letterbox scale");
    }
    if (confidence_threshold < 0.0F || confidence_threshold > 1.0F || nms_threshold < 0.0F ||
        nms_threshold > 1.0F) {
        throw std::invalid_argument("YOLOX confidence and NMS thresholds must be in [0, 1]");
    }

    constexpr int kStrides[] = {8, 16, 32};
    std::size_t expected_predictions = 0;
    for (const int stride : kStrides) {
        if (input_width % stride != 0 || input_height % stride != 0) {
            throw std::invalid_argument("YOLOX input dimensions must be divisible by 32");
        }
        expected_predictions += static_cast<std::size_t>(input_width / stride) *
            static_cast<std::size_t>(input_height / stride);
    }
    if (prediction_count != expected_predictions) {
        throw std::invalid_argument("unexpected YOLOX prediction count");
    }

    std::vector<Candidate> candidates;
    candidates.reserve(prediction_count / 8U);
    std::size_t index = 0;
    for (const int stride : kStrides) {
        const int grid_width = input_width / stride;
        const int grid_height = input_height / stride;
        for (int grid_y = 0; grid_y < grid_height; ++grid_y) {
            for (int grid_x = 0; grid_x < grid_width; ++grid_x, ++index) {
                const float* row = output + index * 6U;
                const double score = static_cast<double>(row[4]) * static_cast<double>(row[5]);
                if (!std::isfinite(score) || score < confidence_threshold) {
                    continue;
                }

                const double center_x = (static_cast<double>(row[0]) + grid_x) * stride / transform.scale;
                const double center_y = (static_cast<double>(row[1]) + grid_y) * stride / transform.scale;
                const double width = std::exp(static_cast<double>(row[2])) * stride / transform.scale;
                const double height = std::exp(static_cast<double>(row[3])) * stride / transform.scale;
                if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width) ||
                    !std::isfinite(height) || width <= 0.0 || height <= 0.0) {
                    continue;
                }

                const double x1 = std::clamp(center_x - width / 2.0, 0.0,
                    static_cast<double>(original_image_size.width));
                const double y1 = std::clamp(center_y - height / 2.0, 0.0,
                    static_cast<double>(original_image_size.height));
                const double x2 = std::clamp(center_x + width / 2.0, 0.0,
                    static_cast<double>(original_image_size.width));
                const double y2 = std::clamp(center_y + height / 2.0, 0.0,
                    static_cast<double>(original_image_size.height));
                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }

                Candidate candidate;
                candidate.detection.bbox = {x1, y1, x2 - x1, y2 - y1};
                candidate.detection.score = score;
                candidate.x2 = x2;
                candidate.y2 = y2;
                candidates.push_back(candidate);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        return left.detection.score > right.detection.score;
    });

    std::vector<Detection> detections;
    detections.reserve(candidates.size());
    std::vector<bool> removed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        detections.push_back(candidates[i].detection);
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (!removed[j] && intersection_over_union(candidates[i], candidates[j]) > nms_threshold) {
                removed[j] = true;
            }
        }
    }
    return detections;
}

}  // namespace exhibitflow
