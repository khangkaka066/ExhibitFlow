#include "exhibitflow/dma_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace exhibitflow {
namespace {

constexpr float kNeutralCosineDistance = 0.5F;
constexpr double kMinPositive = 1e-12;

struct Corners {
    double x1;
    double y1;
    double x2;
    double y2;
};

struct Cholesky4 {
    std::array<double, 16> lower{};
    bool valid = false;
};

Corners to_corners(const BoundingBox& box) {
    return {box.x, box.y, box.x + box.width, box.y + box.height};
}

std::array<double, 4> to_xyah(const BoundingBox& box) {
    const double width = std::max(box.width, kMinPositive);
    const double height = std::max(box.height, kMinPositive);
    return {
        box.x + box.width / 2.0,
        box.y + box.height / 2.0,
        width / height,
        height,
    };
}

double iou(const BoundingBox& track_box, const BoundingBox& detection_box) {
    const Corners a = to_corners(track_box);
    const Corners b = to_corners(detection_box);
    const double ix1 = std::max(a.x1, b.x1);
    const double iy1 = std::max(a.y1, b.y1);
    const double ix2 = std::min(a.x2, b.x2);
    const double iy2 = std::min(a.y2, b.y2);
    const double intersection = std::max(0.0, ix2 - ix1) * std::max(0.0, iy2 - iy1);
    const double area_a = std::max(kMinPositive, (a.x2 - a.x1) * (a.y2 - a.y1));
    const double area_b = std::max(kMinPositive, (b.x2 - b.x1) * (b.y2 - b.y1));
    const double union_area = area_a + area_b - intersection;
    return intersection / std::max(union_area, kMinPositive);
}

float cosine_distance(
    const std::vector<float>& track_feature,
    const std::vector<float>& detection_feature
) {
    if (track_feature.empty() || detection_feature.empty() ||
        track_feature.size() != detection_feature.size()) {
        return kNeutralCosineDistance;
    }

    double dot = 0.0;
    double track_norm = 0.0;
    double detection_norm = 0.0;
    for (std::size_t index = 0; index < track_feature.size(); ++index) {
        const double track_value = track_feature[index];
        const double detection_value = detection_feature[index];
        if (!std::isfinite(track_value) || !std::isfinite(detection_value)) {
            return kNeutralCosineDistance;
        }
        dot += track_value * detection_value;
        track_norm += track_value * track_value;
        detection_norm += detection_value * detection_value;
    }

    if (track_norm <= kMinPositive || detection_norm <= kMinPositive) {
        return kNeutralCosineDistance;
    }

    const double similarity = dot / (std::sqrt(track_norm) * std::sqrt(detection_norm));
    return static_cast<float>(std::clamp(1.0 - similarity, 0.0, 1.0));
}

Cholesky4 factor_covariance(const std::array<double, 16>& covariance) {
    Cholesky4 result;

    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = covariance[row * 4 + column];
            for (std::size_t index = 0; index < column; ++index) {
                value -= result.lower[row * 4 + index] * result.lower[column * 4 + index];
            }

            if (row == column) {
                if (!std::isfinite(value) || value <= kMinPositive) {
                    return result;
                }
                result.lower[row * 4 + column] = std::sqrt(value);
            } else {
                const double diagonal = result.lower[column * 4 + column];
                if (!std::isfinite(diagonal) || diagonal <= kMinPositive) {
                    return result;
                }
                result.lower[row * 4 + column] = value / diagonal;
            }
        }
    }

    result.valid = true;
    return result;
}

double mahalanobis_squared(
    const Cholesky4& covariance,
    const std::array<double, 4>& predicted_xyah,
    const std::array<double, 4>& detection_xyah
) {
    std::array<double, 4> solution{};
    for (std::size_t row = 0; row < 4; ++row) {
        double value = detection_xyah[row] - predicted_xyah[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -= covariance.lower[row * 4 + column] * solution[column];
        }
        solution[row] = value / covariance.lower[row * 4 + row];
    }

    double distance = 0.0;
    for (const double value : solution) {
        distance += value * value;
    }
    return distance;
}

float normalised_mahalanobis(
    const DmaTrackState& track,
    const std::array<double, 4>& detection_xyah,
    const Cholesky4* cached_covariance = nullptr
) {
    const Cholesky4 local_covariance = cached_covariance == nullptr
        ? factor_covariance(track.innovation_covariance)
        : *cached_covariance;
    if (!local_covariance.valid) {
        return 1.0F;
    }

    const double distance = mahalanobis_squared(
        local_covariance,
        track.predicted_xyah,
        detection_xyah
    );
    if (!std::isfinite(distance)) {
        return 1.0F;
    }
    return static_cast<float>(std::clamp(distance / kDmaChi2Inv95FourDof, 0.0, 1.0));
}

float covariance_trace_log(const DmaTrackState& track) {
    const double trace = track.innovation_covariance[0]
        + track.innovation_covariance[5]
        + track.innovation_covariance[10]
        + track.innovation_covariance[15];
    if (!std::isfinite(trace)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(std::log1p(trace));
}

float tracklet_length_norm(const DmaTrackState& track) {
    return static_cast<float>(std::clamp(
        static_cast<double>(track.tracklet_len) / kDmaMaxTrackletLength,
        0.0,
        1.0
    ));
}

std::vector<float> normalised_feature(const std::vector<float>& feature) {
    if (feature.empty()) {
        return {};
    }

    double squared_norm = 0.0;
    for (const float value : feature) {
        if (!std::isfinite(value)) {
            return {};
        }
        squared_norm += static_cast<double>(value) * value;
    }
    if (squared_norm <= kMinPositive) {
        return {};
    }

    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    std::vector<float> result;
    result.reserve(feature.size());
    for (const float value : feature) {
        result.push_back(static_cast<float>(value * inverse_norm));
    }
    return result;
}

float normalised_cosine_distance(
    const std::vector<float>& normalised_track_feature,
    const std::vector<float>& normalised_detection_feature
) {
    if (normalised_track_feature.empty() || normalised_detection_feature.empty() ||
        normalised_track_feature.size() != normalised_detection_feature.size()) {
        return kNeutralCosineDistance;
    }

    double similarity = 0.0;
    for (std::size_t index = 0; index < normalised_track_feature.size(); ++index) {
        similarity += static_cast<double>(normalised_track_feature[index])
            * normalised_detection_feature[index];
    }
    return static_cast<float>(std::clamp(1.0 - similarity, 0.0, 1.0));
}

DmaFeatureVector make_pair_features(
    const DmaTrackState& track,
    const DmaDetection& detection,
    const Cholesky4* cached_covariance
) {
    const std::array<double, 4> detection_xyah = to_xyah(detection.bbox);
    const double width = std::max(detection.bbox.width, 1.0);
    const double height = std::max(detection.bbox.height, 1.0);

    return {
        static_cast<float>(1.0 - iou(track.predicted_bbox, detection.bbox)),
        normalised_mahalanobis(track, detection_xyah, cached_covariance),
        covariance_trace_log(track),
        cosine_distance(track.smooth_feature, detection.current_feature),
        static_cast<float>(std::log1p(width * height) / 15.0),
        tracklet_length_norm(track),
    };
}

}  // namespace

const DmaFeatureVector& DmaFeatureMatrix::at(
    std::size_t track_index,
    std::size_t detection_index
) const {
    if (track_index >= track_count || detection_index >= detection_count) {
        throw std::out_of_range("DMA feature matrix index out of range");
    }
    return values[track_index * detection_count + detection_index];
}

DmaFeatureVector extract_dma_pair_features(
    const DmaTrackState& track,
    const DmaDetection& detection
) {
    return make_pair_features(track, detection, nullptr);
}

DmaFeatureMatrix extract_dma_batch_features(
    const std::vector<DmaTrackState>& tracks,
    const std::vector<DmaDetection>& detections
) {
    DmaFeatureMatrix result;
    result.track_count = tracks.size();
    result.detection_count = detections.size();
    result.values.resize(tracks.size() * detections.size());

    if (tracks.empty() || detections.empty()) {
        return result;
    }

    std::vector<std::array<double, 4>> detection_xyah;
    detection_xyah.reserve(detections.size());
    for (const auto& detection : detections) {
        detection_xyah.push_back(to_xyah(detection.bbox));
    }

    std::vector<Cholesky4> factored_covariances;
    std::vector<float> track_covariance_trace_logs;
    std::vector<float> tracklet_lengths;
    std::vector<float> detection_area_logs;
    std::vector<std::vector<float>> normalised_track_features;
    std::vector<std::vector<float>> normalised_detection_features;
    factored_covariances.reserve(tracks.size());
    track_covariance_trace_logs.reserve(tracks.size());
    tracklet_lengths.reserve(tracks.size());
    normalised_track_features.reserve(tracks.size());
    for (const auto& track : tracks) {
        factored_covariances.push_back(factor_covariance(track.innovation_covariance));
        track_covariance_trace_logs.push_back(covariance_trace_log(track));
        tracklet_lengths.push_back(tracklet_length_norm(track));
        normalised_track_features.push_back(normalised_feature(track.smooth_feature));
    }
    detection_area_logs.reserve(detections.size());
    normalised_detection_features.reserve(detections.size());
    for (const auto& detection : detections) {
        const double width = std::max(detection.bbox.width, 1.0);
        const double height = std::max(detection.bbox.height, 1.0);
        detection_area_logs.push_back(static_cast<float>(std::log1p(width * height) / 15.0));
        normalised_detection_features.push_back(normalised_feature(detection.current_feature));
    }

    // Keep track-only and detection-only work outside the pair loop. The
    // remaining work is the association matrix calculation: IoU, one
    // triangular solve, and a dot product when both embeddings are present.
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const auto& track = tracks[track_index];
            const auto& detection = detections[detection_index];
            const std::size_t output_index = track_index * detections.size() + detection_index;

            result.values[output_index] = {
                static_cast<float>(1.0 - iou(track.predicted_bbox, detection.bbox)),
                normalised_mahalanobis(
                    track,
                    detection_xyah[detection_index],
                    &factored_covariances[track_index]
                ),
                track_covariance_trace_logs[track_index],
                normalised_cosine_distance(
                    normalised_track_features[track_index],
                    normalised_detection_features[detection_index]
                ),
                detection_area_logs[detection_index],
                tracklet_lengths[track_index],
            };
        }
    }

    return result;
}

}  // namespace exhibitflow
