#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "exhibitflow/types.hpp"

namespace exhibitflow {

inline constexpr std::size_t kDmaFeatureDim = 6;
inline constexpr double kDmaChi2Inv95FourDof = 9.4877;
inline constexpr double kDmaMaxTrackletLength = 30.0;

// The order is part of the DMA model contract. Keep it in sync with
// external/ByteTrack-DMA-LTC-Motion-Tracker/yolox/DMA/features.py.
using DmaFeatureVector = std::array<float, kDmaFeatureDim>;

struct DmaTrackState {
    // Kalman-predicted box in image pixels, represented as x/y/width/height.
    BoundingBox predicted_bbox;

    // Predicted measurement [center_x, center_y, aspect_ratio, height].
    std::array<double, 4> predicted_xyah{};

    // Innovation covariance for the four-dimensional measurement above,
    // row-major. The caller owns the Kalman projection/noise step.
    std::array<double, 16> innovation_covariance{};

    // EMA-smoothed appearance embedding. Empty means unavailable.
    std::vector<float> smooth_feature;

    std::int64_t tracklet_len = 0;
};

struct DmaDetection {
    BoundingBox bbox;

    // Current appearance embedding. Empty means unavailable.
    std::vector<float> current_feature;
};

struct DmaFeatureMatrix {
    std::size_t track_count = 0;
    std::size_t detection_count = 0;
    std::vector<DmaFeatureVector> values;

    const DmaFeatureVector& at(std::size_t track_index, std::size_t detection_index) const;
};

// Scalar reference implementation, useful for correctness checks and small
// batches. Missing/invalid motion or appearance signals use the same neutral
// fallbacks as the Python implementation.
DmaFeatureVector extract_dma_pair_features(
    const DmaTrackState& track,
    const DmaDetection& detection
);

// Batch implementation for the association hot path. Track-only quantities
// are computed once per track, detection-only quantities once per detection,
// and each track covariance is factorised once before solving all detections.
DmaFeatureMatrix extract_dma_batch_features(
    const std::vector<DmaTrackState>& tracks,
    const std::vector<DmaDetection>& detections
);

}  // namespace exhibitflow
