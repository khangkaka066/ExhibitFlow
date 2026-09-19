#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "exhibitflow/dma_features.hpp"

namespace {

exhibitflow::DmaTrackState make_track() {
    exhibitflow::DmaTrackState track;
    track.predicted_bbox = {100.0, 200.0, 60.0, 160.0};
    track.predicted_xyah = {130.0, 280.0, 0.375, 160.0};
    track.innovation_covariance = {
        4.0, 0.0, 0.0, 0.0,
        0.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 0.01, 0.0,
        0.0, 0.0, 0.0, 4.0,
    };
    track.smooth_feature = {1.0F, 0.0F, 0.0F};
    track.tracklet_len = 15;
    return track;
}

exhibitflow::DmaDetection make_detection() {
    exhibitflow::DmaDetection detection;
    detection.bbox = {100.0, 200.0, 60.0, 160.0};
    detection.current_feature = {1.0F, 0.0F, 0.0F};
    return detection;
}

void assert_close(float actual, float expected) {
    if (std::fabs(actual - expected) > 1e-5F) {
        std::cerr << "expected " << expected << ", got " << actual << "\n";
        std::exit(1);
    }
}

void test_pair_features() {
    const auto features = exhibitflow::extract_dma_pair_features(make_track(), make_detection());

    assert_close(features[0], 0.0F);  // motion_cost: identical boxes
    assert_close(features[1], 0.0F);  // mahalanobis_norm: identical measurements
    assert_close(features[2], static_cast<float>(std::log1p(12.01))); // covariance trace
    assert_close(features[3], 0.0F);  // identical embeddings
    assert_close(features[4], static_cast<float>(std::log1p(60.0 * 160.0) / 15.0));
    assert_close(features[5], 0.5F);
}

void test_batch_reuses_the_same_contract() {
    const auto track = make_track();
    auto second_track = track;
    second_track.predicted_bbox.x += 10.0;
    second_track.predicted_xyah[0] += 10.0;
    second_track.tracklet_len = 60;

    const auto detection = make_detection();
    auto second_detection = detection;
    second_detection.bbox.x += 200.0;
    second_detection.current_feature = {0.0F, 1.0F, 0.0F};

    const std::vector<exhibitflow::DmaTrackState> tracks = {track, second_track};
    const std::vector<exhibitflow::DmaDetection> detections = {detection, second_detection};
    const auto matrix = exhibitflow::extract_dma_batch_features(tracks, detections);

    assert(matrix.track_count == 2);
    assert(matrix.detection_count == 2);
    assert(matrix.values.size() == 4);
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const auto scalar = exhibitflow::extract_dma_pair_features(
                tracks[track_index], detections[detection_index]
            );
            const auto& batch = matrix.at(track_index, detection_index);
            for (std::size_t feature_index = 0; feature_index < exhibitflow::kDmaFeatureDim; ++feature_index) {
                assert_close(batch[feature_index], scalar[feature_index]);
            }
        }
    }

    assert_close(matrix.at(0, 1)[3], 1.0F); // orthogonal appearance vectors
    assert_close(matrix.at(1, 0)[5], 1.0F); // tracklet length is capped at 30
}

void test_fallbacks_and_empty_batches() {
    auto track = make_track();
    track.innovation_covariance.fill(0.0);
    auto detection = make_detection();
    detection.current_feature.clear();

    const auto features = exhibitflow::extract_dma_pair_features(track, detection);
    assert_close(features[1], 1.0F); // invalid covariance fallback
    assert_close(features[3], 0.5F); // missing appearance fallback

    const auto empty = exhibitflow::extract_dma_batch_features({}, {detection});
    assert(empty.track_count == 0);
    assert(empty.detection_count == 1);
    assert(empty.values.empty());
}

}  // namespace

int main() {
    test_pair_features();
    test_batch_reuses_the_same_contract();
    test_fallbacks_and_empty_batches();
    return 0;
}
