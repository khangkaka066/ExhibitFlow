#include <cassert>
#include <cstdlib>
#include <iostream>

#include "exhibitflow/tracker.hpp"

namespace {

void test_mock_tracker_ids_and_reset() {
    auto tracker = exhibitflow::create_tracker("mock");

    exhibitflow::FrameContext context;
    context.schema_version = "0.1";
    context.sequence_id = "demo_01";
    context.camera_id = "cam_01";
    context.image_size.width = 1920;
    context.image_size.height = 1080;

    exhibitflow::Detection detection;
    detection.bbox.x = 100.0;
    detection.bbox.y = 200.0;
    detection.bbox.width = 60.0;
    detection.bbox.height = 160.0;
    detection.score = 0.95;

    const auto first_tracks = tracker->update(context, {detection});
    assert(first_tracks.size() == 1);
    assert(first_tracks[0].track_id == 1);
    assert(first_tracks[0].point_image_x == 130.0);
    assert(first_tracks[0].point_image_y == 360.0);

    const auto second_tracks = tracker->update(context, {detection});
    assert(second_tracks.size() == 1);
    assert(second_tracks[0].track_id == 2);

    tracker->reset();
    const auto reset_tracks = tracker->update(context, {detection});
    assert(reset_tracks.size() == 1);
    assert(reset_tracks[0].track_id == 1);
}

void test_unknown_tracker_fails() {
    bool failed = false;
    try {
        (void)exhibitflow::create_tracker("bytetrack");
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    if (!failed) {
        std::cerr << "expected unknown tracker to fail\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    test_mock_tracker_ids_and_reset();
    test_unknown_tracker_fails();
    return 0;
}
