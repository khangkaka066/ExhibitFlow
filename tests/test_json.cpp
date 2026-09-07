#include <cassert>
#include <string>

#include "exhibitflow/json.hpp"

namespace {

void test_parse_detection_frame() {
    const std::string line =
        "{\"schema_version\":\"0.1\",\"sequence_id\":\"demo_01\",\"camera_id\":\"cam_01\","
        "\"frame_id\":0,\"timestamp_ms\":0,\"image_size\":{\"width\":1920,\"height\":1080},"
        "\"detections\":[{\"bbox\":[100,200,60,160],\"score\":0.95}]}";

    const exhibitflow::DetectionFrame frame = exhibitflow::parse_detection_frame(line, 1);
    assert(frame.context.schema_version == "0.1");
    assert(frame.context.sequence_id == "demo_01");
    assert(frame.context.camera_id == "cam_01");
    assert(frame.context.frame_id == 0);
    assert(frame.context.timestamp_ms == 0);
    assert(frame.context.image_size.width == 1920);
    assert(frame.context.image_size.height == 1080);
    assert(frame.detections.size() == 1);
    assert(frame.detections[0].bbox.x == 100.0);
    assert(frame.detections[0].bbox.y == 200.0);
    assert(frame.detections[0].bbox.width == 60.0);
    assert(frame.detections[0].bbox.height == 160.0);
    assert(frame.detections[0].score == 0.95);
}

void test_invalid_bbox_fails() {
    const std::string line =
        "{\"schema_version\":\"0.1\",\"sequence_id\":\"demo_01\",\"camera_id\":\"cam_01\","
        "\"frame_id\":0,\"timestamp_ms\":0,\"image_size\":{\"width\":100,\"height\":100},"
        "\"detections\":[{\"bbox\":[90,90,20,20],\"score\":0.95}]}";

    bool failed = false;
    try {
        (void)exhibitflow::parse_detection_frame(line, 1);
    } catch (const exhibitflow::ContractError&) {
        failed = true;
    }
    assert(failed);
}

void test_config() {
    assert(exhibitflow::parse_tracker_name_from_config("{\"schema_version\":\"0.1\",\"tracker\":\"mock\"}") == "mock");
}

}  // namespace

int main() {
    test_parse_detection_frame();
    test_invalid_bbox_fails();
    test_config();
    return 0;
}
