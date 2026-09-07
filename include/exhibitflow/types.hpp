#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace exhibitflow {

struct ImageSize {
    int width = 0;
    int height = 0;
};

struct BoundingBox {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct Detection {
    BoundingBox bbox;
    double score = 0.0;
};

struct FrameContext {
    std::string schema_version;
    std::string sequence_id;
    std::string camera_id;
    std::int64_t frame_id = 0;
    std::int64_t timestamp_ms = 0;
    ImageSize image_size;
};

struct TrackObservation {
    std::int64_t track_id = 0;
    BoundingBox bbox;
    double point_image_x = 0.0;
    double point_image_y = 0.0;
    double score = 0.0;
};

struct DetectionFrame {
    FrameContext context;
    std::vector<Detection> detections;
};

struct TrackFrame {
    FrameContext context;
    std::vector<TrackObservation> tracks;
};

}  // namespace exhibitflow
