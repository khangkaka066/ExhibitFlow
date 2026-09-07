#include "exhibitflow/track_writer.hpp"

namespace exhibitflow {

TrackObservation make_track_observation(std::int64_t track_id, const Detection& detection) {
    TrackObservation observation;
    observation.track_id = track_id;
    observation.bbox = detection.bbox;
    observation.point_image_x = detection.bbox.x + detection.bbox.width / 2.0;
    observation.point_image_y = detection.bbox.y + detection.bbox.height;
    observation.score = detection.score;
    return observation;
}

}  // namespace exhibitflow
