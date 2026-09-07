#pragma once

#include "exhibitflow/types.hpp"

namespace exhibitflow {

TrackObservation make_track_observation(std::int64_t track_id, const Detection& detection);

}  // namespace exhibitflow
