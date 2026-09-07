#pragma once

#include <memory>
#include <string>
#include <vector>

#include "exhibitflow/types.hpp"

namespace exhibitflow {

class ITracker {
public:
    virtual ~ITracker() = default;

    virtual std::vector<TrackObservation> update(
        const FrameContext& context,
        const std::vector<Detection>& detections
    ) = 0;

    virtual void reset() = 0;
};

std::unique_ptr<ITracker> create_tracker(const std::string& tracker_name);

}  // namespace exhibitflow
