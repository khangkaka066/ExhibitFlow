#include "exhibitflow/tracker.hpp"

#include <stdexcept>

#include "exhibitflow/track_writer.hpp"

namespace exhibitflow {
namespace {

class MockTracker final : public ITracker {
public:
    std::vector<TrackObservation> update(
        const FrameContext&,
        const std::vector<Detection>& detections
    ) override {
        std::vector<TrackObservation> tracks;
        tracks.reserve(detections.size());

        for (const auto& detection : detections) {
            tracks.push_back(make_track_observation(next_track_id_, detection));
            ++next_track_id_;
        }

        return tracks;
    }

    void reset() override {
        next_track_id_ = 1;
    }

private:
    std::int64_t next_track_id_ = 1;
};

}  // namespace

std::unique_ptr<ITracker> create_tracker(const std::string& tracker_name) {
    if (tracker_name == "mock") {
        return std::make_unique<MockTracker>();
    }

    throw std::invalid_argument("unknown tracker backend: " + tracker_name);
}

}  // namespace exhibitflow
