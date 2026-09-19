#include <cassert>
#include <cmath>
#include <vector>

#include "exhibitflow/yolox_postprocess.hpp"

int main() {
    // A 32x32 input has 4x4 + 2x2 + 1x1 YOLOX anchors.
    std::vector<float> output(21U * 6U, 0.0F);
    output[0] = 0.5F;
    output[1] = 0.5F;
    output[2] = std::log(2.0F);
    output[3] = std::log(2.0F);
    output[4] = 0.9F;
    output[5] = 0.8F;

    // Overlapping, lower-confidence duplicate in the same grid neighbourhood.
    output[6] = -0.45F;
    output[7] = 0.5F;
    output[8] = std::log(2.0F);
    output[9] = std::log(2.0F);
    output[10] = 0.7F;
    output[11] = 0.7F;

    const exhibitflow::LetterboxTransform transform{0.5};
    const auto detections = exhibitflow::decode_yolox_person_output(
        output.data(), output.size() / 6U, 32, 32, transform, {64, 64}, 0.1F, 0.45F
    );
    assert(detections.size() == 1);
    assert(std::abs(detections[0].score - 0.72) < 1e-6);
    assert(std::abs(detections[0].bbox.x - 8.0) < 1e-6);
    assert(std::abs(detections[0].bbox.y - 8.0) < 1e-6);
    assert(std::abs(detections[0].bbox.width - 32.0) < 1e-6);
    assert(std::abs(detections[0].bbox.height - 32.0) < 1e-6);
    return 0;
}
