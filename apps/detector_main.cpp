#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "exhibitflow/detector.hpp"
#include "exhibitflow/json.hpp"

namespace {

struct Options {
    std::string backend;
    std::string model_path;
    std::string video_path;
    std::string output_path;
    std::string sequence_id;
    std::string camera_id = "cam_01";
    int input_width = 640;
    int input_height = 640;
    float confidence = 0.01F;
    float nms = 0.45F;
    int max_frames = -1;
};

void print_help() {
    std::cout << "Usage: exhibitflow_detector --backend onnxruntime|tensorrt --model model.onnx|model.engine "
                 "--video input.mp4 --output detections.jsonl [options]\n"
                 "  --sequence-id ID    Defaults to video filename stem\n"
                 "  --camera-id ID      Defaults to cam_01\n"
                 "  --input-width N     Defaults to 640\n"
                 "  --input-height N    Defaults to 640\n"
                 "  --conf N            Defaults to 0.01\n"
                 "  --nms N             Defaults to 0.45\n"
                 "  --max-frames N      Defaults to all frames\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--backend") options.backend = value;
        else if (argument == "--model") options.model_path = value;
        else if (argument == "--video") options.video_path = value;
        else if (argument == "--output") options.output_path = value;
        else if (argument == "--sequence-id") options.sequence_id = value;
        else if (argument == "--camera-id") options.camera_id = value;
        else if (argument == "--input-width") options.input_width = std::stoi(value);
        else if (argument == "--input-height") options.input_height = std::stoi(value);
        else if (argument == "--conf") options.confidence = std::stof(value);
        else if (argument == "--nms") options.nms = std::stof(value);
        else if (argument == "--max-frames") options.max_frames = std::stoi(value);
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.backend.empty() || options.model_path.empty() || options.video_path.empty() || options.output_path.empty()) {
        throw std::invalid_argument("--backend, --model, --video, and --output are required");
    }
    if (options.backend != "onnxruntime" && options.backend != "tensorrt") {
        throw std::invalid_argument("--backend must be onnxruntime or tensorrt");
    }
    if (options.input_width <= 0 || options.input_height <= 0 || options.confidence < 0.0F ||
        options.confidence > 1.0F || options.nms < 0.0F || options.nms > 1.0F || options.max_frames == 0) {
        throw std::invalid_argument("invalid detector dimensions, thresholds, or max frames");
    }
    return options;
}

std::unique_ptr<exhibitflow::IDetector> create_detector(const Options& options) {
    const exhibitflow::DetectorOptions detector_options{
        options.input_width, options.input_height, options.confidence, options.nms
    };
    if (options.backend == "onnxruntime") {
#if defined(EXHIBITFLOW_WITH_ONNXRUNTIME)
        return exhibitflow::create_onnxruntime_yolox_detector(options.model_path, detector_options);
#else
        throw std::runtime_error("this build does not include ONNX Runtime");
#endif
    }
#if defined(EXHIBITFLOW_WITH_TENSORRT)
    return exhibitflow::create_tensorrt_yolox_detector(options.model_path, detector_options);
#else
    throw std::runtime_error("this build does not include TensorRT");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        namespace fs = std::filesystem;
        if (fs::exists(options.output_path)) {
            throw std::runtime_error("output already exists: " + options.output_path);
        }
        cv::VideoCapture capture(options.video_path);
        if (!capture.isOpened()) {
            throw std::runtime_error("could not open video: " + options.video_path);
        }
        const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 30.0;
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("video has invalid dimensions");
        }
        auto detector = create_detector(options);
        const fs::path temporary_path = options.output_path + ".tmp";
        std::ofstream output(temporary_path);
        if (!output) {
            throw std::runtime_error("could not open output: " + temporary_path.string());
        }
        const std::string sequence_id = options.sequence_id.empty() ? fs::path(options.video_path).stem().string() : options.sequence_id;
        cv::Mat frame;
        std::int64_t frame_id = 0;
        while (capture.read(frame) && (options.max_frames < 0 || frame_id < options.max_frames)) {
            const exhibitflow::BgrImageView image{
                frame.data, frame.cols, frame.rows, static_cast<int>(frame.step)
            };
            exhibitflow::DetectionFrame detection_frame;
            detection_frame.context = {"0.1", sequence_id, options.camera_id, frame_id,
                static_cast<std::int64_t>(std::llround(frame_id * 1000.0 / fps)), {width, height}};
            detection_frame.detections = detector->detect(image);
            output << exhibitflow::serialize_detection_frame(detection_frame) << "\n";
            ++frame_id;
        }
        output.close();
        if (!output) {
            throw std::runtime_error("failed writing detector output");
        }
        fs::rename(temporary_path, options.output_path);
        std::cerr << "processed_frames=" << frame_id << " backend=" << options.backend << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "detector error: " << error.what() << "\n";
        return 1;
    }
}
