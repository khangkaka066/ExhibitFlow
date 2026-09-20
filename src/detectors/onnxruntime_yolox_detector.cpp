#include "exhibitflow/detector.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>

#include "exhibitflow/yolox_postprocess.hpp"
#include "exhibitflow/yolox_preprocess.hpp"

namespace exhibitflow {
namespace {

Ort::Env& ort_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "exhibitflow_detector");
    return environment;
}

class OnnxRuntimeYoloXDetector final : public IDetector {
public:
    OnnxRuntimeYoloXDetector(const std::string& model_path, DetectorOptions options)
        : options_(options), session_options_() {
        if (model_path.empty()) {
            throw std::invalid_argument("ONNX model path must not be empty");
        }
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.SetIntraOpNumThreads(4);
        if (options_.use_cuda) {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = options_.cuda_device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        }
        const std::filesystem::path native_model_path(model_path);
        session_ = std::make_unique<Ort::Session>(
            ort_environment(), native_model_path.c_str(), session_options_
        );

        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
            throw std::runtime_error("YOLOX ONNX model must expose exactly one input and one output");
        }
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        const auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
            (input_shape[2] > 0 && input_shape[2] != options_.input_height) ||
            (input_shape[3] > 0 && input_shape[3] != options_.input_width)) {
            throw std::runtime_error("ONNX model input shape does not match detector options");
        }
    }

    std::vector<Detection> detect(const BgrImageView& image) override {
        PreparedYoloXInput input = prepare_yolox_input(image, options_);
        const std::array<std::int64_t, 4> input_shape = {
            1, 3, options_.input_height, options_.input_width
        };
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, input.chw.data(), input.chw.size(), input_shape.data(), input_shape.size()
        );
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto results = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1
        );
        if (results.size() != 1 || !results[0].IsTensor()) {
            throw std::runtime_error("ONNX Runtime returned an invalid YOLOX output");
        }
        const auto info = results[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (shape.size() != 3 || shape[0] != 1 || shape[2] != 6) {
            throw std::runtime_error("unexpected YOLOX ONNX output shape");
        }
        return decode_yolox_person_output(
            results[0].GetTensorData<float>(), static_cast<std::size_t>(shape[1]),
            options_.input_width, options_.input_height, input.transform,
            {image.width, image.height}, options_.confidence_threshold, options_.nms_threshold
        );
    }

private:
    DetectorOptions options_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace

std::unique_ptr<IDetector> create_onnxruntime_yolox_detector(
    const std::string& model_path,
    const DetectorOptions& options
) {
    return std::make_unique<OnnxRuntimeYoloXDetector>(model_path, options);
}

}  // namespace exhibitflow
