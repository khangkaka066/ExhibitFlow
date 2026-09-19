#include "exhibitflow/detector.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "exhibitflow/yolox_postprocess.hpp"
#include "exhibitflow/yolox_preprocess.hpp"

namespace exhibitflow {
namespace {

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "TensorRT: %s\n", message);
        }
    }
};

TensorRtLogger& tensorrt_logger() {
    static TensorRtLogger logger;
    return logger;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
struct TensorRtDeleter {
    void operator()(T* object) const noexcept {
        if (object != nullptr) {
            object->destroy();
        }
    }
};

class CudaBuffer final {
public:
    CudaBuffer() = default;
    CudaBuffer(std::size_t byte_count, const char* label) { allocate(byte_count, label); }
    ~CudaBuffer() { reset(); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)) {}
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    void* data() const { return data_; }

private:
    void allocate(std::size_t byte_count, const char* label) {
        if (byte_count == 0U) {
            throw std::invalid_argument(std::string(label) + " buffer must not be empty");
        }
        check_cuda(cudaMalloc(&data_, byte_count), label);
    }
    void reset() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    void* data_ = nullptr;
};

class CudaStream final {
public:
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

std::vector<char> read_engine(const std::string& engine_path) {
    std::ifstream stream(engine_path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("could not open TensorRT engine: " + engine_path);
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_path);
    }
    std::vector<char> data(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(data.data(), size)) {
        throw std::runtime_error("could not read TensorRT engine: " + engine_path);
    }
    return data;
}

std::size_t tensor_element_count(const nvinfer1::Dims& dimensions, const char* tensor_name) {
    std::size_t count = 1U;
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0) {
            throw std::runtime_error(std::string("TensorRT tensor has unresolved dimensions: ") + tensor_name);
        }
        count *= static_cast<std::size_t>(dimensions.d[index]);
    }
    return count;
}

class TensorRtYoloXDetector final : public IDetector {
public:
    TensorRtYoloXDetector(const std::string& engine_path, DetectorOptions options)
        : options_(options), stream_() {
        if (engine_path.empty()) {
            throw std::invalid_argument("TensorRT engine path must not be empty");
        }
        const std::vector<char> serialized_engine = read_engine(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(tensorrt_logger()));
        if (!runtime_) {
            throw std::runtime_error("could not create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
        if (!engine_) {
            throw std::runtime_error("could not deserialize TensorRT engine; rebuild it on this NVIDIA GPU");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("could not create TensorRT execution context");
        }

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) throw std::runtime_error("YOLOX TensorRT engine must have one input");
                input_name_ = name;
            } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                if (!output_name_.empty()) throw std::runtime_error("YOLOX TensorRT engine must have one output");
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("YOLOX TensorRT engine must expose one input and one output");
        }
        if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("YOLOX TensorRT engine must use FP32 input/output tensors");
        }

        const nvinfer1::Dims4 input_shape{1, 3, options_.input_height, options_.input_width};
        if (!context_->setInputShape(input_name_.c_str(), input_shape)) {
            throw std::runtime_error("TensorRT engine input shape does not match detector options");
        }
        const nvinfer1::Dims output_shape = context_->getTensorShape(output_name_.c_str());
        if (output_shape.nbDims != 3 || output_shape.d[0] != 1 || output_shape.d[1] <= 0 ||
            output_shape.d[2] != 6) {
            throw std::runtime_error("unexpected TensorRT YOLOX output shape");
        }
        prediction_count_ = static_cast<std::size_t>(output_shape.d[1]);
        const std::size_t expected_input = static_cast<std::size_t>(options_.input_width) *
            static_cast<std::size_t>(options_.input_height) * 3U;
        const std::size_t output_count = tensor_element_count(output_shape, output_name_.c_str());
        if (output_count != prediction_count_ * 6U) {
            throw std::runtime_error("unexpected TensorRT YOLOX output element count");
        }
        input_device_ = CudaBuffer(expected_input * sizeof(float), "cudaMalloc input");
        output_device_ = CudaBuffer(output_count * sizeof(float), "cudaMalloc output");
        output_host_.resize(output_count);
        if (!context_->setTensorAddress(input_name_.c_str(), input_device_.data()) ||
            !context_->setTensorAddress(output_name_.c_str(), output_device_.data())) {
            throw std::runtime_error("could not bind TensorRT input/output buffers");
        }
    }

    std::vector<Detection> detect(const BgrImageView& image) override {
        PreparedYoloXInput input = prepare_yolox_input(image, options_);
        check_cuda(cudaMemcpyAsync(input_device_.data(), input.chw.data(), input.chw.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream_.get()), "TensorRT input upload");
        if (!context_->enqueueV3(stream_.get())) {
            throw std::runtime_error("TensorRT inference failed");
        }
        check_cuda(cudaMemcpyAsync(output_host_.data(), output_device_.data(), output_host_.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream_.get()), "TensorRT output download");
        check_cuda(cudaStreamSynchronize(stream_.get()), "TensorRT synchronization");
        return decode_yolox_person_output(
            output_host_.data(), prediction_count_, options_.input_width, options_.input_height,
            input.transform, {image.width, image.height}, options_.confidence_threshold, options_.nms_threshold
        );
    }

private:
    DetectorOptions options_;
    CudaStream stream_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDeleter<nvinfer1::IExecutionContext>> context_;
    std::string input_name_;
    std::string output_name_;
    CudaBuffer input_device_;
    CudaBuffer output_device_;
    std::vector<float> output_host_;
    std::size_t prediction_count_ = 0U;
};

}  // namespace

std::unique_ptr<IDetector> create_tensorrt_yolox_detector(
    const std::string& engine_path,
    const DetectorOptions& options
) {
    return std::make_unique<TensorRtYoloXDetector>(engine_path, options);
}

}  // namespace exhibitflow
