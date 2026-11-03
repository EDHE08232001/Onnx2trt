#include "trt_inference.hpp"

#include <NvInferRuntime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;
using namespace nvinfer1;

namespace trt_inference {
namespace {

// Owns a block of GPU memory.
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
};

// Owns a CUDA stream.
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreate(&s_)); }
    ~Stream() { cudaStreamDestroy(s_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return s_; }

private:
    cudaStream_t s_ = nullptr;
};

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open plan file: " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

size_t volume(const Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) v *= static_cast<size_t>(dims.d[i]);
    return v;
}

// Arg-max over the class axis of [N, C, H, W] scores, scaled to 0-255 and saved as PNGs.
void saveSegmentationMasks(const std::vector<float>& scores, const Dims& dims, const std::string& outputDir) {
    const int batch = dims.d[0], classes = dims.d[1], height = dims.d[2], width = dims.d[3];
    const size_t plane = static_cast<size_t>(height) * width;
    const int scale = classes > 1 ? 255 / (classes - 1) : 0;

    fs::create_directories(outputDir);
    for (int b = 0; b < batch; ++b) {
        cv::Mat mask(height, width, CV_8UC1);
        for (size_t p = 0; p < plane; ++p) {
            int best = 0;
            float bestScore = scores[(static_cast<size_t>(b) * classes) * plane + p];
            for (int c = 1; c < classes; ++c) {
                const float s = scores[(static_cast<size_t>(b) * classes + c) * plane + p];
                if (s > bestScore) {
                    bestScore = s;
                    best = c;
                }
            }
            mask.data[p] = static_cast<uchar>(best * scale);
        }

        const std::string file = (fs::path(outputDir) / ("seg_" + std::to_string(b) + ".png")).string();
        if (cv::imwrite(file, mask)) {
            std::cout << "Saved " << file << std::endl;
        } else {
            std::cerr << "Failed to save " << file << std::endl;
        }
    }
}

}  // namespace

bool run_segmentation(const Options& opt, const Tensor4D& input) {
    try {
        TrtLogger logger;
        const std::vector<char> planData = readFile(opt.planPath);

        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(planData.data(), planData.size()));
        if (!engine) {
            std::cerr << "Failed to deserialize the plan. Was it built with the same TensorRT version and GPU?" << std::endl;
            return false;
        }
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create an execution context." << std::endl;
            return false;
        }

        // ---- Locate bindings (single input, any number of outputs) -----------
        int inputIndex = -1;
        std::vector<int> outputIndices;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (engine->bindingIsInput(i)) {
                if (inputIndex >= 0) {
                    std::cerr << "Engines with more than one input are not supported." << std::endl;
                    return false;
                }
                inputIndex = i;
            } else {
                outputIndices.push_back(i);
            }
        }
        if (inputIndex < 0 || outputIndices.empty()) {
            std::cerr << "The engine has no input or no output." << std::endl;
            return false;
        }

        // ---- Set the input shape (only needed for dynamic engines) -----------
        const Dims4 inputDims(input.n, input.c, input.h, input.w);
        const Dims engineInput = engine->getBindingDimensions(inputIndex);
        const bool isDynamic = std::any_of(engineInput.d, engineInput.d + engineInput.nbDims,
                                           [](int d) { return d < 0; });
        if (isDynamic) {
            if (!context->setBindingDimensions(inputIndex, inputDims)) {
                std::cerr << "Input shape " << inputDims << " is outside the range the engine was built for "
                          << "(" << engineInput << " with its optimisation profile)." << std::endl;
                return false;
            }
        } else if (volume(engineInput) != volume(inputDims)) {
            std::cerr << "Input shape " << inputDims << " does not match the engine's fixed shape "
                      << engineInput << "." << std::endl;
            return false;
        }

        // ---- Allocate buffers -------------------------------------------------
        Stream stream;
        std::vector<void*> bindings(engine->getNbBindings(), nullptr);

        const size_t inputBytes = input.size() * sizeof(float);
        DeviceBuffer deviceInput(inputBytes);
        bindings[inputIndex] = deviceInput.get();
        CUDA_CHECK(cudaMemcpyAsync(deviceInput.get(), input.data.data(), inputBytes,
                                   cudaMemcpyHostToDevice, stream.get()));

        std::vector<std::unique_ptr<DeviceBuffer>> deviceOutputs;
        std::vector<Dims> outputDims;
        for (int idx : outputIndices) {
            const Dims dims = context->getBindingDimensions(idx);  // fully resolved after setting the input shape
            outputDims.push_back(dims);
            deviceOutputs.push_back(std::make_unique<DeviceBuffer>(volume(dims) * sizeof(float)));
            bindings[idx] = deviceOutputs.back()->get();
        }

        // ---- Benchmark --------------------------------------------------------
        for (int i = 0; i < opt.warmupIterations; ++i) {
            context->enqueueV2(bindings.data(), stream.get(), nullptr);
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream.get()));
        for (int i = 0; i < opt.numTrials; ++i) {
            TRT_NVTX_PUSH(("frame" + std::to_string(i)).c_str());
            const bool ok = context->enqueueV2(bindings.data(), stream.get(), nullptr);
            TRT_NVTX_POP();
            if (!ok) {
                std::cerr << "TensorRT enqueueV2 failed." << std::endl;
                return false;
            }
        }
        CUDA_CHECK(cudaEventRecord(stop, stream.get()));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float totalMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cout << "Average latency over " << opt.numTrials << " runs (batch " << input.n << "): "
                  << totalMs / static_cast<float>(opt.numTrials) << " ms" << std::endl;

        // ---- Read back the last output and post-process ----------------------
        const Dims& lastDims = outputDims.back();
        std::vector<float> scores(volume(lastDims));
        CUDA_CHECK(cudaMemcpyAsync(scores.data(), deviceOutputs.back()->get(), scores.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        const auto [minIt, maxIt] = std::minmax_element(scores.begin(), scores.end());
        const float mean = std::accumulate(scores.begin(), scores.end(), 0.0f) / static_cast<float>(scores.size());
        std::cout << "Output " << lastDims << " - min: " << *minIt << ", max: " << *maxIt << ", mean: " << mean << std::endl;

        if (lastDims.nbDims == 4) {
            saveSegmentationMasks(scores, lastDims, opt.outputDir);
        } else {
            std::cout << "Output is not 4-D (N, classes, H, W); skipping mask export." << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace trt_inference
