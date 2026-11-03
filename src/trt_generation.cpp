#include "trt_generation.hpp"

#include <NvOnnxParser.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "image_utils.hpp"

namespace fs = std::filesystem;
using namespace nvinfer1;

namespace trt_generation {
namespace {

// Feeds calibration images to TensorRT when building an INT8 engine.
// Supports networks with a single input.
class Int8Calibrator : public IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(std::vector<std::string> imagePaths, int batchSize, int channels, int height, int width)
        : imagePaths_(std::move(imagePaths)),
          batchSize_(batchSize),
          channels_(channels),
          height_(height),
          width_(width) {
        CUDA_CHECK(cudaMalloc(&deviceInput_, batchBytes()));
    }

    ~Int8Calibrator() override { cudaFree(deviceInput_); }

    int getBatchSize() const noexcept override { return batchSize_; }

    bool getBatch(void* bindings[], const char* /*names*/[], int /*nbBindings*/) noexcept override {
        if (nextImage_ >= imagePaths_.size()) return false;

        try {
            // Take up to `batchSize_` images; the last batch is zero-padded.
            std::vector<std::string> paths;
            for (int i = 0; i < batchSize_ && nextImage_ + i < imagePaths_.size(); ++i) {
                paths.push_back(imagePaths_[nextImage_ + i]);
            }
            Tensor4D batch = image_utils::load_batch(paths, /*grayscale=*/channels_ == 1,
                                                     cv::Size(width_, height_));
            batch.data.resize(batchBytes() / sizeof(float), 0.0f);

            CUDA_CHECK(cudaMemcpy(deviceInput_, batch.data.data(), batchBytes(), cudaMemcpyHostToDevice));
            nextImage_ += paths.size();
            bindings[0] = deviceInput_;
            std::cout << "Calibration: " << nextImage_ << "/" << imagePaths_.size() << " images" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Calibration batch failed: " << e.what() << std::endl;
            return false;
        }
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        length = cache_.size();
        return length ? cache_.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        cache_.assign(static_cast<const char*>(cache), static_cast<const char*>(cache) + length);
    }

private:
    size_t batchBytes() const {
        return static_cast<size_t>(batchSize_) * channels_ * height_ * width_ * sizeof(float);
    }

    std::vector<std::string> imagePaths_;
    int batchSize_, channels_, height_, width_;
    size_t nextImage_ = 0;
    void* deviceInput_ = nullptr;
    std::vector<char> cache_;
};

// Replaces every dynamic (-1) axis of `dims` with the matching value from `shape`.
Dims resolveDims(Dims dims, const Dims& shape) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1) dims.d[i] = shape.d[i];
    }
    return dims;
}

}  // namespace

bool build_plan(const BuildOptions& opt) {
    if (opt.precision != "fp32" && opt.precision != "fp16" && opt.precision != "int8") {
        std::cerr << "Unknown precision '" << opt.precision << "' (use fp32, fp16 or int8)." << std::endl;
        return false;
    }

    TrtLogger logger;
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    // ---- Parse the ONNX model -------------------------------------------------
    if (!parser->parseFromFile(opt.onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model: " << opt.onnxPath << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    if (network->getNbInputs() <= 0 || network->getNbOutputs() <= 0) {
        std::cerr << "The network has no inputs or outputs." << std::endl;
        return false;
    }

    // ---- Precision ------------------------------------------------------------
    // fp32 needs no flag (TF32 is enabled by default on supported GPUs).
    std::unique_ptr<Int8Calibrator> calibrator;
    if (opt.precision == "fp16") {
        config->setFlag(BuilderFlag::kFP16);
    } else if (opt.precision == "int8") {
        if (!builder->platformHasFastInt8()) {
            std::cerr << "INT8 is not supported on this GPU." << std::endl;
            return false;
        }
        if (network->getNbInputs() != 1) {
            std::cerr << "INT8 calibration only supports networks with a single input." << std::endl;
            return false;
        }

        // Calibration images are resized to the shape the engine is optimised for.
        const Dims inputDims = network->getInput(0)->getDimensions();
        Dims calibDims = opt.dynamic ? resolveDims(inputDims, opt.shapeRange.optDims) : inputDims;
        if (calibDims.nbDims != 4 || calibDims.d[0] < 1 || calibDims.d[2] < 1 || calibDims.d[3] < 1) {
            std::cerr << "INT8 calibration needs a fixed NCHW input shape " << calibDims
                      << " - enable dynamic shapes and provide opt dims." << std::endl;
            return false;
        }

        const auto images = image_utils::list_images(opt.calibrationDir);
        if (images.empty()) {
            std::cerr << "No calibration images found in " << opt.calibrationDir << std::endl;
            return false;
        }
        std::cout << "INT8 calibration: " << images.size() << " images, batch size " << calibDims.d[0] << std::endl;

        calibrator = std::make_unique<Int8Calibrator>(images, calibDims.d[0], calibDims.d[1],
                                                      calibDims.d[2], calibDims.d[3]);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kGPU_FALLBACK);  // layers without INT8 support run in higher precision
        config->setProfilingVerbosity(ProfilingVerbosity::kVERBOSE);
        config->setInt8Calibrator(calibrator.get());

        // Fixed ranges for the network I/O (inputs are ImageNet-normalised).
        for (int i = 0; i < network->getNbInputs(); ++i) network->getInput(i)->setDynamicRange(-2.0f, 2.0f);
        for (int i = 0; i < network->getNbOutputs(); ++i) network->getOutput(i)->setDynamicRange(-2.0f, 2.0f);
    }

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);  // 4 GiB
    config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);                  // for reproducible benchmarking
    config->setFlag(BuilderFlag::kSTRICT_TYPES);                          // deprecated, kept for identical engines

    // ---- Dynamic shapes: one profile covering all inputs ----------------------
    if (opt.dynamic) {
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); ++i) {
            ITensor* input = network->getInput(i);
            const Dims dims = input->getDimensions();
            std::cout << "Input '" << input->getName() << "' ONNX shape: " << dims << std::endl;

            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, resolveDims(dims, opt.shapeRange.minDims));
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, resolveDims(dims, opt.shapeRange.optDims));
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, resolveDims(dims, opt.shapeRange.maxDims));
        }
        config->addOptimizationProfile(profile);
    }

    // ---- Build and save -------------------------------------------------------
    std::cout << "Building TensorRT engine (" << opt.precision << "), this can take several minutes..." << std::endl;
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Engine build failed." << std::endl;
        return false;
    }

    const fs::path planPath(opt.planPath);
    if (planPath.has_parent_path()) fs::create_directories(planPath.parent_path());

    std::ofstream out(planPath, std::ios::binary);
    out.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    out.close();
    if (!out) {
        std::cerr << "Could not write plan file: " << opt.planPath << std::endl;
        return false;
    }

    std::cout << "Saved plan (" << plan->size() / (1024.0 * 1024.0) << " MiB): " << opt.planPath << std::endl;
    return true;
}

}  // namespace trt_generation
