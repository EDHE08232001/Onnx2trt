// Shared helpers used across the project: TensorRT logger, CUDA error checking,
// NVTX markers (optional) and a tiny 4-D tensor type.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// NVTX ranges show up in Nsight Systems. They are optional: if the header is not
// available the macros below compile to nothing.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define TRT_NVTX_PUSH(name) nvtxRangePushA(name)
#define TRT_NVTX_POP() nvtxRangePop()
#else
#define TRT_NVTX_PUSH(name) ((void)0)
#define TRT_NVTX_POP() ((void)0)
#endif

// Throws std::runtime_error if a CUDA call fails.
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error in " #call ": ") +      \
                                     cudaGetErrorString(err_));                       \
        }                                                                             \
    } while (0)

// Prints TensorRT messages. Warnings and errors only, unless verbose is set.
class TrtLogger : public nvinfer1::ILogger {
public:
    explicit TrtLogger(bool verbose = false) : verbose_(verbose) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (verbose_ || severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }

private:
    bool verbose_;
};

// A dense float tensor in NCHW layout stored on the host.
struct Tensor4D {
    int n = 0, c = 0, h = 0, w = 0;
    std::vector<float> data;

    size_t size() const { return static_cast<size_t>(n) * c * h * w; }
};

// Lives in nvinfer1 so argument-dependent lookup finds it from any namespace.
namespace nvinfer1 {
inline std::ostream& operator<<(std::ostream& os, const Dims& dims) {
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        os << dims.d[i] << (i + 1 < dims.nbDims ? ", " : "");
    }
    return os << ")";
}
}  // namespace nvinfer1
