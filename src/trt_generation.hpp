// Builds a serialized TensorRT engine ("plan") from an ONNX model.
#pragma once

#include <NvInfer.h>

#include <string>

namespace trt_generation {

// Min / opt / max shapes (NCHW) for an optimisation profile with dynamic axes.
struct DynamicShapeRange {
    nvinfer1::Dims4 minDims{1, 3, 256, 256};
    nvinfer1::Dims4 optDims{1, 3, 512, 512};
    nvinfer1::Dims4 maxDims{1, 3, 1024, 1024};
};

struct BuildOptions {
    std::string onnxPath;
    std::string planPath;
    std::string precision = "fp16";  // "fp32", "fp16" or "int8"

    bool dynamic = false;            // build with dynamic input shapes
    DynamicShapeRange shapeRange;    // only used when `dynamic` is true

    // INT8 only: folder of representative images used for calibration.
    std::string calibrationDir;
};

// Parses the ONNX model, builds an engine and writes it to `options.planPath`.
// Returns true on success; errors are reported on stderr.
bool build_plan(const BuildOptions& options);

}  // namespace trt_generation
