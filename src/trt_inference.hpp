// Runs a serialized TensorRT engine, reports latency and saves segmentation masks.
#pragma once

#include <string>

#include "common.hpp"

namespace trt_inference {

struct Options {
    std::string planPath;
    std::string outputDir = "output";  // where the result masks are written
    int numTrials = 20;                // timed iterations
    int warmupIterations = 10;         // untimed iterations before measuring
};

// Runs the engine on `input` (NCHW, float) and prints the average latency.
// The last output tensor is treated as per-class scores [N, classes, H, W]:
// the arg-max class map is scaled to 0-255 and saved as `seg_<i>.png`.
// Returns true on success; errors are reported on stderr.
bool run_segmentation(const Options& options, const Tensor4D& input);

}  // namespace trt_inference
