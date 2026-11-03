// onnx2trt - build TensorRT engines from ONNX models and benchmark them.
//
//   onnx2trt build --onnx <file> --plan <file> [options]
//   onnx2trt infer --plan <file> --input <image|dir> [options]
//
// Run `onnx2trt --help` for the full list of flags.
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "image_utils.hpp"
#include "trt_generation.hpp"
#include "trt_inference.hpp"

namespace fs = std::filesystem;

namespace {

const char* kUsage = R"(onnx2trt - build TensorRT engines from ONNX models and benchmark them

Usage:
  onnx2trt build --onnx <file> --plan <file> [options]
  onnx2trt infer --plan <file> --input <image|directory> [options]
  onnx2trt --help

build options:
  --onnx <file>          input ONNX model                                (required)
  --plan <file>          where to write the TensorRT plan                (required)
  --precision <p>        fp32 | fp16 | int8                              (default: fp16)
  --min <N C H W>        } optimisation profile for dynamic shapes.
  --opt <N C H W>        } Give all three to enable dynamic shapes;
  --max <N C H W>        } omit all three for a static engine.
  --calib-dir <dir>      calibration images, required for --precision int8
                         (batch/image size are taken from --opt, or the model if static)

infer options:
  --plan <file>          TensorRT plan to run                            (required)
  --input <path>         an image file, or a directory of images         (required)
  --output-dir <dir>     where to write result masks                     (default: output)
  --trials <n>           timed iterations                                (default: 20)
  --warmup <n>           untimed warm-up iterations                      (default: 10)
  --resize <W H>         resize every image first (needed for mixed sizes in a directory)

Relative paths are resolved against the current working directory.
)";

// Flag -> list of values, e.g. {"--min": {"1","3","256","256"}}.
using Args = std::map<std::string, std::vector<std::string>>;

// Collects "--flag v1 v2 ..." groups. A flag's values are the tokens up to the next "--flag".
Args parseFlags(int argc, char** argv, int first) {
    Args args;
    std::string current;
    for (int i = first; i < argc; ++i) {
        const std::string token = argv[i];
        if (token.rfind("--", 0) == 0) {
            current = token;
            args[current];  // flags without values are legal but will fail validation below
        } else if (!current.empty()) {
            args[current].push_back(token);
        } else {
            throw std::runtime_error("Unexpected argument: " + token);
        }
    }
    return args;
}

void rejectUnknownFlags(const Args& args, const std::vector<std::string>& allowed) {
    for (const auto& [flag, values] : args) {
        if (std::find(allowed.begin(), allowed.end(), flag) == allowed.end()) {
            throw std::runtime_error("Unknown option: " + flag + " (see --help)");
        }
    }
}

bool has(const Args& args, const std::string& flag) { return args.count(flag) > 0; }

// Returns the single value of a flag, or `fallback` when absent. An empty `fallback`
// marks the flag as required.
std::string single(const Args& args, const std::string& flag, const std::string& fallback = "") {
    const auto it = args.find(flag);
    if (it == args.end()) {
        if (fallback.empty()) throw std::runtime_error("Missing required option: " + flag);
        return fallback;
    }
    if (it->second.size() != 1) throw std::runtime_error(flag + " expects exactly one value");
    return it->second.front();
}

int toInt(const std::string& flag, const std::string& text) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        throw std::runtime_error(flag + ": '" + text + "' is not an integer");
    }
}

// Reads "--flag N C H W".
nvinfer1::Dims4 dims4(const Args& args, const std::string& flag) {
    const auto& v = args.at(flag);
    if (v.size() != 4) throw std::runtime_error(flag + " expects four integers: N C H W");
    return nvinfer1::Dims4(toInt(flag, v[0]), toInt(flag, v[1]), toInt(flag, v[2]), toInt(flag, v[3]));
}

// ------------------------------------------------------------ subcommands ---

int runBuild(const Args& args) {
    rejectUnknownFlags(args, {"--onnx", "--plan", "--precision", "--min", "--opt", "--max", "--calib-dir"});

    trt_generation::BuildOptions opt;
    opt.onnxPath = single(args, "--onnx");
    opt.planPath = single(args, "--plan");
    opt.precision = single(args, "--precision", "fp16");

    // Dynamic shapes need all three profile points.
    const int profileFlags = has(args, "--min") + has(args, "--opt") + has(args, "--max");
    if (profileFlags != 0 && profileFlags != 3) {
        throw std::runtime_error("Dynamic shapes need --min, --opt and --max together.");
    }
    if (profileFlags == 3) {
        opt.dynamic = true;
        opt.shapeRange.minDims = dims4(args, "--min");
        opt.shapeRange.optDims = dims4(args, "--opt");
        opt.shapeRange.maxDims = dims4(args, "--max");
    }

    if (opt.precision == "int8") opt.calibrationDir = single(args, "--calib-dir");

    return trt_generation::build_plan(opt) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int runInfer(const Args& args) {
    rejectUnknownFlags(args, {"--plan", "--input", "--output-dir", "--trials", "--warmup", "--resize"});

    trt_inference::Options opt;
    opt.planPath = single(args, "--plan");
    opt.outputDir = single(args, "--output-dir", "output");
    opt.numTrials = toInt("--trials", single(args, "--trials", "20"));
    opt.warmupIterations = toInt("--warmup", single(args, "--warmup", "10"));
    if (opt.numTrials < 1) throw std::runtime_error("--trials must be at least 1");

    cv::Size resize;  // empty = keep original size
    if (has(args, "--resize")) {
        const auto& v = args.at("--resize");
        if (v.size() != 2) throw std::runtime_error("--resize expects two integers: W H");
        resize = cv::Size(toInt("--resize", v[0]), toInt("--resize", v[1]));
    }

    // A directory is run as one batch, a file as a single image.
    const std::string input = single(args, "--input");
    const Tensor4D tensor = fs::is_directory(input)
                                ? image_utils::load_batch(image_utils::list_images(input), false, resize)
                                : image_utils::load_image(input, false, resize);
    std::cout << "Input tensor (NCHW): " << tensor.n << " x " << tensor.c << " x " << tensor.h << " x "
              << tensor.w << std::endl;

    return trt_inference::run_segmentation(opt, tensor) ? EXIT_SUCCESS : EXIT_FAILURE;
}

void setEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::cout << kUsage;
        return argc < 2 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    // Load CUDA kernels on demand: faster start-up, lower memory use.
    setEnv("CUDA_MODULE_LOADING", "LAZY");

    try {
        const std::string command = argv[1];
        const Args args = parseFlags(argc, argv, 2);
        if (command == "build") return runBuild(args);
        if (command == "infer") return runInfer(args);
        throw std::runtime_error("Unknown command '" + command + "' (expected build or infer)");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\nRun 'onnx2trt --help' for usage." << std::endl;
        return EXIT_FAILURE;
    }
}
