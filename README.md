# onnx2trt

A small C++ command-line tool that converts **ONNX** models into **NVIDIA TensorRT** engines
(FP32 / FP16 / INT8, static or dynamic shapes) and benchmarks them on images. A Python helper
exports PyTorch models to ONNX.

It builds with **CMake + VS Code** on Windows and Linux. Visual Studio (the IDE) is not required.

```
PyTorch model ──(python/onnx_converter.py)──▶ model.onnx ──(onnx2trt build)──▶ model.plan ──(onnx2trt infer)──▶ latency + masks
```

## Repository layout

```
.
├── CMakeLists.txt          Build definition
├── CMakePresets.json       Ready-made configure/build presets (Windows + Linux)
├── LocalConfig.cmake.example  Template for machine-specific dependency paths
├── src/
│   ├── main.cpp            Command-line interface (build / infer)
│   ├── trt_generation.*    ONNX -> TensorRT plan, incl. INT8 calibration
│   ├── trt_inference.*     Load a plan, benchmark it, save segmentation masks
│   ├── image_utils.*       Image discovery and preprocessing (OpenCV)
│   └── common.hpp          Logger, CUDA error checks, Tensor4D
├── python/
│   ├── onnx_converter.py   PyTorch -> ONNX export
│   └── requirements.txt
├── models/                 Put .onnx files here (contents are git-ignored)
├── plans/                  TensorRT plans are written here (git-ignored)
├── data/calibration/       Images for INT8 calibration (git-ignored)
└── .vscode/                Shared VS Code settings, tasks and debug configs
```

The `models/`, `plans/` and `data/` folders are only suggestions. Every path is passed on the
command line, so files can live anywhere.

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| NVIDIA GPU + driver | | |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 11.x / 12.x | Must match your TensorRT build |
| [TensorRT](https://developer.nvidia.com/tensorrt) | **8.x** (tested 8.5.3.1) | TensorRT 10 is *not* supported (the 8.x binding API is used) |
| [OpenCV](https://opencv.org/releases/) | 4.x | The stock build is enough, no contrib modules |
| CMake | 3.21+ | |
| Ninja | any | Bundled with the VS Code CMake Tools extension on most setups |
| C++17 compiler | | **Windows:** MSVC via *Build Tools for Visual Studio* (command-line tools only, no IDE).<br>**Linux:** GCC or Clang |
| Python 3.9+ | | Only for exporting PyTorch models |

## Setup

### 1. Tell CMake where the dependencies are

Pick **one** of these. CMake checks them in this order: `-D` flag, then `LocalConfig.cmake`, then environment variable.

**Option A - `LocalConfig.cmake` (no environment variables, recommended on Windows)**

Copy the template and edit the paths. The copy is git-ignored, so it stays machine-specific.

```powershell
copy LocalConfig.cmake.example LocalConfig.cmake
```

```cmake
set(TENSORRT_ROOT "C:/TensorRT-8.5.3.1")   # folder containing include/ and lib/
set(OpenCV_DIR    "C:/opencv/build")       # folder containing OpenCVConfig.cmake
```

Use forward slashes. CUDA is found automatically from the CUDA installer; if not, set `CUDAToolkit_ROOT` in the same file.

**Option B - environment variables**

| Variable | Points to | Example |
|---|---|---|
| `TENSORRT_ROOT` | TensorRT folder containing `include/` and `lib/` | `C:\TensorRT-8.5.3.1` or `/opt/TensorRT-8.5.3.1` |
| `OpenCV_DIR` | Folder containing `OpenCVConfig.cmake` | `C:\opencv\build` or `/usr/lib/cmake/opencv4` |
| `CUDA_PATH` | CUDA install folder (set by the CUDA installer on Windows) | `/usr/local/cuda` |

```powershell
setx TENSORRT_ROOT "C:\TensorRT-8.5.3.1"      # Windows; restart VS Code afterwards
```
```bash
export TENSORRT_ROOT=/opt/TensorRT-8.5.3.1    # Linux
```

**Option C - `-D` flags**, see "Without presets" below.

### 2. Build

**In VS Code (recommended)**

1. Install the extensions suggested on opening the folder: *C/C++*, *CMake Tools*, *Python*.
2. Windows only: install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   with the **Desktop development with C++** workload. No Visual Studio IDE is needed.
3. Open the folder, then `Ctrl+Shift+P` → **CMake: Select Configure Preset** → pick
   `windows-release` or `linux-release`.
4. Press **F7** (or **CMake: Build**).

The preset uses Ninja and, on Windows, CMake Tools sets up the MSVC environment for you.
If you use environment variables and VS Code was started before they were set, restart it.

**From a terminal**

```bash
cmake --preset linux-release          # or windows-release (use a "Developer PowerShell for VS" on Windows)
cmake --build --preset linux-release
```

Without presets, or to override a location explicitly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTENSORRT_ROOT=/opt/TensorRT-8.5.3.1 -DOpenCV_DIR=/usr/lib/cmake/opencv4
cmake --build build
```

The executable is `build/<preset>/onnx2trt` (`onnx2trt.exe` on Windows). On Windows the
TensorRT and OpenCV DLLs are copied next to it automatically after each build.

## Usage

### Step 1 - Export a PyTorch model to ONNX

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r python/requirements.txt

python python/onnx_converter.py --output models/fcn.onnx --input-shape 1 3 512 512 --dynamic
```

| Flag | Description |
|---|---|
| `--output <file>` | Where to write the `.onnx` file (required) |
| `--input-shape N C H W` | Example input used for tracing (default `1 3 512 512`) |
| `--dynamic` | Make batch, height and width dynamic |
| `--weights <file>` | Optional PyTorch checkpoint to load instead of the pretrained torchvision weights |

The script exports torchvision's **FCN-ResNet50**. To export your own network, edit
`build_model()` in `python/onnx_converter.py`. You can skip this step if you already have an ONNX file.

### Step 2 - Build a TensorRT plan

```bash
onnx2trt build --onnx models/fcn.onnx --plan plans/fcn_fp16.plan --precision fp16
```

Dynamic shapes: give all three of `--min`, `--opt`, `--max` (each `N C H W`):

```bash
onnx2trt build --onnx models/fcn.onnx --plan plans/fcn_dyn.plan \
    --min 1 3 256 256 --opt 1 3 512 512 --max 4 3 1024 1024
```

INT8: provide representative images; they are resized to the engine's input shape
(the `--opt` shape for dynamic engines):

```bash
onnx2trt build --onnx models/fcn.onnx --plan plans/fcn_int8.plan --precision int8 \
    --calib-dir data/calibration \
    --min 1 3 256 256 --opt 4 3 512 512 --max 4 3 1024 1024
```

| Flag | Description |
|---|---|
| `--onnx <file>` | Input ONNX model (required) |
| `--plan <file>` | Output plan path; parent folders are created (required) |
| `--precision` | `fp32`, `fp16` (default) or `int8` |
| `--min` / `--opt` / `--max` | Optimisation profile for dynamic shapes (all three or none) |
| `--calib-dir <dir>` | Calibration images, required for `int8` |

> Plans are specific to the GPU model and TensorRT version they were built with.

### Step 3 - Run inference

```bash
onnx2trt infer --plan plans/fcn_fp16.plan --input data/sample.jpg
onnx2trt infer --plan plans/fcn_dyn.plan  --input data/images/ --resize 512 512 --output-dir output
```

A directory is processed as **one batch**; the engine must support that batch size.
The tool prints the average latency, the output tensor statistics, and writes the arg-max
class map of the last output as `output/seg_<i>.png`.

| Flag | Description |
|---|---|
| `--plan <file>` | Plan to run (required) |
| `--input <path>` | Image file or directory of images (required) |
| `--output-dir <dir>` | Where masks are saved (default `output`) |
| `--trials <n>` | Timed iterations (default 20) |
| `--warmup <n>` | Warm-up iterations (default 10) |
| `--resize W H` | Resize inputs first; required if images in a directory differ in size |

Images are converted to RGB, scaled to `[0, 1]` and normalised with the ImageNet mean/std,
which is what torchvision models expect. Change `load_image()` in `src/image_utils.cpp` for other models.

`onnx2trt --help` prints the same flag reference.

## Debugging in VS Code

`.vscode/launch.json` contains **onnx2trt build** and **onnx2trt infer** configurations.
Edit the `args` to point at your files, select a Debug preset (`windows-debug` / `linux-debug`),
and press **F5**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TensorRT not found` | Set `TENSORRT_ROOT` (`LocalConfig.cmake`, env var or `-DTENSORRT_ROOT=...`) and re-run CMake; delete the build folder if a stale cache persists |
| `TensorRT 10.x found, but this project needs 8.x` | Install TensorRT 8.5.x |
| `Could not find nvcc` | Install the CUDA Toolkit or set `CUDA_PATH` / `CUDAToolkit_ROOT` |
| `OpenCV` not found | Set `OpenCV_DIR` to the folder containing `OpenCVConfig.cmake` |
| Windows: program exits immediately / missing `nvinfer.dll` | Rebuild so the DLL copy step runs, or add `%TENSORRT_ROOT%\lib` to `PATH` |
| `Failed to deserialize the plan` | The plan was built with a different TensorRT version or GPU; rebuild it |
| `Input shape ... is outside the range` | Build with `--min/--opt/--max` that cover your input, or use `--resize` |
| `Image sizes differ within the batch` | Pass `--resize W H` |
| Windows: `cl.exe` not found | Install Build Tools for Visual Studio, or launch VS Code from a *Developer PowerShell* |

## Limitations

- Engines with a single input tensor; inference post-processing assumes a segmentation output `[N, classes, H, W]`.
- INT8 calibration keeps its cache in memory only; it is recomputed on every build.
- Uses the TensorRT 8.x API (`enqueueV2`, binding indices).
