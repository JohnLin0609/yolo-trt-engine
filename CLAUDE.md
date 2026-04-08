# CLAUDE.md

## Project Overview

YOLO TensorRT C++ inference engine with Python bindings (pybind11). Built for AOI production on Nvidia Jetson ORIN.

**Hybrid architecture**: C++/CUDA handles the performance-critical inference pipeline, Python handles everything else (camera, RS232, UI, business logic).

## Tech Stack

- **C++17 / CUDA** — engine core (`cpp/src/`)
- **TensorRT 10.x** — neural network inference
- **pybind11** — zero-copy Python <-> C++ bridge
- **OpenCV** — image I/O only (preprocessing is done by custom CUDA kernel)
- **Python 3.10** — host language
- **CMake 3.18+** — build system
- **Target hardware**: Nvidia Jetson ORIN (SM 8.7), also supports Xavier (SM 7.2)

## Key Files

| File | Purpose |
|---|---|
| `cpp/src/preprocess.cu` | Fused CUDA kernel: resize + BGR→RGB + HWC→CHW + /255.0 in one pass |
| `cpp/src/yolo_engine.cpp` | TensorRT engine loading, buffer management, inference pipeline |
| `cpp/src/postprocess.cpp` | Confidence filter + NMS with pre-allocated buffers |
| `cpp/src/pybind_wrapper.cpp` | Python bindings — numpy ↔ cv::Mat zero-copy bridge |
| `YoloEngine.py` | Python import wrapper |
| `yolo_engine_cpp.pyi` | Type stub for IDE autocomplete |
| `build.sh` | Build script — compiles C++ and copies .so to project root |

## Build

Must be built on the Jetson ORIN (requires TensorRT, CUDA):

```bash
pip install -r requirements.txt
bash build.sh
```

## Architecture Decisions

- **Fused preprocessing kernel** instead of separate OpenCV calls — eliminates intermediate GPU buffers and reduces kernel launches from 4 to 1
- **Pre-allocated all buffers at init** — zero heap allocation during inference for deterministic latency
- **Pinned host memory** (`cudaMallocHost`) for output — faster async DMA transfers
- **Lazy GPU source buffer** — only reallocates if image size changes (fixed cameras in AOI)
- **pybind11 zero-copy** — numpy array wraps as cv::Mat sharing the same pointer, no data copy at boundary
- **Python wrapper returns `list[dict]`** — same format as the original `YoloV9_TRT10.py` for drop-in replacement

## Coding Conventions

- C++ files use `snake_case` for functions, `PascalCase` for classes, trailing underscore for member variables (`input_w_`)
- CUDA kernels use `__restrict__` pointers for compiler optimization
- Python follows the existing AOI project style (matches `RS-Bushing-Production`)
- The return format `{"classid": int, "score": float, "bbox": ndarray}` must stay compatible with existing AOI projects

## Important Notes

- The CUDA kernel channel order (BGR→RGB) matches the original `YoloV9_TRT10.py` preprocessing — do not change without verifying the model's expected input format
- `CMAKE_CUDA_ARCHITECTURES` is set to `72 87` (Xavier + ORIN) — update if targeting other Jetson hardware
- The `.engine` file is platform-specific — must be generated on the same GPU architecture where it will run
- This project cannot be built on the dev machine (x86_64) — it requires ORIN's TensorRT/CUDA libraries
