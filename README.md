# YOLO TensorRT C++ Engine for Python

High-performance YOLO inference engine written in C++/CUDA, callable from Python via pybind11.

Designed for AOI (Automated Optical Inspection) on Nvidia Jetson ORIN.

## Why

The TensorRT inference and pre/post processing run in C++ with a fused CUDA kernel, eliminating Python overhead where it matters. All other logic (camera, RS232, UI, pass/fail) stays in Python for flexibility.

| Component | Language | Why |
|---|---|---|
| Preprocess (resize + RGB + CHW + normalize) | CUDA kernel | 1 kernel vs 3-4 separate ops in Python |
| TensorRT inference | C++ | Direct API, no PyCUDA wrapper overhead |
| NMS postprocess | C++ | Pre-allocated buffers, zero malloc per frame |
| Camera, UI, RS232, logic | Python | Easy to maintain, fast to change on-site |

## Architecture

```
Python (your code)                        C++ / CUDA (this engine)
┌────────────────────────┐               ┌─────────────────────────────────┐
│ camera.getframe()      │               │                                 │
│         │              │   one call    │  1. cudaMemcpyAsync (H2D)       │
│         v              │ ──────────>   │  2. preprocess CUDA kernel      │
│ engine.inference(frame)│               │  3. TensorRT enqueueV3          │
│         │              │   returns     │  4. cudaMemcpyAsync (D2H)       │
│         v              │ <──────────   │  5. NMS postprocess             │
│ pass/fail logic        │  list[dict]   │                                 │
│ RS232 send result      │               └─────────────────────────────────┘
│ UI update              │                numpy ↔ cv::Mat is zero-copy
│ save image             │                via pybind11 (shared pointer)
└────────────────────────┘
```

### How It Works

When you call `engine.inference(image)` from Python, the following happens entirely in C++/CUDA with zero Python overhead:

**Step 1 — Upload image to GPU**

The numpy array from OpenCV is wrapped as a `cv::Mat` via pybind11 **without copying** (they share the same memory pointer). The raw pixel data is then uploaded to GPU memory using `cudaMemcpyAsync` on a dedicated CUDA stream.

**Step 2 — Fused CUDA preprocessing kernel** (`preprocess.cu`)

A single custom CUDA kernel performs **all** preprocessing in one GPU pass:
- **Bilinear resize** — each thread computes one output pixel by interpolating 4 source pixels
- **BGR to RGB** — channel swap during the same read/write
- **HWC to CHW** — layout transformation (height-width-channel to channel-height-width)
- **Normalize /255.0** — integer to float conversion

In Python, this requires 3-4 separate operations (cv2.resize, cv2.cvtColor, np.transpose, /255.0), each allocating intermediate buffers. The fused kernel does it all with **zero intermediate memory** and **one kernel launch**.

**Step 3 — TensorRT inference** (`yolo_engine.cpp`)

The preprocessed tensor is fed to TensorRT via `context->enqueueV3(stream)`. This runs the YOLO neural network on the GPU. The engine file (`.engine`) is loaded and deserialized once at construction time. All I/O buffer addresses are bound once and never change.

**Step 4 — Download results**

The raw model output is copied back to CPU pinned memory (`cudaMallocHost`) via async DMA transfer. Pinned memory enables faster transfers because it bypasses the OS page cache.

**Step 5 — NMS postprocessing** (`postprocess.cpp`)

Post-processing runs on CPU with pre-allocated buffers:
1. **Confidence filter** — scan all detection candidates, keep only those above threshold
2. **Coordinate conversion** — transform from model coordinates (cx, cy, w, h) to image coordinates (x1, y1, x2, y2)
3. **Non-Maximum Suppression** — remove overlapping boxes per class using IoU

The `candidates` vector is pre-reserved (512 slots) to avoid heap allocation during inference. The `results` vector is reused across frames.

### Key C++ Techniques

| Technique | Where | What it does |
|---|---|---|
| **Fused CUDA kernel** | `preprocess.cu` | Combines 4 operations into 1 kernel launch with zero intermediate buffers |
| **Bilinear interpolation on GPU** | `preprocess.cu` | Each CUDA thread reads 4 source pixels and computes weighted average |
| **Pinned (page-locked) memory** | `yolo_engine.cpp` | `cudaMallocHost` for output buffer — enables faster async DMA transfers |
| **CUDA stream async** | `yolo_engine.cpp` | Upload, inference, and download are queued on a stream, overlapping where possible |
| **Pre-allocated buffers** | `yolo_engine.cpp`, `postprocess.cpp` | All GPU and CPU buffers allocated once in constructor, reused every frame |
| **Lazy GPU source buffer** | `yolo_engine.cpp` | Source image GPU buffer only reallocates if image size changes (rare in AOI with fixed cameras) |
| **Zero-copy pybind11 bridge** | `pybind_wrapper.cpp` | numpy array wraps directly as `cv::Mat` — no data copy at the Python/C++ boundary |
| **`std::vector::reserve`** | `postprocess.cpp` | Pre-sized candidate buffer avoids heap allocations during NMS |
| **`__restrict__` pointers** | `preprocess.cu` | Tells CUDA compiler that input/output don't alias, enabling more aggressive optimization |

### Python vs C++ Performance Comparison

| Stage | Pure Python (PyCUDA) | This C++ Engine |
|---|---|---|
| Preprocess | ~3-5 ms (3 GPU ops + CPU transpose + CPU normalize) | ~0.5-1 ms (1 fused kernel) |
| H2D transfer | ~0.3 ms (PyCUDA wrapper) | ~0.2 ms (direct CUDA API) |
| TRT inference | ~5 ms | ~5 ms (same GPU work) |
| D2H transfer | ~0.1 ms | ~0.1 ms |
| Postprocess | ~1-3 ms (NumPy, ~15 allocations) | ~0.2-0.5 ms (zero allocation) |
| **Total** | **~10-14 ms** | **~6-7 ms** |
| **Worst case** | ~18 ms (GC spike) | ~7 ms (deterministic) |

## Project Structure

```
best_use_engine/
├── build.sh                    # Build script (run on ORIN)
├── requirements.txt            # Python dependencies
├── YoloEngine.py               # Python import wrapper
├── yolo_engine_cpp.pyi         # Type stub for IDE autocomplete
├── example.py                  # Usage examples
└── cpp/
    ├── CMakeLists.txt
    └── src/
        ├── preprocess.cu       # Fused CUDA preprocessing kernel
        ├── preprocess.cuh
        ├── postprocess.cpp     # NMS with pre-allocated buffers
        ├── postprocess.h
        ├── yolo_engine.cpp     # TensorRT engine wrapper
        ├── yolo_engine.h
        └── pybind_wrapper.cpp  # Python <-> C++ bridge (zero-copy)
```

## Prerequisites

- Nvidia Jetson ORIN (JetPack 6.x)
- TensorRT 10.x
- CUDA
- OpenCV (with CUDA support recommended)
- Python 3.10
- CMake >= 3.18

## Build

On the Jetson ORIN:

```bash
pip install -r requirements.txt
bash build.sh
```

This compiles the C++ code and places `yolo_engine_cpp.*.so` in the project root.

## Usage

### Drop-in replacement

In your existing project, change one line:

```python
# Before (pure Python TensorRT)
from YoloV9_TRT10 import YoloEngine

# After (C++ engine)
from YoloEngine import YoloEngine
```

Everything else stays the same.

### Quick start

```python
import cv2
from YoloEngine import YoloEngine

engine = YoloEngine("weights/best.engine", conf_thresh=0.7, iou_thresh=0.1)

image = cv2.imread("test.jpg")
results = engine.inference(image)

for det in results:
    print(det["classid"], det["score"], det["bbox"])
```

### Return format

`engine.inference(image)` returns a list of dicts:

```python
[
    {"classid": 0, "score": 0.95, "bbox": array([x1, y1, x2, y2], dtype=float32)},
    {"classid": 1, "score": 0.87, "bbox": array([x1, y1, x2, y2], dtype=float32)},
]
```

### API

```python
engine = YoloEngine(engine_path, conf_thresh=0.7, iou_thresh=0.1)

engine.inference(image)    # image: numpy BGR uint8 (H, W, 3)
engine.input_width         # model input width
engine.input_height        # model input height
```
