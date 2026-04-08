"""
YOLO TensorRT Engine — Python interface to C++ backend.

Usage:
    from YoloEngine import YoloEngine

    engine = YoloEngine("model.engine")
    results = engine.inference(image)

    for det in results:
        print(det["classid"], det["score"], det["bbox"])

Build the C++ module first:
    bash build.sh
"""
import os
import sys

# Add current directory to path so yolo_engine_cpp.so can be found
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from yolo_engine_cpp import YoloEngine

__all__ = ["YoloEngine"]
