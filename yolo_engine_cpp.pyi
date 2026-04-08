"""Type stub for the C++ yolo_engine_cpp module (built from cpp/src/)."""
import numpy as np
from typing import List, TypedDict

class DetectionResult(TypedDict):
    classid: int
    score: float
    bbox: np.ndarray  # shape (4,), float32, [x1, y1, x2, y2]

class YoloEngine:
    """YOLO TensorRT C++ inference engine with fused CUDA preprocessing."""

    input_width: int
    input_height: int

    def __init__(
        self,
        engine_path: str,
        conf_thresh: float = 0.7,
        iou_thresh: float = 0.1,
    ) -> None: ...

    def inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on a BGR uint8 numpy image.

        Args:
            image: numpy array (H, W, 3), uint8, BGR format (from cv2).

        Returns:
            List of dicts with keys: classid (int), score (float), bbox (ndarray).
        """
        ...
