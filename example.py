"""
Example: C++ YOLO engine used from Python.

Same interface as YoloV9_TRT10.py — drop-in replacement.
Just change the import line in your existing project.

Before:  from YoloV9_TRT10 import YoloEngine
After:   from YoloEngine import YoloEngine
"""
import cv2
import time
import numpy as np
from YoloEngine import YoloEngine


# =====================================================
# 1. Create engine (same as before)
# =====================================================
engine = YoloEngine(
    engine_path="weights/best.engine",
    conf_thresh=0.7,
    iou_thresh=0.1,
)
print(f"Model input: {engine.input_width}x{engine.input_height}")


# =====================================================
# 2. Single image inference
# =====================================================
image = cv2.imread("test.jpg")
results = engine.inference(image)

# Results format is identical to YoloV9_TRT10:
#   [{"classid": int, "score": float, "bbox": ndarray[x1,y1,x2,y2]}, ...]
for det in results:
    print(f"  class={det['classid']}  score={det['score']:.3f}  "
          f"bbox={det['bbox'].astype(int)}")


# =====================================================
# 3. Camera loop (typical AOI usage)
# =====================================================
categories = ["NG", "OK"]  # your class names

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference — C++ handles GPU preprocess + TRT + NMS
    results = engine.inference(frame)

    # Your Python logic — pass/fail, RS232, save, etc.
    for det in results:
        x1, y1, x2, y2 = det["bbox"].astype(int)
        label = f"{categories[det['classid']]}: {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("AOI", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# =====================================================
# 4. Benchmark
# =====================================================
image = cv2.imread("test.jpg")

# Warmup
for _ in range(10):
    engine.inference(image)

# Timed runs
times = []
for _ in range(100):
    t0 = time.perf_counter()
    engine.inference(image)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times = np.array(times)
print(f"\nBenchmark (100 runs):")
print(f"  Mean:   {times.mean():.2f} ms")
print(f"  Median: {np.median(times):.2f} ms")
print(f"  Min:    {times.min():.2f} ms")
print(f"  FPS:    {1000 / times.mean():.1f}")
