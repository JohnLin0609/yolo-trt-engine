#pragma once
#include <vector>
#include <cstdint>

struct Detection {
    float x1, y1, x2, y2;  // Bounding box (in original image coordinates)
    float score;
    int class_id;
};

// Post-process raw YOLO output into final detections.
// output: raw engine output buffer, shape (num_classes+4) * num_detections (flattened)
// num_detections: number of detection candidates (e.g. 8400)
// num_classes: number of object classes
// origin_h, origin_w: original image dimensions (for coordinate scaling)
// input_h, input_w: model input dimensions
// conf_thresh: confidence threshold
// iou_thresh: NMS IoU threshold
// results: output vector (pre-allocated, cleared inside)
void postprocess(
    const float* output,
    int num_detections,
    int num_classes,
    int origin_h, int origin_w,
    int input_h, int input_w,
    float conf_thresh,
    float iou_thresh,
    std::vector<Detection>& results);
