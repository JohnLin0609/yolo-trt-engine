#include "postprocess.h"
#include <algorithm>
#include <numeric>
#include <cmath>

static float iou(const Detection& a, const Detection& b)
{
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (area_a + area_b - inter + 1e-16f);
}

void postprocess(
    const float* output,
    int num_detections,
    int num_classes,
    int origin_h, int origin_w,
    int input_h, int input_w,
    float conf_thresh,
    float iou_thresh,
    std::vector<Detection>& results)
{
    results.clear();

    // output layout: (4 + num_classes) rows x num_detections cols, row-major
    // Row 0: cx, Row 1: cy, Row 2: w, Row 3: h, Row 4...: class scores
    int row_len = num_detections;

    float scale_x = static_cast<float>(input_w) / origin_w;
    float scale_y = static_cast<float>(input_h) / origin_h;

    // --- Pass 1: confidence filter + coordinate conversion ---
    // Pre-sized to avoid repeated reallocation
    std::vector<Detection> candidates;
    candidates.reserve(512);  // Typical case: far fewer than num_detections pass threshold

    for (int d = 0; d < num_detections; ++d) {
        // Find best class
        int best_cls = 0;
        float best_score = output[4 * row_len + d];
        for (int c = 1; c < num_classes; ++c) {
            float s = output[(4 + c) * row_len + d];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }

        if (best_score < conf_thresh) continue;

        // Read box (cx, cy, w, h) in model coordinates
        float cx = output[0 * row_len + d];
        float cy = output[1 * row_len + d];
        float bw = output[2 * row_len + d];
        float bh = output[3 * row_len + d];

        // Convert to (x1, y1, x2, y2) in original image coordinates
        Detection det;
        det.x1 = std::clamp((cx - bw * 0.5f) / scale_x, 0.0f, (float)(origin_w - 1));
        det.y1 = std::clamp((cy - bh * 0.5f) / scale_y, 0.0f, (float)(origin_h - 1));
        det.x2 = std::clamp((cx + bw * 0.5f) / scale_x, 0.0f, (float)(origin_w - 1));
        det.y2 = std::clamp((cy + bh * 0.5f) / scale_y, 0.0f, (float)(origin_h - 1));
        det.score = best_score;
        det.class_id = best_cls;

        candidates.push_back(det);
    }

    if (candidates.empty()) return;

    // --- Pass 2: sort by score descending ---
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    // --- Pass 3: per-class NMS ---
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id != candidates[j].class_id) continue;
            if (iou(candidates[i], candidates[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
}
