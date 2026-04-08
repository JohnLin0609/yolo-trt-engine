#pragma once
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include "postprocess.h"

class YoloEngine {
public:
    // Load engine from file. All GPU buffers are allocated once here.
    explicit YoloEngine(const std::string& engine_path,
                        float conf_thresh = 0.7f,
                        float iou_thresh = 0.1f);
    ~YoloEngine();

    // Not copyable, but movable
    YoloEngine(const YoloEngine&) = delete;
    YoloEngine& operator=(const YoloEngine&) = delete;

    // Run full pipeline: preprocess (GPU) -> inference -> postprocess
    // image: BGR uint8 cv::Mat (any size)
    // Returns detections in original image coordinates
    std::vector<Detection> inference(const cv::Mat& image);

    // Accessors
    int input_width()  const { return input_w_; }
    int input_height() const { return input_h_; }

private:
    void load_engine(const std::string& path);
    void allocate_buffers();

    // TensorRT
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA
    cudaStream_t stream_ = nullptr;

    // I/O buffers (GPU)
    void* d_input_  = nullptr;  // Model input tensor (float32, CHW)
    void* d_output_ = nullptr;  // Model output tensor (float32)

    // Source image buffer (GPU) for preprocessing kernel
    uint8_t* d_src_ = nullptr;
    size_t d_src_capacity_ = 0;  // Current allocated bytes for d_src_

    // Host output buffer (pinned memory for fast D2H)
    float* h_output_ = nullptr;

    // Model dimensions
    int input_w_ = 0;
    int input_h_ = 0;
    int num_detections_ = 0;
    int num_classes_ = 0;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;

    // Thresholds
    float conf_thresh_;
    float iou_thresh_;

    // Reusable results buffer (avoid reallocation per frame)
    std::vector<Detection> results_;
};
