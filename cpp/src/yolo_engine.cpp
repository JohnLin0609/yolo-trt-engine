#include "yolo_engine.h"
#include "preprocess.cuh"
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstring>

// TensorRT logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};
static TrtLogger g_logger;

// ---- Custom deleters for TRT unique_ptrs ----
struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
};

// =============================================================================
// Constructor
// =============================================================================
YoloEngine::YoloEngine(const std::string& engine_path,
                       float conf_thresh, float iou_thresh)
    : conf_thresh_(conf_thresh), iou_thresh_(iou_thresh)
{
    cudaStreamCreate(&stream_);
    load_engine(engine_path);
    allocate_buffers();

    std::cout << "[YoloEngine] Ready. Input: " << input_w_ << "x" << input_h_
              << ", Detections: " << num_detections_
              << ", Classes: " << num_classes_ << std::endl;
}

// =============================================================================
// Destructor — free all GPU resources
// =============================================================================
YoloEngine::~YoloEngine()
{
    if (h_output_) cudaFreeHost(h_output_);
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (d_src_)    cudaFree(d_src_);
    if (stream_)   cudaStreamDestroy(stream_);
}

// =============================================================================
// Load serialized engine
// =============================================================================
void YoloEngine::load_engine(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open engine file: " + path);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    std::cout << "[YoloEngine] Loading engine: " << path
              << " (" << size / (1024 * 1024) << " MB)" << std::endl;

    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
}

// =============================================================================
// Allocate I/O buffers (once, at init)
// =============================================================================
void YoloEngine::allocate_buffers()
{
    int num_io = engine_->getNbIOTensors();

    for (int i = 0; i < num_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto shape = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);

        // Compute total element count
        size_t vol = 1;
        for (int d = 0; d < shape.nbDims; ++d) {
            vol *= shape.d[d];
        }
        size_t bytes = vol * sizeof(float);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            // Input: (1, 3, H, W)
            input_h_ = shape.d[2];
            input_w_ = shape.d[3];
            input_bytes_ = bytes;

            cudaMalloc(&d_input_, bytes);
            context_->setTensorAddress(name, d_input_);

            std::cout << "[YoloEngine] Input '" << name << "': "
                      << shape.d[0] << "x" << shape.d[1] << "x"
                      << shape.d[2] << "x" << shape.d[3] << std::endl;
        } else {
            // Output: (1, 4+num_classes, num_detections)
            num_classes_ = shape.d[1] - 4;
            num_detections_ = shape.d[2];
            output_bytes_ = bytes;

            cudaMalloc(&d_output_, bytes);
            // Pinned host memory for fast async D2H transfer
            cudaMallocHost(&h_output_, bytes);
            context_->setTensorAddress(name, d_output_);

            std::cout << "[YoloEngine] Output '" << name << "': "
                      << shape.d[0] << "x" << shape.d[1] << "x"
                      << shape.d[2] << std::endl;
        }
    }
}

// =============================================================================
// Full inference pipeline
// =============================================================================
std::vector<Detection> YoloEngine::inference(const cv::Mat& image)
{
    int src_h = image.rows;
    int src_w = image.cols;
    size_t src_bytes = (size_t)src_h * src_w * 3 * sizeof(uint8_t);

    // --- Ensure GPU source buffer is large enough ---
    // Only reallocate if the image size changed (rare in AOI — fixed camera)
    if (src_bytes > d_src_capacity_) {
        if (d_src_) cudaFree(d_src_);
        cudaMalloc(&d_src_, src_bytes);
        d_src_capacity_ = src_bytes;
    }

    // --- Upload source image to GPU (async) ---
    cudaMemcpyAsync(d_src_, image.data, src_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // --- Fused GPU preprocessing ---
    // Single kernel: resize + BGR→RGB + HWC→CHW + /255.0
    preprocess_gpu(d_src_, src_h, src_w,
                   (float*)d_input_, input_h_, input_w_,
                   stream_);

    // --- TensorRT inference ---
    context_->enqueueV3(stream_);

    // --- Download output (async) ---
    cudaMemcpyAsync(h_output_, d_output_, output_bytes_,
                    cudaMemcpyDeviceToHost, stream_);

    // --- Synchronize ---
    cudaStreamSynchronize(stream_);

    // --- CPU post-processing ---
    postprocess(h_output_, num_detections_, num_classes_,
                src_h, src_w, input_h_, input_w_,
                conf_thresh_, iou_thresh_, results_);

    return results_;
}
