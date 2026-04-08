#pragma once
#include <cuda_runtime.h>

// Fused preprocessing kernel:
//   resize (bilinear) + BGR→RGB + HWC→CHW + normalize(/255) in ONE kernel launch.
//   Zero intermediate buffers.
//
// src: input image (uint8, HWC, BGR)
// dst: output tensor (float32, CHW, RGB, normalized [0,1])
void preprocess_gpu(
    const uint8_t* d_src, int src_h, int src_w,
    float* d_dst, int dst_h, int dst_w,
    cudaStream_t stream);
