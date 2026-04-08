#include "preprocess.cuh"

// =============================================================================
// Fused CUDA kernel: resize + BGR→RGB + HWC→CHW + /255.0
//
// Each thread computes ONE pixel in the output tensor.
// Bilinear interpolation reads from source image directly.
// Output is written in CHW planar layout (R plane, G plane, B plane).
// =============================================================================
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src, int src_h, int src_w,
    float* __restrict__ dst, int dst_h, int dst_w)
{
    // Each thread handles one (x, y) pixel in the destination image
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // dst x
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // dst y

    if (dx >= dst_w || dy >= dst_h) return;

    // Scale factors: map dst pixel back to src coordinates
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    // Source coordinates (floating point)
    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    // Bilinear interpolation: 4 nearest source pixels
    int x0 = __float2int_rd(sx);  // floor
    int y0 = __float2int_rd(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image bounds
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    float fx = sx - __float2int_rd(sx);  // fractional part
    float fy = sy - __float2int_rd(sy);
    fx = fmaxf(0.0f, fminf(fx, 1.0f));
    fy = fmaxf(0.0f, fminf(fy, 1.0f));

    // Read 4 source pixels (BGR, 3 channels)
    int src_stride = src_w * 3;
    const uint8_t* p00 = src + y0 * src_stride + x0 * 3;
    const uint8_t* p01 = src + y0 * src_stride + x1 * 3;
    const uint8_t* p10 = src + y1 * src_stride + x0 * 3;
    const uint8_t* p11 = src + y1 * src_stride + x1 * 3;

    // Bilinear weights
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    // Interpolate each BGR channel, normalize to [0,1]
    float b = (p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11) / 255.0f;
    float g = (p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11) / 255.0f;
    float r = (p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11) / 255.0f;

    // Write in CHW planar format (channel-first): R, G, B planes
    // Source is BGR (OpenCV), output is RGB (YOLO model expects RGB)
    // This matches the original Python: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    // which on BGR input effectively swaps B<->R producing RGB order.
    int plane_size = dst_h * dst_w;
    int pixel_idx = dy * dst_w + dx;

    dst[0 * plane_size + pixel_idx] = r;  // Channel 0 (R)
    dst[1 * plane_size + pixel_idx] = g;  // Channel 1 (G)
    dst[2 * plane_size + pixel_idx] = b;  // Channel 2 (B)
}

void preprocess_gpu(
    const uint8_t* d_src, int src_h, int src_w,
    float* d_dst, int dst_h, int dst_w,
    cudaStream_t stream)
{
    // 16x16 = 256 threads per block (good occupancy on Orin SM 8.7)
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_src, src_h, src_w,
        d_dst, dst_h, dst_w);
}
