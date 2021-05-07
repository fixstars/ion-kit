#include <iostream>

__global__ void inc_kernel(int32_t *in, int32_t width, int32_t height, int32_t v, int32_t *out) {
    int gx = threadIdx.x + blockDim.x * blockIdx.x;
    int gy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gx < width && gy < height) {
        out[gy * width + gx] = in[gy * width + gx] + v;
    }
}

extern "C"
void call_inc_kernel(int32_t *in, int32_t width, int32_t height, int32_t v, int32_t *out) {
    dim3 block_size(16, 16);
    int bx = (width + block_size.x - 1)/block_size.x;
    int by = (height + block_size.y - 1)/block_size.y;
    dim3 grid_size(bx, by);

    inc_kernel<<<grid_size, block_size>>>(in, width, height, v, out);
}

