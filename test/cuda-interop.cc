#include <cassert>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ion/ion.h"

#define CUDA_SAFE_CALL(x)                                                                                            \
    do {                                                                                                             \
        cudaError_t err = x;                                                                                         \
        if (err != cudaSuccess) {                                                                                    \
            std::stringstream ss;                                                                                    \
            ss << "CUDA error: " << cudaGetErrorString(err) << "(" << err << ") at " << __FILE__ << ":" << __LINE__; \
            throw  std::runtime_error(ss.str());                                                                     \
        }                                                                                                            \
    } while (0)

#define CU_SAFE_CALL(x)                                                                              \
    do {                                                                                             \
        CUresult err = x;                                                                            \
        if (err != CUDA_SUCCESS) {                                                                   \
            const char *err_str;                                                                     \
            cuGetErrorString(err, &err_str);                                                         \
            std::stringstream ss;                                                                    \
            ss << "CUDA error: " << err_str << "(" << err << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str());                                                      \
        }                                                                                            \
    } while (0)


struct Sqrt : ion::BuildingBlock<Sqrt> {
    ion::Input<Halide::Func> input{"input0", Int(32), 2};
    ion::Output<Halide::Func> output{"output0", Int(32), 2};
    Halide::Var x, y;

    void generate() {
        output(x, y) = cast<int>(sqrt(input(x, y)));
    }

    virtual void schedule() {
        Target target = get_target();
        if (target.has_gpu_feature()) {
            Var block, thread;
            if (target.has_feature(Target::OpenCL)) {
                std::cout << "Using OpenCL" << std::endl;

            } else if (target.has_feature(Target::CUDA)) {
                std::cout << "Using CUDA" << std::endl;
            } else if (target.has_feature(Target::Metal)) {
                std::cout << "Using Metal" << std::endl;
            }
            output.compute_root().gpu_tile(x, y, block, thread, 16, 16);
        }

        // Fallback to CPU scheduling
        else {
            output.compute_root().parallel(y).vectorize(x, 8);
        }
    }
};

ION_REGISTER_BUILDING_BLOCK(Sqrt, Sqrt_gen)

struct CudaState : public Halide::JITUserContext {
    void *cuda_context = nullptr, *cuda_stream = nullptr;
    std::atomic<int> acquires = 0, releases = 0;

    static int my_cuda_acquire_context(JITUserContext *ctx, void **cuda_ctx, bool create) {
        CudaState *state = (CudaState *)ctx;
        *cuda_ctx = state->cuda_context;
        state->acquires++;
        return 0;
    }

    static int my_cuda_release_context(JITUserContext *ctx) {
        CudaState *state = (CudaState *)ctx;
        state->releases++;
        return 0;
    }

    static int my_cuda_get_stream(JITUserContext *ctx, void *cuda_ctx, void **stream) {
        CudaState *state = (CudaState *)ctx;
        *stream = state->cuda_stream;
        return 0;
    }

    CudaState() {
        handlers.custom_cuda_acquire_context = my_cuda_acquire_context;
        handlers.custom_cuda_release_context = my_cuda_release_context;
        handlers.custom_cuda_get_stream = my_cuda_get_stream;
    }
};

int main() {
    using namespace Halide;

    constexpr int width = 16, height = 16;
    bool flip = true;
    std::vector<int32_t> input_vec(width * height, 2024);
    std::vector<int32_t> output_vec(width * height, 0);
    try {
        // CUDA setup
        CudaState state;

        // Ensure to initialize cuda Context under the hood
        CU_SAFE_CALL(cuInit(0));

        CUdevice device;
        CU_SAFE_CALL(cuDeviceGet(&device, 0));
        
        CU_SAFE_CALL(cuCtxCreate(reinterpret_cast<CUcontext *>(&state.cuda_context), 0, device));

        std::cout << "CUcontext is created on application side : " << state.cuda_context << std::endl;
        
        CU_SAFE_CALL(cuCtxSetCurrent(reinterpret_cast<CUcontext>(state.cuda_context)));
        
        // CUstream is interchangeable with cudaStream_t (ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html)
        CU_SAFE_CALL(cuStreamCreate(reinterpret_cast<CUstream *>(&state.cuda_stream), CU_STREAM_DEFAULT));

        constexpr int size = width * height * sizeof(int32_t);
        void *src;
        CUDA_SAFE_CALL(cudaMalloc(&src, size));
        CUDA_SAFE_CALL(cudaMemcpy(src, input_vec.data(), size, cudaMemcpyHostToDevice));

        void *dst;
        CUDA_SAFE_CALL(cudaMalloc(&dst, size));

        // ION execution
        {

            Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::TracePipeline).with_feature(Target::Debug);
            auto device_interface = get_device_interface_for_device_api(DeviceAPI::CUDA, target);

            Halide::Buffer<> inputBuffer(Halide::Int(32), nullptr, height, width);
            inputBuffer.device_wrap_native(device_interface, reinterpret_cast<uint64_t>(src), &state);
            inputBuffer.set_device_dirty(true);

            Halide::Buffer<> outputBuffer(Halide::Int(32), nullptr, height, width);
            outputBuffer.device_wrap_native(device_interface, reinterpret_cast<uint64_t>(dst), &state);

            ion::Builder b;
            b.set_target(target);
            b.set_jit_context(&state);

            ion::Graph graph(b);

            ion::Node cn = graph.add("Sqrt_gen")(inputBuffer);
            cn["output0"].bind(outputBuffer);

            b.run();

            cudaMemcpy(output_vec.data(), dst, size, cudaMemcpyDeviceToHost);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    std::cerr << output_vec[i * width + j] << " ";
                    if (44 != output_vec[i * width + j]) {
                        return -1;
                    }
                }
                std::cerr << std::endl;
            }
        }

        // CUDA cleanup
        CUDA_SAFE_CALL(cudaFree(src));
        CUDA_SAFE_CALL(cudaFree(dst));
        
        CU_SAFE_CALL(cuStreamDestroy(reinterpret_cast<CUstream>(state.cuda_stream)));
        CU_SAFE_CALL(cuCtxDestroy(reinterpret_cast<CUcontext>(state.cuda_context)));

    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
