#include <cassert>
#include <vector>

#include <cuda_runtime.h>

#include "ion/ion.h"

#define CUDA_SAFE_CALL(x)                                                                                                 \
    do {                                                                                                                  \
        cudaError_t err = x;                                                                                              \
        if (err != cudaSuccess) {                                                                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                                      \
        }                                                                                                                 \
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

int main() {
    using namespace Halide;

    constexpr int width = 16, height = 16;
    bool flip = true;
    std::vector<int32_t> input_vec(width * height, 2024);
    std::vector<int32_t> output_vec(width * height, 0);
    try {
        constexpr int size = width * height * sizeof(int32_t);
        // void *src = subsystem->malloc(size);
        void *src;
        CUDA_SAFE_CALL(cudaMalloc(&src, size));
        CUDA_SAFE_CALL(cudaMemcpy(src, input_vec.data(), size, cudaMemcpyHostToDevice));

        // void *dst = subsystem->malloc(size);
        void *dst;
        CUDA_SAFE_CALL(cudaMalloc(&dst, size));

        Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::TracePipeline).with_feature(Target::Debug);
        // CudaState state(subsystem->getContext(), subsystem->getStream());

        // subsystem->memcpy(src, input_vec.data(), size, RocaMemcpyKind::kMemcpyHostToDevice);
        auto device_interface = get_device_interface_for_device_api(DeviceAPI::CUDA, target);

        assert(device_interface);

        Halide::Buffer<> inputBuffer(Halide::Int(32), nullptr, height, width);
        inputBuffer.device_wrap_native(device_interface, reinterpret_cast<uint64_t>(src));
        
        // inputBuffer.device_wrap_native(device_interface, (uintptr_t)src);
        // inputBuffer.set_device_dirty(true);
        // inputBuffer.set_host_dirty(false);
        // Halide::Buffer<int32_t> outputBuffer(Halide::Int(32), height, width);
        // outputBuffer.device_malloc(device_interface);
         // Halide::Buffer<> outputBuffer(Halide::Int(32), reinterpret_cast<void*>(1), height, width);
         Halide::Buffer<> outputBuffer(Halide::Int(32), nullptr, height, width);
         outputBuffer.device_wrap_native(device_interface, reinterpret_cast<uint64_t>(dst));
        // outputBuffer.set_device_dirty(true);

        ion::Port input{"input0", Int(32), 2};
        ion::Builder b;
        b.set_target(target);
        ion::Graph graph(b);

        ion::Node cn = graph.add("Sqrt_gen")(inputBuffer);
        // input.bind(inputBuffer);
        // cn(inputBuffer);
        cn["output0"].bind(outputBuffer);

        b.run();
        outputBuffer.device_sync();  // Figure out how to replace halide's stream
        // outputBuffer.copy_to_host();

        cudaMemcpy(output_vec.data(), dst, size, cudaMemcpyDeviceToHost);
        // memcpy(output_vec.data(), outputBuffer.data(), size);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                std::cerr << output_vec[i * width + j] << " ";
                if (44 != output_vec[i * width +j]) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

        // subsystem->free(src);
        CUDA_SAFE_CALL(cudaFree(src));

        // subsystem->free(dst);
        CUDA_SAFE_CALL(cudaFree(dst));

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
