#define NOMINMAX
#include "test-bb.h"
#include "test-rt.h"

#include "dynamic_module.h"
#include "log.h"

extern "C" void register_externs(std::map<std::string, Halide::JITExtern> &externs) {
    externs.insert({"consume", Halide::JITExtern(consume)});
    externs.insert({"branch", Halide::JITExtern(branch)});
    externs.insert({"inc", Halide::JITExtern(inc)});
}

extern "C" int consume_dispose(const char *id) {
    ion::log::info("consume_dispose is called with id={}", id);
    return 0;
}

extern "C" int consume(halide_buffer_t *in, halide_buffer_t *id_buf, int desired_min0, int desired_extent0, int desired_min1, int desired_extent1, int32_t v, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = desired_min0;
        in->dim[0].extent = desired_extent0;
        in->dim[1].min = desired_min1;
        in->dim[1].extent = desired_extent1;
    } else {
        ion::log::info("consume is called with id={}", reinterpret_cast<const char *>(id_buf->host));
        Halide::Runtime::Buffer<int32_t> ibuf(*in);
        for (int y = 0; y < in->dim[1].extent; ++y) {
            for (int x = 0; x < in->dim[0].extent; ++x) {
                std::cout << ibuf(x, y) + v << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}

extern "C" int branch(halide_buffer_t *in, int32_t input_width, int32_t input_height, halide_buffer_t *out0, halide_buffer_t *out1) {
    if (in->is_bounds_query() || out0->is_bounds_query() || out1->is_bounds_query()) {
        if (out0->is_bounds_query()) {
            out0->dim[0].min = 0;
            out0->dim[0].extent = input_width;
            out0->dim[1].min = 0;
            out0->dim[1].extent = input_height / 2;
        }
        if (out1->is_bounds_query()) {
            out1->dim[0].min = 0;
            out1->dim[0].extent = input_width;
            out1->dim[1].min = 0;
            out1->dim[1].extent = input_height / 2;
        }
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = input_width;
            in->dim[1].min = 0;
            in->dim[1].extent = input_height;
        }
    } else {
        Halide::Runtime::Buffer<int32_t> ibuf(*in);
        Halide::Runtime::Buffer<int32_t> obuf0(*out0);
        Halide::Runtime::Buffer<int32_t> obuf1(*out1);
        for (int y = 0; y < input_height / 2; ++y) {
            for (int x = 0; x < input_width; ++x) {
                obuf0(x, y) = ibuf(x, y);
                obuf1(x, y) = ibuf(x, y + input_height / 2);
            }
        }
    }

    return 0;
}

using call_inc_kernel_t = void (*)(int32_t *in, int32_t width, int32_t height, int32_t v, int32_t *out);

extern "C" int inc(halide_buffer_t *in, int32_t width, int32_t height, int32_t v, bool use_gpu, halide_buffer_t *out) {
    using namespace Halide;

    if (in->is_bounds_query()) {
        in->dim[0].min = out->dim[0].min;
        in->dim[0].extent = out->dim[0].extent;
        in->dim[1].min = out->dim[1].min;
        in->dim[1].extent = out->dim[1].extent;
    } else {

        ion::log::debug("in->host({:#x}), in->device({:#x}), out->host({:#x}), out->device({:#x})", reinterpret_cast<uint64_t>(in->host), in->device, reinterpret_cast<uint64_t>(out->host), out->device);

        Runtime::Buffer<int32_t> ibuf(*in);
        Runtime::Buffer<int32_t> obuf(*out);

        if (use_gpu) {
            auto device_api = get_device_interface_for_device_api(DeviceAPI::CUDA, get_host_target().with_feature(Target::CUDA));

            if (!ibuf.has_device_allocation()) {
                ibuf.device_malloc(device_api);
                ibuf.copy_to_device(device_api);
            }

            bool copy_to_host = false;
            if (!obuf.has_device_allocation()) {
                obuf.device_malloc(device_api);
                copy_to_host = true;
            }

            static ion::DynamicModule dm("gpu-extern-lib");
            call_inc_kernel_t call_inc_kernel = dm.get_symbol<call_inc_kernel_t>("call_inc_kernel");
            call_inc_kernel(reinterpret_cast<int32_t *>(ibuf.raw_buffer()->device), obuf.extent(0), obuf.extent(1), v,
                            reinterpret_cast<int32_t *>(obuf.raw_buffer()->device));

            if (copy_to_host) {
                obuf.set_host_dirty(false);
                obuf.set_device_dirty();
                obuf.copy_to_host();
            }
        } else {
            for (int y = obuf.min(1); y < obuf.extent(1); ++y) {
                for (int x = obuf.min(0); x < obuf.extent(0); ++x) {
                    obuf(x, y) = ibuf(x, y) + v;
                }
            }
        }
    }

    return 0;
}
