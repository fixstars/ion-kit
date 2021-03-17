#ifndef TEST_RT_H
#define TEST_RT_H

#include <iostream>
#include <HalideBuffer.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT
int consume(halide_buffer_t *in, int desired_min0, int desired_extent0, int desired_min1, int desired_extent1, int32_t v, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = desired_min0;
        in->dim[0].extent = desired_extent0;
        in->dim[1].min = desired_min1;
        in->dim[1].extent = desired_extent1;
    } else {
        Halide::Runtime::Buffer<int32_t> ibuf(*in);
        for (int y=0; y<in->dim[1].extent; ++y) {
            for (int x=0; x<in->dim[0].extent; ++x) {
                std::cout << ibuf(x, y) + v << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}

extern "C" DLLEXPORT
int branch(halide_buffer_t *in, int32_t input_width, int32_t input_height, halide_buffer_t *out0, halide_buffer_t *out1) {
    if (in->is_bounds_query() || out0->is_bounds_query() || out1->is_bounds_query()) {
        if (out0->is_bounds_query()) {
            out0->dim[0].min = 0;
            out0->dim[0].extent = input_width;
            out0->dim[1].min = 0;
            out0->dim[1].extent = input_height/2;
        }
        if (out1->is_bounds_query()) {
            out1->dim[0].min = 0;
            out1->dim[0].extent = input_width;
            out1->dim[1].min = 0;
            out1->dim[1].extent = input_height/2;
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
        for (int y=0; y<input_height/2; ++y) {
            for (int x=0; x<input_width; ++x) {
                obuf0(x, y) = ibuf(x, y);
                obuf1(x, y) = ibuf(x, y + input_height/2);
            }
        }
    }

    return 0;
}


#undef DLLEXPORT

#endif
