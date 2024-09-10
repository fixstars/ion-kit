#ifndef TEST_RT_H
#define TEST_RT_H

#include <Halide.h>
#include <HalideBuffer.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT int consume_dispose(const char *id);

extern "C" DLLEXPORT int consume(halide_buffer_t *in, halide_buffer_t *id_buf, int desired_min0, int desired_extent0, int desired_min1, int desired_extent1, int32_t v, halide_buffer_t *out);

extern "C" DLLEXPORT int branch(halide_buffer_t *in, int32_t input_width, int32_t input_height, halide_buffer_t *out0, halide_buffer_t *out1);

extern "C" DLLEXPORT int inc(halide_buffer_t *in, int32_t width, int32_t height, int32_t v, bool use_gpu, halide_buffer_t *out);

#undef DLLEXPORT

#endif
