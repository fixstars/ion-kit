#ifndef ION_BB_DNN_RT__H
#define ION_BB_DNN_RT__H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>

#include <HalideBuffer.h>

#include "rt_tfl.h"
#include "rt_ort.h"

#include "yolov4_tfl.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

extern "C" ION_EXPORT int ion_bb_dnn_generic_object_detection(halide_buffer_t *in,
                                                              halide_buffer_t *session_id_buf,
                                                              halide_buffer_t *model_root_url_buf,
                                                              halide_buffer_t *cache_root_buf,
                                                              bool cuda_enable,
                                                              halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        // Input is (N)CHW, output is (N)HWC
        in->dim[0].min = out->dim[1].min;
        in->dim[0].extent = out->dim[1].extent;
        in->dim[1].min = out->dim[2].min;
        in->dim[1].extent = out->dim[2].extent;
        in->dim[2].min = out->dim[0].min;
        in->dim[2].extent = out->dim[0].extent;
        if (in->dimensions == 4) {
            in->dim[3].min = out->dim[3].min;
            in->dim[3].extent = out->dim[3].extent;
        }
        return 0;
    }

    Halide::Runtime::Buffer<float> in_buf(*in);
    in_buf.copy_to_host();

    std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));

    using namespace ion::bb::dnn;

    if (is_tfl_available()) {
        return object_detection_tfl(in, model_root_url, out);
    } else if (is_ort_available()) {
        std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));
        return object_detection_ort(in, session_id, model_root_url, cache_root, cuda_enable, out);
    } else {
        return -1;
    }
}

#undef ION_EXPORT

#endif  // ION_BB_DNN_RT__H
