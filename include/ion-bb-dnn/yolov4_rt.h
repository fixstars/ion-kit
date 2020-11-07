#ifndef ION_BB_DNN_YOLOV4_RT_H
#define ION_BB_DNN_YOLOV4_RT_H

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

#include "yolov4_utils.h"
#include "yolov4_ort.h"
#include "yolov4_tfl.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

extern "C" ION_EXPORT int yolov4_object_detection(halide_buffer_t *in,
                                                  halide_buffer_t *session_id_buf,
                                                  halide_buffer_t *model_buf,
                                                  halide_buffer_t *cache_path_buf,
                                                  int height, int width,
                                                  bool cuda_enable,
                                                  halide_buffer_t *boxes,
                                                  halide_buffer_t *confs) {
    bool is_bound_query = false;

    if (boxes->dimensions != confs->dimensions ||
        in->dimensions != boxes->dimensions + 1) {
        return 1;
    }

    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = width;
        in->dim[1].min = 0;
        in->dim[1].extent = height;
        in->dim[2].min = 0;
        in->dim[2].extent = 3;
        if (in->dimensions == 4) {
            in->dim[3].min = 0;
            in->dim[3].extent = boxes->dim[2].extent;
        }
        is_bound_query = true;
    }

    if (is_bound_query) {
        return 0;
    }

    Halide::Runtime::Buffer<float> in_buf(*in);
    in_buf.copy_to_host();

    using namespace ion::bb::dnn;
    std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
    std::string cache_root(reinterpret_cast<const char *>(cache_path_buf->host));

    const uint8_t *model_data = reinterpret_cast<const uint8_t *>(model_buf->host);
    int model_size = model_buf->size_in_bytes();

    if (is_tfl_available()) {
        return yolov4_object_detection_tfl(in, session_id, cache_root, model_data, model_size, height, width, cuda_enable, boxes, confs);
    } else if (is_ort_available()) {
        return yolov4_object_detection_ort(in, session_id, cache_root, model_data, model_size, height, width, cuda_enable, boxes, confs);
    } else {
        return -1;
    }

    return 0;
}

extern "C" ION_EXPORT int yolov4_box_rendering(
    halide_buffer_t *image,
    halide_buffer_t *boxes,
    halide_buffer_t *confs,
    int height, int width,
    int num, int num_classes,
    halide_buffer_t *out) {

    bool is_bound_query = false;

    if (boxes->dimensions != confs->dimensions ||
        image->dimensions != boxes->dimensions + 1 ||
        out->dimensions != image->dimensions) {
        return 1;
    }

    if (image->is_bounds_query()) {
        image->dim[0].min = 0;
        image->dim[0].extent = 3;
        image->dim[1].min = 0;
        image->dim[1].extent = width;
        image->dim[2].min = 0;
        image->dim[2].extent = height;
        if (image->dimensions == 4) {
            image->dim[3].min = 0;
            image->dim[3].extent = out->dim[3].extent;
        }
        is_bound_query = true;
    }

    if (boxes->is_bounds_query()) {
        boxes->dim[0].min = 0;
        // it's becuase Halide define_extern's restriction
        // originally, boxes shape is [4, num] but it needs to match to confs
        boxes->dim[0].extent = num_classes;
        boxes->dim[1].min = 0;
        boxes->dim[1].extent = num;
        if (boxes->dimensions == 3) {
            boxes->dim[2].min = 0;
            boxes->dim[2].extent = out->dim[3].extent;
        }
        is_bound_query = true;
    }

    if (confs->is_bounds_query()) {
        confs->dim[0].min = 0;
        confs->dim[0].extent = num_classes;
        confs->dim[1].min = 0;
        confs->dim[1].extent = num;
        if (confs->dimensions == 3) {
            confs->dim[2].min = 0;
            confs->dim[2].extent = out->dim[3].extent;
        }
        is_bound_query = true;
    }

    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = 3;
        out->dim[1].min = 0;
        out->dim[1].extent = width;
        out->dim[2].min = 0;
        out->dim[2].extent = height;
        is_bound_query = true;
    }

    if (is_bound_query) {
        return 0;
    }

    Halide::Runtime::Buffer<uint8_t> image_buf(*image);
    Halide::Runtime::Buffer<float> boxes_buf(*boxes);
    Halide::Runtime::Buffer<float> confs_buf(*confs);

    image_buf.copy_to_host();
    boxes_buf.copy_to_host();
    confs_buf.copy_to_host();

    const int out_num = out->dimensions == 3 ? 1 : out->dim[3].extent;
    const int image_stride = out->dimensions == 3 ? 0 /*UNUSED*/ : image->dim[3].stride;
    const int boxes_stride = out->dimensions == 3 ? 0 /*UNUSED*/ : boxes->dim[2].stride;
    const int confs_stride = out->dimensions == 3 ? 0 /*UNUSED*/ : confs->dim[2].stride;
    const int out_stride = out->dimensions == 3 ? 0 /*UNUSED*/ : out->dim[3].stride;

    for (int i = 0; i < out_num; i++) {
        const auto prediceted_boxes = post_processing(reinterpret_cast<float *>(boxes->host) + boxes_stride * i, reinterpret_cast<float *>(confs->host) + confs_stride * i, num, num_classes);
        cv::Mat frame(height, width, CV_8UC3, image->host + image_stride * i);
        const auto image_with_bb = copy_with_boxes(frame, prediceted_boxes, height, width);

        memcpy(out->host + out_stride * i, image_with_bb.data, image_with_bb.total() * image_with_bb.elemSize());
    }

    return 0;
}

#undef ION_EXPORT

#endif  // ION_BB_DNN_YOLOV4_RT_H
