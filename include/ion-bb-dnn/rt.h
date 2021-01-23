#ifndef ION_BB_DNN_RT_H
#define ION_BB_DNN_RT_H

#include <HalideBuffer.h>

#include <ion/json.hpp>

#include "rt_opencv.h"
#include "rt_tfl.h"
#include "rt_ort.h"
#include "rt_trt.h"
#include "rt_json.h"

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
    try {

        if (in->is_bounds_query()) {
            // Both input and output is (N)HWC
            for (int i=0; i<in->dimensions; ++i) {
                in->dim[i].min = out->dim[i].min;
                in->dim[i].extent = out->dim[i].extent;
            }
            return 0;
        }

        Halide::Runtime::Buffer<float> in_buf(*in);
        in_buf.copy_to_host();

        std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));

        using namespace ion::bb::dnn;

        if (is_tfl_available()) {
            return object_detection_tfl(in, model_root_url, cache_root, out);
        } else if (is_ort_available()) {
            std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            return object_detection_ort(in, session_id, model_root_url, cache_root, cuda_enable, out);
        } else {
            std::cerr << "No available runtime" << std::endl;
            return -1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

extern "C" ION_EXPORT int ion_bb_dnn_tlt_object_detection_ssd(halide_buffer_t *in,
                                                              halide_buffer_t *session_id_buf,
                                                              halide_buffer_t *model_root_url_buf,
                                                              halide_buffer_t *cache_root_buf,
                                                              halide_buffer_t *out) {
    try {

        if (in->is_bounds_query()) {
            // Both input and output is (N)HWC
            for (int i=0; i<in->dimensions; ++i) {
                in->dim[i].min = out->dim[i].min;
                in->dim[i].extent = out->dim[i].extent;
            }
            return 0;
        }

        Halide::Runtime::Buffer<float> in_buf(*in);
        in_buf.copy_to_host();

        std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));

        using namespace ion::bb::dnn;

        if (trt::is_available()) {
            std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            return trt::object_detection_ssd(in, session_id, model_root_url, cache_root, out);
        } else {
            std::cerr << "No available runtime" << std::endl;
            return -1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

extern "C" ION_EXPORT int ion_bb_dnn_tlt_peoplenet(halide_buffer_t *in,
                                                   halide_buffer_t *session_id_buf,
                                                   halide_buffer_t *model_root_url_buf,
                                                   halide_buffer_t *cache_root_buf,
                                                   halide_buffer_t *out) {
    try {

        if (in->is_bounds_query()) {
            // Both input and output is (N)HWC
            for (int i=0; i<in->dimensions; ++i) {
                in->dim[i].min = out->dim[i].min;
                in->dim[i].extent = out->dim[i].extent;
            }
            return 0;
        }

        Halide::Runtime::Buffer<float> in_buf(*in);
        in_buf.copy_to_host();

        std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));

        using namespace ion::bb::dnn;

        if (trt::is_available()) {
            std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            return trt::peoplenet(in, session_id, model_root_url, cache_root, out);
        } else {
            std::cerr << "No available runtime" << std::endl;
            return -1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

extern "C" ION_EXPORT int ion_bb_dnn_tlt_peoplenet_md(halide_buffer_t *in,
                                                      int32_t input_width,
                                                      int32_t input_height,
                                                      int32_t output_size,
                                                      halide_buffer_t *session_id_buf,
                                                      halide_buffer_t *model_root_url_buf,
                                                      halide_buffer_t *cache_root_buf,
                                                      halide_buffer_t *out) {
    try {

        if (in->is_bounds_query()) {
            // Input is (N)HWC
            in->dim[0].min = 0;
            in->dim[0].extent = 3;
            in->dim[1].min = 0;
            in->dim[1].extent = input_width;
            in->dim[2].min = 0;
            in->dim[2].extent = input_height;
            return 0;
        }

        Halide::Runtime::Buffer<float> in_buf(*in);
        in_buf.copy_to_host();

        std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));

        using namespace ion::bb::dnn;

        if (trt::is_available()) {
            std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            return trt::peoplenet_md(in, output_size, session_id, model_root_url, cache_root, out);
        } else {
            std::cerr << "No available runtime" << std::endl;
            return -1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

extern "C" ION_EXPORT int ion_bb_dnn_classify_gender(halide_buffer_t *in_img,
                                                     uint32_t input_img_width,
                                                     uint32_t input_img_height,
                                                     halide_buffer_t *in_md,
                                                     uint32_t input_md_size,
                                                     uint32_t output_size,
                                                     halide_buffer_t *session_id_buf,
                                                     halide_buffer_t *model_root_url_buf,
                                                     halide_buffer_t *cache_root_buf,
                                                     halide_buffer_t *out) {
    try {

        if (in_img->is_bounds_query() || in_md->is_bounds_query()) {
            if (in_img->is_bounds_query()) {
                // Input is (N)HWC
                in_img->dim[0].min = 0;
                in_img->dim[0].extent = 3;
                in_img->dim[1].min = 0;
                in_img->dim[1].extent = input_img_width;
                in_img->dim[2].min = 0;
                in_img->dim[2].extent = input_img_height;
            }

            if (in_md->is_bounds_query()) {
                in_md->dim[0].min = 0;
                in_md->dim[0].extent = input_md_size;
            }

            return 0;
        }

        Halide::Runtime::Buffer<float> in_img_buf(*in_img);
        in_img_buf.copy_to_host();

        Halide::Runtime::Buffer<uint8_t> in_md_buf(*in_md);
        in_md_buf.copy_to_host();

        std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
        std::string model_root_url(reinterpret_cast<const char *>(model_root_url_buf->host));
        std::string cache_root(reinterpret_cast<const char *>(cache_root_buf->host));

        ion::bb::dnn::opencv::classify_gender(in_img, in_md, output_size, session_id, model_root_url, cache_root, out);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

extern "C" ION_EXPORT int ion_bb_dnn_json_dict_average_regurator(halide_buffer_t *in,
                                                                 uint32_t io_md_size,
                                                                 halide_buffer_t *session_id_buf,
                                                                 uint32_t period_in_sec,
                                                                 halide_buffer_t *out) {
    try {

        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = io_md_size;
            return 0;
        }

        Halide::Runtime::Buffer<uint8_t> in_buf(*in);
        in_buf.copy_to_host();

        std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));

        auto& r = ion::bb::dnn::json::DictAverageRegurator::get_instance(session_id, period_in_sec);
        auto output_string = r.process(nlohmann::json::parse(reinterpret_cast<const char*>(in->host))).dump();

        if (output_string.size()+1 >= io_md_size) {
            throw std::runtime_error("Output buffer size is not sufficient");
        }

        std::memcpy(out->host, output_string.c_str(), output_string.size());
        out->host[output_string.size()] = 0;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}


extern "C" ION_EXPORT int ion_bb_dnn_ifttt_webhook_uploader(halide_buffer_t *in_md,
                                                            uint32_t input_md_size,
                                                            halide_buffer_t *session_id_buf,
                                                            halide_buffer_t *ifttt_webhook_url_buf,
                                                            halide_buffer_t *out) {
    try {

        if (in_md->is_bounds_query()) {
            in_md->dim[0].min = 0;
            in_md->dim[0].extent = input_md_size;

            return 0;
        }

        Halide::Runtime::Buffer<uint8_t> in_md_buf(*in_md);
        in_md_buf.copy_to_host();

        std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
        std::string ifttt_webhook_url(reinterpret_cast<const char *>(ifttt_webhook_url_buf->host));

        auto& uploader = ion::bb::dnn::json::WebHookUploader::get_instance(session_id, ifttt_webhook_url);
        uploader.upload(nlohmann::json::parse(in_md->host));

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

#undef ION_EXPORT

#endif  // ION_BB_DNN_BB_H
