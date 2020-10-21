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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <HalideBuffer.h>

#include "rt_onnxruntime.h"
#include "yolov4_utils.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

namespace ion {
namespace bb {
namespace dnn {

class OrtSessionManager {
public:
    OrtSessionManager(void *model, int model_size, const std ::string &cache_root, bool cuda_enable)
        : ort_{new ONNXRuntime()} {
        const OrtApi *api = ort_->get_api();
        ort_->check_status(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YOLOv4", &env));
        ort_->check_status(api->CreateSessionOptions(&session_options));
        ort_->check_status(api->SetIntraOpNumThreads(session_options, 1));
        ort_->check_status(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));
        if (cuda_enable && check_tensorrt_enable()) {
            set_tensorrt_cache_env(cache_root);
            ort_->enable_tensorrt_provider(session_options, 0);
        }
        ort_->check_status(api->CreateSessionFromArray(env, model, model_size, session_options, &session));
    }

    inline const ONNXRuntime *get_ort() const {
        return ort_.get();
    }

    inline const OrtApi *get_ort_api() const {
        return ort_->get_api();
    }

    inline OrtSession *get_ort_session() const {
        return session;
    }

    static OrtSessionManager *make(const std::string &uuid, void *model, int model_size, const std ::string &cache_root, bool cuda_enable) {
        static std::map<std::string, std::unique_ptr<OrtSessionManager>> map_;
        OrtSessionManager *ort_manager;
        if (map_.count(uuid) == 0) {
            map_[uuid] = std::unique_ptr<OrtSessionManager>(new OrtSessionManager(model, model_size, cache_root, cuda_enable));
        }
        return map_[uuid].get();
    }

private:
    bool check_tensorrt_enable() {
        bool tensorrt_enable = false;
        void *handle = dlopen("libnvinfer.so", RTLD_LAZY);
        if (handle != NULL) {
            tensorrt_enable = true;
            dlclose(handle);
        }

        return tensorrt_enable;
    }

    int set_tensorrt_cache_env(const std ::string &cache_root) const {
        static char env_buf[3][200];
        snprintf(env_buf[0], sizeof(env_buf[0]), "ORT_TENSORRT_ENGINE_CACHE_ENABLE=1");
        if (putenv(env_buf[0]) == -1) {
            std::cout << "set ORT_TENSORRT_ENGINE_CACHE_ENABLE failed..." << std::endl;
        }
        snprintf(env_buf[1], sizeof(env_buf[1]), "ORT_TENSORRT_FP16_ENABLE=1");
        if (putenv(env_buf[1]) == -1) {
            std::cout << "set ORT_TENSORRT_FP16_ENABLE failed..." << std::endl;
        }
        const std::string ort_cache_path = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + cache_root;
        snprintf(env_buf[2], sizeof(env_buf[2]), ort_cache_path.c_str());
        if (putenv(env_buf[2]) == -1) {
            std::cout << "set ORT_TENSORRT_ENGINE_CACHE_PATH failed..." << std::endl;
        }
    }

    std::unique_ptr<ONNXRuntime> ort_;
    OrtEnv *env;
    OrtSession *session;
    OrtSessionOptions *session_options;
};

}  // namespace dnn
}  // namespace bb
}  // namespace ion

extern "C" ION_EXPORT int yolov4_object_detection(halide_buffer_t *in,
                                                  halide_buffer_t *session_id_buf,
                                                  halide_buffer_t *model,
                                                  halide_buffer_t *cache_path_buf,
                                                  int model_size,
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
    std::string session_id = reinterpret_cast<const char *>(session_id_buf->host);
    std::string cache_root = reinterpret_cast<const char *>(cache_path_buf->host);

    OrtSessionManager *session_manager = OrtSessionManager::make(session_id, model->host, model_size, cache_root, cuda_enable);
    const ONNXRuntime *ort = session_manager->get_ort();
    const OrtApi *api = session_manager->get_ort_api();
    OrtSession *session = session_manager->get_ort_session();

    size_t num_input_nodes;
    OrtAllocator *allocator;
    ort->check_status(api->GetAllocatorWithDefaultOptions(&allocator));

    ort->check_status(api->SessionGetInputCount(session, &num_input_nodes));
    char *input_name;
    ort->check_status(api->SessionGetInputName(session, 0, allocator, &input_name));

    OrtTypeInfo *typeinfo;
    ort->check_status(api->SessionGetInputTypeInfo(session, 0, &typeinfo));

    const OrtTensorTypeAndShapeInfo *tensor_info;
    ort->check_status(api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    ONNXTensorElementDataType type;
    ort->check_status(api->GetTensorElementType(tensor_info, &type));

    size_t num_dims;
    ort->check_status(api->GetDimensionsCount(tensor_info, &num_dims));

    std::vector<int64_t> input_node_dims;
    input_node_dims.resize(num_dims);
    ort->check_status(api->GetDimensions(tensor_info, (int64_t *)input_node_dims.data(), num_dims));
    size_t input_size;
    ort->check_status(api->GetTensorShapeElementCount(tensor_info, &input_size));

    api->ReleaseTypeInfo(typeinfo);

    const int out_num = boxes->dimensions == 2 ? 1 : boxes->dim[2].extent;
    const int input_stride = boxes->dimensions == 2 ? 0 /*UNUSED*/ : in->dim[3].stride;
    const int boxes_stride = boxes->dimensions == 2 ? 0 /*UNUSED*/ : boxes->dim[2].stride;
    const int confs_stride = boxes->dimensions == 2 ? 0 /*UNUSED*/ : confs->dim[2].stride;

    const int batch = input_node_dims.data()[0];
    if (out_num != batch) {
        std::cout << "Batch size must be " << batch << " (ONNX expected) but " << out_num << " given..." << std::endl;
        exit(-1);
    }

    int i = 0;
    float *input_tensor_ptr = reinterpret_cast<float *>(in->host) + input_stride * i;
    std::vector<const char *> output_tensor_names = {"boxes", "confs"};

    OrtMemoryInfo *memory_info;
    ort->check_status(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue *input_tensor = NULL;
    ort->check_status(api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_ptr, input_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    int is_tensor;
    ort->check_status(api->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    api->ReleaseMemoryInfo(memory_info);

    std::vector<OrtValue *> output_tensors(2);
    ort->check_status(api->Run(session, NULL, &input_name, (const OrtValue *const *)&input_tensor, 1, output_tensor_names.data(), 2, output_tensors.data()));
    ort->check_status(api->IsTensor(output_tensors[0], &is_tensor));
    assert(is_tensor);
    ort->check_status(api->IsTensor(output_tensors[1], &is_tensor));
    assert(is_tensor);

    float *boxes_ptr, *confs_ptr = NULL;
    ort->check_status(api->GetTensorMutableData(output_tensors[0], reinterpret_cast<void **>(&boxes_ptr)));
    ort->check_status(api->GetTensorMutableData(output_tensors[1], reinterpret_cast<void **>(&confs_ptr)));

    OrtTensorTypeAndShapeInfo *boxes_info, *confs_info;
    ort->check_status(api->GetTensorTypeAndShape(output_tensors[0], &boxes_info));
    ort->check_status(api->GetTensorTypeAndShape(output_tensors[1], &confs_info));

    size_t boxes_size, confs_size;
    ort->check_status(api->GetTensorShapeElementCount(boxes_info, &boxes_size));
    ort->check_status(api->GetTensorShapeElementCount(confs_info, &confs_size));

    for (int i = 0; i < out_num; i++) {
        int real_box_size = boxes_size / out_num;
        memcpy(reinterpret_cast<float *>(boxes->host) + boxes_stride * i, boxes_ptr + real_box_size * i, sizeof(float) * real_box_size);
        int real_conf_size = confs_size / out_num;
        memcpy(reinterpret_cast<float *>(confs->host) + confs_stride * i, confs_ptr + real_conf_size * i, sizeof(float) * real_conf_size);
    }

    api->ReleaseValue(output_tensors[0]);
    api->ReleaseValue(output_tensors[1]);
    api->ReleaseValue(input_tensor);

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
