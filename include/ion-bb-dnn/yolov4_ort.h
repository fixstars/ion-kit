#ifndef ION_BB_DNN_YOLOV$_ORT_H
#define ION_BB_DNN_YOLOV$_ORT_H

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

#include "onnxruntime_c.h"

namespace ion {
namespace bb {
namespace dnn {

class OrtSessionManager {
public:
    OrtSessionManager(const uint8_t *model_data, int model_size, const std ::string &cache_root, bool cuda_enable)
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

    static OrtSessionManager *make(const std::string &uuid, const uint8_t *model_data, int model_size, const std ::string &cache_root, bool cuda_enable) {
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

int yolov4_object_detection_ort(halide_buffer_t *in,
                                const std::string& session_id,
                                const std::string& cache_root,
                                const uint8_t *model_data,
                                int model_size,
                                int height, int width,
                                bool cuda_enable,
                                halide_buffer_t *boxes,
                                halide_buffer_t *confs) {

    OrtSessionManager *session_manager = OrtSessionManager::make(session_id, model_data, model_size, cache_root, cuda_enable);
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


}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif  // ION_BB_DNN_YOLOV4_RT_H
