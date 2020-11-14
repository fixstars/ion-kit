#ifndef ION_BB_DNN_RT_ORT_H
#define ION_BB_DNN_RT_ORT_H

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
#include "rt_yolo.h"

namespace ion {
namespace bb {
namespace dnn {

class OrtSessionManager {
public:
    OrtSessionManager(const std::string& model_root_url, const std ::string &cache_root, bool cuda_enable)
        : ort_{new ONNXRuntime()} {
        const OrtApi *api = ort_->get_api();
        ort_->check_status(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ion_bb_dnn_ort", &env));
        ort_->check_status(api->CreateSessionOptions(&session_options));
        ort_->check_status(api->SetIntraOpNumThreads(session_options, 1));
        ort_->check_status(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));
        if (cuda_enable && check_tensorrt_enable()) {
            set_tensorrt_cache_env(cache_root);
            ort_->enable_tensorrt_provider(session_options, 0);
        }

        //std::string model_url = model_root_url + "yolov4-tiny_416_416.onnx";
        std::string model_name = "ssd_mobilenet_v2_coco_2018_03_29.onnx";

        std::ifstream ifs(cache_root + model_name, std::ios::binary);
        if (ifs.is_open()) {
            auto begin = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            auto end = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            model_.resize(end-begin);
            ifs.read(reinterpret_cast<char*>(model_.data()), model_.size());
        } else {
            std::string model_url = model_root_url + model_name;
            std::string host_name;
            std::string path_name;
            std::tie(host_name, path_name) = parse_url(model_url);
            if (host_name.empty() || path_name.empty()) {
                std::cerr << "Invalid model URL : " << model_url << std::endl;
                return;
            }

            httplib::Client cli(host_name.c_str());
            cli.set_follow_location(true);
            auto res = cli.Get(path_name.c_str());
            if (!res || res->status != 200) {
                std::cerr << "Failed to download model : " << model_url << std::endl;
                return;
            }

            model_.resize(res->body.size());
            std::memcpy(model_.data(), res->body.c_str(), res->body.size());

            std::ofstream ofs (cache_root + model_name, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(model_.data()), model_.size());
        }

        ort_->check_status(api->CreateSessionFromArray(env, model_.data(), model_.size(), session_options, &session));
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

    static OrtSessionManager *make(const std::string &uuid, const std::string& model_root_url, const std ::string &cache_root, bool cuda_enable) {
        static std::map<std::string, std::unique_ptr<OrtSessionManager>> map_;
        OrtSessionManager *ort_manager;
        if (map_.count(uuid) == 0) {
            map_[uuid] = std::unique_ptr<OrtSessionManager>(new OrtSessionManager(model_root_url, cache_root, cuda_enable));
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
        if (setenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1", 1) == -1) {
            std::cerr << "set ORT_TENSORRT_ENGINE_CACHE_ENABLE failed..." << std::endl;
        }
        if (setenv("ORT_TENSORRT_FP16_ENABLE", "1", 1) == -1) {
            std::cerr << "set ORT_TENSORRT_FP16_ENABLE failed..." << std::endl;
        }
        if (setenv("ORT_TENSORRT_ENGINE_CACHE_PATH", cache_root.c_str(), 1) == -1) {
            std::cerr << "set ORT_TENSORRT_ENGINE_CACHE_PATH failed..." << std::endl;
        }
    }

    std::unique_ptr<ONNXRuntime> ort_;
    OrtEnv *env;
    OrtSession *session;
    OrtSessionOptions *session_options;
    std::vector<uint8_t> model_;
};

bool is_ort_available() {
    ONNXRuntime ort;
    return ort.get_api() != nullptr;
}

int object_detection_ort(halide_buffer_t *in,
                         const std::string& session_id,
                         const std::string& model_root_url,
                         const std::string& cache_root,
                         bool cuda_enable,
                         halide_buffer_t *out) {

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    OrtSessionManager *session_manager = OrtSessionManager::make(session_id, model_root_url, cache_root, cuda_enable);
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
    if (num_dims != 4) {
        std::cerr << "This model is not supported." << std::endl;
        return -1;
    }

    std::vector<int64_t> input_node_dims(num_dims);
    ort->check_status(api->GetDimensions(tensor_info, reinterpret_cast<int64_t *>(input_node_dims.data()), num_dims));
    size_t input_size;
    ort->check_status(api->GetTensorShapeElementCount(tensor_info, &input_size));

    api->ReleaseTypeInfo(typeinfo);

    int num_images = in->dimensions == 3 ? 1 : in->dim[3].extent;

    for (int i=0; i<num_images; ++i) {
        int offset = input_size * i;
        cv::Mat in_(height, width, CV_32FC3, in->host + offset);

        const int internal_width = input_node_dims.at(2);
        const int internal_height = input_node_dims.at(1);

        cv::Mat resized(internal_height, internal_width, CV_32FC3);
        cv::resize(in_, resized, resized.size());

        // cv::Mat input_tensor_data(std::vector<int>{3, internal_height*internal_width}, CV_32FC1);

        // cv::transpose(resized.reshape(1, internal_width*internal_height), input_tensor_data);

        cv::Mat input_tensor_data(internal_height, internal_width, CV_8UC3);

        resized.convertTo(input_tensor_data, CV_8UC3, 255.0);

        uint8_t *input_tensor_ptr = reinterpret_cast<uint8_t*>(input_tensor_data.ptr());

        OrtMemoryInfo *memory_info;
        ort->check_status(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        OrtValue *input_tensor = NULL;
        ort->check_status(api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_ptr, input_size * sizeof(uint8_t), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_tensor));
        int is_tensor;
        ort->check_status(api->IsTensor(input_tensor, &is_tensor));
        assert(is_tensor);
        api->ReleaseMemoryInfo(memory_info);

        //std::vector<const char *> output_tensor_names = {"boxes", "confs"};
        std::vector<const char *> output_tensor_names = {"detection_boxes:0", "detection_classes:0", "detection_scores:0", "num_detections:0"};
        std::vector<OrtValue *> output_tensors(4);
        ort->check_status(api->Run(session, NULL, &input_name, (const OrtValue *const *)&input_tensor, 1, output_tensor_names.data(), 4, output_tensors.data()));
        ort->check_status(api->IsTensor(output_tensors[0], &is_tensor));
        assert(is_tensor);
        ort->check_status(api->IsTensor(output_tensors[1], &is_tensor));
        assert(is_tensor);
        ort->check_status(api->IsTensor(output_tensors[2], &is_tensor));
        assert(is_tensor);
        ort->check_status(api->IsTensor(output_tensors[3], &is_tensor));
        assert(is_tensor);

        float *boxes_ptr, *classes_ptr, *scores_ptr, *nums_ptr;
        ort->check_status(api->GetTensorMutableData(output_tensors[0], reinterpret_cast<void **>(&boxes_ptr)));
        ort->check_status(api->GetTensorMutableData(output_tensors[1], reinterpret_cast<void **>(&classes_ptr)));
        ort->check_status(api->GetTensorMutableData(output_tensors[2], reinterpret_cast<void **>(&scores_ptr)));
        ort->check_status(api->GetTensorMutableData(output_tensors[3], reinterpret_cast<void **>(&nums_ptr)));

        OrtTensorTypeAndShapeInfo *boxes_info, *classes_info, *scores_info, *nums_info;
        ort->check_status(api->GetTensorTypeAndShape(output_tensors[0], &boxes_info));
        ort->check_status(api->GetTensorTypeAndShape(output_tensors[1], &classes_info));
        ort->check_status(api->GetTensorTypeAndShape(output_tensors[2], &scores_info));
        ort->check_status(api->GetTensorTypeAndShape(output_tensors[3], &nums_info));

        size_t boxes_size, classes_size, scores_size, nums_size;
        ort->check_status(api->GetTensorShapeElementCount(boxes_info, &boxes_size));
        ort->check_status(api->GetTensorShapeElementCount(classes_info, &classes_size));
        ort->check_status(api->GetTensorShapeElementCount(scores_info, &scores_size));
        ort->check_status(api->GetTensorShapeElementCount(nums_info, &nums_size));

        // const int num = 2535;
        // const int num_classes = 80;

        //const auto prediceted_boxes = yolo_post_processing(boxes_ptr, confs_ptr, num, num_classes);
        const auto prediceted_boxes = ssd_post_processing(boxes_ptr, classes_ptr, scores_ptr, static_cast<int>(lround(*nums_ptr)));
        cv::Mat out_(height, width, CV_32FC3, out->host + offset);
        in_.copyTo(out_);

        coco_render_boxes(out_, prediceted_boxes, width, height);

        api->ReleaseValue(output_tensors[0]);
        api->ReleaseValue(output_tensors[1]);
        api->ReleaseValue(output_tensors[2]);
        api->ReleaseValue(output_tensors[3]);
        api->ReleaseValue(input_tensor);
    }

    return 0;
}

}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif // ION_BB_DNN_RT_ORT_H
