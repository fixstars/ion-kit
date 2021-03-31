#ifndef ION_BB_DNN_RT_DNNDK_H
#define ION_BB_DNN_RT_DNNDK_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

#include <HalideBuffer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "httplib.h"
#include "n2cube.h"
#include "rt_ssd.h"
#include "rt_util.h"

namespace ion {
namespace bb {
namespace dnn {
namespace dnndk {

class SessionManager {
public:
    static SessionManager &get_instance() {
        static SessionManager instance;
        return instance;
    }

    struct Model {
    public:
        Model() {
        }

        Model(std::shared_ptr<std::vector<uint8_t>> model_data, const std::string &library_name, const std::string &kernel_name)
            : is_available_(false), library_path_("/usr/lib/" + library_name), kernel_(nullptr), task_(nullptr) {
            // Copy to usr/lib/
            std::ofstream ofs(library_path_, std::ios::binary);
            if (!ofs.is_open()) {
                std::cerr << "Failed to open ofstream for copy DPU model : " << library_path_ << std::endl;
                return;
            }
            ofs.write(reinterpret_cast<const char *>(model_data->data()), model_data->size());

            kernel_ = dpuLoadKernel(kernel_name.c_str());
            if (kernel_ == nullptr) {
                std::cerr << "Failed to load DPU kernel : " << kernel_name << std::endl;
                return;
            }

            task_ = dpuCreateTask(kernel_, 0);
            if (task_ == nullptr) {
                std::cerr << "Failed to create DPU task : " << kernel_name << "[0]" << std::endl;
                return;
            }

            is_available_ = true;
        }

        ~Model() {
            dpuDestroyTask(task_);
            dpuDestroyKernel(kernel_);
            std::remove(library_path_.c_str());
        }

        bool is_available() const {
            return is_available_;
        }

        template<typename ElemT>
        void set_input_tensor(const std::string &node_name, const cv::Mat &image, ElemT mag = 1, ElemT offset = 0) {
            constexpr int channel = 3;
            assert(image.channels() == channel);

            // Gen the input tensor and parameters.
            auto tensor = dpuGetInputTensor(task(), node_name.c_str(), 0);
            auto input_addr = dpuGetTensorAddress(tensor);
            const auto scale = dpuGetTensorScale(tensor);
            const auto height = dpuGetTensorHeight(tensor);
            const auto width = dpuGetTensorWidth(tensor);
            assert(dpuGetTensorChannel(tensor) == channel);

            cv::Mat resized;
            cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

            for (int y = 0; y < resized.rows; y++) {
                for (int x = 0; x < resized.cols; x++) {
                    for (int c = 0; c < channel; c++) {
                        const auto normalized = resized.at<cv::Vec<ElemT, channel>>(y, x)[c] * mag + offset;
                        input_addr[y * resized.cols * channel + x * channel + c] = static_cast<int8_t>(normalized * scale);
                    }
                }
            }
        }

        void run() {
            dpuRunTask(task());
        }

        std::vector<float> get_output_scaled_tensor(const std::string &node_name) {
            auto tensor = dpuGetOutputTensor(task(), node_name.c_str(), 0);
            const auto addr = dpuGetTensorAddress(tensor);
            const auto scale = dpuGetTensorScale(tensor);
            const auto size = dpuGetTensorSize(tensor);

            std::vector<float> scaled(size);
            for (auto i = decltype(size)(0); i < size; ++i) {
                scaled[i] = addr[i] * scale;
            }

            return scaled;
        }

    private:
        bool is_available_;
        std::string library_path_;
        // shared_ptr can't be constructed from a imcomplete type, so I have them as raw pointers.
        DPUKernel *kernel_;
        DPUTask *task_;

        DPUTask *task() {
            return task_;
        }
    };

    Model *create_model(const std::string &library_name, const std::string &model_root_url, const std::string &cache_root, const std::string &kernel_name) {
        const std::string model_url = model_root_url + library_name;
        const std::string cache_path = cache_root + library_name;
        if (models_.count(model_url)) {
            return models_[model_url].get();
        }

        // Download model.
        std::shared_ptr<std::vector<uint8_t>> model_data;
        std::ifstream ifs(cache_root + library_name, std::ios::binary);
        if (ifs.is_open()) {
            auto begin = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            auto end = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            model_data = std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(end - begin));
            ifs.read(reinterpret_cast<char *>(model_data->data()), model_data->size());
        } else {
            std::string host_name;
            std::string path_name;
            std::tie(host_name, path_name) = parse_url(model_url);
            if (host_name.empty() || path_name.empty()) {
                std::cerr << "Invalid model URL : " << model_url << std::endl;
                return nullptr;
            }

            httplib::Client cli(host_name.c_str());
            cli.set_follow_location(true);
            auto res = cli.Get(path_name.c_str());
            if (!res || res->status != 200) {
                std::cerr << "Failed to download model : " << model_url << std::endl;
                return nullptr;
            }

            model_data = std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(res->body.size()));
            std::memcpy(model_data->data(), res->body.c_str(), res->body.size());

            std::ofstream ofs(cache_root + library_name, std::ios::binary);
            ofs.write(reinterpret_cast<const char *>(model_data->data()), model_data->size());
        }

        models_[model_url] = std::unique_ptr<Model>(new Model(model_data, library_name, kernel_name));

        return models_[model_url].get();
    }

    bool is_available() const {
        return is_available_dnndk_ && is_available_dpu_;
    }

private:
    SessionManager()
        : is_available_dnndk_(false), is_available_dpu_(false) {
        if (!dnndk_init()) {
            return;
        }
        is_available_dnndk_ = true;

        if (!dpu_load()) {
            return;
        }

        // TODO: error handling
        if (dpuOpen() != 0) {
            return;
        }
        is_available_dpu_ = true;
    }

    ~SessionManager() {
        // TODO: error handling
        if (dpuClose) {
            dpuClose();
        }
    }

    bool dpu_load() {
        int ret = system("sudo python3 -c \"from pynq_dpu import DpuOverlay; overlay = DpuOverlay(\"dpu.bit\")\"");
        return ret == 0;
    }

    bool is_available_dnndk_;
    bool is_available_dpu_;
    std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};

bool is_dnndk_available() {
    return SessionManager::get_instance().is_available();
}

int object_detection(halide_buffer_t *in,
                     const std::string &model_root_url,
                     const std::string &cache_root,
                     halide_buffer_t *out) {

    // Input parameters
    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;
    const size_t input_size = channel * width * height * sizeof(float);
    const int num_images = in->dimensions == 3 ? 1 : in->dim[3].extent;

    // Build anchors
    const int internal_height = 300;
    const int internal_width = 300;
    const size_t layer_num = 6;
    const float min_scale = 0.2f;
    const float max_scale = 0.95f;
    const std::vector<float> aspect_ratios = {1.f, 2.f, 0.5f, 3.f, 0.3333f};
    const Vec2d<float> base_anchor_size{1.f, 1.f};
    const Anchor scale_factor{10.f, 10.f, 5.f, 5.f};
    const std::vector<Vec2d<int>> feature_map_shape_list = {{19, 19}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}};
    const bool reduce_boxes_in_lower_layer = true;
    const float interpolated_scale_aspect_ratio = 1.f;
    const auto anchors = build_anchors(
        internal_width, internal_height, layer_num, min_scale, max_scale,
        aspect_ratios, base_anchor_size, feature_map_shape_list,
        reduce_boxes_in_lower_layer, interpolated_scale_aspect_ratio);

    // Postprocess parameters
    const size_t label_num = 91;
    const float nms_threshold = 0.6f;
    const float conf_threshold = 0.2f;
    const size_t top_k_per_class = 100;
    const size_t top_k = 100;

    // Construct the model
    const std::string library_name = "libdpumodeltf_ssd_mobilenetv2_coco.so";
    const std::string kernel_name = "tf_ssd_mobilenetv2_coco";
    auto model = SessionManager::get_instance().create_model(library_name, model_root_url, cache_root, kernel_name);
    if (!model->is_available()) {
        std::cerr << "Can't load model : " << library_name << std::endl;
        return -1;
    }

    for (int i = 0; i < num_images; ++i) {
        int offset = input_size * i;
        const cv::Mat in_(height, width, CV_32FC3, in->host + offset);

        model->set_input_tensor<float>("FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold", in_, 2.f, -1.f);

        model->run();

        const auto boxes = model->get_output_scaled_tensor("concat");
        const auto scores = model->get_output_scaled_tensor("concat_1");

        const auto detected_boxes = ssd_post_processing_dnndk(
            boxes, scores,
            label_num, anchors, scale_factor,
            nms_threshold, conf_threshold,
            top_k_per_class, top_k);

        cv::Mat out_(height, width, CV_32FC3, out->host + offset);
        in_.copyTo(out_);

        coco_render_boxes(out_, detected_boxes, width, height);
    }

    return 0;
}

}  // namespace dnndk
}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif  // ION_BB_DNN_RT_ORT_H
