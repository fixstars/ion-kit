#ifndef ION_BB_DNN_RT_DNNDK_H
#define ION_BB_DNN_RT_DNNDK_H

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

        Model(const std::string &model_name, const std::string &kernel_name)
            : dm_(ion::bb::dnn::DynamicModule(model_name, true, false)) {

            kernel_ = dpuLoadKernel(kernel_name.c_str());
            if (kernel_ == nullptr) {
                return;
            }

            task_ = dpuCreateTask(kernel_, 0);
            if (task_ == nullptr) {
                return;
            }
        }

        ~Model() {
            dpuDestroyTask(task_);
            dpuDestroyKernel(kernel_);
        }

        DPUTask *task() {
            return task_;
        }

        void set_input_image(const std::string &node_name, const cv::Mat &image, float *means) {
            auto tensor = dpuGetInputTensor(task(), node_name.c_str(), 0);

            const auto height = dpuGetTensorHeight(tensor);
            const auto width = dpuGetTensorWidth(tensor);
            const auto channel = dpuGetTensorChannel(tensor);
            assert(channel == 3);

            cv::Mat resized(height, width, CV_32FC3);
            cv::resize(image, resized, resized.size(), 0, 0, cv::INTER_LINEAR);

            int8_t *input_addr = dpuGetTensorAddress(tensor);
            const float scale = dpuGetTensorScale(tensor);

            for (int y = 0; y < resized.rows; y++) {
                for (int x = 0; x < resized.cols; x++) {
                    for (int c = 0; c < 3; c++) {
                        // NOTE: input is normalized as [0.f, 1.f).
                        const auto value = static_cast<int>((image.at<cv::Vec3b>(y, x)[c] - 1.f - means[c]) * scale);
                        // TODO: Is it the correct order?
                        input_addr[y * image.cols * 3 + x * 3 + c] = static_cast<char>(value);
                    }
                }
            }
        }

        std::vector<float> get_output_tensor(const std::string &node_name) {
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
        ion::bb::dnn::DynamicModule dm_;
        // shared_ptr can't be construct from imcomplete type, so have them as raw pointer.
        DPUKernel *kernel_;
        DPUTask *task_;
    };

    DPUTask *create_task(const std::string &model_name, const std::string &model_root_url, const std::string &cache_root, const std::string &kernel_name) {
        const std::string model_url = model_root_url + model_name;
        if (models_.count(model_url)) {
            return models_[model_url].task();
        }

        // Download model.
        std::shared_ptr<std::vector<uint8_t>> model_data;
        // std::ifstream ifs(cache_root + model_name, std::ios::binary);
        // if (ifs.is_open()) {
        //     auto begin = ifs.tellg();
        //     ifs.seekg(0, std::ios::end);
        //     auto end = ifs.tellg();
        //     ifs.seekg(0, std::ios::beg);
        //     model_data = std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(end - begin));
        //     ifs.read(reinterpret_cast<char *>(model_data->data()), model_data->size());
        // } else {
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
        // write to .so
        std::ofstream ofs(model_name, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(model_data->data()), model_data->size());

        models_[model_url] = Model(model_name, kernel_name);

        return models_[model_url].task();
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

        // TODO: error handling
        if (dpuOpen() != 0) {
            return;
        }
        is_available_dpu_ = true;
    }

    ~SessionManager() {
        // TODO: error handling
        dpuClose();
    }

    bool is_available_dnndk_;
    bool is_available_dpu_;
    std::unordered_map<std::string, Model> models_;
};

bool is_dnndk_available() {
    return SessionManager::get_instance().is_available();
}

int object_detection(halide_buffer_t *in,
                     const std::string &model_root_url,
                     const std::string &cache_root,
                     halide_buffer_t *out) {

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;
    const std::string model_name = "tf_ssd_mobilenetv2_coco";
    const std::string kernel_name = "tf_ssd_mobilenetv2_coco";

    size_t input_size = 3 * width * height * sizeof(float);

    int num_images = in->dimensions == 3 ? 1 : in->dim[3].extent;

    // Build anchors
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
        width, height, layer_num, min_scale, max_scale,
        aspect_ratios, base_anchor_size, scale_factor, feature_map_shape_list,
        reduce_boxes_in_lower_layer, interpolated_scale_aspect_ratio);

    // postprocess parameters
    const size_t label_num = 91;
    const float nms_threshold = 0.6;
    const float conf_threshold = 0.5;

    for (int i = 0; i < num_images; ++i) {
        int offset = input_size * i;
        cv::Mat in_(height, width, CV_32FC3, in->host + offset);

        auto task = SessionManager::get_instance().create_task(model_name, model_root_url, cache_root, kernel_name);

        // Set Input
        {
            DPUTensor *input = dpuGetInputTensor(task, "FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold", 0);
            const auto height = dpuGetTensorHeight(input);
            const auto width = dpuGetTensorWidth(input);
            const auto channel = dpuGetTensorChannel(input);
            assert(channel == 3);

            cv::Mat resized(height, width, CV_32FC3);
            cv::resize(in_, resized, resized.size(), 0, 0, cv::INTER_LINEAR);

            int8_t *input_addr = dpuGetTensorAddress(input);
            const float scale = dpuGetTensorScale(input);

            for (int y = 0; y < resized.rows; y++) {
                for (int x = 0; x < resized.cols; x++) {
                    for (int c = 0; c < 3; c++) {
                        // NOTE: input is normalized as [0.f, 1.f).
                        const auto value = static_cast<int>((in_.at<cv::Vec3b>(y, x)[c] - 1.f) * scale);
                        // TODO: It is correct order?
                        input_addr[y * in_.cols * 3 + x * 3 + c] = static_cast<char>(value);
                    }
                }
            }
        }

        dpuRunTask(task);

        // Get Output
        DPUTensor *boxes = dpuGetOutputTensor(task, "concat", 0);
        DPUTensor *scores = dpuGetOutputTensor(task, "concat_1", 0);
        const int8_t *boxes_data = dpuGetTensorAddress(boxes);
        const float boxes_scale = dpuGetTensorScale(boxes);
        const int8_t *scores_data = dpuGetTensorAddress(scores);
        const float scores_scale = dpuGetTensorScale(scores);
        const int scores_size = dpuGetTensorSize(scores);
        const size_t label_num = 91;

        const auto detected_boxes = ssd_post_processing_dnndk(
            boxes_data, boxes_scale,
            scores_data, scores_scale, scores_size,
            label_num, anchors,
            nms_threshold, conf_threshold);

        cv::Mat out_(height, width, CV_32FC3, out->host + offset);
        in_.copyTo(out_);

        // NOTE: Specifying 1 as id_offset because of the model is trained by tweaked dataset.
        coco_render_boxes(out_, detected_boxes, width, height, 1);
    }

    return 0;
}

}  // namespace dnndk
}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif  // ION_BB_DNN_RT_ORT_H
