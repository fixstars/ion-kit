#ifndef ION_BB_DNN_RT_TFL_H
#define ION_BB_DNN_RT_TFL_H

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "httplib.h"
#include "edgetpu_c.h"
#include "tensorflowlite_c.h"

namespace ion {
namespace bb {
namespace dnn {

class TflSessionManager {
 public:
     static TflSessionManager& get_instance() {
         static TflSessionManager instance;
         return instance;
     }

     struct TfLiteObjects {
         // Need to hold model data permanently
         std::shared_ptr<std::vector<uint8_t>> model_data;
         std::shared_ptr<TfLiteModel> model;
         std::shared_ptr<TfLiteDelegate> delegate;
         std::shared_ptr<TfLiteInterpreterOptions> options;
         std::shared_ptr<TfLiteInterpreter> interpreter;
     };

     std::shared_ptr<TfLiteInterpreter> get_interpreter(const std::string& model_root_url) {
        std::string model_url(model_root_url);
        if (is_available_edgetpu_) {
            model_url += "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
        } else {
            model_url += "ssd_mobilenet_v2_coco_quant_postprocess.tflite";
        }

        if (objects_.count(model_url)) {
            return objects_[model_url].interpreter;
        }

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

        std::shared_ptr<std::vector<uint8_t>> model_data(new std::vector<uint8_t>(res->body.size()));
        std::memcpy(model_data->data(), res->body.c_str(), res->body.size());
        std::shared_ptr<TfLiteModel> model(TfLiteModelCreate(model_data->data(), model_data->size()), TfLiteModelDelete);
        if (model == nullptr) {
            std::cerr << "Illegal model format : " << model_url << std::endl;
            return nullptr;
        }

        std::shared_ptr<TfLiteInterpreterOptions> options(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
        TfLiteInterpreterOptionsSetNumThreads(options.get(), 1);
        std::shared_ptr<TfLiteDelegate> delegate;
        if (is_available_edgetpu_) {
            // Determin device
            size_t num_devices;
            std::unique_ptr<edgetpu_device, decltype(edgetpu_free_devices)>
                devices(edgetpu_list_devices(&num_devices), edgetpu_free_devices);
            if (num_devices == 0) {
                std::cerr << "No device found" << std::endl;
                return nullptr;
            }
            const auto& device = devices.get()[0];

            // Create EdgeTpu delegate
            delegate = std::shared_ptr<TfLiteDelegate>(edgetpu_create_delegate(device.type, device.path, nullptr, 0), edgetpu_free_delegate);

            TfLiteInterpreterOptionsAddDelegate(options.get(), delegate.get());
        }

        // Build interpreter
        std::shared_ptr<TfLiteInterpreter>
            interpreter(TfLiteInterpreterCreate(model.get(), options.get()), TfLiteInterpreterDelete);
        if (interpreter == nullptr) {
            std::cerr << "Failed to build interpreter" << std::endl;
            return nullptr;
        }

        if (TfLiteInterpreterAllocateTensors(interpreter.get())!= kTfLiteOk) {
            std::cerr << "Failed to allocate tensors." << std::endl;
            return nullptr;
        }

        objects_[model_url] = TfLiteObjects{
            model_data, model, delegate, options, interpreter
        };
        return interpreter;
     }

     bool is_available() {
         return is_available_tflite_;
     }

 private:
    TflSessionManager()
        : is_available_tflite_(false), is_available_edgetpu_(false)
    {
        if (!tensorflowlite_init()) {
            return;
        }
        is_available_tflite_ = true;

        if (!edgetpu_init()) {
            return;
        }
        is_available_edgetpu_ = true;
    }

    bool is_available_tflite_;
    bool is_available_edgetpu_;

    std::unordered_map<std::string, TfLiteObjects> objects_;
};

bool is_tfl_available() {
    return TflSessionManager::get_instance().is_available();
}

int ssd_render_boxes(cv::Mat &frame, float *boxes, float *classes, float *scores, int nums, int32_t w, int32_t h) {

    static const std::map<int, std::pair<const char *, cv::Scalar>> label_color_map = {
        { 0, {"person", cv::Scalar(111, 221, 142)}},
        { 1, {"bicycle", cv::Scalar(199, 151, 121)}},
        { 2, {"car", cv::Scalar(145, 233, 34)}},
        { 3, {"motorbike", cv::Scalar(110, 131, 63)}},
        { 4, {"aeroplane", cv::Scalar(251, 141, 195)}},
        { 5, {"bus", cv::Scalar(136, 137, 194)}},
        { 6, {"train", cv::Scalar(114, 27, 34)}},
        { 7, {"truck", cv::Scalar(172, 221, 65)}},
        { 8, {"boat", cv::Scalar(7, 30, 178)}},
        { 9, {"traffic light", cv::Scalar(31, 28, 230)}},
        {10, {"fire hydrant", cv::Scalar(66, 214, 26)}},
        {12, {"stop sign", cv::Scalar(133, 39, 182)}},
        {13, {"parking meter", cv::Scalar(33, 20, 48)}},
        {14, {"bench", cv::Scalar(174, 253, 25)}},
        {15, {"bird", cv::Scalar(212, 160, 0)}},
        {16, {"cat", cv::Scalar(88, 78, 255)}},
        {17, {"dog", cv::Scalar(183, 35, 220)}},
        {18, {"horse", cv::Scalar(118, 157, 99)}},
        {19, {"sheep", cv::Scalar(81, 39, 129)}},
        {20, {"cow", cv::Scalar(253, 97, 253)}},
        {21, {"elephant", cv::Scalar(208, 170, 203)}},
        {22, {"bear", cv::Scalar(209, 175, 193)}},
        {23, {"zebra", cv::Scalar(43, 32, 163)}},
        {24, {"giraffe", cv::Scalar(246, 162, 213)}},
        {26, {"backpack", cv::Scalar(150, 199, 251)}},
        {27, {"umbrella", cv::Scalar(225, 165, 42)}},
        {30, {"handbag", cv::Scalar(56, 139, 51)}},
        {31, {"tie", cv::Scalar(235, 82, 61)}},
        {32, {"suitcase", cv::Scalar(219, 129, 248)}},
        {33, {"frisbee", cv::Scalar(120, 74, 139)}},
        {34, {"skis", cv::Scalar(164, 201, 240)}},
        {35, {"snowboard", cv::Scalar(238, 83, 85)}},
        {36, {"sports ball", cv::Scalar(134, 120, 102)}},
        {37, {"kite", cv::Scalar(166, 149, 183)}},
        {38, {"baseball bat", cv::Scalar(243, 13, 18)}},
        {39, {"baseball glove", cv::Scalar(56, 182, 85)}},
        {40, {"skateboard", cv::Scalar(117, 60, 48)}},
        {41, {"surfboard", cv::Scalar(109, 204, 30)}},
        {42, {"tennis racket", cv::Scalar(245, 221, 109)}},
        {43, {"bottle", cv::Scalar(74, 27, 47)}},
        {45, {"wine glass", cv::Scalar(229, 166, 29)}},
        {46, {"cup", cv::Scalar(158, 219, 241)}},
        {47, {"fork", cv::Scalar(95, 153, 84)}},
        {48, {"knife", cv::Scalar(218, 183, 12)}},
        {49, {"spoon", cv::Scalar(146, 37, 136)}},
        {50, {"bowl", cv::Scalar(63, 212, 25)}},
        {51, {"banana", cv::Scalar(174, 9, 96)}},
        {52, {"apple", cv::Scalar(180, 104, 193)}},
        {53, {"sandwich", cv::Scalar(160, 117, 33)}},
        {54, {"orange", cv::Scalar(224, 42, 115)}},
        {55, {"broccoli", cv::Scalar(9, 49, 96)}},
        {56, {"carrot", cv::Scalar(124, 213, 203)}},
        {57, {"hot dog", cv::Scalar(187, 193, 196)}},
        {58, {"pizza", cv::Scalar(57, 25, 171)}},
        {59, {"donut", cv::Scalar(189, 74, 145)}},
        {60, {"cake", cv::Scalar(73, 119, 11)}},
        {61, {"chair", cv::Scalar(37, 253, 178)}},
        {62, {"sofa", cv::Scalar(83, 223, 49)}},
        {63, {"pottedplant", cv::Scalar(111, 216, 113)}},
        {64, {"bed", cv::Scalar(167, 152, 203)}},
        {66, {"diningtable", cv::Scalar(99, 144, 184)}},
        {69, {"toilet", cv::Scalar(100, 204, 167)}},
        {71, {"tvmonitor", cv::Scalar(203, 87, 87)}},
        {72, {"laptop", cv::Scalar(139, 188, 41)}},
        {73, {"mouse", cv::Scalar(23, 84, 185)}},
        {74, {"remote", cv::Scalar(79, 160, 205)}},
        {75, {"keyboard", cv::Scalar(63, 7, 87)}},
        {76, {"cell phone", cv::Scalar(197, 255, 152)}},
        {77, {"microwave", cv::Scalar(199, 123, 207)}},
        {78, {"oven", cv::Scalar(211, 86, 200)}},
        {79, {"toaster", cv::Scalar(232, 184, 61)}},
        {80, {"sink", cv::Scalar(226, 254, 156)}},
        {81, {"refrigerator", cv::Scalar(195, 207, 141)}},
        {83, {"book", cv::Scalar(238, 101, 223)}},
        {84, {"clock", cv::Scalar(24, 84, 233)}},
        {85, {"vase", cv::Scalar(39, 104, 233)}},
        {86, {"scissors", cv::Scalar(49, 115, 78)}},
        {87, {"teddy bear", cv::Scalar(199, 193, 20)}},
        {88, {"hair drier", cv::Scalar(156, 85, 108)}},
        {89, {"toothbrush", cv::Scalar(189, 59, 8)}},
    };

    for (int i = 0; i < nums; ++i) {
        const auto lc = label_color_map.at(static_cast<int>(classes[i]));
        const auto label = lc.first;
        const auto color = lc.second / 255.0;
        const float top = boxes[4 * i + 0];
        const float left = boxes[4 * i + 1];
        const float bottom = boxes[4 * i + 2];
        const float right = boxes[4 * i + 3];
        const int x1 = left * w;
        const int y1 = top * h;
        const int x2 = right * w;
        const int y2 = bottom * h;
        const cv::Point2d p1(x1, y1);
        const cv::Point2d p2(x2, y2);
        cv::rectangle(frame, p1, p2, color);
        // s += ":" + std::to_string(scores[i]);
        cv::putText(frame, label, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);
    }

    return 0;
}

std::vector<DetectionBox> post_processing(const float *boxes, const float *classes, const float *scores, const int num, const float conf_thresh = 0.4, const float nms_thresh = 0.4) {
    std::vector<DetectionBox> all_boxes;

    for (int i = 0; i < num; i++) {
        const auto max_conf = scores[i];
        const auto max_id = classes[i];

        if (max_conf > conf_thresh) {
            DetectionBox b;
            b.max_conf = max_conf;
            b.max_id = max_id;
            b.x1 = boxes[i * 4 + 1];
            b.y1 = boxes[i * 4 + 0];
            b.x2 = boxes[i * 4 + 3];
            b.y2 = boxes[i * 4 + 2];
            all_boxes.push_back(b);
        }
    }

    std::vector<bool> is_valid(all_boxes.size(), true);

    for (int i = 0; i < all_boxes.size(); i++) {
        if (!is_valid[i]) continue;
        const auto main = all_boxes[i];
        for (int j = i + 1; j < all_boxes.size(); j++) {
            if (!is_valid[j]) continue;
            const auto other = all_boxes[j];
            const auto iou = intersection(main, other) / union_(main, other);
            is_valid[j] = iou <= nms_thresh;
        }
    }

    std::vector<DetectionBox> detected_boxes;
    for (int i = 0; i < all_boxes.size(); i++) {
        if (is_valid[i]) detected_boxes.push_back(all_boxes[i]);
    }

    return detected_boxes;
}

int object_detection_tfl(halide_buffer_t *in,
                         const std::string& model_root_url,
                         halide_buffer_t *out) {

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    cv::Mat in_(height, width, CV_32FC3, in->host);

    auto interpreter = TflSessionManager::get_instance().get_interpreter(model_root_url);

    // Prepare input
    TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

    if (channel != TfLiteTensorDim(input, 3)) {
        std::cerr << "Input channel mismatches: "
            << channel << " vs " << TfLiteTensorDim(input, 3) << std::endl;
        return -1;
    }

    const int internal_width = TfLiteTensorDim(input, 2);
    const int internal_height = TfLiteTensorDim(input, 1);

    cv::Mat resized(internal_height, internal_width, CV_32FC3);
    cv::resize(in_, resized, resized.size());

    cv::Mat input_tensor_data(internal_height, internal_width, CV_8UC3);

    resized.convertTo(input_tensor_data, CV_8UC3, 255.0);

    if ((3*input_tensor_data.total()) != TfLiteTensorByteSize(input)) {
        std::cerr << "Input size mismatches: "
            << 3*input_tensor_data.total() << " vs " << TfLiteTensorByteSize(input)
            << std::endl;
        return -1;
    }

    std::memcpy(TfLiteTensorData(input), input_tensor_data.ptr(), TfLiteTensorByteSize(input));

    // Invoke
    if (TfLiteInterpreterInvoke(interpreter.get()) != kTfLiteOk) {
        std::cerr << "Failed to invoke" << std::endl;
        return -1;
    }

    // Prepare output
    const int num_outputs = TfLiteInterpreterGetOutputTensorCount(interpreter.get());
    if (num_outputs != 4) {
        std::cerr << "Unexpected number of output" << std::endl;
        return -1;
    }

    const TfLiteTensor* boxes = TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);
    const TfLiteTensor* classes = TfLiteInterpreterGetOutputTensor(interpreter.get(), 1);
    const TfLiteTensor* scores = TfLiteInterpreterGetOutputTensor(interpreter.get(), 2);
    const TfLiteTensor* num = TfLiteInterpreterGetOutputTensor(interpreter.get(), 3);

    float *boxes_ptr = reinterpret_cast<float*>(TfLiteTensorData(boxes));
    float *classes_ptr = reinterpret_cast<float*>(TfLiteTensorData(classes));
    float *scores_ptr = reinterpret_cast<float*>(TfLiteTensorData(scores));
    float *num_ptr = reinterpret_cast<float*>(TfLiteTensorData(num));

    const auto detected_boxes = post_processing(boxes_ptr, classes_ptr, scores_ptr, static_cast<int>(*num_ptr));

    cv::Mat out_(height, width, CV_32FC3, out->host);
    in_.copyTo(out_);

    render_boxes(out_, detected_boxes, width, height);
    //ssd_render_boxes(out_, boxes_ptr, classes_ptr, scores_ptr, static_cast<int>(*num_ptr), width, height);

    return 0;
}


}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif // ION_BB_DNN_RT_TFL_H
