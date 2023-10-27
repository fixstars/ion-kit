#ifndef ION_BB_DNN_RT_TFL_H
#define ION_BB_DNN_RT_TFL_H

#include <memory>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rt_ssd.h"
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

     std::shared_ptr<TfLiteInterpreter> get_interpreter(const std::string& model_root_url, const std::string& cache_root) {
         std::string model_name;
        if (is_available_edgetpu_) {
            model_name = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
        } else {
            model_name = "ssd_mobilenet_v2_coco_quant_postprocess.tflite";
        }

        std::string model_url = model_root_url + model_name;
        if (objects_.count(model_url)) {
            return objects_[model_url].interpreter;
        }

        std::shared_ptr<std::vector<uint8_t>> model_data;
        std::ifstream ifs(cache_root + model_name, std::ios::binary);
        if (ifs.is_open()) {
            auto begin = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            auto end = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            model_data = std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(end-begin));
            ifs.read(reinterpret_cast<char *>(model_data->data()), model_data->size());
        }  else {
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

            std::ofstream ofs (cache_root + model_name, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(model_data->data()), model_data->size());
        }

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

int object_detection_tfl(halide_buffer_t *in,
                         const std::string& model_root_url,
                         const std::string& cache_root,
                         halide_buffer_t *out) {

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    size_t input_size = 3 * width * height * sizeof(float);

    int num_images = in->dimensions == 3 ? 1 : in->dim[3].extent;

    for (int i=0; i<num_images; ++i) {
        int offset = input_size * i;
        cv::Mat in_(height, width, CV_32FC3, in->host + offset);

        auto interpreter = TflSessionManager::get_instance().get_interpreter(model_root_url, cache_root);

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

        const auto detected_boxes = ssd_post_processing(boxes_ptr, classes_ptr, scores_ptr, static_cast<int>(*num_ptr));

        cv::Mat out_(height, width, CV_32FC3, out->host + offset);
        in_.copyTo(out_);

        // NOTE: Specifying 1 as id_offset because of the model is trained by tweaked dataset.
        coco_render_boxes(out_, detected_boxes, width, height, 1);
    }

    return 0;
}

}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif // ION_BB_DNN_RT_TFL_H
