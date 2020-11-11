#ifndef ION_BB_DNN_RT_TFL_H
#define ION_BB_DNN_RT_TFL_H

#include <memory>

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

     TfLiteInterpreter *get_interpreter(const std::string& model_root_url) {
        std::string model_url(model_root_url);
        if (is_available_edgetpu_) {
            model_url += "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
        } else {
            model_url += "ssd_mobilenet_v2_coco_quant_postprocess.tflite";
        }

        if (interpreters_.count(model_url)) {
            return interpreters_[model_url].get();
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

        std::vector<char> model_data(res->body.size());
        std::memcpy(model_data.data(), res->body.c_str(), res->body.size());
        std::unique_ptr<TfLiteModel, decltype(TfLiteModelDelete)> model(
            TfLiteModelCreate(model_data.data(), model_data.size()), TfLiteModelDelete);
        if (model == nullptr) {
            std::cerr << "Illegal model format : " << model_url << std::endl;
            return nullptr;
        }

        std::unique_ptr<TfLiteInterpreterOptions, decltype(TfLiteInterpreterOptionsDelete)>
            options(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
        TfLiteInterpreterOptionsSetNumThreads(options.get(), 1);
        std::unique_ptr<TfLiteDelegate, decltype(edgetpu_free_delegate)> delegate(nullptr, edgetpu_free_delegate);
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
            delegate = std::unique_ptr<TfLiteDelegate, decltype(edgetpu_free_delegate)>(
                edgetpu_create_delegate(device.type, device.path, nullptr, 0), edgetpu_free_delegate);

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

        interpreters_[model_url] = interpreter;
        return interpreter.get();
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

    std::tuple<std::string, std::string> parse_url(const std::string &url) {
        auto protocol_end_pos = url.find("://");
        if (protocol_end_pos == std::string::npos) {
            return std::tuple<std::string, std::string>("", "");
        }
        auto host_name_pos = protocol_end_pos + 3;
        auto path_name_pos = url.find("/", host_name_pos);
        auto host_name = url.substr(0, path_name_pos);
        auto path_name = url.substr(path_name_pos);
        return std::tuple<std::string, std::string>(host_name, path_name);
    }

    bool is_available_tflite_;
    bool is_available_edgetpu_;

    std::unordered_map<std::string, std::shared_ptr<TfLiteInterpreter>> interpreters_;
};

bool is_tfl_available() {
    return TflSessionManager::get_instance().is_available();
}

int object_detection_tfl(halide_buffer_t *in,
                         const std::string& model_root_url,
                         halide_buffer_t *out) {

    auto& session = TflSessionManager::get_instance();

    TfLiteInterpreter *interpreter = session.get_interpreter(model_root_url);

    // Prepare input
    TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
#if 0
  if (TfLiteTensorByteSize(tensor) != in->size_in_bytes()) {
      std::cerr << "Input size mismatches: "
          << in->size_in_bytes() << " vs " << TfLiteTensorByteSize(tensor)
          << std::endl;
      return -1;
  }

  if (TfLiteTensorDim(tensor, 1) != height ||
      TfLiteTensorDim(tensor, 2) != width ||
      TfLiteTensorDim(tensor, 3) != 3) {
      std::cerr << "Input size mismatches: "
              << "channels: " << 3 << " vs " << TfLiteTensorDim(tensor, 3)
              << ", width: " << width << " vs " << TfLiteTensorDim(tensor, 2)
              << ", height: " << height << " vs " << TfLiteTensorDim(tensor, 1)
              << std::endl;
      return -1;
  }
  std::memcpy(reinterpret_cast<uint8_t*>(TfLiteTensorData(tensor)),
              in->host,
              in->size_in_bytes());
#endif

  // Invoke
  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
      std::cerr << "Failed to invoke" << std::endl;
      return -1;
  }

  // Prepare output
  const int num_outputs = TfLiteInterpreterGetOutputTensorCount(interpreter);
  if (num_outputs != 4) {
      std::cerr << "Unexpected number of output" << std::endl;
      return -1;
  }

  const TfLiteTensor* boxes = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  const TfLiteTensor* classes = TfLiteInterpreterGetOutputTensor(interpreter, 1);
  const TfLiteTensor* scores = TfLiteInterpreterGetOutputTensor(interpreter, 2);
  const TfLiteTensor* num = TfLiteInterpreterGetOutputTensor(interpreter, 3);

  return 0;
}


}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif // ION_BB_DNN_RT_TFL_H
