#ifndef ION_BB_DNN_RT_TRT_H
#define ION_BB_DNN_RT_TRT_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

//
// CUDA
//
// NOTE: Guard with anonymous ns to prevent symbol collision
namespace {

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

enum cudaDeviceAttr {
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
};

using cudaMalloc_t = int (*)(void **devPtr, size_t size);
using cudaMemset_t = int (*)(void *devPtr, int value, size_t count);
using cudaMemcpy_t = int (*)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
using cudaFree_t = int (*)(void *devPtr);
using cudaGetDevice_t = int (*)(int *device);
using cudaDeviceGetAttribute_t = int (*)(int *value, enum cudaDeviceAttr attr, int device);

cudaMalloc_t cudaMalloc = nullptr;
cudaMemset_t cudaMemset = nullptr;
cudaMemcpy_t cudaMemcpy = nullptr;
cudaFree_t cudaFree = nullptr;
cudaGetDevice_t cudaGetDevice = nullptr;
cudaDeviceGetAttribute_t cudaDeviceGetAttribute = nullptr;

} // anonymous

//
// NvInfer
//
using createInferRuntime_INTERNAL_t = void* (*)(void* logger, int32_t version);
using createInferRefitter_INTERNAL_t = void* (*)(void* engine, void* logger, int32_t version);

createInferRuntime_INTERNAL_t createInferRuntime_INTERNAL = nullptr;
createInferRefitter_INTERNAL_t createInferRefitter_INTERNAL = nullptr;

#include "NvInferRuntime.h"

//
// NvInferPlugin
//
using initLibNvInferPlugins_t = bool (*)(void* logger, const char* libNamespace);
initLibNvInferPlugins_t initLibNvInferPlugins;

namespace ion {
namespace bb {
namespace dnn {
namespace trt {

std::vector<uint8_t> load(const std::string& path)
{
    std::vector<uint8_t> buffer;

    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("File not found : " + path);
    }

    ifs.seekg(0, std::ifstream::end);
    std::ifstream::pos_type end = ifs.tellg();

    ifs.seekg(0, std::ifstream::beg);
    std::ifstream::pos_type beg = ifs.tellg();

    std::size_t size_in_byte = end-beg;

    buffer.resize(size_in_byte);

    ifs.read(reinterpret_cast<char*>(buffer.data()), size_in_byte);

    return buffer;
}

class Logger : public nvinfer1::ILogger {
 public:
     void log(Severity severity, const char* msg) override
     {
         std::cerr << msg << std::endl;
     }
};

class SessionManager {
 public:
     static SessionManager& get_instance(const std::string& uuid, const std::string& model_root_url, const std::string& cache_root) {
         static std::map<std::string, std::unique_ptr<SessionManager>> map_;
         SessionManager *sess;
         if (map_.count(uuid) == 0) {
             map_[uuid] = std::unique_ptr<SessionManager>(new SessionManager(model_root_url, cache_root));
         }
         return *map_[uuid].get();
     }

     ~SessionManager() {
         for (auto b : buffers_) {
             cudaFree(std::get<0>(b));
         }
         buffers_.clear();
     }

     nvinfer1::IExecutionContext *get_context() {
         return context_;
     }

     std::vector<std::tuple<void*, size_t>> get_buffers() {
         return buffers_;
     }

 private:
     SessionManager(const std::string& model_root_url, const std::string& cache_root)
         : cudart_dm_("cudart"), nvinfer_dm_("nvinfer"), nvinfer_plugin_dm_("nvinfer_plugin")
     {
         using namespace nvinfer1;

         if (!init()) {
             throw std::runtime_error("Failed to initialize runtime libraries");
         }

         int device = 0;
         if (cudaGetDevice(&device) != 0) {
             throw std::runtime_error("Failed to get CUDA device");
         }

         int major = 0;
         if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != 0) {
             throw std::runtime_error("Failed to get CUDA device attribute");
         }

         int minor = 0;
         if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != 0) {
             throw std::runtime_error("Failed to get CUDA device attribute");
         }

         std::string model_name;
         if (major == 7 && minor == 5) {
             // Assumes latest x86 environment
             model_name = "trt721_arch75_fp16.engine";
         } else if (major == 7 && minor == 2) {
             // Jetson NX + JetPack 4.4
             model_name = "trt713_arch72_fp16.engine";
         } else if (major == 5 && minor == 3) {
             // Jetson Nano + JetPack 4.4
             model_name = "trt713_arch53_fp16.engine";
         } else {
             throw std::runtime_error("Unsupported CUDA device");
         }

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
                 throw std::runtime_error("Invalid model URL : " + model_url);
             }

             httplib::Client cli(host_name.c_str());
             cli.set_follow_location(true);
             auto res = cli.Get(path_name.c_str());
             if (!res || res->status != 200) {
                 throw std::runtime_error("Failed to download model : " + model_url);
             }

             model_.resize(res->body.size());
             std::memcpy(model_.data(), res->body.c_str(), res->body.size());

             std::ofstream ofs (cache_root + model_name, std::ios::binary);
             ofs.write(reinterpret_cast<const char*>(model_.data()), model_.size());
         }

         if (!initLibNvInferPlugins(nullptr, "")) {
             throw std::runtime_error("Failed to initialize TensorRT plugin");
         }

         IRuntime *runtime = createInferRuntime(logger_);
         if (runtime == nullptr) {
             throw std::runtime_error("Failed to create TensorRT runtime");
         }

         ICudaEngine *engine = runtime->deserializeCudaEngine(model_.data(), model_.size(), nullptr);
         if (engine == nullptr) {
             throw std::runtime_error("Failed to create TensorRT inference engine");
         }

         context_ = engine->createExecutionContext();
         if (context_ == nullptr) {
             throw std::runtime_error("Failed to create TensorRT inference context");
         }


         for (int i=0; i<engine->getNbBindings(); ++i) {
             auto type = engine->getBindingDataType(i);
             size_t size_in_bytes = 0;
             switch (type) {
             case DataType::kFLOAT: size_in_bytes = 4; break;
             case DataType::kHALF:  size_in_bytes = 2; break;
             case DataType::kINT8:  size_in_bytes = 1; break;
             case DataType::kINT32: size_in_bytes = 4; break;
             case DataType::kBOOL:  size_in_bytes = 1; break;
             default: throw std::runtime_error("Unknown data type");
             }

             auto dim = engine->getBindingDimensions(i);
             for (int j=0; j<dim.nbDims; ++j) {
                 size_in_bytes *= static_cast<size_t>(dim.d[j]);
             }

             void *ptr = nullptr;
             if (cudaMalloc(reinterpret_cast<void**>(&ptr), size_in_bytes) != 0) {
                 throw std::runtime_error("Failed to allocate I/O buffer");
             }
             buffers_.push_back(std::make_tuple(ptr, size_in_bytes));
         }
     }

     bool init() {
         if (!cudart_dm_.is_available()) {
             return false;
         }

         if (!nvinfer_dm_.is_available()) {
             return false;
         }

         if (!nvinfer_plugin_dm_.is_available()) {
             return false;
         }

#define RESOLVE_SYMBOL(DM, SYM_NAME)                               \
          SYM_NAME = DM.get_symbol<SYM_NAME ## _t>(#SYM_NAME);      \
          if (SYM_NAME == nullptr) {                                \
              throw std::runtime_error(                             \
                  #SYM_NAME " is unavailable on your edgetpu DSO"); \
          }

        RESOLVE_SYMBOL(cudart_dm_, cudaMalloc);
        RESOLVE_SYMBOL(cudart_dm_, cudaMemset);
        RESOLVE_SYMBOL(cudart_dm_, cudaMemcpy);
        RESOLVE_SYMBOL(cudart_dm_, cudaFree);
        RESOLVE_SYMBOL(cudart_dm_, cudaGetDevice);
        RESOLVE_SYMBOL(cudart_dm_, cudaDeviceGetAttribute);
        RESOLVE_SYMBOL(nvinfer_dm_, createInferRuntime_INTERNAL);
        RESOLVE_SYMBOL(nvinfer_dm_, createInferRefitter_INTERNAL);
        RESOLVE_SYMBOL(nvinfer_plugin_dm_, initLibNvInferPlugins);

#undef RESOLVE_SYMBOL

        return true;
     }

     ion::bb::dnn::DynamicModule cudart_dm_;
     ion::bb::dnn::DynamicModule nvinfer_dm_;
     ion::bb::dnn::DynamicModule nvinfer_plugin_dm_;

     nvinfer1::IExecutionContext *context_;
     std::vector<std::tuple<void*, size_t>> buffers_;

     Logger logger_;

     std::vector<uint8_t> model_;
};

bool is_available() {
    void *handle = nullptr;

    handle = dlopen("libcudart.so", RTLD_LAZY);
    if (handle == NULL) {
        return false;
    }

    handle = dlopen("libnvinfer.so", RTLD_LAZY);
    if (handle == NULL) {
        return false;
    }

    handle = dlopen("libnvinfer_plugin.so", RTLD_LAZY);
    if (handle == NULL) {
        return false;
    }

    return true;
}

int object_detection_ssd(halide_buffer_t *in,
                         const std::string& session_id,
                         const std::string& model_root_url,
                         const std::string& cache_root,
                         halide_buffer_t *out) {
    using namespace nvinfer1;

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    auto& session = SessionManager::get_instance(session_id, model_root_url, cache_root);

    // auto in_ = cv::imread("in.png");
    // std::cout << "in.empty() : " << in.empty() << std::endl;

    cv::Mat in_(height, width, CV_32FC3, in->host);

    cv::Mat out_(height, width, CV_32FC3, out->host);
    in_.copyTo(out_);

    const int internal_width = 1248;
    const int internal_height = 384;

    const float internal_ratio = static_cast<float>(internal_width) / static_cast<float>(internal_height);
    const float ratio = static_cast<float>(width) / static_cast<float>(height);
    if (ratio > internal_ratio) {
        cv::resize(in_, in_, cv::Size(internal_width, internal_width / ratio));
    } else {
        cv::resize(in_, in_, cv::Size(ratio * internal_height, internal_height));
    }

    const float resize_ratio = static_cast<float>(width) / static_cast<float>(in_.cols);

    int top  = std::max((internal_height - in_.rows) / 2, 0);
    int bottom = internal_height - in_.rows - top;
    int left = std::max((internal_width - in_.cols) / 2, 0);
    int right = internal_width - in_.cols - left;

    cv::copyMakeBorder(in_, in_, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

    cv::normalize(in_, in_, 0, 255.0, cv::NORM_MINMAX, CV_32FC3);

    in_ = in_.reshape(1, internal_width*internal_height).t();

    // cv::Mat test;
    // cv::imwrite("test.png", in_.reshape(1, 384*3));

    IExecutionContext *context = session.get_context();

    auto buffers = session.get_buffers();
    auto input = buffers.at(0);
    auto output = buffers.at(1);

    if (cudaMemcpy(std::get<0>(input), in_.ptr(), std::get<1>(input), cudaMemcpyHostToDevice) != 0) {
        throw std::runtime_error("Failed to copy input data");
    }

    std::vector<void*> bindings;
    for (auto b : buffers) {
        bindings.push_back(std::get<0>(b));
    }
    const int32_t batch_size = 1;
    if (!context->execute(batch_size, bindings.data())) {
        throw std::runtime_error("Failed to execute TensorRT infererence");
    }

    std::vector<float> output_host(std::get<1>(output)/sizeof(float));
    if (cudaMemcpy(output_host.data(), std::get<0>(output), std::get<1>(output), cudaMemcpyDeviceToHost) != 0) {
        throw std::runtime_error("Failed to copy output data");
    }

    // 200 * [image_id, label, confidence, xmin, ymin, xmax, ymax]
    for (int i=0; i<200; ++i) {
        auto confidence = output_host[i * 7 + 2];
        if (confidence > 0.4) {
            int x1 = output_host[i * 7 + 3] * internal_width;
            int y1 = output_host[i * 7 + 4] * internal_height;
            int x2 = output_host[i * 7 + 5] * internal_width;
            int y2 = output_host[i * 7 + 6] * internal_height;

            x1 -= left;
            y1 -= top;
            x2 -= left;
            y2 -= top;

            x1 *= resize_ratio;
            y1 *= resize_ratio;
            x2 *= resize_ratio;
            y2 *= resize_ratio;

            const cv::Point2d p1(x1, y1);
            const cv::Point2d p2(x2, y2);
            const cv::Scalar color = cv::Scalar(1.0, 0, 0);
            cv::rectangle(out_, p1, p2, color);
        }
        // cv::putText(frame, label, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);
    }

    return 0;
}

} // trt
} // dnn
} // bb
} // ion

#endif // ION_BB_DNN_RT_TRT_H
