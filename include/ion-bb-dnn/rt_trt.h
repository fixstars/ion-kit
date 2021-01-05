#if 0
#include <iostream>
#include <exception>
#include <vector>
#include <fstream>
#include <cassert>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class Logger : public nvinfer1::ILogger {
 public:
     void log(Severity severity, const char* msg) override
     {
         std::cerr << msg << std::endl;
     }
};

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

int main()
{
    using namespace nvinfer1;

    Logger logger;

    auto in = cv::imread("in.png");

    std::cout << "in.empty() : " << in.empty() << std::endl;

    cv::Mat in2;
    // cv::resize(in, in2, cv::Size(1248, 384), 0, 0);
    int top  = std::max((384 - in.rows) / 2, 0);
    int bottom = 384 - in.rows - top;
    int left = std::max((1248 - in.cols) / 2, 0);
    int right = 1248 - in.cols - left;

    cv::copyMakeBorder(in, in2, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

    cv::Mat in3;
    cv::normalize(in2, in3, 0, 255.0, cv::NORM_MINMAX, CV_32FC3);

    cv::Mat in4 = in3.reshape(1, 384*1248);
    cv::Mat in5 = in4.t();
    // std::cout << in3.size[0] << " " << in3.size[1] << std::endl;
    // std::cout << in4.size[0] << " " << in4.size[1] << std::endl;
    // std::cout << in5.size[0] << " " << in5.size[1] << std::endl;

    cv::Mat test;
    cv::normalize(in5.reshape(1, 384*3), test, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("test.png", test);

    auto result = initLibNvInferPlugins(nullptr, "");
    assert(result);

    auto buffer = load("trt.engine");

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    assert(engine != nullptr);

    std::cout << "MaxBatchSize : " << engine->getMaxBatchSize() << std::endl;

#if 1
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    int ret = cudaSuccess;
    size_t size = 0;
    std::vector<void *> buffers;

    float *input = nullptr;
    size = 3 * 384 * 1248 * sizeof(float);
    ret = cudaMalloc(reinterpret_cast<void**>(&input), size);
    assert(ret == cudaSuccess);
    ret = cudaMemcpy(input, in5.ptr(), 3 * 384 * 1248 * sizeof(float), cudaMemcpyHostToDevice);
    // ret = cudaMemset(input, 0, size);
    assert(ret == cudaSuccess);
    buffers.push_back(reinterpret_cast<void*>(input));

    float *output0 = nullptr;
    size = 1 * 200 * 7 * sizeof(float);
    ret = cudaMalloc(reinterpret_cast<void**>(&output0), size);
    assert(ret == cudaSuccess);
    ret = cudaMemset(output0, 0, size);
    assert(ret == cudaSuccess);
    buffers.push_back(reinterpret_cast<void*>(output0));

    float *output1 = nullptr;
    size = 1 * 1 * 1 * sizeof(float);
    ret = cudaMalloc(reinterpret_cast<void**>(&output1), size);
    assert(ret == cudaSuccess);
    ret = cudaMemset(output1, 0, size);
    assert(ret == cudaSuccess);
    buffers.push_back(reinterpret_cast<void*>(output1));

    const int32_t batch_size = 1;
    context->execute(batch_size, buffers.data());

    std::vector<float> output0_host(1 * 200 * 7);
    ret = cudaMemcpy(output0_host.data(), output0, 1 * 200 * 7 * sizeof(float), cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    // 200 * [image_id, label, confidence, xmin, ymin, xmax, ymax]
    for (int i=0; i<200; ++i) {
        auto confidence = output0_host[i * 7 + 2];
        if (confidence > 0.4) {
            const int x1 = output0_host[i * 7 + 3] * 1284;
            const int y1 = output0_host[i * 7 + 4] * 384;
            const int x2 = output0_host[i * 7 + 5] * 1284;
            const int y2 = output0_host[i * 7 + 6] * 384;
            const cv::Point2d p1(x1, y1);
            const cv::Point2d p2(x2, y2);
            const cv::Scalar color = cv::Scalar(255, 0, 0);
            cv::rectangle(in2, p1, p2, color);
        }
        // cv::putText(frame, label, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);


        //if (output0_host[i * 7 + 2] > 0.7)  {
        //    std::cout << output0_host[i * 7 + 2] << std::endl;
        //}
    }

    cv::imwrite("out.png", in2);

    // std::cout << output0_host[0] << std::endl;

    cudaFree(input);
    cudaFree(output0);
    cudaFree(output1);

    std::cout << "Done" << std::endl;

#else
    std::cout << engine->getNbBindings() << std::endl;
    std::cout << engine->getBindingName(0) << std::endl;
    {
        auto dim = engine->getBindingDimensions(0);
        std::cout << "type: " << static_cast<int32_t>(engine->getBindingDataType(0)) << std::endl;
        std::cout << "[";
        for (int i=0; i<dim.nbDims; ++i) {
            std::cout << dim.d[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << engine->getBindingName(1) << std::endl;
    {
        auto dim = engine->getBindingDimensions(1);
        std::cout << "type: " << static_cast<int32_t>(engine->getBindingDataType(1)) << std::endl;
        std::cout << "[";
        for (int i=0; i<dim.nbDims; ++i) {
            std::cout << dim.d[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << engine->getBindingName(2) << std::endl;
    {
        auto dim = engine->getBindingDimensions(2);
        std::cout << "type: " << static_cast<int32_t>(engine->getBindingDataType(2)) << std::endl;
        std::cout << "[";
        for (int i=0; i<dim.nbDims; ++i) {
            std::cout << dim.d[i] << " ";
        }
        std::cout << "]" << std::endl;
    }
#endif
return 0;
}
#else

#ifndef ION_BB_DNN_RT_TRT_H
#define ION_BB_DNN_RT_TRT_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

//
// CUDA
//
enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

using cudaMalloc_t = int (*)(void **devPtr, size_t size);
using cudaMemset_t = int (*)(void *devPtr, int value, size_t count);
using cudaMemcpy_t = int (*)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
using cudaFree_t = int (*)(void *devPtr);

cudaMalloc_t cudaMalloc = nullptr;
cudaMemset_t cudaMemset = nullptr;
cudaMemcpy_t cudaMemcpy = nullptr;
cudaFree_t cudaFree = nullptr;

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
     static SessionManager& get_instance(const std::string& uuid, const std::string& model_root_url) {
         static std::map<std::string, std::unique_ptr<SessionManager>> map_;
         SessionManager *sess;
         if (map_.count(uuid) == 0) {
             map_[uuid] = std::unique_ptr<SessionManager>(new SessionManager(model_root_url));
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
     SessionManager(const std::string& model_root_url)
         : cudart_dm_("cudart"), nvinfer_dm_("nvinfer"), nvinfer_plugin_dm_("nvinfer_plugin")
     {
         using namespace nvinfer1;

         if (!init()) {
             throw std::runtime_error("Failed to initialize runtime libraries");
         }

         if (!initLibNvInferPlugins(nullptr, "")) {
             throw std::runtime_error("Failed to initialize TensorRT plugin");
         }

         IRuntime *runtime = createInferRuntime(logger_);
         if (runtime == nullptr) {
             throw std::runtime_error("Failed to create TensorRT runtime");
         }

         auto engine_data = load("trt.engine");

         ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
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

int object_detection(halide_buffer_t *in,
                     const std::string& session_id,
                     const std::string& model_root_url,
                     halide_buffer_t *out) {
    using namespace nvinfer1;

    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    auto& session = SessionManager::get_instance(session_id, model_root_url);

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

    cv::cvtColor(in_, in_, cv::COLOR_BGR2RGB);

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
#endif
