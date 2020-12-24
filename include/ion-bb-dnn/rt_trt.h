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
