#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <HalideBuffer.h>

#include "dnn.h"

int main(int argc, char *argv[]) {
    try {
        const int input_height = 512;
        const int input_width = 512;
        const int input_channel = 3;

        Halide::Runtime::Buffer<uint8_t> out_buf(input_channel, input_width, input_height);

        halide_reuse_device_allocations(nullptr, true);
        dnn(out_buf);
        halide_profiler_reset();

        cv::Mat predicted(input_height, input_width, CV_8UC3, out_buf.data());
        cv::imwrite("predicted.png", predicted);

        std::cout << "dnn run done!!!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
