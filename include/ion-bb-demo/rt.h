#ifndef ION_BB_DEMO_RT_H
#define ION_BB_DEMO_RT_H

#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include <HalideBuffer.h>

#include "rt_realsense.h"
#include "rt_v4l2.h"

extern "C" ION_EXPORT int ion_bb_demo_gui_display(halide_buffer_t *in, int width, int height, int idx, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = 3;  // RGB
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    } else {
        if (getenv("DISPLAY")) {
            Halide::Runtime::Buffer<uint8_t> ibuf(*in);
            ibuf.copy_to_host();
            cv::Mat img(std::vector<int>{height, width}, CV_8UC3, ibuf.data());
            cv::imshow("img" + std::to_string(idx), img);
            cv::waitKey(1);
        } else {
            // This is shimulation mode. Just sleep 1/1000 second.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    return 0;
}

#endif
