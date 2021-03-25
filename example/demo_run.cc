#include "demo.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-image-io/rt.h"
#include "ion-bb-image-processing/rt.h"
#include "ion-bb-sgm/rt.h"

#include <HalideBuffer.h>

#include <iostream>
#include <vector>

using namespace ion;

int main() {
    try {
        int d435_width = 1280;
        int d435_height = 720;
        int32_t imx219_width = 3264;
        int32_t imx219_height = 2464;
        int32_t yolo_width = 416;
        int32_t yolo_height = 416;

        // Pipeline parameters
        float offset_r = 1.f / 16.f;
        float offset_g = 1.f / 16.f;
        float offset_b = 1.f / 16.f;
        float shading_correction_slope_r = 0.7f;
        float shading_correction_slope_g = 0.2f;
        float shading_correction_slope_b = 0.1f;
        float shading_correction_offset_r = 1.f;
        float shading_correction_offset_g = 1.f;
        float shading_correction_offset_b = 1.f;
        float gain_r = 2.5f;
        float gain_g = 2.0f;
        float gain_b = 3.2f;
        float gamma = 1 / 2.2f;

        Halide::Runtime::Buffer<uint16_t> depth_buf(std::vector<int>{d435_width, d435_height});
        Halide::Runtime::Buffer<uint8_t> sgm_buf(std::vector<int>{d435_width, d435_height});
        Halide::Runtime::Buffer<uint8_t> yolo_buf(std::vector<int>{3, yolo_width * 3, yolo_height * 2});

        // Initial execution
        halide_reuse_device_allocations(nullptr, true);
        demo(gain_b, gain_g, gain_r, gamma, offset_b, offset_g, offset_r, shading_correction_offset_b, shading_correction_offset_g, shading_correction_offset_r, shading_correction_slope_b, shading_correction_slope_g, shading_correction_slope_r, yolo_buf, depth_buf, sgm_buf);
        halide_profiler_reset();

        const int iter = 5;
        for (int i = 0; i < iter; ++i) {
            demo(gain_b, gain_g, gain_r, gamma, offset_b, offset_g, offset_r, shading_correction_offset_b, shading_correction_offset_g, shading_correction_offset_r, shading_correction_slope_b, shading_correction_slope_g, shading_correction_slope_r, yolo_buf, depth_buf, sgm_buf);
        }

        depth_buf.copy_to_host();
        sgm_buf.copy_to_host();
        yolo_buf.copy_to_host();

        // cv::Mat depth_img(std::vector<int>{depth_height, depth_width}, CV_U16C1, depth_buf.data());
        cv::Mat sgm_img(std::vector<int>{d435_height, d435_width}, CV_8UC1, sgm_buf.data());
        cv::Mat yolo_img(std::vector<int>{yolo_height * 2, yolo_width * 3}, CV_8UC3, yolo_buf.data());
        // cv::imwrite("demo-depth.png", depth_img);
        cv::imwrite("demo-sgm.png", sgm_img);
        cv::imwrite("demo-yolo.png", yolo_img);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
