#include <cassert>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace ion;

void save_image(Buffer<float> buffer, std::string filename) {
    int width = buffer.width();
    int height = buffer.height();
    int channels = buffer.channels();
    cv::Mat img_out;
    if (channels == 3) {
        cv::Mat img_float;
        cv::merge(std::vector<cv::Mat>{
                      cv::Mat(height, width, CV_32F, buffer.data() + width * height * 2),
                      cv::Mat(height, width, CV_32F, buffer.data() + width * height * 1),
                      cv::Mat(height, width, CV_32F, buffer.data())},
                  img_float);
        img_float.convertTo(img_out, CV_8U, 255);
    } else {
        cv::Mat img_float(height, width, CV_32F, buffer.data());
        img_float.convertTo(img_out, CV_8U, 255);
    }

    cv::imwrite(filename, img_out);
}

int main(int argc, char *argv[]) {
    try {
        constexpr int32_t width = 3264;
        constexpr int32_t height = 2464;
        constexpr float resize_scale = 0.4f;

        float offset_r = 1.f / 16.f;
        float offset_g = 1.f / 16.f;
        float offset_b = 1.f / 16.f;
        float gain_r = 2.5f;
        float gain_g = 2.0f;
        float gain_b = 3.2f;
        float shading_correction_slope_r = 0.7f;
        float shading_correction_slope_g = 0.2f;
        float shading_correction_slope_b = 0.1f;
        float shading_correction_offset_r = 1.f;
        float shading_correction_offset_g = 1.f;
        float shading_correction_offset_b = 1.f;
        float coef_color = 100.f;
        float coef_space = 0.03f;
        float gamma = 1.f / 2.2f;
        float k1 = 0.f;
        float k2 = 0.f;
        float k3 = 0.f;
        float scale = resize_scale;
        float p1 = 0.f;
        float p2 = 0.f;
        float output_scale = 1.f;
        float fx = static_cast<float>(sqrt(width * width + height * height) / 2);
        float fy = static_cast<float>(sqrt(width * width + height * height) / 2);
        float cx = width * 0.5f;
        float cy = height * 0.6f;

        Builder b;
        b.set_target(get_target_from_environment());
        b.with_bb_module("ion-bb");

        Param bayer_pattern("bayer_pattern", "RGGB");

        Node loader, normalize, offset, shading_correction, white_balance, demosaic, luminance, filtered_luminance, luminance_filter, noise_reduction;
        Node color_matrix, color_conversion, gamma_correction, distortion_lut, distortion_correction, resize;
        Node debug_output;

        loader = b.add("image_io_grayscale_data_loader")
            .set_param(
                Param("width", 3264),
                Param("height", 2464),
                Param("url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/IMX219-3264x2464-RG10.raw"));

        normalize = b.add("image_processing_normalize_raw_image")
            .set_param(
                Param("bit_width", 10),
                Param("bit_shift", 0))(
                    loader["output"]);

        offset = b.add("image_processing_bayer_offset")
            .set_param(
                bayer_pattern)(
                    &offset_r,
                    &offset_g,
                    &offset_b,
                    normalize["output"]);

        shading_correction = b.add("image_processing_lens_shading_correction_linear")
            .set_param(
                bayer_pattern,
                Param("width", width),
                Param("height", height))(
                    &shading_correction_slope_r,
                    &shading_correction_slope_g,
                    &shading_correction_slope_b,
                    &shading_correction_offset_r,
                    &shading_correction_offset_g,
                    &shading_correction_offset_b,
                    offset["output"]);
        white_balance = b.add("image_processing_bayer_white_balance")
            .set_param(
                bayer_pattern)(
                    &gain_r,
                    &gain_g,
                    &gain_b,
                    shading_correction["output"]);
        demosaic = b.add("image_processing_bayer_demosaic_filter")
            .set_param(
                bayer_pattern,
                Param("width", width),
                Param("height", height))(
                    white_balance["output"]);
        luminance = b.add("image_processing_calc_luminance")
            .set_param(
                Param("luminance_method", "Average"))(
                    demosaic["output"]);
        luminance_filter = b.add("base_constant_buffer_2d_float")
            .set_param(
                Param("values", 0.04),
                Param("extent0", 5),
                Param("extent1", 5));
        filtered_luminance = b.add("image_processing_convolution_2d")
            .set_param(
                Param("boundary_conditions_method", "MirrorInterior"),
                Param("window_size", 2),
                Param("width", width),
                Param("height", height))(
                    luminance_filter["output"],
                    luminance["output"]);
        noise_reduction = b.add("image_processing_bilateral_filter_3d")
            .set_param(
                Param("color_difference_method", "Average"),
                Param("window_size", 2),
                Param("width", width),
                Param("height", height))(
                    &coef_color,
                    &coef_space,
                    filtered_luminance["output"],
                    demosaic["output"]);
        color_matrix = b.add("base_constant_buffer_2d_float")
            .set_param(
                Param{"values", "2.20213000 -1.27425000 0.07212000 "
                "-0.25650000 1.45961000 -0.20311000 "
                "0.07458000 -1.35791000 2.28333000"},
                Param("extent0", 3),
                Param("extent1", 3));
        color_conversion = b.add("image_processing_color_matrix")(
            color_matrix["output"],
            noise_reduction["output"]);
        distortion_correction = b.add("image_processing_lens_distortion_correction_model_3d")
            .set_param(
                Param("width", width),
                Param("height", height))(
                    &k1,
                    &k2,
                    &k3,
                    &p1,
                    &p2,
                    &fx,
                    &fy,
                    &cx,
                    &cy,
                    &output_scale,
                    color_conversion["output"]);
        resize = b.add("image_processing_resize_area_average_3d")
            .set_param(
                Param("width", width),
                Param("height", height),
                Param("scale", resize_scale))(
                    distortion_correction["output"]);
        gamma_correction = b.add("image_processing_gamma_correction_3d")(
            &gamma,
            resize["output"]);

        Buffer<float> obuf(width * resize_scale, height * resize_scale, 3);
        gamma_correction["output"].bind(obuf);

        b.run();
        obuf.copy_to_host();
        save_image(obuf, "output.png");

    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
