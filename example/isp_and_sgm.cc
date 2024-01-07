#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace ion;

int main(int argc, char *argv[]) {

    try {
        int32_t raw_width = 5184;
        int32_t raw_height = 1944;
        int32_t buffer_width = raw_width / 2;
        int32_t buffer_height = raw_height;

        // ISP Parameters for OV5647
        float offset_r_l = 1.f / 64.f;
        float offset_g_l = 1.f / 64.f;
        float offset_b_l = 1.f / 64.f;
        float gain_r_l = 1.0f;
        float gain_g_l = 1.0f;
        float gain_b_l = 1.5f;
        float shading_correction_slope_r_l = 1.5f;
        float shading_correction_slope_g_l = 1.0f;
        float shading_correction_slope_b_l = 2.5f;
        float shading_correction_offset_r_l = 1.f;
        float shading_correction_offset_g_l = 1.f;
        float shading_correction_offset_b_l = 1.f;
        float coef_color_l = 100.f;
        float coef_space_l = 0.03;
        float gamma_l = 1.f / 2.2f;
        float k1_l = 0.f;
        float k2_l = 0.f;
        float k3_l = 0.f;
        float p1_l = 0.f;
        float p2_l = 0.f;
        float output_scale_l = 1.f;
        float fx_l = static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height / 2));
        float fy_l = static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height / 2));
        float cx_l = buffer_width * 0.5f;
        float cy_l = buffer_height * 0.6f;
        Param bayer_pattern_l{"bayer_pattern", "GRBG"};

        float offset_r_r = 1.f / 64.f;
        float offset_g_r = 1.f / 64.f;
        float offset_b_r = 1.f / 64.f;
        float gain_r_r = 1.0f;
        float gain_g_r = 1.0f;
        float gain_b_r = 1.5f;
        float shading_correction_slope_r_r = 1.5f;
        float shading_correction_slope_g_r = 1.0f;
        float shading_correction_slope_b_r = 2.5f;
        float shading_correction_offset_r_r = 1.f;
        float shading_correction_offset_g_r = 1.f;
        float shading_correction_offset_b_r = 1.f;
        float coef_color_r = 100.f;
        float coef_space_r = 0.03;
        float gamma_r = 1.f / 2.2f;
        float k1_r = 0.f;
        float k2_r = 0.f;
        float k3_r = 0.f;
        float p1_r = 0.f;
        float p2_r = 0.f;
        float output_scale_r = 1.f;
        float fx_r = static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height / 2));
        float fy_r = static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height / 2));
        float cx_r = buffer_width * 0.5f;
        float cy_r = buffer_height * 0.6f;
        Param bayer_pattern_r{"bayer_pattern", "GRBG"};

        float resize_scale_l = 0.2f;
        float resize_scale_r = 0.2f;

        // SGM Parameters
        int output_width = static_cast<int>(buffer_width * resize_scale_l);
        int output_height = static_cast<int>(buffer_height * resize_scale_l);
        const int disp = 192;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        // ISP Nodes
        Node loader, normalize;
        Node crop_l, crop_r;
        Node offset_l, shading_correction_l, white_balance_l, demosaic_l, luminance_l, filtered_luminance_l, luminance_filter_l, noise_reduction_l;
        Node color_matrix_l, color_conversion_l, gamma_correction_l, distortion_lut_l, distortion_correction_l, resize_l, final_luminance_l;
        Node offset_r, shading_correction_r, white_balance_r, demosaic_r, luminance_r, filtered_luminance_r, luminance_filter_r, noise_reduction_r;
        Node color_matrix_r, color_conversion_r, gamma_correction_r, distortion_lut_r, distortion_correction_r, resize_r, final_luminance_r;

        loader = b.add("image_io_grayscale_data_loader")
            .set_param(
                Param("width", raw_width),
                Param("height", raw_height),
                Param("url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/OV5647x2-5184x1944-GB10.raw"));

        normalize = b.add("image_processing_normalize_raw_image")
            .set_param(
                Param("bit_width", "10"),
                Param("bit_shift", "6"))(
                    loader["output"]);

        crop_l = b.add("image_processing_crop_image_2d_float")
                     .set_param(
                         Param("x_dim", "0"),
                         Param("y_dim", "1"),
                         Param("input_width", "0"),
                         Param("input_height", "0"),
                         Param("top", "0"),
                         Param("left", "0"),
                         Param("output_width", buffer_width),
                         Param("output_height", buffer_height))(
                         normalize["output"]);
        offset_l = b.add("image_processing_bayer_offset")
                       .set_param(
                           bayer_pattern_l)(
                           &offset_r_l,
                           &offset_g_l,
                           &offset_b_l,
                           normalize["output"]);
        shading_correction_l = b.add("image_processing_lens_shading_correction_linear")
                                   .set_param(
                                       bayer_pattern_l,
                                       Param("width", buffer_width),
                                       Param("height", buffer_height))(
                                       &shading_correction_slope_r_l,
                                       &shading_correction_slope_g_l,
                                       &shading_correction_slope_b_l,
                                       &shading_correction_offset_r_l,
                                       &shading_correction_offset_g_l,
                                       &shading_correction_offset_b_l,
                                       offset_l["output"]);
        white_balance_l = b.add("image_processing_bayer_white_balance")
                              .set_param(
                                  bayer_pattern_l)(
                                  &gain_r_l,
                                  &gain_g_l,
                                  &gain_b_l,
                                  shading_correction_l["output"]);
        demosaic_l = b.add("image_processing_bayer_demosaic_filter")
                         .set_param(
                             bayer_pattern_l,
                             Param("width", buffer_width),
                             Param("height", buffer_height))(
                             white_balance_l["output"]);
        luminance_l = b.add("image_processing_calc_luminance")
                          .set_param(
                              Param("luminance_method", "Average"))(
                              demosaic_l["output"]);
        luminance_filter_l = b.add("base_constant_buffer_2d_float")
                                 .set_param(
                                     Param("values", "0.04"),
                                     Param("extent0", "5"),
                                     Param("extent1", "5"));
        filtered_luminance_l = b.add("image_processing_convolution_2d")
                                   .set_param(
                                       Param("boundary_conditions_method", "MirrorInterior"),
                                       Param("window_size", "2"),
                                       Param("width", buffer_width),
                                       Param("height", buffer_height))(
                                       luminance_filter_l["output"],
                                       luminance_l["output"]);
        noise_reduction_l = b.add("image_processing_bilateral_filter_3d")
                                .set_param(
                                    Param("color_difference_method", "Average"),
                                    Param("window_size", "2"),
                                    Param("width", buffer_width),
                                    Param("height", buffer_height))(
                                    &coef_color_l,
                                    &coef_space_l,
                                    filtered_luminance_l["output"],
                                    demosaic_l["output"]);
        color_matrix_l = b.add("base_constant_buffer_2d_float")
                             .set_param(
                                 Param{"values", "1.5 -0.25 -0.25 "
                                                 "-0.25 1.5 -0.25 "
                                                 "-0.25 -0.25 1.5"},
                                 Param("extent0", "3"),
                                 Param("extent1", "3"));
        color_conversion_l = b.add("image_processing_color_matrix")(
            color_matrix_l["output"],
            noise_reduction_l["output"]);
        distortion_correction_l = b.add("image_processing_lens_distortion_correction_model_3d")
                                      .set_param(
                                          Param("width", buffer_width),
                                          Param("height", buffer_height))(
                                          &k1_l,
                                          &k2_l,
                                          &k3_l,
                                          &p1_l,
                                          &p2_l,
                                          &fx_l,
                                          &fy_l,
                                          &cx_l,
                                          &cy_l,
                                          &output_scale_l,
                                          color_conversion_l["output"]);
        resize_l = b.add("image_processing_resize_area_average_3d")
                       .set_param(
                           Param("width", buffer_width),
                           Param("height", buffer_height),
                           Param("scale", resize_scale_l))(
                           distortion_correction_l["output"]);
        final_luminance_l = b.add("image_processing_calc_luminance")
                                .set_param(
                                    Param("luminance_method", "Y"))(
                                    resize_l["output"]);
        gamma_correction_l = b.add("image_processing_gamma_correction_2d")(
            &gamma_l,
            final_luminance_l["output"]);

        crop_r = b.add("image_processing_crop_image_2d_float")
                     .set_param(
                         Param("x_dim", "0"),
                         Param("y_dim", "1"),
                         Param("input_width", "0"),
                         Param("input_height", "0"),
                         Param("top", "0"),
                         Param("left", "0"),
                         Param("output_width", buffer_width),
                         Param("output_height", buffer_height))(
                         normalize["output"]);
        offset_r = b.add("image_processing_bayer_offset")
                       .set_param(
                           bayer_pattern_r)(
                           &offset_r_r,
                           &offset_g_r,
                           &offset_b_r,
                           normalize["output"]);
        shading_correction_r = b.add("image_processing_lens_shading_correction_linear")
                                   .set_param(
                                       bayer_pattern_r,
                                       Param("width", buffer_width),
                                       Param("height", buffer_height))(
                                       &shading_correction_slope_r_r,
                                       &shading_correction_slope_g_r,
                                       &shading_correction_slope_b_r,
                                       &shading_correction_offset_r_r,
                                       &shading_correction_offset_g_r,
                                       &shading_correction_offset_b_r,
                                       offset_r["output"]);
        white_balance_r = b.add("image_processing_bayer_white_balance")
                              .set_param(
                                  bayer_pattern_r)(
                                  &gain_r_r,
                                  &gain_g_r,
                                  &gain_b_r,
                                  shading_correction_r["output"]);
        demosaic_r = b.add("image_processing_bayer_demosaic_filter")
                         .set_param(
                             bayer_pattern_r,
                             Param("width", buffer_width),
                             Param("height", buffer_height))(
                             white_balance_r["output"]);
        luminance_r = b.add("image_processing_calc_luminance")
                          .set_param(
                              Param("luminance_method", "Average"))(
                              demosaic_r["output"]);
        luminance_filter_r = b.add("base_constant_buffer_2d_float")
                                 .set_param(
                                     Param("values", "0.04"),
                                     Param("extent0", "5"),
                                     Param("extent1", "5"));
        filtered_luminance_r = b.add("image_processing_convolution_2d")
                                   .set_param(
                                       Param("boundary_conditions_method", "MirrorInterior"),
                                       Param("window_size", "2"),
                                       Param("width", buffer_width),
                                       Param("height", buffer_height))(
                                       luminance_filter_r["output"],
                                       luminance_r["output"]);
        noise_reduction_r = b.add("image_processing_bilateral_filter_3d")
                                .set_param(
                                    Param("color_difference_method", "Average"),
                                    Param("window_size", "2"),
                                    Param("width", buffer_width),
                                    Param("height", buffer_height))(
                                    &coef_color_r,
                                    &coef_space_r,
                                    filtered_luminance_r["output"],
                                    demosaic_r["output"]);
        color_matrix_r = b.add("base_constant_buffer_2d_float")
                             .set_param(
                                 Param{"values", "1.5 -0.25 -0.25 "
                                                 "-0.25 1.5 -0.25 "
                                                 "-0.25 -0.25 1.5"},
                                 Param("extent0", "3"),
                                 Param("extent1", "3"));
        color_conversion_r = b.add("image_processing_color_matrix")(
            color_matrix_r["output"],
            noise_reduction_r["output"]);
        distortion_correction_r = b.add("image_processing_lens_distortion_correction_model_3d")
                                      .set_param(
                                          Param("width", buffer_width),
                                          Param("height", buffer_height))(
                                          &k1_r,
                                          &k2_r,
                                          &k3_r,
                                          &p1_r,
                                          &p2_r,
                                          &fx_r,
                                          &fy_r,
                                          &cx_r,
                                          &cy_r,
                                          &output_scale_r,
                                          color_conversion_r["output"]);
        resize_r = b.add("image_processing_resize_area_average_3d")
                       .set_param(
                           Param("width", buffer_width),
                           Param("height", buffer_height),
                           Param("scale", resize_scale_r))(
                           distortion_correction_r["output"]);
        final_luminance_r = b.add("image_processing_calc_luminance")
                                .set_param(
                                    Param("luminance_method", "Y"))(
                                    resize_r["output"]);
        gamma_correction_r = b.add("image_processing_gamma_correction_2d")(
            &gamma_r,
            final_luminance_r["output"]);

        Node ln = b.add("base_denormalize_2d_uint8")(gamma_correction_l["output"]);
        ln = b.add("sgm_census")(ln["output"]).set_param(Param("width", output_width), Param("height", output_height));

        Node rn = b.add("base_denormalize_2d_uint8")(gamma_correction_r["output"]);
        rn = b.add("sgm_census")(rn["output"]).set_param(Param("width", output_width), Param("height", output_height));

        Node n = b.add("sgm_matching_cost")(ln["output"], rn["output"]).set_param(Param("width", output_width), Param("height", output_height));

        Node up = b.add("sgm_scan_cost")(n["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp), Param("dx", 0), Param("dy", 1));
        Node lp = b.add("sgm_scan_cost")(n["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp), Param("dx", 1), Param("dy", 0));
        Node rp = b.add("sgm_scan_cost")(n["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp), Param("dx", -1), Param("dy", 0));
        Node dp = b.add("sgm_scan_cost")(n["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp), Param("dx", 0), Param("dy", -1));

        n = b.add("sgm_add_cost4")(up["output"], lp["output"], rp["output"], dp["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp), Param("num", 4));
        n = b.add("sgm_disparity")(n["output"]).set_param(Param("width", output_width), Param("height", output_height), Param("disp", disp));

        Halide::Buffer<uint8_t> obuf(std::vector<int>{output_width, output_height});
        // Halide::Buffer<uint8_t> obuf1(std::vector<int>{output_width, output_height});
        // Halide::Buffer<uint8_t> obuf2(std::vector<int>{output_width, output_height});

        n["output"].bind(obuf);
        // pm.set(ln["output"], obuf1);
        // pm.set(rn["output"], obuf2);

        b.run();

        obuf.copy_to_host();
        cv::Mat img(std::vector<int>{output_height, output_width}, CV_8UC1, obuf.data());
        cv::imwrite("sgm-out.png", img);

        // obuf1.copy_to_host();
        // obuf2.copy_to_host();
        // cv::Mat img1(std::vector<int>{output_height, output_width}, CV_8UC1, obuf1.data());
        // cv::Mat img2(std::vector<int>{output_height, output_width}, CV_8UC1, obuf2.data());
        // cv::imwrite("isp-out_l.png", img1);
        // cv::imwrite("isp-out_r.png", img2);

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
