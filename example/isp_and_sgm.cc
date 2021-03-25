#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ion-bb-core/bb.h"
#include "ion-bb-image-processing/bb.h"
#include "ion-bb-sgm/bb.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-image-processing/rt.h"
#include "ion-bb-sgm/rt.h"

using namespace ion;

// Load raw image
Halide::Buffer<float> load_raw(std::string filename, int32_t width, int32_t height, int32_t bit_width, int32_t bit_shift) {
    assert(width > 0 && height > 0);
    std::ifstream ifs(filename, std::ios_base::binary);

    assert(ifs.is_open());

    std::vector<uint16_t> data(width * height);

    ifs.read(reinterpret_cast<char *>(data.data()), width * height * sizeof(uint16_t));

    Halide::Buffer<float> buffer(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            buffer(x, y) = static_cast<float>(data[y * width + x] >> bit_shift) / ((1 << bit_width) - 1);
        }
    }

    return buffer;
}

int main(int argc, char *argv[]) {
    assert(argc >= 6);
    try {
        int32_t raw_width = std::atoi(argv[2]);
        int32_t raw_height = std::atoi(argv[3]);
        int32_t buffer_width = raw_width / 2;
        int32_t buffer_height = raw_height;
        Halide::Buffer<float> buffer = load_raw(argv[1], raw_width, raw_height, std::atoi(argv[4]), std::atoi(argv[5]));

        // ISP Parameters for OV5647
        Port input{"input", Halide::type_of<float>(), 2};
        Port offset_r_l{"offset_r_l", Halide::type_of<float>()};
        Port offset_g_l{"offset_g_l", Halide::type_of<float>()};
        Port offset_b_l{"offset_b_l", Halide::type_of<float>()};
        Port gain_r_l{"gain_r_l", Halide::type_of<float>()};
        Port gain_g_l{"gain_g_l", Halide::type_of<float>()};
        Port gain_b_l{"gain_b_l", Halide::type_of<float>()};
        Port shading_correction_slope_r_l{"shading_correction_slope_r_l", Halide::type_of<float>()};
        Port shading_correction_slope_g_l{"shading_correction_slope_g_l", Halide::type_of<float>()};
        Port shading_correction_slope_b_l{"shading_correction_slope_b_l", Halide::type_of<float>()};
        Port shading_correction_offset_r_l{"shading_correction_offset_r_l", Halide::type_of<float>()};
        Port shading_correction_offset_g_l{"shading_correction_offset_g_l", Halide::type_of<float>()};
        Port shading_correction_offset_b_l{"shading_correction_offset_b_l", Halide::type_of<float>()};
        Port coef_color_l{"coef_color_l", Halide::type_of<float>()};
        Port coef_space_l{"coef_space_l", Halide::type_of<float>()};
        Port gamma_l{"gamma_l", Halide::type_of<float>()};
        Port k1_l{"k1_l", Halide::type_of<float>()};
        Port k2_l{"k2_l", Halide::type_of<float>()};
        Port k3_l{"k3_l", Halide::type_of<float>()};
        Port p1_l{"p1_l", Halide::type_of<float>()};
        Port p2_l{"p2_l", Halide::type_of<float>()};
        Port fx_l{"fx_l", Halide::type_of<float>()};
        Port fy_l{"fy_l", Halide::type_of<float>()};
        Port cx_l{"cx_l", Halide::type_of<float>()};
        Port cy_l{"cy_l", Halide::type_of<float>()};
        Port output_scale_l{"output_scale_l", Halide::type_of<float>()};
        Param bayer_pattern_l{"bayer_pattern", "2"};  // GRBG

        Port offset_r_r{"offset_r_r", Halide::type_of<float>()};
        Port offset_g_r{"offset_g_r", Halide::type_of<float>()};
        Port offset_b_r{"offset_b_r", Halide::type_of<float>()};
        Port gain_r_r{"gain_r_r", Halide::type_of<float>()};
        Port gain_g_r{"gain_g_r", Halide::type_of<float>()};
        Port gain_b_r{"gain_b_r", Halide::type_of<float>()};
        Port shading_correction_slope_r_r{"shading_correction_slope_r_r", Halide::type_of<float>()};
        Port shading_correction_slope_g_r{"shading_correction_slope_g_r", Halide::type_of<float>()};
        Port shading_correction_slope_b_r{"shading_correction_slope_b_r", Halide::type_of<float>()};
        Port shading_correction_offset_r_r{"shading_correction_offset_r_r", Halide::type_of<float>()};
        Port shading_correction_offset_g_r{"shading_correction_offset_g_r", Halide::type_of<float>()};
        Port shading_correction_offset_b_r{"shading_correction_offset_b_r", Halide::type_of<float>()};
        Port coef_color_r{"coef_color_r", Halide::type_of<float>()};
        Port coef_space_r{"coef_space_r", Halide::type_of<float>()};
        Port gamma_r{"gamma_r", Halide::type_of<float>()};
        Port k1_r{"k1_r", Halide::type_of<float>()};
        Port k2_r{"k2_r", Halide::type_of<float>()};
        Port k3_r{"k3_r", Halide::type_of<float>()};
        Port p1_r{"p1_r", Halide::type_of<float>()};
        Port p2_r{"p2_r", Halide::type_of<float>()};
        Port fx_r{"fx_r", Halide::type_of<float>()};
        Port fy_r{"fy_r", Halide::type_of<float>()};
        Port cx_r{"cx_r", Halide::type_of<float>()};
        Port cy_r{"cy_r", Halide::type_of<float>()};
        Port output_scale_r{"output_scale_r", Halide::type_of<float>()};
        Param bayer_pattern_r{"bayer_pattern", "2"};  // GRBG

        float resize_scale_l = 0.2f;
        float resize_scale_r = 0.2f;

        PortMap pm;
        pm.set(input, buffer);

        pm.set(offset_r_l, 1.f / 64.f);
        pm.set(offset_g_l, 1.f / 64.f);
        pm.set(offset_b_l, 1.f / 64.f);
        pm.set(gain_r_l, 1.0f);
        pm.set(gain_g_l, 1.0f);
        pm.set(gain_b_l, 1.5f);
        pm.set(shading_correction_slope_r_l, 1.5f);
        pm.set(shading_correction_slope_g_l, 1.0f);
        pm.set(shading_correction_slope_b_l, 2.5f);
        pm.set(shading_correction_offset_r_l, 1.f);
        pm.set(shading_correction_offset_g_l, 1.f);
        pm.set(shading_correction_offset_b_l, 1.f);
        pm.set(coef_color_l, 100.f);
        pm.set(coef_space_l, 0.03);
        pm.set(gamma_l, 1.f / 2.2f);
        pm.set(k1_l, 0.f);
        pm.set(k2_l, 0.f);
        pm.set(k3_l, 0.f);
        pm.set(p1_l, 0.f);
        pm.set(p2_l, 0.f);
        pm.set(output_scale_l, 1.f);
        pm.set(fx_l, static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height) / 2));
        pm.set(fy_l, static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height) / 2));
        pm.set(cx_l, buffer_width * 0.5f);
        pm.set(cy_l, buffer_height * 0.6f);

        pm.set(offset_r_r, 1.f / 64.f);
        pm.set(offset_g_r, 1.f / 64.f);
        pm.set(offset_b_r, 1.f / 64.f);
        pm.set(gain_r_r, 1.0f);
        pm.set(gain_g_r, 1.0f);
        pm.set(gain_b_r, 1.5f);
        pm.set(shading_correction_slope_r_r, 1.5f);
        pm.set(shading_correction_slope_g_r, 1.0f);
        pm.set(shading_correction_slope_b_r, 2.5f);
        pm.set(shading_correction_offset_r_r, 1.f);
        pm.set(shading_correction_offset_g_r, 1.f);
        pm.set(shading_correction_offset_b_r, 1.f);
        pm.set(coef_color_r, 100.f);
        pm.set(coef_space_r, 0.03);
        pm.set(gamma_r, 1.f / 2.2f);
        pm.set(k1_r, 0.f);
        pm.set(k2_r, 0.f);
        pm.set(k3_r, 0.f);
        pm.set(p1_r, 0.f);
        pm.set(p2_r, 0.f);
        pm.set(output_scale_r, 1.f);
        pm.set(fx_r, static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height) / 2));
        pm.set(fy_r, static_cast<float>(sqrt(buffer_width * buffer_width + buffer_height * buffer_height) / 2));
        pm.set(cx_r, buffer_width * 0.5f);
        pm.set(cy_r, buffer_height * 0.6f);

        // SGM Parameters
        int output_width = static_cast<int>(buffer_width * resize_scale_l);
        int output_height = static_cast<int>(buffer_height * resize_scale_l);
        const int disp = 192;

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        // ISP Nodes
        Node crop_l, crop_r;
        Node offset_l, shading_correction_l, white_balance_l, demosaic_l, luminance_l, filtered_luminance_l, luminance_filter_l, noise_reduction_l;
        Node color_matrix_l, color_conversion_l, gamma_correction_l, distortion_lut_l, distortion_correction_l, resize_l, final_luminance_l;
        Node offset_r, shading_correction_r, white_balance_r, demosaic_r, luminance_r, filtered_luminance_r, luminance_filter_r, noise_reduction_r;
        Node color_matrix_r, color_conversion_r, gamma_correction_r, distortion_lut_r, distortion_correction_r, resize_r, final_luminance_r;

        crop_l = b.add("image_processing_crop_image_2d_float")
                     .set_param(
                         Param{"x_dim", "0"},
                         Param{"y_dim", "1"},
                         Param{"input_width", "0"},
                         Param{"input_height", "0"},
                         Param{"top", "0"},
                         Param{"left", "0"},
                         Param{"output_width", std::to_string(buffer_width)},
                         Param{"output_height", std::to_string(buffer_height)})(
                         input);
        offset_l = b.add("image_processing_bayer_offset")
                       .set_param(
                           bayer_pattern_l)(
                           offset_r_l,
                           offset_g_l,
                           offset_b_l,
                           input);
        shading_correction_l = b.add("image_processing_lens_shading_correction_linear")
                                   .set_param(
                                       bayer_pattern_l,
                                       Param{"width", std::to_string(buffer_width)},
                                       Param{"height", std::to_string(buffer_height)})(
                                       shading_correction_slope_r_l,
                                       shading_correction_slope_g_l,
                                       shading_correction_slope_b_l,
                                       shading_correction_offset_r_l,
                                       shading_correction_offset_g_l,
                                       shading_correction_offset_b_l,
                                       offset_l["output"]);
        white_balance_l = b.add("image_processing_bayer_white_balance")
                              .set_param(
                                  bayer_pattern_l)(
                                  gain_r_l,
                                  gain_g_l,
                                  gain_b_l,
                                  shading_correction_l["output"]);
        demosaic_l = b.add("image_processing_bayer_demosaic_filter")
                         .set_param(
                             bayer_pattern_l,
                             Param{"width", std::to_string(buffer_width)},
                             Param{"height", std::to_string(buffer_height)})(
                             white_balance_l["output"]);
        luminance_l = b.add("image_processing_calc_luminance")
                          .set_param(
                              Param{"luminance_method", "1"})(  // Average
                              demosaic_l["output"]);
        luminance_filter_l = b.add("core_constant_buffer_2d_float")
                                 .set_param(
                                     Param{"values", "0.04"},
                                     Param{"extent0", "5"},
                                     Param{"extent1", "5"});
        filtered_luminance_l = b.add("image_processing_convolution_2d")
                                   .set_param(
                                       Param{"boundary_conditions_method", "3"},  // MirrorInterior
                                       Param{"window_size", "2"},
                                       Param{"width", std::to_string(buffer_width)},
                                       Param{"height", std::to_string(buffer_height)})(
                                       luminance_filter_l["output"],
                                       luminance_l["output"]);
        noise_reduction_l = b.add("image_processing_bilateral_filter_3d")
                                .set_param(
                                    Param{"color_difference_method", "1"},  // Average
                                    Param{"window_size", "2"},
                                    Param{"width", std::to_string(buffer_width)},
                                    Param{"height", std::to_string(buffer_height)})(
                                    coef_color_l,
                                    coef_space_l,
                                    filtered_luminance_l["output"],
                                    demosaic_l["output"]);
        color_matrix_l = b.add("core_constant_buffer_2d_float")
                             .set_param(
                                 Param{"values", "1.5 -0.25 -0.25 "
                                                 "-0.25 1.5 -0.25 "
                                                 "-0.25 -0.25 1.5"},
                                 Param{"extent0", "3"},
                                 Param{"extent1", "3"});
        color_conversion_l = b.add("image_processing_color_matrix")(
            color_matrix_l["output"],
            noise_reduction_l["output"]);
        distortion_correction_l = b.add("image_processing_lens_distortion_correction_model_3d")
                                      .set_param(
                                          Param{"width", std::to_string(buffer_width)},
                                          Param{"height", std::to_string(buffer_height)})(
                                          k1_l,
                                          k2_l,
                                          k3_l,
                                          p1_l,
                                          p2_l,
                                          fx_l,
                                          fy_l,
                                          cx_l,
                                          cy_l,
                                          output_scale_l,
                                          color_conversion_l["output"]);
        resize_l = b.add("image_processing_resize_area_average_3d")
                       .set_param(
                           Param{"width", std::to_string(buffer_width)},
                           Param{"height", std::to_string(buffer_height)},
                           Param{"scale", std::to_string(resize_scale_l)})(
                           distortion_correction_l["output"]);
        final_luminance_l = b.add("image_processing_calc_luminance")
                                .set_param(
                                    Param{"luminance_method", "3"})(  // Y
                                    resize_l["output"]);
        gamma_correction_l = b.add("image_processing_gamma_correction_2d")(
            gamma_l,
            final_luminance_l["output"]);

        crop_r = b.add("image_processing_crop_image_2d_float")
                     .set_param(
                         Param{"x_dim", "0"},
                         Param{"y_dim", "1"},
                         Param{"input_width", "0"},
                         Param{"input_height", "0"},
                         Param{"top", "0"},
                         Param{"left", "0"},
                         Param{"output_width", std::to_string(buffer_width)},
                         Param{"output_height", std::to_string(buffer_height)})(
                         input);
        offset_r = b.add("image_processing_bayer_offset")
                       .set_param(
                           bayer_pattern_r)(
                           offset_r_r,
                           offset_g_r,
                           offset_b_r,
                           input);
        shading_correction_r = b.add("image_processing_lens_shading_correction_linear")
                                   .set_param(
                                       bayer_pattern_r,
                                       Param{"width", std::to_string(buffer_width)},
                                       Param{"height", std::to_string(buffer_height)})(
                                       shading_correction_slope_r_r,
                                       shading_correction_slope_g_r,
                                       shading_correction_slope_b_r,
                                       shading_correction_offset_r_r,
                                       shading_correction_offset_g_r,
                                       shading_correction_offset_b_r,
                                       offset_r["output"]);
        white_balance_r = b.add("image_processing_bayer_white_balance")
                              .set_param(
                                  bayer_pattern_r)(
                                  gain_r_r,
                                  gain_g_r,
                                  gain_b_r,
                                  shading_correction_r["output"]);
        demosaic_r = b.add("image_processing_bayer_demosaic_filter")
                         .set_param(
                             bayer_pattern_r,
                             Param{"width", std::to_string(buffer_width)},
                             Param{"height", std::to_string(buffer_height)})(
                             white_balance_r["output"]);
        luminance_r = b.add("image_processing_calc_luminance")
                          .set_param(
                              Param{"luminance_method", "1"})(  // Average
                              demosaic_r["output"]);
        luminance_filter_r = b.add("core_constant_buffer_2d_float")
                                 .set_param(
                                     Param{"values", "0.04"},
                                     Param{"extent0", "5"},
                                     Param{"extent1", "5"});
        filtered_luminance_r = b.add("image_processing_convolution_2d")
                                   .set_param(
                                       Param{"boundary_conditions_method", "3"},  // MirrorInterior
                                       Param{"window_size", "2"},
                                       Param{"width", std::to_string(buffer_width)},
                                       Param{"height", std::to_string(buffer_height)})(
                                       luminance_filter_r["output"],
                                       luminance_r["output"]);
        noise_reduction_r = b.add("image_processing_bilateral_filter_3d")
                                .set_param(
                                    Param{"color_difference_method", "1"},  // Average
                                    Param{"window_size", "2"},
                                    Param{"width", std::to_string(buffer_width)},
                                    Param{"height", std::to_string(buffer_height)})(
                                    coef_color_r,
                                    coef_space_r,
                                    filtered_luminance_r["output"],
                                    demosaic_r["output"]);
        color_matrix_r = b.add("core_constant_buffer_2d_float")
                             .set_param(
                                 Param{"values", "1.5 -0.25 -0.25 "
                                                 "-0.25 1.5 -0.25 "
                                                 "-0.25 -0.25 1.5"},
                                 Param{"extent0", "3"},
                                 Param{"extent1", "3"});
        color_conversion_r = b.add("image_processing_color_matrix")(
            color_matrix_r["output"],
            noise_reduction_r["output"]);
        distortion_correction_r = b.add("image_processing_lens_distortion_correction_model_3d")
                                      .set_param(
                                          Param{"width", std::to_string(buffer_width)},
                                          Param{"height", std::to_string(buffer_height)})(
                                          k1_r,
                                          k2_r,
                                          k3_r,
                                          p1_r,
                                          p2_r,
                                          fx_r,
                                          fy_r,
                                          cx_r,
                                          cy_r,
                                          output_scale_r,
                                          color_conversion_r["output"]);
        resize_r = b.add("image_processing_resize_area_average_3d")
                       .set_param(
                           Param{"width", std::to_string(buffer_width)},
                           Param{"height", std::to_string(buffer_height)},
                           Param{"scale", std::to_string(resize_scale_r)})(
                           distortion_correction_r["output"]);
        final_luminance_r = b.add("image_processing_calc_luminance")
                                .set_param(
                                    Param{"luminance_method", "3"})(  // Y
                                    resize_r["output"]);
        gamma_correction_r = b.add("image_processing_gamma_correction_2d")(
            gamma_r,
            final_luminance_r["output"]);

        Node ln = b.add("core_denormalize_2d_uint8")(gamma_correction_l["output"]);
        ln = b.add("sgm_census")(ln["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)});

        Node rn = b.add("core_denormalize_2d_uint8")(gamma_correction_r["output"]);
        rn = b.add("sgm_census")(rn["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)});

        Node n = b.add("sgm_matching_cost")(ln["output"], rn["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)});

        Node up = b.add("sgm_scan_cost")(n["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)}, Param{"dx", std::to_string(0)}, Param{"dy", std::to_string(1)});
        Node lp = b.add("sgm_scan_cost")(n["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)}, Param{"dx", std::to_string(1)}, Param{"dy", std::to_string(0)});
        Node rp = b.add("sgm_scan_cost")(n["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)}, Param{"dx", std::to_string(-1)}, Param{"dy", std::to_string(0)});
        Node dp = b.add("sgm_scan_cost")(n["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)}, Param{"dx", std::to_string(0)}, Param{"dy", std::to_string(-1)});

        n = b.add("sgm_add_cost4")(up["output"], lp["output"], rp["output"], dp["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)}, Param{"num", std::to_string(4)});
        n = b.add("sgm_disparity")(n["output"]).set_param(Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)}, Param{"disp", std::to_string(disp)});

        Halide::Buffer<uint8_t> obuf(std::vector<int>{output_width, output_height});
        // Halide::Buffer<uint8_t> obuf1(std::vector<int>{output_width, output_height});
        // Halide::Buffer<uint8_t> obuf2(std::vector<int>{output_width, output_height});

        pm.set(n["output"], obuf);
        // pm.set(ln["output"], obuf1);
        // pm.set(rn["output"], obuf2);

        b.run(pm);

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
