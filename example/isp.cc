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

#include "ion-bb-core/bb.h"
#include "ion-bb-image-processing/bb.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-image-processing/rt.h"

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

void save_image(Halide::Buffer<float> buffer, std::string filename) {
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
    assert(argc >= 6);

    Builder b;
    b.set_target(Halide::get_target_from_environment());

    Halide::Buffer<float> buffer = load_raw(argv[1], std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));

    // Parameters for IMX219
    Port input{"input", Halide::type_of<float>(), 2};
    Port offset_r{"offset_r", Halide::type_of<float>()};
    Port offset_g{"offset_g", Halide::type_of<float>()};
    Port offset_b{"offset_b", Halide::type_of<float>()};
    Port gain_r{"gain_r", Halide::type_of<float>()};
    Port gain_g{"gain_g", Halide::type_of<float>()};
    Port gain_b{"gain_b", Halide::type_of<float>()};
    Port shading_correction_slope_r{"shading_correction_slope_r", Halide::type_of<float>()};
    Port shading_correction_slope_g{"shading_correction_slope_g", Halide::type_of<float>()};
    Port shading_correction_slope_b{"shading_correction_slope_b", Halide::type_of<float>()};
    Port shading_correction_offset_r{"shading_correction_offset_r", Halide::type_of<float>()};
    Port shading_correction_offset_g{"shading_correction_offset_g", Halide::type_of<float>()};
    Port shading_correction_offset_b{"shading_correction_offset_b", Halide::type_of<float>()};
    Port coef_color{"coef_color", Halide::type_of<float>()};
    Port coef_space{"coef_space", Halide::type_of<float>()};
    Port gamma{"gamma", Halide::type_of<float>()};
    Port k1{"k1", Halide::type_of<float>()};
    Port k2{"k2", Halide::type_of<float>()};
    Port k3{"k3", Halide::type_of<float>()};
    Port p1{"p1", Halide::type_of<float>()};
    Port p2{"p2", Halide::type_of<float>()};
    Port fx{"fx", Halide::type_of<float>()};
    Port fy{"fy", Halide::type_of<float>()};
    Port cx{"cx", Halide::type_of<float>()};
    Port cy{"cy", Halide::type_of<float>()};
    Port output_scale{"output_scale", Halide::type_of<float>()};
    Port scale{"scale", Halide::type_of<float>()};

    int32_t width = buffer.width();
    int32_t height = buffer.height();
    float resize_scale = 0.4f;

    PortMap pm;
    pm.set(offset_r, 1.f / 16.f);
    pm.set(offset_g, 1.f / 16.f);
    pm.set(offset_b, 1.f / 16.f);
    pm.set(gain_r, 2.5f);
    pm.set(gain_g, 2.0f);
    pm.set(gain_b, 3.2f);
    pm.set(shading_correction_slope_r, 0.7f);
    pm.set(shading_correction_slope_g, 0.2f);
    pm.set(shading_correction_slope_b, 0.1f);
    pm.set(shading_correction_offset_r, 1.f);
    pm.set(shading_correction_offset_g, 1.f);
    pm.set(shading_correction_offset_b, 1.f);
    pm.set(coef_color, 100.f);
    pm.set(coef_space, 0.03);
    pm.set(gamma, 1.f / 2.2f);
    pm.set(k1, 0.f);
    pm.set(k2, 0.f);
    pm.set(k3, 0.f);
    pm.set(scale, resize_scale);
    pm.set(p1, 0.f);
    pm.set(p2, 0.f);
    pm.set(output_scale, 1.f);
    pm.set(input, buffer);
    pm.set(fx, static_cast<float>(sqrt(buffer.width() * buffer.width() + buffer.height() * buffer.height()) / 2));
    pm.set(fy, static_cast<float>(sqrt(buffer.width() * buffer.width() + buffer.height() * buffer.height()) / 2));
    pm.set(cx, buffer.width() * 0.5f);
    pm.set(cy, buffer.height() * 0.6f);

    Param bayer_pattern{"bayer_pattern", "0"};  // RGGB

    Node offset, shading_correction, white_balance, demosaic, luminance, filtered_luminance, luminance_filter, noise_reduction;
    Node color_matrix, color_conversion, gamma_correction, distortion_lut, distortion_correction, resize;
    Node debug_output;

    offset = b.add("image_processing_bayer_offset")
                 .set_param(
                     bayer_pattern)(
                     offset_r,
                     offset_g,
                     offset_b,
                     input);
    shading_correction = b.add("image_processing_lens_shading_correction_linear")
                             .set_param(
                                 bayer_pattern,
                                 Param{"width", std::to_string(width)},
                                 Param{"height", std::to_string(height)})(
                                 shading_correction_slope_r,
                                 shading_correction_slope_g,
                                 shading_correction_slope_b,
                                 shading_correction_offset_r,
                                 shading_correction_offset_g,
                                 shading_correction_offset_b,
                                 offset["output"]);
    white_balance = b.add("image_processing_bayer_white_balance")
                        .set_param(
                            bayer_pattern)(
                            gain_r,
                            gain_g,
                            gain_b,
                            shading_correction["output"]);
    demosaic = b.add("image_processing_bayer_demosaic_filter")
                   .set_param(
                       bayer_pattern,
                       Param{"width", std::to_string(width)},
                       Param{"height", std::to_string(height)})(
                       white_balance["output"]);
    luminance = b.add("image_processing_calc_luminance")
                    .set_param(
                        Param{"luminance_method", "1"})(  // Average
                        demosaic["output"]);
    luminance_filter = b.add("core_constant_buffer_2d_float")
                           .set_param(
                               Param{"values", "0.04"},
                               Param{"extent0", "5"},
                               Param{"extent1", "5"});
    filtered_luminance = b.add("image_processing_convolution_2d")
                             .set_param(
                                 Param{"boundary_conditions_method", "3"},  // MirrorInterior
                                 Param{"window_size", "2"},
                                 Param{"width", std::to_string(width)},
                                 Param{"height", std::to_string(height)})(
                                 luminance_filter["output"],
                                 luminance["output"]);
    noise_reduction = b.add("image_processing_bilateral_filter_3d")
                          .set_param(
                              Param{"color_difference_method", "1"},  // Average
                              Param{"window_size", "2"},
                              Param{"width", std::to_string(width)},
                              Param{"height", std::to_string(height)})(
                              coef_color,
                              coef_space,
                              filtered_luminance["output"],
                              demosaic["output"]);
    color_matrix = b.add("core_constant_buffer_2d_float")
                       .set_param(
                           Param{"values", "2.20213000 -1.27425000 0.07212000 "
                                           "-0.25650000 1.45961000 -0.20311000 "
                                           "0.07458000 -1.35791000 2.28333000"},
                           Param{"extent0", "3"},
                           Param{"extent1", "3"});
    color_conversion = b.add("image_processing_color_matrix")(
        color_matrix["output"],
        noise_reduction["output"]);
    distortion_correction = b.add("image_processing_lens_distortion_correction_model_3d")
                                .set_param(
                                    Param{"width", std::to_string(width)},
                                    Param{"height", std::to_string(height)})(
                                    k1,
                                    k2,
                                    k3,
                                    p1,
                                    p2,
                                    fx,
                                    fy,
                                    cx,
                                    cy,
                                    output_scale,
                                    color_conversion["output"]);
    resize = b.add("image_processing_resize_area_average_3d")
                 .set_param(
                     Param{"width", std::to_string(width)},
                     Param{"height", std::to_string(height)},
                     Param{"scale", std::to_string(resize_scale)})(
                     distortion_correction["output"]);
    gamma_correction = b.add("image_processing_gamma_correction_3d")(
        gamma,
        resize["output"]);

    Halide::Buffer<float> obuf(buffer.width() * resize_scale, buffer.height() * resize_scale, 3);
    pm.set(gamma_correction["output"], obuf);

    b.run(pm);
    obuf.copy_to_host();
    save_image(obuf, "output.png");

    return 0;
}
