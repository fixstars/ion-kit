#include <ion/ion.h>

#include "ion-bb-core/bb.h"
#include "ion-bb-dnn/bb.h"
#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-processing/bb.h"
#include "ion-bb-sgm/bb.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-image-io/rt.h"
#include "ion-bb-image-processing/rt.h"
#include "ion-bb-sgm/rt.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        // Parameters
        int imx_width = 3264;
        int imx_height = 2464;

        const int d435_width = 1280;
        const int d435_height = 720;

        const int disp = 16;

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        // ISP
        Port offset_r{"offset_r", Halide::type_of<float>()};
        Port offset_g{"offset_g", Halide::type_of<float>()};
        Port offset_b{"offset_b", Halide::type_of<float>()};
        Port shading_correction_slope_r{"shading_correction_slope_r", Halide::type_of<float>()};
        Port shading_correction_slope_g{"shading_correction_slope_g", Halide::type_of<float>()};
        Port shading_correction_slope_b{"shading_correction_slope_b", Halide::type_of<float>()};
        Port shading_correction_offset_r{"shading_correction_offset_r", Halide::type_of<float>()};
        Port shading_correction_offset_g{"shading_correction_offset_g", Halide::type_of<float>()};
        Port shading_correction_offset_b{"shading_correction_offset_b", Halide::type_of<float>()};
        Port gain_r{"gain_r", Halide::type_of<float>()};
        Port gain_g{"gain_g", Halide::type_of<float>()};
        Port gain_b{"gain_b", Halide::type_of<float>()};
        Port gamma{"gamma", Halide::type_of<float>()};

        int32_t bit_width = 10;
        int32_t bit_shift = 6;
        int32_t downscale_factor = 2;
        int32_t raw_width = imx_width;
        int32_t raw_height = imx_height;
        int32_t width = raw_width / downscale_factor;
        int32_t height = raw_height / downscale_factor;
        int32_t half_width = width / 2;
        int32_t half_height = height / 2;
        float scale = 416.f / half_height;
        int32_t scaled_width = half_width * scale;
        int32_t scaled_height = half_height * scale;
        int32_t output_width = 416;
        int32_t output_height = 416;

        // DNN
        int32_t yolo_width = 416;
        int32_t yolo_height = 416;

        Port dnn_inputs[6];

        for (int i = 0; i < 6; i++) {
            // IMX219
            Node imx = b.add("image_io_imx219")
                           .set_param(Param{"index", std::to_string(i)},
                                      Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/pedestrian.jpg"});

            // ISP
            Node downscale = b.add("image_processing_bayer_downscale_uint16")
                                 .set_param(
                                     Param{"input_width", std::to_string(raw_width)},
                                     Param{"input_height", std::to_string(raw_height)},
                                     Param{"downscale_factor", std::to_string(downscale_factor)})(
                                     imx["output"]);
            Node normalize = b.add("image_processing_normalize_raw_image")
                                 .set_param(
                                     Param{"bit_width", std::to_string(bit_width)},
                                     Param{"bit_shift", std::to_string(bit_shift)})(
                                     downscale["output"]);
            Node offset = b.add("image_processing_bayer_offset")(
                offset_r,
                offset_g,
                offset_b,
                normalize["output"]);
            Node shading_correction = b.add("image_processing_lens_shading_correction_linear")
                                          .set_param(
                                              Param{"width", std::to_string(width)},
                                              Param{"height", std::to_string(height)})(
                                              shading_correction_slope_r,
                                              shading_correction_slope_g,
                                              shading_correction_slope_b,
                                              shading_correction_offset_r,
                                              shading_correction_offset_g,
                                              shading_correction_offset_b,
                                              offset["output"]);
            Node white_balance = b.add("image_processing_bayer_white_balance")(
                gain_r,
                gain_g,
                gain_b,
                shading_correction["output"]);
            Node demosaic = b.add("image_processing_bayer_demosaic_simple")
                                .set_param(
                                    Param{"width", std::to_string(width)},
                                    Param{"height", std::to_string(height)})(
                                    white_balance["output"]);
            Node resize = b.add("image_processing_resize_bilinear_3d")
                              .set_param(
                                  Param{"width", std::to_string(half_width)},
                                  Param{"height", std::to_string(half_height)},
                                  Param{"scale", std::to_string(scale)})(
                                  demosaic["output"]);
            Node gamma_correction = b.add("image_processing_gamma_correction_3d")(
                gamma,
                resize["output"]);

            // DNN
            Node fit_image = b.add("image_processing_fit_image_to_center_3d_float")
                                 .set_param(
                                     Param{"input_width", std::to_string(scaled_width)},
                                     Param{"input_height", std::to_string(scaled_height)},
                                     Param{"output_width", std::to_string(output_width)},
                                     Param{"output_height", std::to_string(output_height)})(
                                     gamma_correction["output"]);
            Node reorder_channel = b.add("image_processing_reorder_color_channel_3d_float")(
                fit_image["output"]);
            Node reorder_chw2hwc = b.add("core_reorder_buffer_3d_float")
                                       .set_param(
                                           Param{"dim0", "2"},
                                           Param{"dim1", "0"},
                                           Param{"dim2", "1"})(
                                           reorder_channel["output"]);
            Node extended = b.add("core_extend_dimension_3d_float")
                                .set_param(
                                    Param{"new_dim", "3"})(
                                    reorder_chw2hwc["output"]);

            dnn_inputs[i] = extended["output"];
        }

        Port packed_dnn_input = dnn_inputs[0];
        for (int i = 1; i < 6; i++) {
            packed_dnn_input = b.add("core_concat_buffer_4d_float")
                                   .set_param(
                                       Param{"dim", "3"},
                                       Param{"input0_extent", std::to_string(i)})(
                                       packed_dnn_input,
                                       dnn_inputs[i])["output"];
        }

        Node object_detection = b.add("dnn_object_detection_array")(packed_dnn_input);

        Port dnn_outputs[6];
        for (int i = 0; i < 6; i++) {
            dnn_outputs[i] = b.add("core_extract_buffer_4d_float")
                                 .set_param(
                                     Param{"dim", "3"},
                                     Param{"index", std::to_string(i)})(
                                     object_detection["output"])["output"];
        }

        Port horizontal_tiled_image[2];
        for (int i = 0; i < 2; i++) {
            horizontal_tiled_image[i] = dnn_outputs[i * 3];
            for (int j = 1; j < 3; j++) {
                horizontal_tiled_image[i] = b.add("image_processing_tile_image_horizontal_3d_float")
                                                .set_param(
                                                    Param{"x_dim", "1"},
                                                    Param{"y_dim", "2"},
                                                    Param{"input0_width", std::to_string(output_width * j)},
                                                    Param{"input0_height", std::to_string(output_height)},
                                                    Param{"input1_width", std::to_string(output_width)},
                                                    Param{"input1_height", std::to_string(output_height)})(
                                                    horizontal_tiled_image[i],
                                                    dnn_outputs[i * 3 + j])["output"];
            }
        }

        Port tiled_image = b.add("image_processing_tile_image_vertical_3d_float")
                               .set_param(
                                   Param{"x_dim", "1"},
                                   Param{"y_dim", "2"},
                                   Param{"input0_width", std::to_string(output_width * 3)},
                                   Param{"input0_height", std::to_string(output_height)},
                                   Param{"input1_width", std::to_string(output_width * 3)},
                                   Param{"input1_height", std::to_string(output_height)})(
                                   horizontal_tiled_image[0],
                                   horizontal_tiled_image[1])["output"];

        Node denormalized = b.add("core_denormalize_3d_uint8")(tiled_image);

        // d435
        auto d435 = b.add("image_io_d435");

        // SGM
        auto sgm = b.add("sgm_sgm")(d435["output_l"], d435["output_r"]).set_param(Param{"disp", std::to_string(disp)}, Param{"width", std::to_string(d435_width)}, Param{"height", std::to_string(d435_height)});

        b.save("demo.json");

        {
            int d435_width_ = 1280;
            int d435_height_ = 720;
            int32_t imx219_width_ = 3264;
            int32_t imx219_height_ = 2464;

            // Pipeline parameters
            float offset_r_ = 1.f / 16.f;
            float offset_g_ = 1.f / 16.f;
            float offset_b_ = 1.f / 16.f;
            float shading_correction_slope_r_ = 0.7f;
            float shading_correction_slope_g_ = 0.2f;
            float shading_correction_slope_b_ = 0.1f;
            float shading_correction_offset_r_ = 1.f;
            float shading_correction_offset_g_ = 1.f;
            float shading_correction_offset_b_ = 1.f;
            float gain_r_ = 2.5f;
            float gain_g_ = 2.0f;
            float gain_b_ = 3.2f;
            float gamma_ = 1 / 2.2f;

            Halide::Buffer<uint16_t> depth_buf(std::vector<int>{d435_width_, d435_height_});
            Halide::Buffer<uint8_t> sgm_buf(std::vector<int>{d435_width_, d435_height_});
            Halide::Buffer<uint8_t> yolo_buf(std::vector<int>{3, yolo_width * 3, yolo_height * 2});

            PortMap pm;
            pm.set(offset_r, offset_r_);
            pm.set(offset_g, offset_g_);
            pm.set(offset_b, offset_b_);
            pm.set(shading_correction_slope_r, shading_correction_slope_r_);
            pm.set(shading_correction_slope_g, shading_correction_slope_g_);
            pm.set(shading_correction_slope_b, shading_correction_slope_b_);
            pm.set(shading_correction_offset_r, shading_correction_offset_r_);
            pm.set(shading_correction_offset_g, shading_correction_offset_g_);
            pm.set(shading_correction_offset_b, shading_correction_offset_b_);
            pm.set(gain_r, gain_r_);
            pm.set(gain_g, gain_g_);
            pm.set(gain_b, gain_b_);
            pm.set(gamma, gamma_);

            pm.set(denormalized["output"], yolo_buf);
            pm.set(d435["output_d"], depth_buf);
            pm.set(sgm["output"], sgm_buf);

            // Execution execution
            b.run(pm);

            depth_buf.copy_to_host();
            sgm_buf.copy_to_host();
            yolo_buf.copy_to_host();

            // cv::Mat depth_img(std::vector<int>{depth_height, depth_width}, CV_U16C1, depth_buf.data());
            cv::Mat sgm_img(std::vector<int>{d435_height_, d435_width_}, CV_8UC1, sgm_buf.data());
            cv::Mat yolo_img(std::vector<int>{yolo_height * 2, yolo_width * 3}, CV_8UC3, yolo_buf.data());
            // cv::imwrite("demo-depth.png", depth_img);
            cv::imwrite("demo-sgm.png", sgm_img);
            cv::imwrite("demo-yolo.png", yolo_img);
        }

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
