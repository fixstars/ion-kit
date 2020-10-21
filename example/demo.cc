#include <ion/ion.h>

#include "ion-bb-demo/bb.h"
#include "ion-bb-dnn/bb.h"
#include "ion-bb-genesis-cloud/bb.h"

#include "ion-bb-demo/rt.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-opencv/rt.h"

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
        std::string yolo_model_root = argv[1];
        std::string yolo_model_name = argv[2];

        Port dnn_inputs[6];
        Port dnn_inputs_opencv[6];

        for (int i = 0; i < 6; i++) {
            // IMX219
            Node imx = b.add("demo_imx219")
                           .set_param(Param{"index", std::to_string(i)});

            // ISP
            Node downscale = b.add("demo_bayer_downscale_uint16")
                                 .set_param(
                                     Param{"input_width", std::to_string(raw_width)},
                                     Param{"input_height", std::to_string(raw_height)},
                                     Param{"downscale_factor", std::to_string(downscale_factor)})(
                                     imx["output"]);
            Node normalize = b.add("demo_normalize_raw_image")
                                 .set_param(
                                     Param{"bit_width", std::to_string(bit_width)},
                                     Param{"bit_shift", std::to_string(bit_shift)})(
                                     downscale["output"]);
            Node offset = b.add("demo_bayer_offset")(
                offset_r,
                offset_g,
                offset_b,
                normalize["output"]);
            Node shading_correction = b.add("demo_lens_shading_correction_linear")
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
            Node white_balance = b.add("demo_bayer_white_balance")(
                gain_r,
                gain_g,
                gain_b,
                shading_correction["output"]);
            Node demosaic = b.add("demo_bayer_demosaic_simple")
                                .set_param(
                                    Param{"width", std::to_string(width)},
                                    Param{"height", std::to_string(height)})(
                                    white_balance["output"]);
            Node resize = b.add("demo_resize_bilinear_3d")
                              .set_param(
                                  Param{"width", std::to_string(half_width)},
                                  Param{"height", std::to_string(half_height)},
                                  Param{"scale", std::to_string(scale)})(
                                  demosaic["output"]);
            Node gamma_correction = b.add("demo_gamma_correction_3d")(
                gamma,
                resize["output"]);

            // DNN
            Node fit_image = b.add("demo_fit_image_to_center_3d_float")
                                 .set_param(
                                     Param{"input_width", std::to_string(scaled_width)},
                                     Param{"input_height", std::to_string(scaled_height)},
                                     Param{"output_width", std::to_string(output_width)},
                                     Param{"output_height", std::to_string(output_height)})(
                                     gamma_correction["output"]);
            Node reorder_chw2hwc = b.add("demo_reorder_image_chw2hwc_float")(
                fit_image["output"]);
            Node reorder_channel = b.add("demo_reorder_color_channel_float")(
                reorder_chw2hwc["output"]);
            Node denormalize = b.add("genesis_cloud_denormalize_u8x3")(
                reorder_channel["output"]);

            dnn_inputs[i] = fit_image["output"];
            dnn_inputs_opencv[i] = denormalize["output"];
        }

        Node pack_dnn_inputs = b.add("demo_pack_6images_3d_float")
                                   .set_param(
                                       Param{"input0_width", std::to_string(output_width)},
                                       Param{"input0_height", std::to_string(output_height)},
                                       Param{"input1_width", std::to_string(output_width)},
                                       Param{"input1_height", std::to_string(output_height)},
                                       Param{"input2_width", std::to_string(output_width)},
                                       Param{"input2_height", std::to_string(output_height)},
                                       Param{"input3_width", std::to_string(output_width)},
                                       Param{"input3_height", std::to_string(output_height)},
                                       Param{"input4_width", std::to_string(output_width)},
                                       Param{"input4_height", std::to_string(output_height)},
                                       Param{"input5_width", std::to_string(output_width)},
                                       Param{"input5_height", std::to_string(output_height)})(
                                       dnn_inputs[0],
                                       dnn_inputs[1],
                                       dnn_inputs[2],
                                       dnn_inputs[3],
                                       dnn_inputs[4],
                                       dnn_inputs[5]);

        Node pack_dnn_inputs_opencv = b.add("demo_pack_6images_3d_uint8_hwc")
                                          .set_param(
                                              Param{"input0_width", std::to_string(output_width)},
                                              Param{"input0_height", std::to_string(output_height)},
                                              Param{"input1_width", std::to_string(output_width)},
                                              Param{"input1_height", std::to_string(output_height)},
                                              Param{"input2_width", std::to_string(output_width)},
                                              Param{"input2_height", std::to_string(output_height)},
                                              Param{"input3_width", std::to_string(output_width)},
                                              Param{"input3_height", std::to_string(output_height)},
                                              Param{"input4_width", std::to_string(output_width)},
                                              Param{"input4_height", std::to_string(output_height)},
                                              Param{"input5_width", std::to_string(output_width)},
                                              Param{"input5_height", std::to_string(output_height)})(
                                              dnn_inputs_opencv[0],
                                              dnn_inputs_opencv[1],
                                              dnn_inputs_opencv[2],
                                              dnn_inputs_opencv[3],
                                              dnn_inputs_opencv[4],
                                              dnn_inputs_opencv[5]);

        Node yolo_object_detection = b.add("yolov4_object_detection_array")
                                         .set_param(
                                             Param{"model_root", yolo_model_root},
                                             Param{"model_name", yolo_model_name},
                                             Param{"height", std::to_string(yolo_height)},
                                             Param{"width", std::to_string(yolo_width)})(
                                             pack_dnn_inputs["output"]);

        Node yolo_box_rendering = b.add("yolov4_box_rendering_array")
                                      .set_param(
                                          Param{"height", std::to_string(yolo_height)},
                                          Param{"width", std::to_string(yolo_width)})(
                                          pack_dnn_inputs_opencv["output"],
                                          yolo_object_detection["boxes"],
                                          yolo_object_detection["confs"]);

        Node tile_images = b.add("demo_tile_6images_3d_array_uint8_hwc")
                               .set_param(
                                   Param{"input_height", std::to_string(output_width)},
                                   Param{"input_width", std::to_string(output_height)})(
                                   yolo_box_rendering["output"]);

        // d435
        auto d435 = b.add("demo_d435");

        // SGM
        auto sgm = b.add("demo_sgm")(d435["output_l"], d435["output_r"]).set_param(Param{"disp", std::to_string(disp)}, Param{"width", std::to_string(d435_width)}, Param{"height", std::to_string(d435_height)});

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

            pm.set(tile_images["output"], yolo_buf);
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
