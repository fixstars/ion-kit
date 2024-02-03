#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        // Parameters
        constexpr int imx_width = 3264;
        constexpr int imx_height = 2464;

        const int d435_width = 1280;
        const int d435_height = 720;

        const int disp = 16;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        // Pipeline parameters
        int d435_width_ = 1280;
        int d435_height_ = 720;
        int32_t imx219_width_ = 3264;
        int32_t imx219_height_ = 2464;
        int32_t yolo_width = 416;
        int32_t yolo_height = 416;

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

        ion::Buffer<uint16_t> depth_buf(std::vector<int>{d435_width_, d435_height_});
        ion::Buffer<uint8_t> sgm_buf(std::vector<int>{d435_width_, d435_height_});
        ion::Buffer<uint8_t> yolo_buf(std::vector<int>{3, yolo_width * 3, yolo_height * 2});

        constexpr int32_t bit_width = 10;
        constexpr int32_t bit_shift = 6;
        constexpr int32_t downscale_factor = 2;
        constexpr int32_t raw_width = imx_width;
        constexpr int32_t raw_height = imx_height;
        constexpr int32_t width = raw_width / downscale_factor;
        constexpr int32_t height = raw_height / downscale_factor;
        constexpr int32_t half_width = width / 2;
        constexpr int32_t half_height = height / 2;
        constexpr float scale = 416.f / half_height;
        constexpr int32_t scaled_width = half_width * scale;
        constexpr int32_t scaled_height = half_height * scale;
        constexpr int32_t output_width = 416;
        constexpr int32_t output_height = 416;

        Port dnn_inputs[6];

        for (int i = 0; i < 6; i++) {
            // IMX219
            Node imx = b.add("image_io_imx219")
                           .set_param(Param("index", i),
                                      Param("url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/pedestrian.png"));

            // ISP
            Node downscale = b.add("image_processing_bayer_downscale_uint16")
                                 .set_param(
                                     Param("input_width", raw_width),
                                     Param("input_height", raw_height),
                                     Param("downscale_factor", downscale_factor))(
                                     imx["output"]);
            Node normalize = b.add("image_processing_normalize_raw_image")
                                 .set_param(
                                     Param("bit_width", bit_width),
                                     Param("bit_shift", bit_shift))(
                                     downscale["output"]);
            Node offset = b.add("image_processing_bayer_offset")(
                &offset_r,
                &offset_g,
                &offset_b,
                normalize["output"]);
            Node shading_correction = b.add("image_processing_lens_shading_correction_linear")
                                          .set_param(
                                              Param("width", width),
                                              Param("height", height))(
                                              &shading_correction_slope_r,
                                              &shading_correction_slope_g,
                                              &shading_correction_slope_b,
                                              &shading_correction_offset_r,
                                              &shading_correction_offset_g,
                                              &shading_correction_offset_b,
                                              offset["output"]);
            Node white_balance = b.add("image_processing_bayer_white_balance")(
                &gain_r,
                &gain_g,
                &gain_b,
                shading_correction["output"]);
            Node demosaic = b.add("image_processing_bayer_demosaic_simple")
                                .set_param(
                                    Param("width", width),
                                    Param("height", height))(
                                    white_balance["output"]);
            Node resize = b.add("image_processing_resize_bilinear_3d")
                              .set_param(
                                  Param("width", half_width),
                                  Param("height", half_height),
                                  Param("scale", scale))(
                                  demosaic["output"]);
            Node gamma_correction = b.add("image_processing_gamma_correction_3d")(
                &gamma,
                resize["output"]);

            // DNN
            Node fit_image = b.add("image_processing_fit_image_to_center_3d_float")
                                 .set_param(
                                     Param("input_width", scaled_width),
                                     Param("input_height", scaled_height),
                                     Param("output_width", output_width),
                                     Param("output_height", output_height))(
                                     gamma_correction["output"]);
            Node reorder_channel = b.add("image_processing_reorder_color_channel_3d_float")(
                fit_image["output"]);
            Node reorder_chw2hwc = b.add("base_reorder_buffer_3d_float")
                                       .set_param(
                                           Param("dim0", 2),
                                           Param("dim1", 0),
                                           Param("dim2", 1))(
                                           reorder_channel["output"]);
            Node extended = b.add("base_extend_dimension_3d_float")
                                .set_param(
                                    Param("new_dim", 3))(
                                    reorder_chw2hwc["output"]);

            dnn_inputs[i] = extended["output"];
        }

        Port packed_dnn_input = dnn_inputs[0];
        for (int i = 1; i < 6; i++) {
            packed_dnn_input = b.add("base_concat_buffer_4d_float")
                                   .set_param(
                                       Param("dim", 3),
                                       Param("input0_extent", i))(
                                       packed_dnn_input,
                                       dnn_inputs[i])["output"];
        }

        Node object_detection = b.add("dnn_object_detection_array")(packed_dnn_input);

        Port dnn_outputs[6];
        for (int i = 0; i < 6; i++) {
            dnn_outputs[i] = b.add("base_extract_buffer_4d_float")
                                 .set_param(
                                     Param("dim", 3),
                                     Param("index", i))(
                                     object_detection["output"])["output"];
        }

        Port horizontal_tiled_image[2];
        for (int i = 0; i < 2; i++) {
            horizontal_tiled_image[i] = dnn_outputs[i * 3];
            for (int j = 1; j < 3; j++) {
                horizontal_tiled_image[i] = b.add("image_processing_tile_image_horizontal_3d_float")
                                                .set_param(
                                                    Param("x_dim", 1),
                                                    Param("y_dim", 2),
                                                    Param("input0_width", output_width * j),
                                                    Param("input0_height", output_height),
                                                    Param("input1_width", output_width),
                                                    Param("input1_height", output_height))(
                                                    horizontal_tiled_image[i],
                                                    dnn_outputs[i * 3 + j])["output"];
            }
        }

        Port tiled_image = b.add("image_processing_tile_image_vertical_3d_float")
                               .set_param(
                                   Param("x_dim", 1),
                                   Param("y_dim", 2),
                                   Param("input0_width", output_width * 3),
                                   Param("input0_height", output_height),
                                   Param("input1_width", output_width * 3),
                                   Param("input1_height", output_height))(
                                   horizontal_tiled_image[0],
                                   horizontal_tiled_image[1])["output"];

        Node denormalized = b.add("base_denormalize_3d_uint8")(tiled_image);

        // d435
        auto d435 = b.add("image_io_d435");

        // SGM
        auto sgm = b.add("sgm_sgm")(d435["output_l"], d435["output_r"]).set_param(Param("disp", disp), Param("width", d435_width), Param("height", d435_height));

        b.save("demo.json");

        denormalized["output"].bind(yolo_buf);
        d435["output_d"].bind(depth_buf);
        sgm["output"].bind(sgm_buf);


        {
            // Execution execution
            b.run();

            depth_buf.copy_to_host();
            sgm_buf.copy_to_host();
            yolo_buf.copy_to_host();

            // cv::Mat depth_img(std::vector<int>{depth_height, depth_width}, CV_U16C1, depth_buf.data());
            cv::Mat sgm_img(std::vector<int>{d435_height_, d435_width_}, CV_8UC1, sgm_buf.data());
            cv::Mat yolo_img(std::vector<int>{yolo_height * 2, yolo_width * 3}, CV_8UC3, yolo_buf.data());
            // cv::imwrite("demo-depth.png", depth_img);
            cv::imwrite("demo-sgm.png", sgm_img);
            cv::imwrite("demo-yolo.png", yolo_img);
            std::cout<<"Passed"<<std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
