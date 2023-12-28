#include <ion/ion.h>

using namespace ion;

int main() {
    try {
        int input_width = 1282;
        int input_height = 1110;
        float scale = 0.5;
        int output_width = static_cast<int>(input_width * scale);
        int output_height = static_cast<int>(input_height * scale);
        const int disp = 16;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node ln = b.add("image_io_color_data_loader").set_params(Param{"url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/aloe_left.jpg"}, Param{"width", std::to_string(input_width)}, Param{"height", std::to_string(input_height)});
        ln = b.add("base_normalize_3d_uint8")(ln["output"]);
        ln = b.add("image_processing_resize_nearest_3d")(ln["output"]).set_params(Param{"width", std::to_string(input_width)}, Param{"height", std::to_string(input_height)}, Param{"scale", std::to_string(scale)});
        ln = b.add("base_schedule")(ln["output"]).set_params(Param{"output_name", "scaled_left"}, Param{"compute_level", "compute_root"});
        ln = b.add("image_processing_calc_luminance")(ln["output"]).set_params(Param{"luminance_method", "Average"});
        ln = b.add("base_denormalize_2d_uint8")(ln["output"]);

        Node rn = b.add("image_io_color_data_loader").set_params(Param{"url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/aloe_right.jpg"}, Param{"width", std::to_string(input_width)}, Param{"height", std::to_string(input_height)});
        rn = b.add("base_normalize_3d_uint8")(rn["output"]);
        rn = b.add("image_processing_resize_nearest_3d")(rn["output"]).set_params(Param{"width", std::to_string(input_width)}, Param{"height", std::to_string(input_height)}, Param{"scale", std::to_string(scale)});
        rn = b.add("base_schedule")(rn["output"]).set_params(Param{"output_name", "scaled_right"}, Param{"compute_level", "compute_root"});
        rn = b.add("image_processing_calc_luminance")(rn["output"]).set_params(Param{"luminance_method", "Average"});
        rn = b.add("base_denormalize_2d_uint8")(rn["output"]);

        Node n = b.add("sgm_sgm")(ln["output"], rn["output"]).set_params(Param{"disp", std::to_string(disp)}, Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)});

        b.compile("sgm");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
