#include <ion/ion.h>
#include <iostream>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int input_height = 512;
        const int input_width = 512;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_color_data_loader").set_param(Param{"url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/pedestrian.jpg"}, Param{"width", std::to_string(input_width)}, Param{"height", std::to_string(input_height)});
        n = b.add("base_normalize_3d_uint8")(n["output"]);
        n = b.add("base_reorder_buffer_3d_float")(n["output"]).set_param(Param{"dim0", "2"}, Param{"dim1", "0"}, Param{"dim2", "1"});  // CHW -> HWC
        n = b.add("dnn_object_detection")(n["output"]);
        n = b.add("base_denormalize_3d_uint8")(n["output"]);

        b.compile("dnn");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
