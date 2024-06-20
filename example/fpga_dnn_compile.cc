#include <ion/ion.h>
#include <iostream>

#include "ion-bb-core/bb.h"
#include "ion-bb-fpga/bb.h"
#include "ion-bb-dnn/bb.h"
#include "ion-bb-image-io/bb.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int input_height = 512;
        const int input_width = 512;

        Builder b;
        b.set_target(Halide::Target("arm-64-linux-vivado_hls-dpu"));
        // b.set_target(Halide::Target("arm-64-linux"));

        Node n;
        n = b.add("image_io_color_data_loader").set_params(Param{"url", "http://ion-kit.s3.us-west-2.amazonaws.com/images/pedestrian.jpg"});
        n = b.add("fpga_normalize_3d_chw_uint8")
            .set_params(
                Param{"width", "512"},
                Param{"height", "512"}
            )(n["output"]);
        n = b.add("base_reorder_buffer_3d_float")(n["output"]).set_params(Param{"dim0", "2"}, Param{"dim1", "0"}, Param{"dim2", "1"});  // CHW -> HWC
        n = b.add("dnn_object_detection")(n["output"]);
        n = b.add("base_denormalize_3d_uint8")(n["output"]);

        b.compile("dnn_fpga");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
