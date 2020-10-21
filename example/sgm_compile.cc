#include <ion/ion.h>

#include "ion-bb-genesis-cloud/bb.h"
#include "ion-bb-genesis-cloud/bb_sgm.h"
#include "ion-bb-internal/bb.h"

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

        Node ln = b.add("genesis_cloud_image_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/aloe_left.jpg"});
        ln = b.add("genesis_cloud_scale_u8x3")(ln["output"]).set_param(Param{"input_width", std::to_string(input_width)}, Param{"input_height", std::to_string(input_height)}, Param{"scale", std::to_string(scale)});
        ln = b.add("internal_schedule")(ln["output"]).set_param(Param{"output_name", "scaled_left"}, Param{"compute_level", "compute_root"});
        ln = b.add("genesis_cloud_color_interpolation_RGB8_to_Mono8")(ln["output"]);

        Node rn = b.add("genesis_cloud_image_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/aloe_right.jpg"});
        rn = b.add("genesis_cloud_scale_u8x3")(rn["output"]).set_param(Param{"input_width", std::to_string(input_width)}, Param{"input_height", std::to_string(input_height)}, Param{"scale", std::to_string(scale)});
        rn = b.add("internal_schedule")(rn["output"]).set_param(Param{"output_name", "scaled_right"}, Param{"compute_level", "compute_root"});
        rn = b.add("genesis_cloud_color_interpolation_RGB8_to_Mono8")(rn["output"]);

        Node n = b.add("genesis_cloud_sgm")(ln["output"], rn["output"]).set_param(Param{"disp", std::to_string(disp)}, Param{"width", std::to_string(output_width)}, Param{"height", std::to_string(output_height)});

        b.compile("sgm");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
