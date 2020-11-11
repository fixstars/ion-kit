#include <ion/ion.h>
#include <iostream>

#include "ion-bb-dnn/bb.h"
#include "ion-bb-genesis-cloud/bb.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int input_height = 512;
        const int input_width = 512;
        const int height = 416;
        const int width = 416;
        const float scale = static_cast<float>(width) / static_cast<float>(input_width);

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n;
        n = b.add("genesis_cloud_image_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/pedestrian.jpg"});
        n = b.add("genesis_cloud_scale_u8x3")(n["output"])
             .set_param(
                 Param{"input_height", std::to_string(input_height)},
                 Param{"input_width", std::to_string(input_width)},
                 Param{"scale", std::to_string(scale)});
        n = b.add("dnn_reorder_hwc2chw")(n["output"]);
        n = b.add("genesis_cloud_normalize_u8x3")(n["output"]);
        n = b.add("dnn_object_detection")(n["output"]);
        n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);


        b.compile("dnn");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
