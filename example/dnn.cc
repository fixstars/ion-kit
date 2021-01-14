#include <ion/ion.h>
#include <iostream>

#include "ion-bb-dnn/bb.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-opencv/bb.h"
#include "ion-bb-opencv/rt.h"
#include "ion-bb-genesis-cloud/bb.h"
#include "ion-bb-genesis-cloud/rt.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int width = 1920;
        const int height = 1080;

        Param wparam("width", std::to_string(width));
        Param hparam("height", std::to_string(height));

        Port wport("width", Halide::type_of<int32_t>());
        Port hport("height", Halide::type_of<int32_t>());

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n;
        //n = b.add("genesis_cloud_image_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/crosswalk-small.png"});
        n = b.add("genesis_cloud_camera").set_param(wparam, hparam);
        n = b.add("genesis_cloud_normalize_u8x3")(n["output"]);
        n = b.add("dnn_tlt_peoplenet_md")(n["output"]).set_param(wparam, hparam);
        // n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);
        // n = b.add("genesis_cloud_color")(n["output"]);
        // n = b.add("opencv_display")(n["output"], wp, hp);

        PortMap pm;
        // pm.set(wport, 1920);
        // pm.set(hport, 1080);

        Halide::Buffer<uint8_t> buf({16*1024*1024});
        pm.set(n["output"], buf);
        for (int i=0; i<1000; ++i) {
            b.run(pm);
        }

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
