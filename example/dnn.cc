#include <ion/ion.h>
#include <ion/json.hpp>
#include <iostream>

#include "ion-bb-dnn/bb.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-opencv/bb.h"
#include "ion-bb-opencv/rt.h"
#include "ion-bb-genesis-cloud/bb.h"
#include "ion-bb-genesis-cloud/rt.h"

using namespace ion;

using json = nlohmann::json;

int main(int argc, char *argv[]) {
    try {
        const int width = 640;
        const int height = 480;

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
        n = b.add("dnn_gender_count")(n["output"]);
        n = b.add("demo_ifttt")(n["output"]);
        // n = b.add("dnn_tlt_peoplenet")(n["output"]);
        // n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);
        // n = b.add("opencv_display")(n["output"], wport, hport);

        PortMap pm;
        pm.set(wport, width);
        pm.set(hport, height);
        Halide::Buffer<uint8_t> buf({16*1024*1024});
        pm.set(n["output"], buf);
        for (int i=0; i<1000; ++i) {
            b.run(pm);
            json j = json::parse(reinterpret_cast<const char*>(buf.data()));
            std::cout << j << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
