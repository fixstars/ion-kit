#include <ion/ion.h>
#include <ion/json.hpp>
#include <iostream>

#include "ion-bb-demo/bb.h"
#include "ion-bb-demo/rt.h"
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
        // TODO: Test with FullHD
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
#if 1
        auto img = n["output"];
        n = b.add("dnn_tlt_peoplenet")(img);
        n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);
        n = b.add("demo_gui_display")(n["output"]).set_param(wparam, hparam);
        Port out_p1 = n["output"];

        n = b.add("dnn_tlt_peoplenet_md")(img).set_param(wparam, hparam);
        n = b.add("dnn_classify_gender")(img, n["output"]);
        n = b.add("dnn_json_dict_average_regulator")(n["output"]).set_param(Param{"period_in_sec", "3"});
        n = b.add("dnn_ifttt_webhook_uploader")(n["output"]).set_param(Param{"ifttt_webhook_url", "http://maker.ifttt.com/trigger/gender_count/with/key/buf--6AoUjTGu868Pva_Q9"});
        Port out_p2 = n["output"];

        PortMap pm;
        pm.set(wport, width);
        pm.set(hport, height);
        Halide::Buffer<int32_t> out1 = Halide::Buffer<int32_t>::make_scalar();
        pm.set(out_p1, out1);
        Halide::Buffer<int32_t> out2 = Halide::Buffer<int32_t>::make_scalar();
        pm.set(out_p2, out2);
        for (int i=0; i<1000; ++i) {
            b.run(pm);
        }
        // for (int i=0; i<1000; ++i) {
        //     b.run(pm);
        //     json j = json::parse(reinterpret_cast<const char*>(buf.data()));
        //     std::cout << j << std::endl;
        // }
#else
        n = b.add("dnn_tlt_peoplenet")(n["output"]);
        n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);
        n = b.add("opencv_display")(n["output"], wport, hport);

        PortMap pm;
        pm.set(wport, width);
        pm.set(hport, height);
        Halide::Buffer<int32_t> buf = Halide::Buffer<int32_t>::make_scalar();
        pm.set(n["output"], buf);
        for (int i=0; i<1000; ++i) {
            b.run(pm);
        }
#endif

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
