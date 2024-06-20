#include <iostream>
#include <ion/ion.h>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        // TODO: Test with FullHD
        const int width = 1280;
        const int height = 720;

        Param wparam("width", width);
        Param hparam("height", height);

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_camera").set_params(wparam, hparam);
        n = b.add("base_normalize_3d_uint8")(n["output"]);
        n = b.add("base_reorder_buffer_3d_float")(n["output"]).set_params(Param("dim0", 2), Param("dim1", 0), Param("dim2", 1));  // CHW -> HWC

        auto img = n["output"];
        n = b.add("dnn_tlt_peoplenet")(img);
        n = b.add("base_reorder_buffer_3d_float")(n["output"]).set_params(Param("dim0", 1), Param("dim1", 2), Param("dim2", 0));  // HWC -> CHW
        n = b.add("base_denormalize_3d_uint8")(n["output"]);
        n = b.add("image_io_gui_display")(n["output"]).set_params(wparam, hparam);
        Port out_p1 = n["output"];

        n = b.add("dnn_tlt_peoplenet_md")(img).set_params(wparam, hparam);
        n = b.add("dnn_classify_gender")(img, n["output"]).set_params(wparam, hparam);
        n = b.add("dnn_json_dict_average_regulator")(n["output"]).set_params(Param("period_in_sec", 10));
        n = b.add("dnn_ifttt_webhook_uploader")(n["output"]).set_params(Param("ifttt_webhook_url", "http://maker.ifttt.com/trigger/gender_count/with/key/buf--6AoUjTGu868Pva_Q9"));
        Port out_p2 = n["output"];

        Halide::Buffer<int32_t> out1 = Halide::Buffer<int32_t>::make_scalar();
        out_p1.bind(out1);
        Halide::Buffer<int32_t> out2 = Halide::Buffer<int32_t>::make_scalar();
        out_p2.bind(out2);
        for (int i=0; i<100; ++i) {
            b.run();
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
