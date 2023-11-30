#include <iostream>
#include <ion/ion.h>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        // TODO: Test with FullHD
        const int width = 1280;
        const int height = 720;

        Param wparam("width", std::to_string(width));
        Param hparam("height", std::to_string(height));

        Port wport("width", Halide::type_of<int32_t>());
        Port hport("height", Halide::type_of<int32_t>());

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_camera").set_param(wparam, hparam);
        n = b.add("base_normalize_3d_uint8")(n["output"]);
        n = b.add("base_reorder_buffer_3d_float")(n["output"]).set_param(Param{"dim0", "2"}, Param{"dim1", "0"}, Param{"dim2", "1"});  // CHW -> HWC

        auto img = n["output"];
        Port out_p1 = n["output"];

        PortMap pm;
        pm.set(wport, width);
        pm.set(hport, height);
        Halide::Buffer<int32_t> out1 = Halide::Buffer<int32_t>::make_scalar();
        pm.set(out_p1, out1);

        for (int i=0; i<10; ++i) {
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
