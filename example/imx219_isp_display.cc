#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include "util.h"

using namespace ion;

int main(int argc, char *argv[]) {

    try {

        const int32_t width_v = 3264;
        const int32_t height_v = 2464;
        const int32_t half_width_v = width_v / 2;
        const int32_t half_height_v = height_v / 2;
        const int32_t quad_width_v = half_width_v / 2;
        const int32_t quad_height_v = half_height_v / 2;

        Builder b;
        b.set_target(Halide::Target{get_target_from_cmdline(argc, argv)});
        b.with_bb_module("ion-bb");

        constexpr int32_t num = 2;
        Node n;
        Node ns[num];
        for (int i = 0; i < num; ++i) {
            n = b.add("image_io_imx219").set_params(Param("index", i));
            n = b.add("image_processing_bayer_downscale_uint16")(n["output"]).set_params(Param("input_width", width_v), Param("input_height", height_v), Param("downscale_factor", "2"));
            n = b.add("image_processing_normalize_raw_image")(n["output"]).set_params(Param("bit_width", "10"), Param("bit_shift", "6"));
            n = b.add("image_processing_bayer_demosaic_simple")(n["output"]).set_params(Param("width", half_width_v), Param("height", half_height_v));
            ns[i] = n;
        }

        n = b.add("image_processing_tile_image_horizontal_3d_float")(ns[0]["output"], ns[1]["output"]).set_params(Param("input0_width", quad_width_v), Param("input0_height", quad_height_v), Param("input1_width", quad_width_v), Param("input1_height", quad_height_v));
        n = b.add("base_denormalize_3d_uint8")(n["output"]);
        n = b.add("image_io_gui_display")(n["output"]).set_params(Param("width", half_width_v), Param("height", quad_height_v));

        auto out = ion::Buffer<int>::make_scalar();

        n["output"].bind(out);

        b.save("graph.json");

        for (int i = 0; i < 100; ++i) {
            b.run();
            std::cout << "." << std::flush;
        }

    } catch (Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
