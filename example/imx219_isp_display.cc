#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include "ion-bb-genesis-cloud/bb.h"
#include "ion-bb-genesis-cloud/rt.h"

#include "ion-bb-demo/bb.h"
#include "ion-bb-demo/rt.h"

#include "util.h"

using namespace ion;

int main(int argc, char *argv[]) {

    const int32_t width_v = 3264;
    const int32_t height_v = 2464;
    const int32_t half_width_v = width_v / 2;
    const int32_t half_height_v = height_v / 2;
    const int32_t quad_width_v = half_width_v / 2;
    const int32_t quad_height_v = half_height_v / 2;

    Builder b;
    b.set_target(Halide::Target{get_target_from_cmdline(argc, argv)});

    Port width{"width", Halide::type_of<int32_t>()};
    Port height{"height", Halide::type_of<int32_t>()};
    Port half_width{"half_width", Halide::type_of<int32_t>()};
    Port half_height{"half_height", Halide::type_of<int32_t>()};
    Port quad_width{"quad_width", Halide::type_of<int32_t>()};
    Port quad_height{"quad_height", Halide::type_of<int32_t>()};
    Port bit_width{"bit_width", Halide::type_of<uint8_t>()};
    Port bit_shift{"bit_shift", Halide::type_of<uint8_t>()};
    Port input1_left{"input1_left", Halide::type_of<int32_t>()};
    Port input1_top{"input1_top", Halide::type_of<int32_t>()};
    Port downscale_factor{"downscale_factor", Halide::type_of<int32_t>()};

    constexpr int32_t num = 2;
    Node n;
    Node ns[num];
    for (int i=0; i<num; ++i) {
        n = b.add("demo_imx219").set_param(Param{"index", std::to_string(i)});
        n = b.add("demo_bayer_downscale_uint16")(width, height, downscale_factor, n["output"]);
        n = b.add("demo_normalize_raw_image")(bit_width, bit_shift, n["output"]);
        n = b.add("demo_bayer_demosaic_simple")(half_width, half_height, n["output"]);
        ns[i] = n;
    }

    n = b.add("demo_merge_image_3d_float")(input1_left, input1_top, quad_width, quad_height, ns[0]["output"], ns[1]["output"]).set_param(Param{"output_width", std::to_string(half_width_v)}, Param{"output_height", std::to_string(quad_height_v)});
    n = b.add("demo_reorder_image_chw2hwc_float")(n["output"]);
    n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);
    n = b.add("demo_gui_display")(n["output"]).set_param(Param{"width", std::to_string(half_width_v)}, Param{"height", std::to_string(quad_height_v)});

    PortMap pm;
    pm.set(width, static_cast<int32_t>(width_v));
    pm.set(height, static_cast<int32_t>(height_v));
    pm.set(half_width, static_cast<int32_t>(half_width_v));
    pm.set(half_height, static_cast<int32_t>(half_height_v));
    pm.set(quad_width, static_cast<int32_t>(quad_width_v));
    pm.set(quad_height, static_cast<int32_t>(quad_height_v));
    pm.set(bit_width, static_cast<uint8_t>(10));
    pm.set(bit_shift, static_cast<uint8_t>(6));
    pm.set(input1_left, static_cast<int32_t>(quad_width_v));
    pm.set(input1_top, static_cast<int32_t>(0));

    Halide::Buffer<int> out = Halide::Buffer<int>::make_scalar();

    pm.set(n["output"], out);

    b.save("graph.json");

    while(true) {
        //b.run({}, pm);
        b.run(pm);
        std::cout << "." << std::flush;
    }

    return 0;
}
