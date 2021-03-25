#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include "ion-bb-core/bb.h"
#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-processing/bb.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-image-io/rt.h"
#include "ion-bb-image-processing/rt.h"

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

    constexpr int32_t num = 2;
    Node n;
    Node ns[num];
    for (int i=0; i<num; ++i) {
        n = b.add("image_io_imx219").set_param(Param{"index", std::to_string(i)});
        n = b.add("image_processing_bayer_downscale_uint16")(n["output"]).set_param(Param{"width", std::to_string(width_v)}, Param{"height", std::to_string(height_v)}, Param{"downscale_factor", "2"});
        n = b.add("image_processing_normalize_raw_image")(n["output"]).set_param(Param{"bit_width", "10"}, Param{"bit_shift", "6"});
        n = b.add("image_processing_bayer_demosaic_simple")(n["output"]).set_param(Param{"width", std::to_string(half_width_v)}, Param{"height", std::to_string(half_height_v)});
        ns[i] = n;
    }

    n = b.add("image_processing_tile_image_horizontal_3d_float")(ns[0]["output"], ns[1]["output"]).set_param(Param{"input0_width", std::to_string(quad_width_v)}, Param{"input0_height", std::to_string(quad_height_v)}, Param{"input1_width", std::to_string(quad_width_v)}, Param{"input1_height", std::to_string(quad_height_v)});
    n = b.add("core_denormalize_3d_uint8")(n["output"]);
    n = b.add("image_io_gui_display")(n["output"]).set_param(Param{"width", std::to_string(half_width_v)}, Param{"height", std::to_string(quad_height_v)});

    PortMap pm;

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
