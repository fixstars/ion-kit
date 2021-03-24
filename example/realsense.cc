#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-io/rt.h"

#include "util.h"

using namespace ion;

int main(int argc, char *argv[]) {
    Builder b;
    b.set_target(Halide::Target{get_target_from_cmdline(argc, argv)});

    Node n = b.add("image_io_d435");

    PortMap pm;
    Halide::Buffer<uint8_t> obuf_l(1280, 720);
    Halide::Buffer<uint8_t> obuf_r(1280, 720);
    Halide::Buffer<uint16_t> obuf_d(1280, 720);
    pm.set(n["output_l"], obuf_l);
    pm.set(n["output_r"], obuf_r);
    pm.set(n["output_d"], obuf_d);

    b.run(pm);

    {
        std::ofstream ofs("out_stereo.bin", std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(obuf_l.data()), obuf_l.size_in_bytes());
        ofs.write(reinterpret_cast<const char *>(obuf_r.data()), obuf_r.size_in_bytes());
    }

    {
        std::ofstream ofs("out_depth.bin", std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(obuf_d.data()), obuf_d.size_in_bytes());
    }

    return 0;
}
