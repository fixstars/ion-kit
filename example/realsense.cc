#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.h"

using namespace ion;

int main(int argc, char *argv[]) {
    Builder b;
    b.set_target(get_target_from_cmdline(argc, argv));
    b.with_bb_module("ion-bb");

    Node n = b.add("image_io_d435");

    Buffer<uint8_t> obuf_l(1280, 720);
    Buffer<uint8_t> obuf_r(1280, 720);
    Buffer<uint16_t> obuf_d(1280, 720);

    n["output_l"].bind(obuf_l);
    n["output_r"].bind(obuf_r);
    n["output_d"].bind(obuf_d);

    b.run();

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
