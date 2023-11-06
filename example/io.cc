#include <fstream>
#include <iostream>
#include <string>

#include <ion/ion.h>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int height = 1392;
        const int width = 512;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_grayscale_data_loader").set_param(Param{"width", std::to_string(width)}, Param{"height", std::to_string(height)}, Param{"url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/20210623_genesis_bayer_image_raw.zip"}, Param{"dynamic_range", "255"});

        Halide::Buffer<uint16_t> out_buf(width, height);

        PortMap pm;
        pm.set(n["output"], out_buf);

        for (int i=0; i<10; ++i) {
            b.run(pm);
            std::ofstream ofs(std::to_string(i) + ".bin");
            ofs.write(reinterpret_cast<const char *>(out_buf.data()), width * height * sizeof(uint16_t));
        }

        std::cout << "yolov4 example done!!!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
