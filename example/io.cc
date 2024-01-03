#include <fstream>
#include <iostream>
#include <string>

#include <ion/ion.h>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        int height = 1392;
        int width = 512;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_grayscale_data_loader").set_param(Param("width", width),
                                                              Param("height", height),
                                                              Param("url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/20210623_genesis_bayer_image_raw.zip"),
                                                              Param("dynamic_range", 255));

        ion::Buffer<uint16_t> out_buf(width, height);

        n["output"].bind(out_buf);

        for (int i=0; i<10; ++i) {
            b.run();
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
