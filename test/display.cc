#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        int32_t width = 1024, height = 1024;

        ion::Buffer<uint8_t> in(std::vector<int>{width, height, 3});
        ion::Buffer<int32_t> r = ion::Buffer<int32_t>::make_scalar();

        for (int y=0; y<height; ++y) {
            for (int x=0; x<width; ++x) {
                in(x, y, 0) = y * width + x;
                in(x, y, 1) = y * width + x;
                in(x, y, 2) = y * width + x;
            }
        }

        Builder b;
        b.with_bb_module("ion-bb");
        b.set_target(ion::get_host_target());

        b.add("image_io_gui_display")(in).set_params(Param("width", width), Param("height", height))["output"].bind(r);

        for (int i=0; i<300; ++i) {
            b.run();
        }

    } catch (Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
