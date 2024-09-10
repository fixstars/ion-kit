#include <ion/ion.h>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int width = 640;
        const int height = 480;

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Port in{"input", Halide::type_of<uint8_t>(), 3};

        Node n;
        n = b.add("opencv_median_blur")(in);
        n = b.add("opencv_display")(n["output"]);

        Halide::Buffer<uint8_t> in_buf(3, width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                in_buf(0, x, y) = y * width + x;
                in_buf(1, x, y) = y * width + x;
                in_buf(2, x, y) = y * width + x;
            }
        }

        Halide::Buffer<int32_t> r = Halide::Buffer<int32_t>::make_scalar();

        in.bind(in_buf);
        n["output"].bind(r);

        for (int i = 0; i < 1000; ++i) {
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
