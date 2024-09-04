#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {
        constexpr int h = 12, w = 10, len = 5;

        ion::Buffer<int32_t> in(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                in(x, y) = y * w + x;
            }
        }

        std::vector<ion::Buffer<int32_t>> outs{
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h}};

        for (auto &b : outs) {
            b.fill(0);
        }

        Builder b;
        b.set_target(ion::get_host_target());

        Node n;
        n = b.add("test_array_output")(in).set_params(Param("len", len));
        n = b.add("test_array_copy")(n["array_output"]).set_params(Param("array_input.size", len));

        for (int i = 0; i < len; ++i) {
            n["array_output"][i].bind(outs[i]);
        }

        b.run();

        for (auto &b : outs) {
            if (b.dimensions() != 2) {
                throw runtime_error("Unexpected out dimension");
            }
            if (b.extent(0) != w) {
                throw runtime_error("Unexpected out extent(0)");
            }
            if (b.extent(1) != h) {
                throw runtime_error("Unexpected out extent(1)");
            }
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (in(x, y) != b(x, y)) {
                        throw runtime_error("Unexpected out value");
                    }
                }
            }
        }

    } catch (Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
