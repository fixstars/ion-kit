#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {
        constexpr size_t h = 12, w = 10, len = 5;

        Port input{"input", Halide::type_of<int32_t>(), 2};
        Builder b;
        b.set_target(Halide::get_host_target());
        auto n = b.add("test_array_output")(input).set_param(Param{"len", std::to_string(len)});

        Halide::Buffer<int32_t> in(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                in(x, y) = y * w + x;
            }
        }

        std::vector<Halide::Buffer<int32_t>> outs{
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h}
        };
        for (auto &b : outs) {
            b.fill(0);
        }

        PortMap pm;
        pm.set(input, in);
        for (size_t i=0; i<len; ++i) {
            pm.set(n["array_output"][i], outs[i]);
        }
        b.run(pm);

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
