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
        auto n = b.add("test_array_input")(input);

        Halide::Buffer<int32_t> in0(w, h), in1(w, h), in2(w, h), in3(w, h), in4(w, h);

        std::vector<Halide::Buffer<int32_t>> ins{
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h},
            Halide::Buffer<int32_t>{w, h}
        };

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (auto &b : ins) {
                    b(x, y) = y * w + x;
                }
            }
        }

        Halide::Buffer<int32_t> out(w, h);

        PortMap pm;
        for (size_t i=0; i<len; ++i) {
            pm.set(input[i], ins[i]);
        }
        pm.set(n["output"], out);

        b.run(pm);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if ((ins[0](x, y) +
                     ins[1](x, y) +
                     ins[2](x, y) +
                     ins[3](x, y) +
                     ins[4](x, y))!= out(x, y)) {
                    throw runtime_error("Unexpected out value");
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
