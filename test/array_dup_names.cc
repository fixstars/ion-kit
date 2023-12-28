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
        auto n = b.add("test_array_output")(input).set_params(Param{"len", std::to_string(len)});
        n = b.add("test_array_input")(n["array_output"]);
        n = b.add("test_array_output")(n["output"]).set_params(Param{"len", std::to_string(len)});
        n = b.add("test_array_input")(n["array_output"]);

        Halide::Buffer<int32_t> in(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                in(x, y) = y * w + x;
            }
        }

        Halide::Buffer<int32_t> out(w, h);
        out.fill(0);

        PortMap pm;
        pm.set(input, in);
        pm.set(n["output"], out);
        b.run(pm);

        if (out.dimensions() != 2) {
            throw runtime_error("Unexpected out dimension");
        }
        if (out.extent(0) != w) {
            throw runtime_error("Unexpected out extent(0)");
        }
        if (out.extent(1) != h) {
            throw runtime_error("Unexpected out extent(1)");
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (len * len * in(x, y) != out(x, y)) {
                    throw runtime_error("Unexpected out value");
                }
            }
        }
    } catch (const Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
