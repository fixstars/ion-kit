#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    constexpr size_t h = 12, w = 10, len = 5;

    Port input{"input", Halide::type_of<int32_t>(), 2};
    Builder b;
    b.set_target(Halide::get_host_target());
    auto n = b.add("test_array_input")(input);

    Halide::Buffer<int32_t> in(w, h);
    Halide::Buffer<int32_t> out(w, h);

    PortMap pm;
    pm.set(input, in);
    pm.set(n["output"], out);

    try {
        b.run(pm);
        std::cerr << "Unexpected result" << std::endl;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    try {
        PortMap pm;
        pm.set(input, std::vector<decltype(in)>{in});
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}
