#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main() {
    try {
        int size = 32;

        Param wp{"width", std::to_string(size)};
        Param hp{"height", std::to_string(size)};
        Param vp{"v", std::to_string(1)};

        Builder b;
        // b.set_target(Halide::get_host_target()); // CPU
        b.set_target(Halide::get_host_target().with_feature(Halide::Target::CUDA));  // GPU

        Node n;
        Port ip{"input", Halide::type_of<int32_t>(), 2};
        n = b.add("test_extern_inc_i32x2")(ip).set_params(wp, hp, vp);
        n = b.add("test_extern_inc_i32x2")(n["output"]).set_params(wp, hp, vp);

        b.compile("gpu_extern");

    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;
    return 0;
}
