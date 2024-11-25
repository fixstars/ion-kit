#include <exception>

#include "ion/ion.h"
#include <vector>
#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {
        constexpr size_t h = 4, w = 4;

        Halide::Buffer<int32_t> in(w, h);
        in.fill(42);

         std::vector<ion::Buffer<int32_t>> outs{
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h},
            ion::Buffer<int32_t>{w, h}};

        for (auto &out : outs) {
            out.fill(0);
        }


        Builder b;
        b.set_target(Halide::get_host_target());

        std::array<int, 4> offsets = {1, 2, 3, 4};
        auto n = b.add("test_scalar_array")(in, &offsets).set_params(Param("input_offsets.size", 4));
        n["output"].bind(outs);
        b.run();

        std::cout<<outs[0](0,0)<<std::endl;
        std::cout<<outs[1](0,0)<<std::endl;
        std::cout<<outs[2](0,0)<<std::endl;
        std::cout<<outs[3](0,0)<<std::endl;

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
