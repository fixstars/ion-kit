#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        int size = 1000;
        ion::Buffer<int32_t> input(std::vector<int>{size, sizse})

        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());
        Node n;
        n = b.add("test_inc_i32x2")(input);

        ion::Buffer<int32_t> r = ion::Buffer<int32_t>::make_scalar();
        n["output"].bind(r);

        b.save("simple_graph.graph");

        for (int i=0; i<5; ++i) {
            std::cout << i << "'th loop" << std::endl;
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
