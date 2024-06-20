#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        // New API
        int32_t min0 = 0, extent0 = 2, min1 = 0, extent1 = 2, v = 1;

        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());
        Node n;
        n = b.add("test_producer").set_params(Param("v", 41));
        n = b.add("test_consumer")(n["output"], &min0, &extent0, &min1, &extent1, &v);

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
