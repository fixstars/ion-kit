#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());

        int32_t size = 16;

        Buffer<int32_t> in0(size, size);
        in0.fill(1);

        Buffer<int32_t> in1(size, size);
        in1.fill(1);

        Buffer<int32_t> out0(size, size);
        out0.fill(0);

        Buffer<int32_t> out1(size, size);
        out1.fill(0);

        Graph g0 = b.add_graph("graph0");
        Node n0 = g0.add("test_inc_i32x2")(in0).set_param(Param("v", 40));
        n0["output"].bind(out0);
        g0.run();

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (out0(x, y) != 41) {
                    return 1;
                }
                if (out1(x, y) != 0) {
                    return 1;
                }
            }
        }

        Graph g1 = b.add_graph("graph1");
        Node n1 = g1.add("test_inc_i32x2")(in1).set_param(Param("v", 41));
        n1["output"].bind(out1);
        g1.run();

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (out0(x, y) != 41) {
                    return 1;
                }
                if (out1(x, y) != 42) {
                    return 1;
                }
            }
        }

        out0.fill(0);
        out1.fill(0);

        Graph g2(g0 + g1);
        g2.run();

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (out0(x, y) != 41) {
                    return 1;
                }
                if (out1(x, y) != 42) {
                    return 1;
                }
            }
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
