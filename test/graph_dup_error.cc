#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());

        int32_t size = 1;
        // Test 1
        Buffer<int32_t> ibuf0(std::vector<int32_t>{1, 1});


        Port ip0{"input", Halide::type_of<int32_t>(), 2};
        Port vp0{"v", Halide::type_of<int32_t>()};

        Graph g0 = b.add_graph("graph0");

        Node n0 = g0.add("test_incx_i32x2")(ip0,vp0);

        ip0.bind(ibuf0);
        int32_t v0 = 0;
        vp0.bind(&v0);

        Buffer<int32_t> obuf0(std::vector<int32_t>{1, 1});
        n0["output"].bind(obuf0);

        ibuf0(0, 0) = 42;
        v0 = 0;
        obuf0(0, 0) = 0;

        g0.run();
        if (obuf0(0, 0) != 42) {
            std::cerr << "Expected: " << 42 << " Actual:" << obuf0(0, 0) << std::endl;
            return 1;
        }

         // Test 2
        Port ip1{"input", Halide::type_of<int32_t>(), 2};
        Port vp1{"v", Halide::type_of<int32_t>()};

        Graph g1 = b.add_graph("graph1");
        Node n1 = g1.add("test_incx_i32x2")(ip1,vp1);

        int32_t v1 = 0;
        vp1.bind(&v1);
        Buffer<int32_t> ibuf1(std::vector<int32_t>{1, 1});
        ip1.bind(ibuf1);
        Buffer<int32_t> obuf1(std::vector<int32_t>{1, 1});
        obuf1.fill(0);
        n1["output"].bind(obuf1);


        ibuf1(0, 0) = 42;
        v1 = 1;
        obuf1(0, 0) = 0;

        g1.run();

        if (obuf1(0, 0) != 43) {
            std::cerr << "Expected: " << 43 << " Actual:" << obuf1(0, 0) << std::endl;
            return 1;
        }

         // Test 3
        Graph g2(g0 + g1);

        g2.run();

        if (obuf1(0, 0) != 43) {
            std::cerr << "Expected: " << 43 << " Actual:" << obuf1(0, 0) << std::endl;
            return 1;
        }
         if (obuf0(0, 0) != 42) {
            std::cerr << "Expected: " << 42 << " Actual:" << obuf0(0, 0) << std::endl;
            return 1;
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
