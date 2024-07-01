#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    try {
        Builder b;
        b.set_target(Halide::get_host_target());

        Node n;
        Port ip{"input", Halide::type_of<int32_t>(), 2};
        Port vp{"v", Halide::type_of<int32_t>()};
        n = b.add("test_incx_i32x2")(ip, vp);

        PortMap pm;

        Halide::Buffer<int32_t> ibuf(std::vector<int32_t>{1, 1});

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{1, 1});

        int32_t v = 0;

        pm.set(ip, ibuf);
        pm.set(n["output"], obuf);


        // Test 1
        ibuf(0, 0) = 42;
        pm.set(vp, 0);
        obuf(0, 0) = 0;

        b.run(pm);
        if (obuf(0, 0) != 42) {
            std::cerr << "Expected: " << 42 << " Actual:" << obuf(0, 0) << std::endl;
            return 1;
        }

        // Test 2
        ibuf(0, 0) = 42;
        pm.set(vp, 1);
        obuf(0, 0) = 0;

        b.run(pm);
        if (obuf(0, 0) != 43) {
            std::cerr << "Expected: " << 43 << " Actual:" << obuf(0, 0) << std::endl;
            return 1;
        }

        // Test 3
        ibuf(0, 0) = 44;
        pm.set(vp, 0);
        obuf(0, 0) = 0;

        b.run(pm);
        if (obuf(0, 0) != 44) {
            std::cerr << "Expected: " << 44 << " Actual:" << obuf(0, 0) << std::endl;
            return 1;
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
