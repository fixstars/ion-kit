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

        Halide::Buffer<int32_t> ibuf(std::vector<int32_t>{1, 1});
        ip.bind(ibuf);

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{1, 1});
        n["output"].bind(obuf);

        int32_t v = 0;
        vp.bind(&v);

        // Test 1
        ibuf(0, 0) = 42;
        v = 0;
        obuf(0, 0) = 0;

        b.run();
        if (obuf(0, 0) != 42) {
            std::cerr << "Expected: " << 42 << " Actual:" << obuf(0, 0) << std::endl;
            return 1;
        }

        // Test 2
        ibuf(0, 0) = 42;
        v = 1;
        obuf(0, 0) = 0;

        b.run();
        if (obuf(0, 0) != 43) {
            std::cerr << "Expected: " << 43 << " Actual:" << obuf(0, 0) << std::endl;
            return 1;
        }

        // Test 3
        ibuf(0, 0) = 44;
        v = 0;
        obuf(0, 0) = 0;

        b.run();
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
