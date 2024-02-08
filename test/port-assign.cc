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
        Port ip0{"input0", Halide::type_of<int32_t>(), 2};
        Port ip1{"input1", Halide::type_of<int32_t>(), 2};
        n = b.add("test_sub_i32x2");

        // ip1 comes first
        n.set_iport(ip1);

        // ip0 comes next
        n.set_iport(ip0);

        Halide::Buffer<int32_t> ibuf0(std::vector<int32_t>{1, 1});
        ibuf0.fill(43);
        ip0.bind(ibuf0);

        Halide::Buffer<int32_t> ibuf1(std::vector<int32_t>{1, 1});
        ibuf1.fill(1);
        ip1.bind(ibuf1);

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{1, 1});
        n["output"].bind(obuf);

        b.run();

        if (obuf(0, 0) != 42) {
            std::cerr << "Expected: " << 42 << " Actual:" << obuf(0, 0) << std::endl;
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
