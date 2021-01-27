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
        n = b.add("test_inc_i32x2")(ip);

        PortMap pm;

        Halide::Buffer<int32_t> ibuf(std::vector<int32_t>{1, 1});
        ibuf(0, 0) = 42;
        pm.set(ip, ibuf);

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{1, 1});
        obuf(0, 0) = 0;
        pm.set(n["output"], obuf);

        b.run(pm);
    } catch (const std::range_error& e) {
        std::cout << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 0;
    }

    return 1;
}
