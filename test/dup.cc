#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    Halide::Type t = Halide::type_of<int32_t>();
    Port min0{"min0", t}, extent0{"extent0", t}, min1{"min1", t}, extent1{"extent1", t}, v{"v", t};
    Param v41{"v", "41"};
    Builder b;
    b.set_target(Halide::get_host_target());
    Node n;
    n = b.add("test_producer").set_param(v41);
    n = b.add("test_dup")(n["output"]);
    Port output0 = n["output0"];
    n = b.add("test_consumer")(n["output1"], min0, extent0, min1, extent1, v);
    Port output1 = n["output"];

    PortMap pm;
    pm.set(min0, 0);
    pm.set(extent0, 2);
    pm.set(min1, 0);
    pm.set(extent1, 2);
    pm.set(v, 1);

    Halide::Buffer<int32_t> obuf0(std::vector<int>{2, 2});
    pm.set(output0, obuf0);

    Halide::Buffer<int32_t> obuf1 = Halide::Buffer<int32_t>::make_scalar();
    pm.set(output1, obuf1);

    b.run(pm);

    for (int y=0; y<2; ++y) {
        for (int x=0; x<2; ++x) {
            std::cout << obuf0(x, y) << " ";
            if (obuf0(x, y) != 41) {
                std::cerr << "Invalid value" << std::endl;
                return 1;
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
