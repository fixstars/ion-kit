#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    Param v41{"v", "41"};
    Builder b;
    b.set_target(Halide::get_host_target());
    Node n;
    n = b.add("test_producer").set_param(v41);
    Port intm = n["output"];
    n = b.add("test_inc_i32x2")(n["output"]);

    PortMap pm;
    Halide::Buffer<int32_t> outBuf0(std::vector<int32_t>{1, 1});
    outBuf0(0, 0) = 0;
    pm.set(n["output"], outBuf0);
    Halide::Buffer<int32_t> outBuf1(std::vector<int32_t>{1, 1});
    outBuf1(0, 0) = 1;
    pm.set(intm, outBuf1);

    for (int i=0; i<10; ++i) {
        b.run(pm);
        if (outBuf0(0, 0) != outBuf1(0, 0)) {
            std::cout << "o0:" << outBuf0(0, 0) << std::endl;
            std::cout << "o1:" << outBuf1(0, 0) << std::endl;
            return 1;
        }
    }

    std::cout << "Success" << std::endl;

    return 0;
}
