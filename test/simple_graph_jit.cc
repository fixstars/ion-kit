#include "ion/ion.h"

using namespace ion;

int main()
{
    Halide::Type t = Halide::type_of<int32_t>();
    Port ip{"input", t, 2};
    Builder b;
    b.with_bb_module("ion-bb-test");
    b.set_target(Halide::get_host_target().with_feature(Halide::Target::Debug).with_feature(Halide::Target::TracePipeline));
    Node n;
    n = b.add("test_inc_i32x2");

    ion::Buffer<int32_t> ibuf{16, 16};
    ion::Buffer<int32_t> obuf{16, 16};

    PortMap pm;
    pm.set(ip, ibuf);
    pm.set(n["output"], obuf);

    try {
    b.prepare(pm);
    b.callable_(ibuf, obuf);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
