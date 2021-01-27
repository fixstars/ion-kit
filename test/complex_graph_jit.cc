#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    Halide::Type t = Halide::type_of<int32_t>();
    Port input{"input", t, 2}, width{"width", t}, height{"height", t};
    Param v0{"v", "0"};
    Param v1{"v", "1"};
    Builder b;
    b.set_target(Halide::get_host_target());
    Node n;
    n = b.add("test_inc_i32x2")(input).set_param(v1);
    n = b.add("test_branch")(n["output"], width, height);
    auto ln = b.add("test_inc_i32x2")(n["output0"]);
    auto rn = b.add("test_inc_i32x2")(n["output1"]).set_param(v1);
    n = b.add("test_merge")(ln["output"], rn["output"], height);
    n = b.add("test_branch")(n["output"], width, height);
    ln = b.add("test_inc_i32x2")(n["output0"]).set_param(v0);
    rn = b.add("test_inc_i32x2")(n["output1"]).set_param(v0);

    b.save("complex_graph.json");

    int32_t size = 16;
    int32_t split_n = 2;

    Halide::Buffer<int32_t> ibuf(std::vector<int>{size, size});
    for (int y=0; y<size; ++y) {
        for (int x=0; x<size; ++x) {
            ibuf(x, y) = 40;
        }
    }

    ion::PortMap pm;
    pm.set(input, ibuf);
    pm.set(width, size);
    pm.set(height, size);

    Halide::Buffer<int32_t> out0(std::vector<int>{size, size});
    Halide::Buffer<int32_t> out1(std::vector<int>{size, size});
    pm.set(ln["output"], out0);
    pm.set(rn["output"], out1);

    b.run(pm);

    for (int y=0; y<size/split_n; ++y) {
        for (int x=0; x<size; ++x) {
            std::cerr << out0(x, y) << " ";
            if (out0(x, y) != 41) {
                return -1;
            }
        }
        std::cerr << std::endl;
    }

    for (int y=0; y<size/split_n; ++y) {
        for (int x=0; x<size; ++x) {
            std::cerr << out1(x, y) << " ";
            if (out1(x, y) != 42) {
                return -1;
            }
        }
        std::cerr << std::endl;
    }

    return 0;
}
