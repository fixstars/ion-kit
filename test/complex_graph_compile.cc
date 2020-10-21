#include "ion/ion.h"

#include "test-bb.h"

using namespace ion;

int main()
{
    Halide::Type t = Halide::type_of<int32_t>();
    Port input{"input", t, 2}, width{"width", t}, height{"height", t};
    Param v0{"v", "0"};
    Param v1{"v", "1"};

    Builder b;
    b.set_target(Halide::get_target_from_environment());

    Node n;
    n = b.add("test_inc_i32x2")(input).set_param(v1);
    n = b.add("test_branch")(n["output"], width, height);
    auto ln = b.add("test_inc_i32x2")(n["output0"]);
    auto rn = b.add("test_inc_i32x2")(n["output1"]).set_param(v1);
    n = b.add("test_merge")(ln["output"], rn["output"], height);
    n = b.add("test_branch")(n["output"], width, height);
    ln = b.add("test_inc_i32x2")(n["output0"]).set_param(v0);
    rn = b.add("test_inc_i32x2")(n["output1"]).set_param(v0);
    b.compile("complex_graph");
    return 0;
}
