#include "ion/ion.h"

#include "test-bb.h"

using namespace ion;

int main()
{
    Halide::Type t = Halide::type_of<int32_t>();
    Port input{"input", t, 2}, width{"width", t}, height{"height", t};

    Builder b;
    b.set_target(Halide::get_target_from_environment());

    Node n;
    n = b.add("test_branch")(n["output"], width, height);
    n = b.add("test_merge")(n["output0"], n["output1"], height);

    b.compile("complex_graph");

    return 0;
}
