#include "ion/ion.h"

#include "test-bb.h"

using namespace ion;

int main()
{
    try {
        Halide::Type t = Halide::type_of<int32_t>();
        Port input{"input", t, 2}, width{"width", t}, height{"height", t};

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n;
        n = b.add("test_branch")(input, width, height);
        n = b.add("test_merge")(n["output0"], n["output1"], height);

        ion::Buffer<int32_t> obuf(16, 16);
        n["output"].bind(obuf);

        b.compile("complex_graph");
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
