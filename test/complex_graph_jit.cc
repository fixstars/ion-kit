#include "ion/ion.h"

using namespace ion;

int main() {
    try {
        int32_t size = 16;

        ion::Buffer<int32_t> in(std::vector<int>{size, size});
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                in(x, y) = 40;
            }
        }
        ion::Buffer<int32_t> out0(std::vector<int>{size, size});
        ion::Buffer<int32_t> out1(std::vector<int>{size, size});

        Param v0("v", 0);
        Param v1("v", 1);

        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());

        Node n;
        n = b.add("test_inc_i32x2")(in).set_params(v1);
        n = b.add("test_branch")(n["output"], &size, &size);
        auto ln = b.add("test_inc_i32x2")(n["output0"]);
        auto rn = b.add("test_inc_i32x2")(n["output1"]).set_params(v1);
        n = b.add("test_merge")(ln["output"], rn["output"], &size);
        n = b.add("test_branch")(n["output"], &size, &size);
        ln = b.add("test_extern_inc_i32x2")(n["output0"]).set_params(v0);
        rn = b.add("test_inc_i32x2")(n["output1"]).set_params(v0);

        b.save("complex_graph.json");

        ln["output"].bind(out0);
        rn["output"].bind(out1);

        b.run();

        int32_t split_n = 2;
        for (int y = 0; y < size / split_n; ++y) {
            for (int x = 0; x < size; ++x) {
                std::cerr << out0(x, y) << " ";
                if (out0(x, y) != 41) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

        for (int y = 0; y < size / split_n; ++y) {
            for (int x = 0; x < size; ++x) {
                std::cerr << out1(x, y) << " ";
                if (out1(x, y) != 42) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
