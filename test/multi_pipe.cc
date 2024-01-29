#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        auto size = 16;


        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());

        Node n;

        Buffer<int32_t> in0(std::vector<int>{size, size});
        in0.fill(0);
        Buffer<int32_t> out0(std::vector<int>{size, size});
        n = b.add("test_inc_i32x2")(in0).set_param(Param{"v", 1});;
        n["output"].bind(out0);

        Buffer<int32_t> in1(std::vector<int>{size, size});
        in1.fill(0);
        Buffer<int32_t> out1(std::vector<int>{size, size});
        n = b.add("test_inc_i32x2")(in1).set_param(Param{"v", 2});
        n["output"].bind(out1);

        b.run();

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                std::cerr << out0(x, y) << " ";
                if (out0(x, y) != 1) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                std::cerr << out1(x, y) << " ";
                if (out1(x, y) != 2) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

        b.compile("multi_pipe");

    } catch (Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
