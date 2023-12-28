#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        int size = 32;

        Param wp{"width", std::to_string(size)};
        Param hp{"height", std::to_string(size)};
        Param vp{"v", std::to_string(1)};

        Builder b;
        b.set_target(Halide::get_host_target().with_feature(Halide::Target::Profile)); // CPU
        b.with_bb_module("ion-bb");
        b.with_bb_module("ion-bb-test");

        Node n;
        Port ip{"input", Halide::type_of<int32_t>(), 2};
        n = b.add("test_extern_inc_i32x2")(ip).set_params(wp, hp, vp);
        n = b.add("base_schedule")(n["output"]).set_params(Param{"output_name", "b1"}, Param{"compute_level", "compute_inline"});
        n = b.add("test_extern_inc_i32x2")(n["output"]).set_params(wp, hp, vp);
        n = b.add("base_schedule")(n["output"]).set_params(Param{"output_name", "b2"}, Param{"compute_level", "compute_inline"});

        PortMap pm;

        Halide::Buffer<int32_t> ibuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                ibuf(x, y) = 42;
            }
        }
        pm.set(ip, ibuf);

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                obuf(x, y) = 0;
            }
        }
        pm.set(n["output"], obuf);

        b.run(pm);

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (obuf(x, y) != 44) {
                    throw std::runtime_error("Invalid value");
                }
            }
        }

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
