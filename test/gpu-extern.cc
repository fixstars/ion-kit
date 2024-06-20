#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    try {
        int size = 32;

        Param wp{"width", std::to_string(size)};
        Param hp{"height", std::to_string(size)};
        Param vp{"v", std::to_string(1)};

        Builder b;
        b.set_target(Halide::get_host_target().with_feature(Halide::Target::CUDA));
        b.with_bb_module("ion-bb");

        Node n;
        Port ip{"input", Halide::type_of<int32_t>(), 2};
        n = b.add("test_extern_inc_i32x2")(ip).set_params(wp, hp, vp);
        n = b.add("test_extern_inc_i32x2")(n["output"]).set_params(wp, hp, vp);



        Halide::Buffer<int32_t> ibuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                ibuf(x, y) = 42;
            }
        }
        ip.bind(ibuf);

        Halide::Buffer<int32_t> obuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                obuf(x, y) = 0;
            }
        }
        n["output"].bind(obuf);
        b.run();

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (obuf(x, y) != 44) {
                    throw std::runtime_error("Invalid value");
                }
            }
        }

        std::cout << "OK" << std::endl;

    } catch (const Halide::Error& e) {
        std::cout << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
