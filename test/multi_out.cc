#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main()
{
    // From same node
    {
        Port input{"input", Halide::type_of<int32_t>(), 2};
        Builder b;
        b.set_target(Halide::get_host_target());
        Node n;
        n = b.add("test_multi_out")(input);

        int ny = 8;//16;
        int nx = 4;//8;
        int nc = 2;//4;
        Halide::Buffer<int32_t> in(std::vector<int>{nx, ny});
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                in(x, y) = y * nx + x;
            }
        }

        Halide::Buffer<int32_t> o0(std::vector<int>{nx});
        Halide::Buffer<int32_t> o1(std::vector<int>{nx, ny});
        Halide::Buffer<int32_t> o2(std::vector<int>{nx, ny, nc});
        for (int c=0; c<nc; ++c) {
            for (int y=0; y<ny; ++y) {
                for (int x=0; x<nx; ++x) {
                    o0(x) = 0;
                    o1(x, y) = 0;
                    o2(x, y, c) = 0;
                }
            }
        }

        PortMap pm;
        pm.set(input, in);
        pm.set(n["output0"], o0);
        pm.set(n["output1"], o1);
        pm.set(n["output2"], o2);

        b.run(pm);

        if (o0.dimensions() != 1) { throw runtime_error("Unexpected o0 dimension"); }
        if (o0.extent(0) != nx) { throw runtime_error("Unexpected o0 extent(0)"); }

        if (o1.dimensions() != 2) { throw runtime_error("Unexpected o1 dimension"); }
        if (o1.extent(0) != nx) { throw runtime_error("Unexpected o1 extent(0)"); }
        if (o1.extent(1) != ny) { throw runtime_error("Unexpected o1 extent(1)"); }

        if (o2.dimensions() != 3) { throw runtime_error("Unexpected o2 dimension"); }
        if (o2.extent(0) != nx) { throw runtime_error("Unexpected o2 extent(0)"); }
        if (o2.extent(1) != ny) { throw runtime_error("Unexpected o2 extent(1)"); }
        if (o2.extent(2) != nc) { throw runtime_error("Unexpected o2 extent(2)"); }

        std::cout << "# Output from same node" << std::endl;

        // o0
        std::cout << "01:" << std::endl;
        for (int x=0; x<nx; ++x) {
            std::cout << o0(x) << " ";
        }
        std::cout << std::endl;


        // o1
        std::cout << "o1:" << std::endl;
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                std::cout << o1(x, y) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // o2
        std::cout << "o2:" << std::endl;
        for (int c=0; c<nc; ++c) {
            for (int y=0; y<ny; ++y) {
                for (int x=0; x<nx; ++x) {
                    std::cout << o2(x, y, c) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "----" << std::endl;;
        }
        std::cout << std::endl;

        for (int c=0; c<nc; ++c) {
            for (int y=0; y<ny; ++y) {
                for (int x=0; x<nx; ++x) {
                    if (in(x, 0) != o0(x)) { throw runtime_error("Unexpected o0 value"); }
                    if (in(x, y) != o1(x, y)) { throw runtime_error("Unexpected o0 value"); }
                    if (in(x, y) != o2(x, y, c)) { throw runtime_error("Unexpected o0 value"); }
                }
            }
        }

    }

    // Differnet node
    {
        Port input{"input", Halide::type_of<int32_t>(), 2};
        Builder b;
        b.set_target(Halide::get_host_target());
        Node n;
        n = b.add("test_dup")(input);

        // Before one come last
        Port op1 = n["output1"];

        n = b.add("test_scale2x")(n["output0"]);

        // Later one come first
        Port op0 = n["output"];

        int ny = 8;
        int nx = 4;
        Halide::Buffer<int32_t> in(std::vector<int>{nx, ny});
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                in(x, y) = y * nx + x;
            }
        }

        int nx2 = nx * 2;
        int ny2 = ny * 2;
        Halide::Buffer<int32_t> o0(std::vector<int>{nx2, ny2});
        for (int y=0; y<ny2; ++y) {
            for (int x=0; x<nx2; ++x) {
                o0(x, y) = 0;
            }
        }

        Halide::Buffer<int32_t> o1(std::vector<int>{nx, ny});
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                o1(x, y) = 0;
            }
        }

        PortMap pm;
        pm.set(input, in);
        pm.set(op0, o0);
        pm.set(op1, o1);

        b.run(pm);

        if (o0.dimensions() != 2) { throw runtime_error("Unexpected o0 dimension"); }
        if (o0.extent(0) != nx2) { throw runtime_error("Unexpected o0 extent(0)"); }
        if (o0.extent(1) != ny2) { throw runtime_error("Unexpected o0 extent(1)"); }

        if (o1.dimensions() != 2) { throw runtime_error("Unexpected o1 dimension"); }
        if (o1.extent(0) != nx) { throw runtime_error("Unexpected o1 extent(0)"); }
        if (o1.extent(1) != ny) { throw runtime_error("Unexpected o1 extent(1)"); }

        std::cout << "# Output from different node" << std::endl;

        // o0
        std::cout << "o0:" << std::endl;
        for (int y=0; y<ny2; ++y) {
            for (int x=0; x<nx2; ++x) {
                std::cout << o0(x, y) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // o1
        std::cout << "o1:" << std::endl;
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                std::cout << o1(x, y) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check
        for (int y=0; y<ny2; ++y) {
            for (int x=0; x<nx2; ++x) {
                if (o0(x, y) != in(x/2, y/2)) { throw runtime_error("Unexpected o0 value"); }
            }
        }

        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                if (o1(x, y) != in(x, y)) { throw runtime_error("Unexpected o1 value"); }
            }
        }

    }

    return 0;
}
