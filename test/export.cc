#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace ion;

int main()
{
    // simple_graph
    {
        {
            Halide::Type t = Halide::type_of<int32_t>();
            Port min0{"min0", t}, extent0{"extent0", t}, min1{"min1", t}, extent1{"extent1", t}, v{"v", t};
            Param v41{"v", "41"};
            Builder b;
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_producer").set_param(v41);
            n = b.add("test_consumer")(n["output"], min0, extent0, min1, extent1, v);
            b.save("simple_graph.json");
        }
        {
            Builder b;
            b.load("simple_graph.json");
            PortMap pm;
            pm.set(Halide::Param<int32_t>{"min0"}, 0);
            pm.set(Halide::Param<int32_t>{"extent0"}, 2);
            pm.set(Halide::Param<int32_t>{"min1"}, 0);
            pm.set(Halide::Param<int32_t>{"extent1"}, 2);
            pm.set(Halide::Param<int32_t>{"v"}, 1);

            Halide::Buffer<int32_t> out = Halide::Buffer<int32_t>::make_scalar();

            auto nodes = b.get_nodes();
            pm.set(nodes.back()["output"], out);

            b.run(pm);
        }
    }

    // complex_graph
    {
        Halide::Type t = Halide::type_of<int32_t>();
        Port input{"input", t, 2}, width{"width", t}, height{"height", t};
        Param v1{"v", "1"};
        Builder b;
        b.set_target(Halide::get_host_target());
        Node n;
        n = b.add("test_inc_i32x2")(input).set_param(v1);
        n = b.add("test_branch")(n["output"], width, height);
        auto ln = b.add("test_inc_i32x2")(n["output0"]);
        auto rn = b.add("test_inc_i32x2")(n["output1"]).set_param(v1);
        n = b.add("test_merge")(ln["output"], rn["output"], height);
        b.save("complex_graph.json");
    }

    {
        Builder b;
        b.load("complex_graph.json");

        int32_t size = 16;
        int32_t split_n = 2;

        Halide::Buffer<int32_t> ibuf(std::vector<int>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                ibuf(x, y) = 40;
            }
        }

        PortMap pm;
        pm.set(Halide::ImageParam{Halide::type_of<int32_t>(), 2, "input"}, ibuf);
        pm.set(Halide::Param<int32_t>{"width"}, size);
        pm.set(Halide::Param<int32_t>{"height"}, size);

        Halide::Buffer<int32_t> out(std::vector<int>{size, size});

        auto nodes = b.get_nodes();
        pm.set(nodes.back()["output"], out);

        b.run(pm);

        int y=0;
        for (; y<size/split_n; ++y) {
            for (int x=0; x<size; ++x) {
                std::cerr << out(x, y) << " ";
                if (out(x, y) != 41) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

        for (; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                std::cerr << out(x, y) << " ";
                if (out(x, y) != 42) {
                    return -1;
                }
            }
            std::cerr << std::endl;
        }

    }

    // Array inout
    {
        constexpr size_t h = 10, w = 10, len = 5;
        {
            Port input{"input", Halide::type_of<int32_t>(), 2};
            Builder b;
            b.set_target(Halide::get_host_target());
            auto n = b.add("test_array_output")(input).set_param(Param{"len", std::to_string(len)});
            n = b.add("test_array_input")(n["array_output"]);
            b.save("array_inout.json");
        }
        {
            Builder b;
            b.load("array_inout.json");

            Halide::Buffer<int32_t> in(w, h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    in(x, y) = y * w + x;
                }
            }

            PortMap pm;
            pm.set(Halide::ImageParam(Halide::type_of<int32_t>(), 2, "input"), in);

            // TODO: Need to resolve issue #16
            // Halide::Buffer<int32_t> out = b.run({w, h}, pm)[0];
            // pm.set(Halide::ImageParam(Halide::type_of<int32_t>(), 2, "output"), out);
            // Halide::Buffer<int32_t> out = b.run({w, h}, pm)[0];

            Halide::Buffer<int32_t> out(std::vector<int>{w, h});

            auto nodes = b.get_nodes();
            pm.set(nodes.back()["output"], out);

            if (out.dimensions() != 2) {
                return -1;
            }
            if (out.extent(0) != h) {
                return -1;
            }
            if (out.extent(1) != w) {
                return -1;
            }

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (len * in(x, y) != out(x, y)) {
                        return -1;
                    }
                }
            }

        }
    }

    return 0;
}
