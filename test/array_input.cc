#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {
        constexpr size_t h = 12, w = 10, len = 5;

        // Index access
        {
            Port input{"input", Halide::type_of<int32_t>(), 2};
            Builder b;
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_array_copy")(input).set_param(Param{"len", std::to_string(len)});
            n = b.add("test_array_input")(n["array_output"]).set_param(Param{"len", std::to_string(len)});

            std::vector<Halide::Buffer<int32_t>> ins{
                Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h}
            };

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (auto &b : ins) {
                        b(x, y) = y * w + x;
                    }
                }
            }

            Halide::Buffer<int32_t> out(w, h);

            PortMap pm;
            for (size_t i=0; i<len; ++i) {
                pm.set(input[i], ins[i]);
            }
            pm.set(n["output"], out);

            b.compile("array_input_index");
            b.run(pm);

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    auto v = (ins[0](x, y) +
                              ins[1](x, y) +
                              ins[2](x, y) +
                              ins[3](x, y) +
                              ins[4](x, y));
                    if (v!= out(x, y)) {
                        throw runtime_error("Unexpected out value at ("  + std::to_string(x) + ", " + std::to_string(y) + "):"
                                            + " expect=" + std::to_string(v)
                                            + " actual=" + std::to_string(out(x, y)));
                    }
                }
            }
        }

        // Array access
        {
            Port input{"input", Halide::type_of<int32_t>(), 2};
            Builder b;
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_array_copy")(input).set_param(Param{"len", std::to_string(len)});
            n = b.add("test_array_input")(n["array_output"]).set_param(Param{"len", std::to_string(len)});

            Halide::Buffer<int32_t> in0(w, h), in1(w, h), in2(w, h), in3(w, h), in4(w, h);

            std::vector<Halide::Buffer<int32_t>> ins{
                Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h},
                    Halide::Buffer<int32_t>{w, h}
            };

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (auto &b : ins) {
                        b(x, y) = y * w + x;
                    }
                }
            }

            Halide::Buffer<int32_t> out(w, h);

            PortMap pm;
            pm.set(input, ins);
            pm.set(n["output"], out);

            b.compile("array_input_array");
            b.run(pm);

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    auto v = (ins[0](x, y) +
                              ins[1](x, y) +
                              ins[2](x, y) +
                              ins[3](x, y) +
                              ins[4](x, y));
                    if (v!= out(x, y)) {
                        throw runtime_error("Unexpected out value at ("  + std::to_string(x) + ", " + std::to_string(y) + "):"
                                            + " expect=" + std::to_string(v)
                                            + " actual=" + std::to_string(out(x, y)));
                    }
                }
            }
        }

    } catch (Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
