#include <ion/ion.h>

using namespace ion;

struct Test : BuildingBlock<Test> {
    // This Building Block takes 1 input, 1 output and 1 parameter.
    Input<Halide::Func> input{"input", Int(32), 1};
    Output<Halide::Func> output{"output", Int(32), 1};
    std::vector<Input<int32_t> *> extra_scalar_inputs;
    GeneratorParam<int32_t> num{"num", 0};

    void configure() {
        for (int32_t i=0; i<num; ++i) {
            extra_scalar_inputs.push_back(add_input<int32_t>("extra_scalar_input_" + std::to_string(i)));
        }
    }

    void generate() {
        Halide::Var i;
        Halide::Expr v = input(i);
        for (int i=0; i<num; ++i) {
            v += *extra_scalar_inputs[i];
        }
        output(i) = v;
    }
};
ION_REGISTER_BUILDING_BLOCK(Test, test);

int main() {
    try {
        int32_t v = 1;
        auto size = 4;

        Buffer<int32_t> input{size};
        input.fill(40);

        // No extra
        {
            Builder b;
            b.set_target(get_host_target());
            Buffer<int32_t> output{size};
            b.add("test")(input)["output"].bind(output);
            b.run();
            for (int i=0; i<size; ++i) {
                if (output(i) != 40) {
                    return 1;
                }
            }
        }

        // Added Extra
        {
            Builder b;
            b.set_target(get_host_target());
            Buffer<int32_t> output{size};
            b.add("test")(input, &v, &v).set_param(Param("num", 2))["output"].bind(output);
            b.compile("x");
            b.run();
            for (int i=0; i<size; ++i) {
                if (output(i) != 42) {
                    return 1;
                }
            }
        }

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
