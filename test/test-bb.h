#ifndef TEST_BB_H
#define TEST_BB_H

#include "ion/ion.h"

namespace ion {
namespace bb {
namespace test {

class Producer : public BuildingBlock<Producer> {
public:
    Output<Halide::Func> output{"output", Int(32), 2};
    GeneratorParam<std::string> string_param{"string_param", "string value"};
    GeneratorParam<int32_t> v{"v", 42};
    void generate() {
        output(x, y) = v;
    }

    void schedule() {
        output.compute_root();
    }
private:
    Halide::Var x, y;
};

class Consumer : public BuildingBlock<Consumer> {
public:
    Input<Halide::Func> input{"input", Int(32), 2};
    Input<int32_t> desired_min0{"desired_min0", 0};
    Input<int32_t> desired_extent0{"desired_extent0", 0};
    Input<int32_t> desired_min1{"desired_min1", 0};
    Input<int32_t> desired_extent1{"desired_extent1", 0};
    Input<int32_t> v{"v", 1};

    Output<int> output{"output"};
    void generate() {
        using namespace Halide;
        Func in;
        in(x, y) = input(x, y);
        in.compute_root();
        std::vector<ExternFuncArgument> params{in, get_id(), desired_min0, desired_extent0, desired_min1, desired_extent1, v};
        Func consume;
        consume.define_extern("consume", params, Int(32), 0);
        consume.compute_root();
        output() = consume();

        register_disposer("consume_dispose");
    }

    void schedule() {
    }
private:
    Halide::Var x, y;
};

class Branch : public BuildingBlock<Branch> {
public:
    Input<Halide::Func> input{"input", Int(32), 2};
    Input<int32_t> input_width{"input_width", 0};
    Input<int32_t> input_height{"input_height", 0};
    Output<Halide::Func> output0{"output0", Int(32), 2};
    Output<Halide::Func> output1{"output1", Int(32), 2};

    void generate() {
        using namespace Halide;
        Var x, y;
        Func in{"branch_in"};
        in(x, y) = input(x, y);
        in.compute_root();
        std::vector<ExternFuncArgument> params{in, input_width, input_height};
        std::vector<Type> types = {Int(32), Int(32)};
        Func branch;
        branch.define_extern("branch", params, types, 2);
        branch.compute_root();
        output0(_) = branch(_)[0];
        output1(_) = branch(_)[1];
    }

    void schedule() {
    }

private:
    Halide::Var x, y;
};

class Merge : public BuildingBlock<Merge> {
public:
    Input<Halide::Func> input0{"input0", Int(32), 2};
    Input<Halide::Func> input1{"input1", Int(32), 2};
    Input<int32_t> output_height{"output_height", 0};
    Output<Halide::Func> output{"output", Int(32), 2};

    void generate() {
        output(x, y) = select(y < Halide::cast<int32_t>(output_height)/2, input0(x, y), input1(x, clamp(y - output_height, 0, output_height)));
    }

    void schedule() {
    }

private:
    Halide::Var x, y;
};

template<typename T, int D>
class Inc : public BuildingBlock<Inc<T, D>> {
public:
    GeneratorParam<T> v{"v", 0};
    Input<Halide::Func> input{"input", Halide::type_of<T>(), D};
    Output<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        output(Halide::_) = input(Halide::_) + v;
    }

    void schedule() {
    }

private:
    Halide::Var x, y;
};
using IncI32x2 = Inc<int32_t,2>;

class Dup : public BuildingBlock<Dup> {
public:
    Input<Halide::Func> input{"input", Int(32), 2};
    Output<Halide::Func> output0{"output0", Int(32), 2};
    Output<Halide::Func> output1{"output1", Int(32), 2};

    void generate() {
        output0(x, y) = input(x, y);
        output1(x, y) = input(x, y);
    }

private:
    Halide::Var x, y;
};

class Scale2x : public BuildingBlock<Scale2x> {
public:
    Input<Halide::Func> input{"input", Int(32), 2};
    Output<Halide::Func> output{"output", Int(32), 2};

    void generate() {
        output(x, y) = input(x/2, y/2);
    }

private:
    Halide::Var x, y;
};

class MultiOut : public BuildingBlock<MultiOut> {
public:
    Input<Halide::Func> input{"input", Int(32), 2};
    Output<Halide::Func> output0{"output0", Int(32), 1};
    Output<Halide::Func> output1{"output1", Int(32), 2};
    Output<Halide::Func> output2{"output2", Int(32), 3};

    void generate() {
        output0(x) = input(x, 0);
        output1(x, y) = input(x, y);
        output2(x, y, c) = input(x, y);
    }

private:
    Halide::Var x, y, c;
};

class ArrayInput : public BuildingBlock<ArrayInput> {
public:
    GeneratorParam<int> len{"len", 5};

    Input<Halide::Func[]> array_input{"array_input", Int(32), 2};
    Output<Halide::Func> output{"output", Int(32), 2};

    void generate() {
        Halide::Expr v = 0;
        for (int i = 0; i < len; ++i) {
             v += array_input[i](x, y);
        }
        output(x, y) = v;
    }

private:
    Halide::Var x, y;
};

class ArrayOutput : public BuildingBlock<ArrayOutput> {
public:
    GeneratorParam<int> len{"len", 5};
    Input<Halide::Func> input{"input", Int(32), 2};
    Output<Halide::Func[]> array_output{"array_output", Int(32), 2};

    void generate() {
        array_output.resize(len);
        for (int i = 0; i < len; ++i) {
            array_output[i](x, y) = input(x, y);
        }
    }

private:
    Halide::Var x, y;
};

class ArrayCopy : public BuildingBlock<ArrayCopy> {
public:
    GeneratorParam<int> len{"len", 5};

    Input<Halide::Func[]> array_input{"array_input", Int(32), 2};
    Output<Halide::Func[]> array_output{"array_output", Int(32), 2};

    void generate() {
        array_output.resize(len);
        for (int i = 0; i < len; ++i) {
            array_output[i](x, y) = array_input[i](x, y);
        }
    }

private:
    Halide::Var x, y;
};


class ExternIncI32x2 : public BuildingBlock<ExternIncI32x2> {
public:
    GeneratorParam<int32_t> v{"v", 0};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    Input<Halide::Func> input{"input", Halide::type_of<int32_t>(), 2};
    Output<Halide::Func> output{"output", Halide::type_of<int32_t>(), 2};

    void generate() {
        using namespace Halide;

        bool use_gpu = get_target().has_gpu_feature();

        Var x, y;
        std::vector<ExternFuncArgument> params{static_cast<Func>(input), cast<int32_t>(width), cast<int32_t>(height), cast<int32_t>(v), use_gpu};
        Func inc;
        inc.define_extern("inc", params, Int(32), 2, NameMangling::C, use_gpu ? DeviceAPI::CUDA : DeviceAPI::Host);
        inc.compute_root();

        output = inc;
    }
};

} // test
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Producer, test_producer);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Consumer, test_consumer);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Branch, test_branch);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Merge, test_merge);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::IncI32x2, test_inc_i32x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Dup, test_dup);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::Scale2x, test_scale2x);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::MultiOut, test_multi_out);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::ArrayInput, test_array_input);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::ArrayOutput, test_array_output);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::ArrayCopy, test_array_copy);
ION_REGISTER_BUILDING_BLOCK(ion::bb::test::ExternIncI32x2, test_extern_inc_i32x2);

#endif
