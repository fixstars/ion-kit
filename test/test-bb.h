#ifndef TEST_BB_H
#define TEST_BB_H

#include "ion/ion.h"

class Producer : public ion::BuildingBlock<Producer> {
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
ION_REGISTER_BUILDING_BLOCK(Producer, test_producer);

class Consumer : public ion::BuildingBlock<Consumer> {
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
        std::vector<ExternFuncArgument> params{in, desired_min0, desired_extent0, desired_min1, desired_extent1, v};
        Func consume;
        consume.define_extern("consume", params, Int(32), 0);
        consume.compute_root();
        output() = consume();
    }

    void schedule() {
    }
private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(Consumer, test_consumer);

class Branch : public ion::BuildingBlock<Branch> {
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
ION_REGISTER_BUILDING_BLOCK(Branch, test_branch);

class Merge : public ion::BuildingBlock<Merge> {
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
ION_REGISTER_BUILDING_BLOCK(Merge, test_merge);

template<typename T, int D>
class Inc : public ion::BuildingBlock<Inc<T, D>> {
public:
    Halide::GeneratorParam<T> v{"v", 0};
    Halide::GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    Halide::GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        output(Halide::_) = input(Halide::_) + v;
    }

    void schedule() {
    }

private:
    Halide::Var x, y;
};
using IncI32x2 = Inc<int32_t,2>;
ION_REGISTER_BUILDING_BLOCK(IncI32x2, test_inc_i32x2);

class Dup : public ion::BuildingBlock<Dup> {
public:
    Halide::GeneratorInput<Halide::Func> input{"input", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output0{"output0", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output1{"output1", Int(32), 2};

    void generate() {
        output0(x, y) = input(x, y);
        output1(x, y) = input(x, y);
    }

private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(Dup, test_dup);

class Scale2x : public ion::BuildingBlock<Scale2x> {
public:
    Halide::GeneratorInput<Halide::Func> input{"input", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output{"output", Int(32), 2};

    void generate() {
        output(x, y) = input(x/2, y/2);
    }

private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(Scale2x, test_scale2x);

class MultiOut : public ion::BuildingBlock<MultiOut> {
public:
    Halide::GeneratorInput<Halide::Func> input{"input", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output0{"output0", Int(32), 1};
    Halide::GeneratorOutput<Halide::Func> output1{"output1", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output2{"output2", Int(32), 3};

    void generate() {
        output0(x) = input(x, 0);
        output1(x, y) = input(x, y);
        output2(x, y, c) = input(x, y);
    }

private:
    Halide::Var x, y, c;
};
ION_REGISTER_BUILDING_BLOCK(MultiOut, test_multi_out);

class ArrayOutput : public ion::BuildingBlock<ArrayOutput> {
public:
    Halide::GeneratorParam<std::size_t> len{"len", 5};

    Halide::GeneratorInput<Halide::Func> input{"input", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func[]> array_output{"array_output", Int(32), 2};

    void generate() {
        array_output.resize(len);
        for (std::size_t i = 0; i < array_output.size(); ++i) {
            array_output[i](x, y) = input(x, y);
        }
    }

private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(ArrayOutput, test_array_output);

class ArrayInput : public ion::BuildingBlock<ArrayInput> {
public:
    Halide::GeneratorInput<Halide::Func[]> array_input{"array_input", Int(32), 2};
    Halide::GeneratorOutput<Halide::Func> output{"output", Int(32), 2};

    void generate() {
        for (std::size_t i = 0; i < array_input.size(); ++i) {
            output(x, y) += array_input[i](x, y);
        }
    }

private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(ArrayInput, test_array_input);

#endif
