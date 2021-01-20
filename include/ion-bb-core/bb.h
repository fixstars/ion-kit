#ifndef ION_BB_CORE_BB_H
#define ION_BB_CORE_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace core {

template<typename X, typename T, int32_t D>
class BufferLoader : public BuildingBlock<X> {
    static_assert(D >= 1 && D <= 4, "D must be between 1 and 4.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_tags{"gc_tags", "input"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<std::string> url{"url", ""};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    virtual std::vector<int32_t> get_extents() = 0;

    void generate() {
        std::string url_str(url);
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<Halide::ExternFuncArgument> params = {url_buf};

        std::vector<int32_t> extents = get_extents();
        for (int i = 0; i < 4; i++) {
            if (i < extents.size()) {
                params.push_back(extents[i]);
            } else {
                params.push_back(0);
            }
        }

        Halide::Func buffer_loader(static_cast<std::string>(gc_prefix) + "buffer_loader");
        buffer_loader.define_extern("ion_bb_core_buffer_loader", params, Halide::type_of<T>(), D);
        buffer_loader.compute_root();
        output(Halide::_) = buffer_loader(Halide::_);
    }
};

template<typename X, typename T>
class BufferLoader1D : public BufferLoader<X, T, 1> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0"};

    GeneratorParam<int32_t> extent0{"extent0", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0};
    }
};

template<typename X, typename T>
class BufferLoader2D : public BufferLoader<X, T, 2> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1};
    }
};

template<typename X, typename T>
class BufferLoader3D : public BufferLoader<X, T, 3> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1), parseInt(v.extent2)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1,extent2"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2};
    }
};

template<typename X, typename T>
class BufferLoader4D : public BufferLoader<X, T, 4> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1), parseInt(v.extent2), parseInt(v.extent3)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1,extent2,extent3"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};
    GeneratorParam<int32_t> extent3{"extent3", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2, extent3};
    }
};

class BufferLoader1DUInt8 : public BufferLoader1D<BufferLoader1DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader1DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 1D UInt8 buffer from specified URL."};
};

class BufferLoader2DUInt8 : public BufferLoader2D<BufferLoader2DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader2DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 2D UInt8 buffer from specified URL."};
};

class BufferLoader3DUInt8 : public BufferLoader3D<BufferLoader3DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader3DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 3D UInt8 buffer from specified URL."};
};

class BufferLoader4DUInt8 : public BufferLoader4D<BufferLoader4DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader4DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 4D UInt8 buffer from specified URL."};
};

class BufferLoader1DUInt16 : public BufferLoader1D<BufferLoader1DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader1DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 1D UInt16 buffer from specified URL."};
};

class BufferLoader2DUInt16 : public BufferLoader2D<BufferLoader2DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader2DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 2D UInt16 buffer from specified URL."};
};

class BufferLoader3DUInt16 : public BufferLoader3D<BufferLoader3DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader3DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 3D UInt16 buffer from specified URL."};
};

class BufferLoader4DUInt16 : public BufferLoader4D<BufferLoader4DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader4DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 4D UInt16 buffer from specified URL."};
};

class BufferLoader1DFloat : public BufferLoader1D<BufferLoader1DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader1DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 1D Float buffer from specified URL."};
};

class BufferLoader2DFloat : public BufferLoader2D<BufferLoader2DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader2DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 2D Float buffer from specified URL."};
};

class BufferLoader3DFloat : public BufferLoader3D<BufferLoader3DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 3D Float buffer from specified URL."};
};

class BufferLoader4DFloat : public BufferLoader4D<BufferLoader4DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferLoader4DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 4D Float buffer from specified URL."};
};

template<typename X, typename T, int32_t D>
class BufferSaver : public BuildingBlock<X> {
    static_assert(D >= 1 && D <= 4, "D must be between 1 and 4.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_tags{"gc_tags", "output"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<std::string> path{"path", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<int32_t> output{"output"};

    virtual std::vector<int32_t> get_extents() = 0;

    void generate() {
        std::string path_str(path);
        Halide::Buffer<uint8_t> path_buf(path_str.size() + 1);
        path_buf.fill(0);
        std::memcpy(path_buf.data(), path_str.c_str(), path_str.size());

        Halide::Func input_(static_cast<std::string>(gc_prefix) + "input");
        input_(Halide::_) = input(Halide::_);
        input_.compute_root();

        std::vector<Halide::ExternFuncArgument> params = {input_, path_buf};

        std::vector<int32_t> extents = get_extents();
        for (int i = 0; i < 4; i++) {
            if (i < extents.size()) {
                params.push_back(extents[i]);
            } else {
                params.push_back(0);
            }
        }

        Halide::Func buffer_saver(static_cast<std::string>(gc_prefix) + "buffer_saver");
        buffer_saver.define_extern("ion_bb_core_buffer_saver", params, Halide::Int(32), 0);
        buffer_saver.compute_root();
        output(Halide::_) = buffer_saver(Halide::_);
    }
};

template<typename X, typename T>
class BufferSaver1D : public BufferSaver<X, T, 1> {
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0"};

    GeneratorParam<int32_t> extent0{"extent0", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0};
    }
};

template<typename X, typename T>
class BufferSaver2D : public BufferSaver<X, T, 2> {
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1};
    }
};

template<typename X, typename T>
class BufferSaver3D : public BufferSaver<X, T, 3> {
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1,extent2"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2};
    }
};

template<typename X, typename T>
class BufferSaver4D : public BufferSaver<X, T, 4> {
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "extent0,extent1,extent2,extent3"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};
    GeneratorParam<int32_t> extent3{"extent3", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2, extent3};
    }
};

class BufferSaver1DUInt8 : public BufferSaver1D<BufferSaver1DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver1DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 1D UInt8 buffer to specified path."};
};

class BufferSaver2DUInt8 : public BufferSaver2D<BufferSaver2DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver2DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 2D UInt8 buffer to specified path."};
};

class BufferSaver3DUInt8 : public BufferSaver3D<BufferSaver3DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver3DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 3D UInt8 buffer to specified path."};
};

class BufferSaver4DUInt8 : public BufferSaver4D<BufferSaver4DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver4DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 4D UInt8 buffer to specified path."};
};

class BufferSaver1DUInt16 : public BufferSaver1D<BufferSaver1DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver1DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 1D UInt16 buffer to specified path."};
};

class BufferSaver2DUInt16 : public BufferSaver2D<BufferSaver2DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver2DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 2D UInt16 buffer to specified path."};
};

class BufferSaver3DUInt16 : public BufferSaver3D<BufferSaver3DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver3DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 3D UInt16 buffer to specified path."};
};

class BufferSaver4DUInt16 : public BufferSaver4D<BufferSaver4DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver4DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 4D UInt16 buffer to specified path."};
};

class BufferSaver1DFloat : public BufferSaver1D<BufferSaver1DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver1DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 1D Float buffer to specified path."};
};

class BufferSaver2DFloat : public BufferSaver2D<BufferSaver2DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver2DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 2D Float buffer to specified path."};
};

class BufferSaver3DFloat : public BufferSaver3D<BufferSaver3DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 3D Float buffer to specified path."};
};

class BufferSaver4DFloat : public BufferSaver4D<BufferSaver4DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "BufferSaver4DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves 4D Float buffer to specified path."};
};

template<typename X, typename T, int32_t D>
class RandomBuffer : public BuildingBlock<X> {
    static_assert(D >= 1 && D <= 4, "D must be between 1 and 4.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_tags{"gc_tags", "input"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> seed{"seed", 0};
    GeneratorParam<T> min{"min", std::numeric_limits<T>::lowest()};
    GeneratorParam<T> max{"max", std::numeric_limits<T>::max()};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    virtual std::vector<int32_t> get_extents() = 0;

    Halide::Buffer<T> get_range() {
        Halide::Buffer<T> range(2);
        range(0) = min;
        range(1) = max;
        return range;
    }

    static int32_t instance_id;

    void generate() {
        std::vector<Halide::ExternFuncArgument> params = {instance_id++, static_cast<int32_t>(seed), get_range()};

        std::vector<int32_t> extents = get_extents();
        for (int i = 0; i < 4; i++) {
            if (i < extents.size()) {
                params.push_back(extents[i]);
            } else {
                params.push_back(0);
            }
        }

        Halide::Func random_buffer(static_cast<std::string>(gc_prefix) + "random_buffer");
        random_buffer.define_extern("ion_bb_core_random_buffer", params, Halide::type_of<T>(), D);
        random_buffer.compute_root();
        output(Halide::_) = random_buffer(Halide::_);
    }
};

template<typename X, typename T, int32_t D>
int32_t RandomBuffer<X, T, D>::instance_id = 0;

template<typename X, typename T>
class RandomBuffer1D : public RandomBuffer<X, T, 1> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "min,max,extent0"};

    GeneratorParam<int32_t> extent0{"extent0", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0};
    }
};

template<typename X, typename T>
class RandomBuffer2D : public RandomBuffer<X, T, 2> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "min,max,extent0,extent1"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1};
    }
};

template<typename X, typename T>
class RandomBuffer3D : public RandomBuffer<X, T, 3> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1), parseInt(v.extent2)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "min,max,extent0,extent1,extent2"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2};
    }
};

template<typename X, typename T>
class RandomBuffer4D : public RandomBuffer<X, T, 4> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.extent0), parseInt(v.extent1), parseInt(v.extent2), parseInt(v.extent3)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "min,max,extent0,extent1,extent2,extent3"};

    GeneratorParam<int32_t> extent0{"extent0", 0};
    GeneratorParam<int32_t> extent1{"extent1", 0};
    GeneratorParam<int32_t> extent2{"extent2", 0};
    GeneratorParam<int32_t> extent3{"extent3", 0};

    std::vector<int32_t> get_extents() override {
        return {extent0, extent1, extent2, extent3};
    }
};

class RandomBuffer1DUInt8 : public RandomBuffer1D<RandomBuffer1DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer1DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 1D UInt8 random buffer."};
};

class RandomBuffer2DUInt8 : public RandomBuffer2D<RandomBuffer2DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer2DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 2D UInt8 random buffer."};
};

class RandomBuffer3DUInt8 : public RandomBuffer3D<RandomBuffer3DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer3DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 3D UInt8 random buffer."};
};

class RandomBuffer4DUInt8 : public RandomBuffer4D<RandomBuffer4DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer4DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 4D UInt8 random buffer."};
};

class RandomBuffer1DUInt16 : public RandomBuffer1D<RandomBuffer1DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer1DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 1D UInt16 random buffer."};
};

class RandomBuffer2DUInt16 : public RandomBuffer2D<RandomBuffer2DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer2DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 2D UInt16 random buffer."};
};

class RandomBuffer3DUInt16 : public RandomBuffer3D<RandomBuffer3DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer3DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 3D UInt16 random buffer."};
};

class RandomBuffer4DUInt16 : public RandomBuffer4D<RandomBuffer4DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer4DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 4D UInt16 random buffer."};
};

class RandomBuffer1DFloat : public RandomBuffer1D<RandomBuffer1DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer1DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 1D Float random buffer."};
};

class RandomBuffer2DFloat : public RandomBuffer2D<RandomBuffer2DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer2DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 2D Float random buffer."};
};

class RandomBuffer3DFloat : public RandomBuffer3D<RandomBuffer3DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 3D Float random buffer."};
};

class RandomBuffer4DFloat : public RandomBuffer4D<RandomBuffer4DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "RandomBuffer4DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This makes 4D Float random buffer."};
};

template<typename X, typename T, int32_t D>
class ReorderBuffer : public BuildingBlock<X> {
    static_assert(D >= 2 && D <= 4, "D must be between 2 and 4.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    virtual std::vector<int32_t> get_order() = 0;

    void generate() {
        std::vector<int32_t> order = get_order();

        // Check order
        for (int i = 0; i < order.size(); i++) {
            if (std::count(order.begin(), order.end(), i) != 1) {
                internal_error << "Invalid order";
            }
        }

        std::vector<Halide::Var> dst_vars(D);
        std::vector<Halide::Var> src_vars;
        for (auto o : order) {
            src_vars.push_back(dst_vars[o]);
        }

        output(dst_vars) = input(src_vars);
    }
};

template<typename X, typename T>
class ReorderBuffer2D : public ReorderBuffer<X, T, 2> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[parseInt(v.dim0)], v.input[parseInt(v.dim1)]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "dim0,dim1"};

    GeneratorParam<int32_t> dim0{"dim0", 0, 0, 1};
    GeneratorParam<int32_t> dim1{"dim1", 1, 0, 1};

    std::vector<int32_t> get_order() override {
        return {dim0, dim1};
    }
};

template<typename X, typename T>
class ReorderBuffer3D : public ReorderBuffer<X, T, 3> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[parseInt(v.dim0)], v.input[parseInt(v.dim1)], v.input[parseInt(v.dim2)]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "dim0,dim1,dim2"};

    GeneratorParam<int32_t> dim0{"dim0", 0, 0, 2};
    GeneratorParam<int32_t> dim1{"dim1", 1, 0, 2};
    GeneratorParam<int32_t> dim2{"dim2", 2, 0, 2};

    std::vector<int32_t> get_order() override {
        return {dim0, dim1, dim2};
    }
};

template<typename X, typename T>
class ReorderBuffer4D : public ReorderBuffer<X, T, 4> {
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[parseInt(v.dim0)], v.input[parseInt(v.dim1)], v.input[parseInt(v.dim2)], v.input[parseInt(v.dim3)]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "dim0,dim1,dim2,dim3"};

    GeneratorParam<int32_t> dim0{"dim0", 0, 0, 3};
    GeneratorParam<int32_t> dim1{"dim1", 1, 0, 3};
    GeneratorParam<int32_t> dim2{"dim2", 2, 0, 3};
    GeneratorParam<int32_t> dim3{"dim3", 3, 0, 3};

    std::vector<int32_t> get_order() override {
        return {dim0, dim1, dim2, dim3};
    }
};

class ReorderBuffer2DUInt8 : public ReorderBuffer2D<ReorderBuffer2DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer2DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 2D UInt8 buffer."};
};

class ReorderBuffer3DUInt8 : public ReorderBuffer3D<ReorderBuffer3DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer3DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 3D UInt8 buffer."};
};

class ReorderBuffer4DUInt8 : public ReorderBuffer4D<ReorderBuffer4DUInt8, uint8_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer4DUInt8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 4D UInt8 buffer."};
};

class ReorderBuffer2DUInt16 : public ReorderBuffer2D<ReorderBuffer2DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer2DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 2D UInt16 buffer."};
};

class ReorderBuffer3DUInt16 : public ReorderBuffer3D<ReorderBuffer3DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer3DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 3D UInt16 buffer."};
};

class ReorderBuffer4DUInt16 : public ReorderBuffer4D<ReorderBuffer4DUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer4DUInt16"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 4D UInt16 buffer."};
};

class ReorderBuffer2DFloat : public ReorderBuffer2D<ReorderBuffer2DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer2DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 2D Float buffer."};
};

class ReorderBuffer3DFloat : public ReorderBuffer3D<ReorderBuffer3DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 3D Float buffer."};
};

class ReorderBuffer4DFloat : public ReorderBuffer4D<ReorderBuffer4DFloat, float> {
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderBuffer4DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "This reorders 4D Float buffer."};
};

template<typename X, typename T, int32_t D>
class Denormalize : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "This denormalize [0..1.0] values into target type range."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};
    void generate() {
        using namespace Halide;
        output(_) = saturating_cast<T>(input(_) * cast<float>((std::numeric_limits<T>::max)()));
    }
};

class Denormalize1DUInt8 : public Denormalize<Denormalize1DUInt8, uint8_t, 1> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize1DUInt8"};
};

class Denormalize2DUInt8 : public Denormalize<Denormalize2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize2DUInt8"};
};

class Denormalize3DUInt8 : public Denormalize<Denormalize3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize3DUInt8"};
};

class Denormalize4DUInt8 : public Denormalize<Denormalize4DUInt8, uint8_t, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize4DUInt8"};
};

class Denormalize1DUInt16 : public Denormalize<Denormalize1DUInt16, uint16_t, 1> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize1DUInt16"};
};

class Denormalize2DUInt16 : public Denormalize<Denormalize2DUInt16, uint16_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize2DUInt16"};
};

class Denormalize3DUInt16 : public Denormalize<Denormalize3DUInt16, uint16_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize3DUInt16"};
};

class Denormalize4DUInt16 : public Denormalize<Denormalize4DUInt16, uint16_t, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize4DUInt16"};
};

template<typename X, typename T, int32_t D>
class Normalize : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "This normalize values into range [0..1.0]."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), D};
    void generate() {
        using namespace Halide;
        output(_) = cast<float>(input(_)) / (std::numeric_limits<T>::max)();
    }
};

class Normalize1DUInt8 : public Normalize<Normalize1DUInt8, uint8_t, 1> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize1DUInt8"};
};

class Normalize2DUInt8 : public Normalize<Normalize2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize2DUInt8"};
};

class Normalize3DUInt8 : public Normalize<Normalize3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize3DUInt8"};
};

class Normalize4DUInt8 : public Normalize<Normalize4DUInt8, uint8_t, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize4DUInt8"};
};

class Normalize1DUInt16 : public Normalize<Normalize1DUInt16, uint16_t, 1> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize1DUInt16"};
};

class Normalize2DUInt16 : public Normalize<Normalize2DUInt16, uint16_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize2DUInt16"};
};

class Normalize3DUInt16 : public Normalize<Normalize3DUInt16, uint16_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize3DUInt16"};
};

class Normalize4DUInt16 : public Normalize<Normalize4DUInt16, uint16_t, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize4DUInt16"};
};

}  // namespace core
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader1DUInt8, core_buffer_loader_1d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader2DUInt8, core_buffer_loader_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader3DUInt8, core_buffer_loader_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader4DUInt8, core_buffer_loader_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader1DUInt16, core_buffer_loader_1d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader2DUInt16, core_buffer_loader_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader3DUInt16, core_buffer_loader_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader4DUInt16, core_buffer_loader_4d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader1DFloat, core_buffer_loader_1d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader2DFloat, core_buffer_loader_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader3DFloat, core_buffer_loader_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferLoader4DFloat, core_buffer_loader_4d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver1DUInt8, core_buffer_saver_1d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver2DUInt8, core_buffer_saver_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver3DUInt8, core_buffer_saver_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver4DUInt8, core_buffer_saver_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver1DUInt16, core_buffer_saver_1d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver2DUInt16, core_buffer_saver_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver3DUInt16, core_buffer_saver_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver4DUInt16, core_buffer_saver_4d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver1DFloat, core_buffer_saver_1d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver2DFloat, core_buffer_saver_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver3DFloat, core_buffer_saver_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::BufferSaver4DFloat, core_buffer_saver_4d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer1DUInt8, core_random_buffer_1d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer2DUInt8, core_random_buffer_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer3DUInt8, core_random_buffer_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer4DUInt8, core_random_buffer_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer1DUInt16, core_random_buffer_1d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer2DUInt16, core_random_buffer_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer3DUInt16, core_random_buffer_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer4DUInt16, core_random_buffer_4d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer1DFloat, core_random_buffer_1d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer2DFloat, core_random_buffer_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer3DFloat, core_random_buffer_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::RandomBuffer4DFloat, core_random_buffer_4d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer2DUInt8, core_reorder_buffer_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer3DUInt8, core_reorder_buffer_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer4DUInt8, core_reorder_buffer_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer2DUInt16, core_reorder_buffer_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer3DUInt16, core_reorder_buffer_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer4DUInt16, core_reorder_buffer_4d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer2DFloat, core_reorder_buffer_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer3DFloat, core_reorder_buffer_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::ReorderBuffer4DFloat, core_reorder_buffer_4d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize1DUInt8, core_denormalize_1d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize2DUInt8, core_denormalize_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize3DUInt8, core_denormalize_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize4DUInt8, core_denormalize_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize1DUInt16, core_denormalize_1d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize2DUInt16, core_denormalize_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize3DUInt16, core_denormalize_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Denormalize4DUInt16, core_denormalize_4d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize1DUInt8, core_normalize_1d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize2DUInt8, core_normalize_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize3DUInt8, core_normalize_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize4DUInt8, core_normalize_4d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize1DUInt16, core_normalize_1d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize2DUInt16, core_normalize_2d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize3DUInt16, core_normalize_3d_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::core::Normalize4DUInt16, core_normalize_4d_uint16);

#endif