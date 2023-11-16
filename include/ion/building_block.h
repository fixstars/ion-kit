#ifndef ION_BUILDING_BLOCK_H
#define ION_BUILDING_BLOCK_H

#include <vector>
#include <Halide.h>

// #include "generator.h"

namespace ion {

enum class IOKind { Unknown,
                    Scalar,
                    Function,
                    Buffer };

class StubInput {
    const IOKind kind_;
    // Exactly one of the following fields should be defined:
    const Halide::Internal::Parameter parameter_;
    const Halide::Func func_;
    const Halide::Expr expr_;

public:
    // *not* explicit.
    // template<typename T2>
    // StubInput(const StubInputBuffer<T2> &b)
    //     : kind_(IOKind::Buffer), parameter_(b.parameter_), func_(), expr_() {
    // }
    StubInput(const Halide::Func &f)
        : kind_(IOKind::Function), parameter_(), func_(f), expr_() {
    }
    StubInput(const Halide::Expr &e)
        : kind_(IOKind::Scalar), parameter_(), func_(), expr_(e) {
    }

private:
    // friend class GeneratorInputBase;

    IOKind kind() const {
        return kind_;
    }

    Halide::Internal::Parameter parameter() const {
        // internal_assert(kind_ == IOKind::Buffer);
        return parameter_;
    }

    Halide::Func func() const {
        // internal_assert(kind_ == IOKind::Function);
        return func_;
    }

    Halide::Expr expr() const {
        // internal_assert(kind_ == IOKind::Scalar);
        return expr_;
    }
};

class InputBase {
 public:

     std::string name() {
         return "";
     }

     bool is_array() {
         return true;
     }

     IOKind kind() {
         return IOKind::Unknown;
     }
};

class OutputBase {
 public:

     std::string name() {
         return "";
     }

     bool is_array() {
         return true;
     }
};

class ParamInfo {
 public:
     const std::vector<InputBase *> &inputs() const {
         return inputs_;
     }

     const std::vector<OutputBase *> &outputs() const {
         return outputs_;
     }
 private:

     std::vector<InputBase *> inputs_;
     std::vector<OutputBase *> outputs_;
};

struct StringOrLoopLevel {
    std::string string_value;
    Halide::LoopLevel loop_level;

    StringOrLoopLevel() = default;
    /*not-explicit*/ StringOrLoopLevel(const char *s)
        : string_value(s) {
    }
    /*not-explicit*/ StringOrLoopLevel(const std::string &s)
        : string_value(s) {
    }
    /*not-explicit*/ StringOrLoopLevel(const Halide::LoopLevel &loop_level)
        : loop_level(loop_level) {
    }
};

using BuildingBlockParamsMap = std::map<std::string, StringOrLoopLevel>;

class BuildingBlockBase {
 public:
     const ParamInfo& param_info() const {
         return param_info_;
     }

     void set_param_values(const BuildingBlockParamsMap& params) {
     }

     std::vector<Halide::Func> get_outputs(const std::string& n) {
        return {};
     }

     std::vector<StubInput> build_input(size_t i, const Halide::Internal::Buffer<T> &arg) {
        return {};
     }

 private:
     ParamInfo param_info_;
};

template<typename T>
class BuildingBlock : BuildingBlockBase {

};

class BuildingBlockContext {

 public:
     BuildingBlockContext(const Halide::Target& t) {
     }
};

class BuildingBlockRegistry {
 public:
    static std::vector<std::string> enumerate() {
        return {};
    }

    static BuildingBlockBase *create(const std::string& name, const BuildingBlockContext& context) {
        return nullptr;
    }
};

} // namespace ion

#define ION_REGISTER_BUILDING_BLOCK(...) ION_REGISTER_GENERATOR(__VA_ARGS__)

#endif
