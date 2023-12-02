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

class IOBase {
public:
    bool array_size_defined() const { return true; }
    size_t array_size() const { return 0; }
    virtual bool is_array() const { return true; }

    const std::string &name() const { return name_; }
    IOKind kind() const { return IOKind::Unknown; }

    bool types_defined() const { return true; }
    const std::vector<Halide::Type> &types() const {return types_; }
    Halide::Type type() const { return Halide::Type(); }

    bool dims_defined() const { return true; }
    int dims() const { return 0; }

    const std::vector<Halide::Func> &funcs() const { return funcs_; }
    const std::vector<Halide::Expr> &exprs() const { return exprs_; }

    virtual ~IOBase() = default;

    void set_type(const Halide::Type &type) {}
    void set_dimensions(int dims) {}
    void set_array_size(int size) {}

protected:
    mutable int array_size_;  // always 1 if is_array() == false.
        // -1 if is_array() == true but unspecified.

    const std::string name_;
    const IOKind kind_;
    mutable std::vector<Halide::Type> types_;  // empty if type is unspecified
    mutable int dims_;                 // -1 if dim is unspecified

    // Exactly one of these will have nonzero length
    std::vector<Halide::Func> funcs_;
    std::vector<Halide::Expr> exprs_;
};

class InputBase : public IOBase {
};

class OutputBase : public IOBase {
};

class BuildingBlockParamBase {
 public:

    inline const std::string &name() const {
        return name_;
    }

    virtual std::string get_default_value() const {
        return "";
    }

     virtual bool is_synthetic_param() const {
        return false;
    }

    virtual std::string get_c_type() const  {
        return "";
    }

    virtual std::string get_type_decls() const {
        return "";
    }

 private:
    const std::string name_;
};

class ParamInfo {
 public:
     const std::vector<InputBase *> &inputs() const {
         return inputs_;
     }

     const std::vector<OutputBase *> &outputs() const {
         return outputs_;
     }

    const std::vector<BuildingBlockParamBase *> &building_block_params() const {
        return filter_building_block_params;
    }
 private:

     std::vector<InputBase *> inputs_;
     std::vector<OutputBase *> outputs_;
     std::vector<BuildingBlockParamBase *> filter_building_block_params;
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

     template<typename T>
     std::vector<StubInput> build_input(size_t i, const Halide::Buffer<T> &arg) {
        return {};
     }

     std::vector<StubInput> build_input(size_t i, const Halide::Expr &arg) {
         return {};
     }

     std::vector<StubInput> build_input(size_t i, const std::vector<Halide::Expr> &arg) {
         return {};
     }

     std::vector<StubInput> build_input(size_t i, const Halide::Func &arg) {
         return {};
     }

     std::vector<StubInput> build_input(size_t i, const std::vector<Halide::Func> &arg) {
         return {};
     }

     template<typename... Args>
         void apply(const Args &...args) {
         }

     Halide::Pipeline get_pipeline() {
         return Halide::Pipeline();
     }

     void fake_configure();

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
