// Halide/src/Generator.cpp
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <utility>

#if defined(_MSC_VER) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ion/generator.h"

#include "Halide.h"

namespace ion {

GeneratorContext::GeneratorContext(const Halide::Target &t, bool auto_schedule,
                                   const Halide::MachineParams &machine_params)
    : target("target", t),
      auto_schedule("auto_schedule", auto_schedule),
      machine_params("machine_params", machine_params),
      externs_map(std::make_shared<ExternsMap>()),
      value_tracker(std::make_shared<Internal::ValueTracker>()) {
}

GeneratorContext::~GeneratorContext() {
    // nothing
}

void GeneratorContext::init_from_context(const ion::GeneratorContext &context) {
    target.set(context.get_target());
    auto_schedule.set(context.get_auto_schedule());
    machine_params.set(context.get_machine_params());
    value_tracker = context.get_value_tracker();
    externs_map = context.get_externs_map();
}

namespace Internal {

namespace {

// Return true iff the name is valid for Generators or Params.
// (NOTE: gcc didn't add proper std::regex support until v4.9;
// we don't yet require this, hence the hand-rolled replacement.)

bool is_alpha(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Note that this includes '_'
bool is_alnum(char c) {
    return is_alpha(c) || (c == '_') || (c >= '0' && c <= '9');
}

// Basically, a valid C identifier, except:
//
// -- initial _ is forbidden (rather than merely "reserved")
// -- two underscores in a row is also forbidden
bool is_valid_name(const std::string &n) {
    if (n.empty()) return false;
    if (!is_alpha(n[0])) return false;
    for (size_t i = 1; i < n.size(); ++i) {
        if (!is_alnum(n[i])) return false;
        if (n[i] == '_' && n[i - 1] == '_') return false;
    }
    return true;
}

std::string compute_base_path(const std::string &output_dir,
                              const std::string &function_name,
                              const std::string &file_base_name) {
    std::vector<std::string> namespaces;
    std::string simple_name = Halide::Internal::extract_namespaces(function_name, namespaces);
    std::string base_path = output_dir + "/" + (file_base_name.empty() ? simple_name : file_base_name);
    return base_path;
}

std::map<Halide::Output, std::string> compute_output_files(const Halide::Target &target,
                                                           const std::string &base_path,
                                                           const std::set<Halide::Output> &outputs) {
    std::map<Halide::Output, const Halide::Internal::OutputInfo> output_info = Halide::Internal::get_output_info(target);

    std::map<Halide::Output, std::string> output_files;
    for (auto o : outputs) {
        output_files[o] = base_path + output_info.at(o).extension;
    }
    return output_files;
}

Halide::Argument to_argument(const Halide::Internal::Parameter &param, const Halide::Expr &default_value) {
    Halide::ArgumentEstimates argument_estimates = param.get_argument_estimates();
    argument_estimates.scalar_def = default_value;
    return Halide::Argument(param.name(),
                    param.is_buffer() ? Halide::Argument::InputBuffer : Halide::Argument::InputScalar,
                    param.type(), param.dimensions(), argument_estimates);
}

Halide::Func make_param_func(const Halide::Internal::Parameter &p, const std::string &name) {
    internal_assert(p.is_buffer());
    Halide::Func f(name + "_im");
    auto b = p.buffer();
    if (b.defined()) {
        // If the Halide::Internal::Parameter has an explicit BufferPtr set, bind directly to it
        f(Halide::_) = b(Halide::_);
    } else {
        std::vector<Halide::Var> args;
        std::vector<Halide::Expr> args_expr;
        for (int i = 0; i < p.dimensions(); ++i) {
            Halide::Var v = Halide::Var::implicit(i);
            args.push_back(v);
            args_expr.push_back(v);
        }
        f(args) = Halide::Internal::Call::make(p, args_expr);
    }
    return f;
}

}  // namespace

std::vector<Halide::Type> parse_halide_type_list(const std::string &types) {
    const auto &e = get_halide_type_enum_map();
    std::vector<Halide::Type> result;
    for (auto t : Halide::Internal::split_string(types, ",")) {
        auto it = e.find(t);
        user_assert(it != e.end()) << "Halide::Type not found: " << t;
        result.push_back(it->second);
    }
    return result;
}

void ValueTracker::track_values(const std::string &name, const std::vector<Halide::Expr> &values) {
    std::vector<std::vector<Halide::Expr>> &history = values_history[name];
    if (history.empty()) {
        for (size_t i = 0; i < values.size(); ++i) {
            history.push_back({values[i]});
        }
        return;
    }

    internal_assert(history.size() == values.size())
        << "Expected values of size " << history.size()
        << " but saw size " << values.size()
        << " for name " << name << "\n";

    // For each item, see if we have a new unique value
    for (size_t i = 0; i < values.size(); ++i) {
        Halide::Expr oldval = history[i].back();
        Halide::Expr newval = values[i];
        if (oldval.defined() && newval.defined()) {
            if (can_prove(newval == oldval)) {
                continue;
            }
        } else if (!oldval.defined() && !newval.defined()) {
            // Halide::Expr::operator== doesn't work with undefined
            // values, but they are equal for our purposes here.
            continue;
        }
        history[i].push_back(newval);
        // If we exceed max_unique_values, fail immediately.
        // TODO: could be useful to log all the entries that
        // overflow max_unique_values before failing.
        // TODO: this could be more helpful about labeling the values
        // that have multiple setttings.
        if (history[i].size() > max_unique_values) {
            std::ostringstream o;
            o << "Saw too many unique values in ValueTracker[" + std::to_string(i) + "]; "
              << "expected a maximum of " << max_unique_values << ":\n";
            for (auto e : history[i]) {
                o << "    " << e << "\n";
            }
            user_error << o.str();
        }
    }
}

std::vector<Halide::Expr> parameter_constraints(const Halide::Internal::Parameter &p) {
    internal_assert(p.defined());
    std::vector<Halide::Expr> values;
    values.emplace_back(p.host_alignment());
    if (p.is_buffer()) {
        for (int i = 0; i < p.dimensions(); ++i) {
            values.push_back(p.min_constraint(i));
            values.push_back(p.extent_constraint(i));
            values.push_back(p.stride_constraint(i));
        }
    } else {
        values.push_back(p.min_value());
        values.push_back(p.max_value());
    }
    return values;
}

class StubEmitter {
public:
    StubEmitter(std::ostream &dest,
                const std::string &generator_registered_name,
                const std::string &generator_stub_name,
                const std::vector<Internal::GeneratorParamBase *> &generator_params,
                const std::vector<Internal::GeneratorInputBase *> &inputs,
                const std::vector<Internal::GeneratorOutputBase *> &outputs)
        : stream(dest),
          generator_registered_name(generator_registered_name),
          generator_stub_name(generator_stub_name),
          generator_params(select_generator_params(generator_params)),
          inputs(inputs),
          outputs(outputs) {
        namespaces = Halide::Internal::split_string(generator_stub_name, "::");
        internal_assert(!namespaces.empty());
        if (namespaces[0].empty()) {
            // We have a name like ::foo::bar::baz; omit the first empty ns.
            namespaces.erase(namespaces.begin());
            internal_assert(namespaces.size() >= 2);
        }
        class_name = namespaces.back();
        namespaces.pop_back();
    }

    void emit();

private:
    std::ostream &stream;
    const std::string generator_registered_name;
    const std::string generator_stub_name;
    std::string class_name;
    std::vector<std::string> namespaces;
    const std::vector<Internal::GeneratorParamBase *> generator_params;
    const std::vector<Internal::GeneratorInputBase *> inputs;
    const std::vector<Internal::GeneratorOutputBase *> outputs;
    int indent_level{0};

    std::vector<Internal::GeneratorParamBase *> select_generator_params(const std::vector<Internal::GeneratorParamBase *> &in) {
        std::vector<Internal::GeneratorParamBase *> out;
        for (auto p : in) {
            // These are always propagated specially.
            if (p->name == "target" ||
                p->name == "auto_schedule" ||
                p->name == "machine_params") continue;
            if (p->is_synthetic_param()) continue;
            out.push_back(p);
        }
        return out;
    }

    /** Emit spaces according to the current indentation level */
    Halide::Internal::Indentation get_indent() const {
        return Halide::Internal::Indentation{indent_level};
    }

    void emit_inputs_struct();
    void emit_generator_params_struct();
};

void StubEmitter::emit_generator_params_struct() {
    const auto &v = generator_params;
    std::string name = "GeneratorParams";
    stream << get_indent() << "struct " << name << " final {\n";
    indent_level++;
    if (!v.empty()) {
        for (auto p : v) {
            stream << get_indent() << p->get_c_type() << " " << p->name << "{ " << p->get_default_value() << " };\n";
        }
        stream << "\n";
    }

    stream << get_indent() << name << "() {}\n";
    stream << "\n";

    if (!v.empty()) {
        stream << get_indent() << name << "(\n";
        indent_level++;
        std::string comma = "";
        for (auto p : v) {
            stream << get_indent() << comma << p->get_c_type() << " " << p->name << "\n";
            comma = ", ";
        }
        indent_level--;
        stream << get_indent() << ") : \n";
        indent_level++;
        comma = "";
        for (auto p : v) {
            stream << get_indent() << comma << p->name << "(" << p->name << ")\n";
            comma = ", ";
        }
        indent_level--;
        stream << get_indent() << "{\n";
        stream << get_indent() << "}\n";
        stream << "\n";
    }

    stream << get_indent() << "inline HALIDE_NO_USER_CODE_INLINE Halide::Internal::GeneratorParamsMap to_generator_params_map() const {\n";
    indent_level++;
    stream << get_indent() << "return {\n";
    indent_level++;
    std::string comma = "";
    for (auto p : v) {
        stream << get_indent() << comma << "{\"" << p->name << "\", ";
        if (p->is_looplevel_param()) {
            stream << p->name << "}\n";
        } else {
            stream << p->call_to_string(p->name) << "}\n";
        }
        comma = ", ";
    }
    indent_level--;
    stream << get_indent() << "};\n";
    indent_level--;
    stream << get_indent() << "}\n";

    indent_level--;
    stream << get_indent() << "};\n";
    stream << "\n";
}

void StubEmitter::emit_inputs_struct() {
    struct InInfo {
        std::string c_type;
        std::string name;
    };
    std::vector<InInfo> in_info;
    for (auto input : inputs) {
        std::string c_type = input->get_c_type();
        if (input->is_array()) {
            c_type = "std::vector<" + c_type + ">";
        }
        in_info.push_back({c_type, input->name()});
    }

    const std::string name = "Inputs";
    stream << get_indent() << "struct " << name << " final {\n";
    indent_level++;
    for (auto in : in_info) {
        stream << get_indent() << in.c_type << " " << in.name << ";\n";
    }
    stream << "\n";

    stream << get_indent() << name << "() {}\n";
    stream << "\n";
    if (!in_info.empty()) {
        stream << get_indent() << name << "(\n";
        indent_level++;
        std::string comma = "";
        for (auto in : in_info) {
            stream << get_indent() << comma << "const " << in.c_type << "& " << in.name << "\n";
            comma = ", ";
        }
        indent_level--;
        stream << get_indent() << ") : \n";
        indent_level++;
        comma = "";
        for (auto in : in_info) {
            stream << get_indent() << comma << in.name << "(" << in.name << ")\n";
            comma = ", ";
        }
        indent_level--;
        stream << get_indent() << "{\n";
        stream << get_indent() << "}\n";

        indent_level--;
    }
    stream << get_indent() << "};\n";
    stream << "\n";
}

void StubEmitter::emit() {
    if (outputs.empty()) {
        // The generator can't support a real stub. Instead, generate an (essentially)
        // empty .stub.h file, so that build systems like Bazel will still get the output file
        // they expected. Note that we deliberately don't emit an ifndef header guard,
        // since we can't reliably assume that the generator_name will be globally unique;
        // on the other hand, since this file is just a couple of comments, it's
        // really not an issue if it's included multiple times.
        stream << "/* MACHINE-GENERATED - DO NOT EDIT */\n";
        stream << "/* The Generator named " << generator_registered_name << " uses ImageParam or Param, thus cannot have a Stub generated. */\n";
        return;
    }

    struct OutputInfo {
        std::string name;
        std::string ctype;
        std::string getter;
    };
    bool all_outputs_are_func = true;
    std::vector<OutputInfo> out_info;
    for (auto output : outputs) {
        std::string c_type = output->get_c_type();
        std::string getter;
        const bool is_func = (c_type == "Halide::Func");
        if (output->is_array()) {
            getter = is_func ? "get_array_output" : "get_array_output_buffer<" + c_type + ">";
        } else {
            getter = is_func ? "get_output" : "get_output_buffer<" + c_type + ">";
        }
        out_info.push_back({output->name(),
                            output->is_array() ? "std::vector<" + c_type + ">" : c_type,
                            getter + "(\"" + output->name() + "\")"});
        if (c_type != "Halide::Func") {
            all_outputs_are_func = false;
        }
    }

    std::ostringstream guard;
    guard << "ION_STUB";
    for (const auto &ns : namespaces) {
        guard << "_" << ns;
    }
    guard << "_" << class_name;

    stream << get_indent() << "#ifndef " << guard.str() << "\n";
    stream << get_indent() << "#define " << guard.str() << "\n";
    stream << "\n";

    stream << get_indent() << "/* MACHINE-GENERATED - DO NOT EDIT */\n";
    stream << "\n";

    stream << get_indent() << "#include <cassert>\n";
    stream << get_indent() << "#include <map>\n";
    stream << get_indent() << "#include <memory>\n";
    stream << get_indent() << "#include <string>\n";
    stream << get_indent() << "#include <utility>\n";
    stream << get_indent() << "#include <vector>\n";
    stream << "\n";
    stream << get_indent() << "#include \"Halide.h\"\n";
    stream << "\n";

    stream << "namespace ion_register_generator {\n";
    stream << "namespace " << generator_registered_name << "_ns {\n";
    stream << "extern std::unique_ptr<ion::Internal::GeneratorBase> factory(const ion::GeneratorContext& context);\n";
    stream << "}  // namespace ion_register_generator\n";
    stream << "}  // namespace " << generator_registered_name << "\n";
    stream << "\n";

    for (const auto &ns : namespaces) {
        stream << get_indent() << "namespace " << ns << " {\n";
    }
    stream << "\n";

    for (auto *p : generator_params) {
        std::string decl = p->get_type_decls();
        if (decl.empty()) continue;
        stream << decl << "\n";
    }

    stream << get_indent() << "class " << class_name << " final : public ion::NamesInterface {\n";
    stream << get_indent() << "public:\n";
    indent_level++;

    emit_inputs_struct();
    emit_generator_params_struct();

    stream << get_indent() << "struct Outputs final {\n";
    indent_level++;
    stream << get_indent() << "// Outputs\n";
    for (const auto &out : out_info) {
        stream << get_indent() << out.ctype << " " << out.name << ";\n";
    }

    stream << "\n";
    stream << get_indent() << "// The Halide::Target used\n";
    stream << get_indent() << "Halide::Target target;\n";

    if (out_info.size() == 1) {
        stream << "\n";
        if (all_outputs_are_func) {
            std::string name = out_info.at(0).name;
            auto output = outputs[0];
            if (output->is_array()) {
                stream << get_indent() << "operator std::vector<Halide::Func>() const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << ";\n";
                indent_level--;
                stream << get_indent() << "}\n";

                stream << get_indent() << "Halide::Func operator[](size_t i) const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << "[i];\n";
                indent_level--;
                stream << get_indent() << "}\n";

                stream << get_indent() << "Halide::Func at(size_t i) const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << ".at(i);\n";
                indent_level--;
                stream << get_indent() << "}\n";

                stream << get_indent() << "// operator operator()() overloads omitted because the sole Halide::Output is array-of-Func.\n";
            } else {
                // If there is exactly one output, add overloads
                // for operator Halide::Func and operator().
                stream << get_indent() << "operator Halide::Func() const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << ";\n";
                indent_level--;
                stream << get_indent() << "}\n";

                stream << "\n";
                stream << get_indent() << "template <typename... Args>\n";
                stream << get_indent() << "Halide::FuncRef operator()(Args&&... args) const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << "(std::forward<Args>(args)...);\n";
                indent_level--;
                stream << get_indent() << "}\n";

                stream << "\n";
                stream << get_indent() << "template <typename ExprOrVar>\n";
                stream << get_indent() << "Halide::FuncRef operator()(std::vector<ExprOrVar> args) const {\n";
                indent_level++;
                stream << get_indent() << "return " << name << "()(args);\n";
                indent_level--;
                stream << get_indent() << "}\n";
            }
        } else {
            stream << get_indent() << "// operator Halide::Func() and operator()() overloads omitted because the sole Halide::Output is not Halide::Func.\n";
        }
    }

    stream << "\n";
    if (all_outputs_are_func) {
        stream << get_indent() << "Halide::s get_pipeline() const {\n";
        indent_level++;
        stream << get_indent() << "return Halide::Pipeline(std::vector<Halide::Func>{\n";
        indent_level++;
        int commas = (int)out_info.size() - 1;
        for (const auto &out : out_info) {
            stream << get_indent() << out.name << (commas-- ? "," : "") << "\n";
        }
        indent_level--;
        stream << get_indent() << "});\n";
        indent_level--;
        stream << get_indent() << "}\n";

        stream << "\n";
        stream << get_indent() << "Halide::Realization realize(std::vector<int32_t> sizes) {\n";
        indent_level++;
        stream << get_indent() << "return get_pipeline().realize(sizes, target);\n";
        indent_level--;
        stream << get_indent() << "}\n";

        stream << "\n";
        stream << get_indent() << "template <typename... Args, typename std::enable_if<Halide::Internal::NoRealizations<Args...>::value>::type * = nullptr>\n";
        stream << get_indent() << "Halide::Realization realize(Args&&... args) {\n";
        indent_level++;
        stream << get_indent() << "return get_pipeline().realize(std::forward<Args>(args)..., target);\n";
        indent_level--;
        stream << get_indent() << "}\n";

        stream << "\n";
        stream << get_indent() << "void realize(Halide::Realization r) {\n";
        indent_level++;
        stream << get_indent() << "get_pipeline().realize(r, target);\n";
        indent_level--;
        stream << get_indent() << "}\n";
    } else {
        stream << get_indent() << "// get_pipeline() and realize() overloads omitted because some Outputs are not Halide::Func.\n";
    }

    indent_level--;
    stream << get_indent() << "};\n";
    stream << "\n";

    stream << get_indent() << "HALIDE_NO_USER_CODE_INLINE static Outputs generate(\n";
    indent_level++;
    stream << get_indent() << "const GeneratorContext& context,\n";
    stream << get_indent() << "const Inputs& inputs,\n";
    stream << get_indent() << "const GeneratorParams& generator_params = GeneratorParams()\n";
    indent_level--;
    stream << get_indent() << ")\n";
    stream << get_indent() << "{\n";
    indent_level++;
    stream << get_indent() << "using Stub = ion::Internal::GeneratorStub;\n";
    stream << get_indent() << "Stub stub(\n";
    indent_level++;
    stream << get_indent() << "context,\n";
    stream << get_indent() << "halide_register_generator::" << generator_registered_name << "_ns::factory,\n";
    stream << get_indent() << "generator_params.to_generator_params_map(),\n";
    stream << get_indent() << "{\n";
    indent_level++;
    for (size_t i = 0; i < inputs.size(); ++i) {
        stream << get_indent() << "Stub::to_stub_input_vector(inputs." << inputs[i]->name() << ")";
        stream << ",\n";
    }
    indent_level--;
    stream << get_indent() << "}\n";
    indent_level--;
    stream << get_indent() << ");\n";

    stream << get_indent() << "return {\n";
    indent_level++;
    for (const auto &out : out_info) {
        stream << get_indent() << "stub." << out.getter << ",\n";
    }
    stream << get_indent() << "stub.generator->get_target()\n";
    indent_level--;
    stream << get_indent() << "};\n";
    indent_level--;
    stream << get_indent() << "}\n";
    stream << "\n";

    stream << get_indent() << "// overload to allow GeneratorContext-pointer\n";
    stream << get_indent() << "inline static Outputs generate(\n";
    indent_level++;
    stream << get_indent() << "const GeneratorContext* context,\n";
    stream << get_indent() << "const Inputs& inputs,\n";
    stream << get_indent() << "const GeneratorParams& generator_params = GeneratorParams()\n";
    indent_level--;
    stream << get_indent() << ")\n";
    stream << get_indent() << "{\n";
    indent_level++;
    stream << get_indent() << "return generate(*context, inputs, generator_params);\n";
    indent_level--;
    stream << get_indent() << "}\n";
    stream << "\n";

    stream << get_indent() << "// overload to allow Halide::Target instead of GeneratorContext.\n";
    stream << get_indent() << "inline static Outputs generate(\n";
    indent_level++;
    stream << get_indent() << "const Halide::Target& target,\n";
    stream << get_indent() << "const Inputs& inputs,\n";
    stream << get_indent() << "const GeneratorParams& generator_params = GeneratorParams()\n";
    indent_level--;
    stream << get_indent() << ")\n";
    stream << get_indent() << "{\n";
    indent_level++;
    stream << get_indent() << "return generate(ion::GeneratorContext(target), inputs, generator_params);\n";
    indent_level--;
    stream << get_indent() << "}\n";
    stream << "\n";

    stream << get_indent() << class_name << "() = delete;\n";

    indent_level--;
    stream << get_indent() << "};\n";
    stream << "\n";

    for (int i = (int)namespaces.size() - 1; i >= 0; --i) {
        stream << get_indent() << "}  // namespace " << namespaces[i] << "\n";
    }
    stream << "\n";

    stream << get_indent() << "#endif  // " << guard.str() << "\n";
}

GeneratorStub::GeneratorStub(const GeneratorContext &context,
                             const GeneratorFactory &generator_factory)
    : generator(generator_factory(context)) {
}

GeneratorStub::GeneratorStub(const GeneratorContext &context,
                             const GeneratorFactory &generator_factory,
                             const GeneratorParamsMap &generator_params,
                             const std::vector<std::vector<Internal::StubInput>> &inputs)
    : GeneratorStub(context, generator_factory) {
    generate(generator_params, inputs);
}

// Return a vector of all Outputs of this Generator; non-array outputs are returned
// as a vector-of-size-1. This method is primarily useful for code that needs
// to iterate through the outputs of unknown, arbitrary Generators (e.g.,
// the Python bindings).
std::vector<std::vector<Halide::Func>> GeneratorStub::generate(const GeneratorParamsMap &generator_params,
                                                       const std::vector<std::vector<Internal::StubInput>> &inputs) {
    generator->set_generator_param_values(generator_params);
    generator->call_configure();
    generator->set_inputs_vector(inputs);
    Halide::Pipeline p = generator->build_pipeline();

    std::vector<std::vector<Halide::Func>> v;
    GeneratorParamInfo &pi = generator->param_info();
    if (!pi.outputs().empty()) {
        for (auto *output : pi.outputs()) {
            const std::string &name = output->name();
            if (output->is_array()) {
                v.push_back(get_array_output(name));
            } else {
                v.push_back(std::vector<Halide::Func>{get_output(name)});
            }
        }
    } else {
        // Generators with build() method can't have Output<>, hence can't have array outputs
        for (auto output : p.outputs()) {
            v.push_back(std::vector<Halide::Func>{output});
        }
    }
    return v;
}

GeneratorStub::Names GeneratorStub::get_names() const {
    auto &pi = generator->param_info();
    Names names;
    for (auto *o : pi.generator_params()) {
        names.generator_params.push_back(o->name);
    }
    for (auto *o : pi.inputs()) {
        names.inputs.push_back(o->name());
    }
    for (auto *o : pi.outputs()) {
        names.outputs.push_back(o->name());
    }
    return names;
}

const std::map<std::string, Halide::Type> &get_halide_type_enum_map() {
    static const std::map<std::string, Halide::Type> halide_type_enum_map{
        {"bool", Halide::Bool()},
        {"int8", Halide::Int(8)},
        {"int16", Halide::Int(16)},
        {"int32", Halide::Int(32)},
        {"uint8", Halide::UInt(8)},
        {"uint16", Halide::UInt(16)},
        {"uint32", Halide::UInt(32)},
        {"float16", Halide::Float(16)},
        {"float32", Halide::Float(32)},
        {"float64", Halide::Float(64)}};
    return halide_type_enum_map;
}

std::string halide_type_to_c_source(const Halide::Type &t) {
    static const std::map<halide_type_code_t, std::string> m = {
        {halide_type_int, "Int"},
        {halide_type_uint, "UInt"},
        {halide_type_float, "Float"},
        {halide_type_handle, "Handle"},
    };
    std::ostringstream oss;
    oss << "Halide::" << m.at(t.code()) << "(" << t.bits() << +")";
    return oss.str();
}

std::string halide_type_to_c_type(const Halide::Type &t) {
    auto encode = [](const Halide::Type &t) -> int { return t.code() << 16 | t.bits(); };
    static const std::map<int, std::string> m = {
        {encode(Halide::Int(8)), "int8_t"},
        {encode(Halide::Int(16)), "int16_t"},
        {encode(Halide::Int(32)), "int32_t"},
        {encode(Halide::Int(64)), "int64_t"},
        {encode(Halide::UInt(1)), "bool"},
        {encode(Halide::UInt(8)), "uint8_t"},
        {encode(Halide::UInt(16)), "uint16_t"},
        {encode(Halide::UInt(32)), "uint32_t"},
        {encode(Halide::UInt(64)), "uint64_t"},
        {encode(Halide::BFloat(16)), "uint16_t"},  // TODO: see Issues #3709, #3967
        {encode(Halide::Float(16)), "uint16_t"},   // TODO: see Issues #3709, #3967
        {encode(Halide::Float(32)), "float"},
        {encode(Halide::Float(64)), "double"},
        {encode(Halide::Handle(64)), "void*"}};
    internal_assert(m.count(encode(t))) << t << " " << encode(t);
    return m.at(encode(t));
}

int generate_filter_main_inner(int argc, char **argv, std::ostream &cerr) {
    const char kUsage[] =
        "gengen \n"
        "  [-g GENERATOR_NAME] [-f FUNCTION_NAME] [-o OUTPUT_DIR] [-r RUNTIME_NAME] [-d 1|0]\n"
        "  [-e EMIT_OPTIONS] [-n FILE_BASE_NAME] [-p PLUGIN_NAME] [-s AUTOSCHEDULER_NAME]\n"
        "       target=target-string[,target-string...] [generator_arg=value [...]]\n"
        "\n"
        " -d  Build a module that is suitable for using for gradient descent calculationn\n"
        "     in TensorFlow or PyTorch. See Generator::build_gradient_module() documentation.\n"
        "\n"
        " -e  A comma separated list of files to emit. Accepted values are:\n"
        "     [assembly, bitcode, c_header, c_source, cpp_stub, featurization,\n"
        "      llvm_assembly, object, python_extension, pytorch_wrapper, registration,\n"
        "      schedule, static_library, stmt, stmt_html, compiler_log].\n"
        "     If omitted, default value is [c_header, static_library, registration].\n"
        "\n"
        " -p  A comma-separated list of shared libraries that will be loaded before the\n"
        "     generator is run. Useful for custom auto-schedulers. The generator must\n"
        "     either be linked against a shared libHalide or compiled with -rdynamic\n"
        "     so that references in the shared library to libHalide can resolve.\n"
        "     (Note that this does not change the default autoscheduler; use the -s flag\n"
        "     to set that value.)"
        "\n"
        " -r   The name of a standalone runtime to generate. Only honors EMIT_OPTIONS 'o'\n"
        "     and 'static_library'. When multiple targets are specified, it picks a\n"
        "     runtime that is compatible with all of the targets, or fails if it cannot\n"
        "     find one. Flags across all of the targets that do not affect runtime code\n"
        "     generation, such as `no_asserts` and `no_runtime`, are ignored.\n"
        "\n"
        " -s  The name of an autoscheduler to set as the default.\n";

    std::map<std::string, std::string> flags_info = {
        {"-d", "0"},
        {"-e", ""},
        {"-f", ""},
        {"-g", ""},
        {"-n", ""},
        {"-o", ""},
        {"-p", ""},
        {"-r", ""},
        {"-s", ""},
    };
    GeneratorParamsMap generator_args;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            std::vector<std::string> v = Halide::Internal::split_string(argv[i], "=");
            if (v.size() != 2 || v[0].empty() || v[1].empty()) {
                cerr << kUsage;
                return 1;
            }
            generator_args[v[0]] = v[1];
            continue;
        }
        auto it = flags_info.find(argv[i]);
        if (it != flags_info.end()) {
            if (i + 1 >= argc) {
                cerr << kUsage;
                return 1;
            }
            it->second = argv[i + 1];
            ++i;
            continue;
        }
        cerr << "Unknown flag: " << argv[i] << "\n";
        cerr << kUsage;
        return 1;
    }

    // It's possible that in the future loaded plugins might change
    // how arguments are parsed, so we handle those first.
    for (const auto &lib : Halide::Internal::split_string(flags_info["-p"], ",")) {
        if (!lib.empty()) {
            Halide::load_plugin(lib);
        }
    }

    if (flags_info["-d"] != "1" && flags_info["-d"] != "0") {
        cerr << "-d must be 0 or 1\n";
        cerr << kUsage;
        return 1;
    }
    const int build_gradient_module = flags_info["-d"] == "1";

    std::string autoscheduler_name = flags_info["-s"];
    if (!autoscheduler_name.empty()) {
        Halide::Pipeline::set_default_autoscheduler_name(autoscheduler_name);
    }

    std::string runtime_name = flags_info["-r"];

    std::vector<std::string> generator_names = GeneratorRegistry::enumerate();
    if (generator_names.empty() && runtime_name.empty()) {
        cerr << "No generators have been registered and not compiling a standalone runtime\n";
        cerr << kUsage;
        return 1;
    }

    std::string generator_name = flags_info["-g"];
    if (generator_name.empty() && runtime_name.empty()) {
        // Require either -g or -r to be specified:
        // no longer infer the name when only one Generator is registered
        cerr << "Either -g <name> or -r must be specified; available Generators are:\n";
        if (!generator_names.empty()) {
            for (const auto &name : generator_names) {
                cerr << "    " << name << "\n";
            }
        } else {
            cerr << "    <none>\n";
        }
        return 1;
    }

    std::string function_name = flags_info["-f"];
    if (function_name.empty()) {
        // If -f isn't specified, assume function name = generator name.
        function_name = generator_name;
    }
    std::string output_dir = flags_info["-o"];
    if (output_dir.empty()) {
        cerr << "-o must always be specified.\n";
        cerr << kUsage;
        return 1;
    }

    // It's ok to omit "target=" if we are generating *only* a cpp_stub
    const std::vector<std::string> emit_flags = Halide::Internal::split_string(flags_info["-e"], ",");
    const bool stub_only = (emit_flags.size() == 1 && emit_flags[0] == "cpp_stub");
    if (!stub_only) {
        if (generator_args.find("target") == generator_args.end()) {
            cerr << "Halide::Target missing\n";
            cerr << kUsage;
            return 1;
        }
    }

    // it's OK for file_base_name to be empty: filename will be based on function name
    std::string file_base_name = flags_info["-n"];

    auto target_strings = Halide::Internal::split_string(generator_args["target"].string_value, ",");
    std::vector<Halide::Target> targets;
    for (const auto &s : target_strings) {
        targets.emplace_back(s);
    }

    // extensions won't vary across multitarget output
    std::map<Halide::Output, const Halide::Internal::OutputInfo> output_info = Halide::Internal::get_output_info(targets[0]);

    std::set<Halide::Output> outputs;
    if (emit_flags.empty() || (emit_flags.size() == 1 && emit_flags[0].empty())) {
        // If omitted or empty, assume .a and .h and registration.cpp
        outputs.insert(Halide::Output::c_header);
        outputs.insert(Halide::Output::registration);
        outputs.insert(Halide::Output::static_library);
    } else {
        // Build a reverse lookup table. Allow some legacy aliases on the command line,
        // to allow legacy build systems to work more easily.
        std::map<std::string, Halide::Output> output_name_to_enum = {
            {"cpp", Halide::Output::c_source},
            {"h", Halide::Output::c_header},
            {"html", Halide::Output::stmt_html},
            {"o", Halide::Output::object},
            {"py.c", Halide::Output::python_extension},
        };
        for (const auto &it : output_info) {
            output_name_to_enum[it.second.name] = it.first;
        }

        for (std::string opt : emit_flags) {
            auto it = output_name_to_enum.find(opt);
            if (it == output_name_to_enum.end()) {
                cerr << "Unrecognized emit option: " << opt << " is not one of [";
                auto end = output_info.cend();
                auto last = std::prev(end);
                for (auto iter = output_info.cbegin(); iter != end; ++iter) {
                    cerr << iter->second.name;
                    if (iter != last) {
                        cerr << " ";
                    }
                }
                cerr << "], ignoring.\n";
                cerr << kUsage;
                return 1;
            }
            outputs.insert(it->second);
        }
    }

    // Allow quick-n-dirty use of compiler logging via HL_DEBUG_COMPILER_LOGGER env var
    const bool do_compiler_logging = outputs.count(Halide::Output::compiler_log) ||
                                     (Halide::Internal::get_env_variable("HL_DEBUG_COMPILER_LOGGER") == "1");

    const bool obfuscate_compiler_logging = Halide::Internal::get_env_variable("HL_OBFUSCATE_COMPILER_LOGGER") == "1";

    const Halide::CompilerLoggerFactory no_compiler_logger_factory =
        [](const std::string &, const Halide::Target &) -> std::unique_ptr<Halide::Internal::CompilerLogger> {
        return nullptr;
    };

    const Halide::CompilerLoggerFactory json_compiler_logger_factory =
        [&](const std::string &function_name, const Halide::Target &target) -> std::unique_ptr<Halide::Internal::CompilerLogger> {
        // rebuild generator_args from the map so that they are always canonical
        std::string generator_args_string;
        std::string sep;
        for (const auto &it : generator_args) {
            if (it.first == "target") continue;
            std::string quote = it.second.string_value.find(" ") != std::string::npos ? "\\\"" : "";
            generator_args_string += sep + it.first + "=" + quote + it.second.string_value + quote;
            sep = " ";
        }
        std::unique_ptr<Halide::Internal::JSONCompilerLogger> t(new Halide::Internal::JSONCompilerLogger(
            obfuscate_compiler_logging ? "" : generator_name,
            obfuscate_compiler_logging ? "" : function_name,
            obfuscate_compiler_logging ? "" : autoscheduler_name,
            obfuscate_compiler_logging ? Halide::Target() : target,
            obfuscate_compiler_logging ? "" : generator_args_string,
            obfuscate_compiler_logging));
        return t;
    };

    const Halide::CompilerLoggerFactory compiler_logger_factory = do_compiler_logging ?
                                                              json_compiler_logger_factory :
                                                              no_compiler_logger_factory;

    if (!runtime_name.empty()) {
        std::string base_path = compute_base_path(output_dir, runtime_name, "");

        Halide::Target gcd_target = targets[0];
        for (size_t i = 1; i < targets.size(); i++) {
            if (!gcd_target.get_runtime_compatible_target(targets[i], gcd_target)) {
                user_error << "Failed to find compatible runtime target for "
                           << gcd_target.to_string()
                           << " and "
                           << targets[i].to_string() << "\n";
            }
        }

        if (targets.size() > 1) {
            Halide::Internal::debug(1) << "Building runtime for computed target: " << gcd_target.to_string() << "\n";
        }

        auto output_files = compute_output_files(gcd_target, base_path, outputs);
        // Runtime doesn't get to participate in the Halide::CompilerLogger party
        compile_standalone_runtime(output_files, gcd_target);
    }

    if (!generator_name.empty()) {
        std::string base_path = compute_base_path(output_dir, function_name, file_base_name);
        Halide::Internal::debug(1) << "Generator " << generator_name << " has base_path " << base_path << "\n";
        if (outputs.count(Halide::Output::cpp_stub)) {
            // When generating cpp_stub, we ignore all generator args passed in, and supply a fake Halide::Target.
            // (Halide::CompilerLogger is never enabled for cpp_stub, for now anyway.)
            auto gen = GeneratorRegistry::create(generator_name, GeneratorContext(Halide::Target()));
            auto stub_file_path = base_path + output_info[Halide::Output::cpp_stub].extension;
            gen->emit_cpp_stub(stub_file_path);
        }

        // Don't bother with this if we're just emitting a cpp_stub.
        if (!stub_only) {
            auto output_files = compute_output_files(targets[0], base_path, outputs);
            auto module_factory = [&generator_name, &generator_args, build_gradient_module](const std::string &name, const Halide::Target &target) -> Halide::Module {
                auto sub_generator_args = generator_args;
                sub_generator_args.erase("target");
                // Must re-create each time since each instance will have a different Halide::Target.
                auto gen = GeneratorRegistry::create(generator_name, GeneratorContext(target));
                gen->set_generator_param_values(sub_generator_args);
                return build_gradient_module ? gen->build_gradient_module(name) : gen->build_module(name);
            };
            compile_multitarget(function_name, output_files, targets, target_strings, module_factory, compiler_logger_factory);
        }
    }

    return 0;
}

#ifdef HALIDE_WITH_EXCEPTIONS
int generate_filter_main(int argc, char **argv, std::ostream &cerr) {
    try {
        return generate_filter_main_inner(argc, argv, cerr);
    } catch (std::runtime_error &err) {
        cerr << "Unhandled exception: " << err.what() << "\n";
        return -1;
    }
}
#else
int generate_filter_main(int argc, char **argv, std::ostream &cerr) {
    return generate_filter_main_inner(argc, argv, cerr);
}
#endif

GeneratorParamBase::GeneratorParamBase(const std::string &name)
    : name(name) {
    Halide::Internal::ObjectInstanceRegistry::register_instance(this, 0, Halide::Internal::ObjectInstanceRegistry::GeneratorParam,
                                              this, nullptr);
}

GeneratorParamBase::~GeneratorParamBase() {
    Halide::Internal::ObjectInstanceRegistry::unregister_instance(this);
}

void GeneratorParamBase::check_value_readable() const {
    // These are always readable.
    if (name == "target") return;
    if (name == "auto_schedule") return;
    if (name == "machine_params") return;
    user_assert(generator && generator->phase >= GeneratorBase::ConfigureCalled)
        << "The GeneratorParam \"" << name << "\" cannot be read before build() or configure()/generate() is called.\n";
}

void GeneratorParamBase::check_value_writable() const {
    // Allow writing when no Generator is set, to avoid having to special-case ctor initing code
    if (!generator) return;
    user_assert(generator->phase < GeneratorBase::GenerateCalled) << "The GeneratorParam \"" << name << "\" cannot be written after build() or generate() is called.\n";
}

void GeneratorParamBase::fail_wrong_type(const char *type) {
    user_error << "The GeneratorParam \"" << name << "\" cannot be set with a value of type " << type << ".\n";
}

/* static */
GeneratorRegistry &GeneratorRegistry::get_registry() {
    static GeneratorRegistry *registry = new GeneratorRegistry;
    return *registry;
}

/* static */
void GeneratorRegistry::register_factory(const std::string &name,
                                         GeneratorFactory generator_factory) {
    user_assert(is_valid_name(name)) << "Invalid Generator name: " << name;
    GeneratorRegistry &registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    internal_assert(registry.factories.find(name) == registry.factories.end())
        << "Duplicate Generator name: " << name;
    registry.factories[name] = std::move(generator_factory);
}

/* static */
void GeneratorRegistry::unregister_factory(const std::string &name) {
    GeneratorRegistry &registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    internal_assert(registry.factories.find(name) != registry.factories.end())
        << "Generator not found: " << name;
    registry.factories.erase(name);
}

/* static */
std::unique_ptr<GeneratorBase> GeneratorRegistry::create(const std::string &name,
                                                         const GeneratorContext &context) {
    GeneratorRegistry &registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto it = registry.factories.find(name);
    if (it == registry.factories.end()) {
        std::ostringstream o;
        o << "Generator not found: " << name << "\n";
        o << "Did you mean:\n";
        for (const auto &n : registry.factories) {
            o << "    " << n.first << "\n";
        }
        user_error << o.str();
    }
    std::unique_ptr<GeneratorBase> g = it->second(context);
    internal_assert(g != nullptr);
    return g;
}

/* static */
std::vector<std::string> GeneratorRegistry::enumerate() {
    GeneratorRegistry &registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    std::vector<std::string> result;
    for (const auto &i : registry.factories) {
        result.push_back(i.first);
    }
    return result;
}

GeneratorBase::GeneratorBase(size_t size, const void *introspection_helper)
    : size(size) {
    Halide::Internal::ObjectInstanceRegistry::register_instance(this, size, Halide::Internal::ObjectInstanceRegistry::Generator, this, introspection_helper);
}

GeneratorBase::~GeneratorBase() {
    Halide::Internal::ObjectInstanceRegistry::unregister_instance(this);
}

GeneratorParamInfo::GeneratorParamInfo(GeneratorBase *generator, const size_t size) {
    std::vector<void *> vf = Halide::Internal::ObjectInstanceRegistry::instances_in_range(
        generator, size, Halide::Internal::ObjectInstanceRegistry::FilterParam);
    user_assert(vf.empty()) << "ImageParam and Param<> are no longer allowed in Generators; use Input<> instead.";

    const auto add_synthetic_params = [this, generator](GIOBase *gio) {
        const std::string &n = gio->name();
        const std::string &gn = generator->generator_registered_name;

        if (gio->kind() != IOKind::Scalar) {
            owned_synthetic_params.push_back(GeneratorParam_Synthetic<Halide::Type>::make(generator, gn, n + ".type", *gio, SyntheticParamType::Type, gio->types_defined()));
            filter_generator_params.push_back(owned_synthetic_params.back().get());

            owned_synthetic_params.push_back(GeneratorParam_Synthetic<int>::make(generator, gn, n + ".dim", *gio, SyntheticParamType::Dim, gio->dims_defined()));
            filter_generator_params.push_back(owned_synthetic_params.back().get());
        }
        if (gio->is_array()) {
            owned_synthetic_params.push_back(GeneratorParam_Synthetic<size_t>::make(generator, gn, n + ".size", *gio, SyntheticParamType::ArraySize, gio->array_size_defined()));
            filter_generator_params.push_back(owned_synthetic_params.back().get());
        }
    };

    std::vector<void *> vi = Halide::Internal::ObjectInstanceRegistry::instances_in_range(
        generator, size, Halide::Internal::ObjectInstanceRegistry::GeneratorInput);
    for (auto v : vi) {
        auto input = static_cast<Internal::GeneratorInputBase *>(v);
        internal_assert(input != nullptr);
        user_assert(is_valid_name(input->name())) << "Invalid Input name: (" << input->name() << ")\n";
        user_assert(!names.count(input->name())) << "Duplicate Input name: " << input->name();
        names.insert(input->name());
        internal_assert(input->generator == nullptr || input->generator == generator);
        input->generator = generator;
        filter_inputs.push_back(input);
        add_synthetic_params(input);
    }

    std::vector<void *> vo = Halide::Internal::ObjectInstanceRegistry::instances_in_range(
        generator, size, Halide::Internal::ObjectInstanceRegistry::GeneratorOutput);
    for (auto v : vo) {
        auto output = static_cast<Internal::GeneratorOutputBase *>(v);
        internal_assert(output != nullptr);
        user_assert(is_valid_name(output->name())) << "Invalid Output name: (" << output->name() << ")\n";
        user_assert(!names.count(output->name())) << "Duplicate Output name: " << output->name();
        names.insert(output->name());
        internal_assert(output->generator == nullptr || output->generator == generator);
        output->generator = generator;
        filter_outputs.push_back(output);
        add_synthetic_params(output);
    }

    std::vector<void *> vg = Halide::Internal::ObjectInstanceRegistry::instances_in_range(
        generator, size, Halide::Internal::ObjectInstanceRegistry::GeneratorParam);
    for (auto v : vg) {
        auto param = static_cast<GeneratorParamBase *>(v);
        internal_assert(param != nullptr);
        user_assert(is_valid_name(param->name)) << "Invalid GeneratorParam name: " << param->name;
        user_assert(!names.count(param->name)) << "Duplicate GeneratorParam name: " << param->name;
        names.insert(param->name);
        internal_assert(param->generator == nullptr || param->generator == generator);
        param->generator = generator;
        filter_generator_params.push_back(param);
    }

    for (auto &g : owned_synthetic_params) {
        g->generator = generator;
    }
}

GeneratorParamInfo &GeneratorBase::param_info() {
    internal_assert(param_info_ptr != nullptr);
    return *param_info_ptr;
}

Halide::Func GeneratorBase::get_output(const std::string &n) {
    check_min_phase(GenerateCalled);
    auto *output = find_output_by_name(n);
    // Call for the side-effect of asserting if the value isn't defined.
    (void)output->array_size();
    user_assert(!output->is_array() && output->funcs().size() == 1) << "Output " << n << " must be accessed via get_array_output()\n";
    Halide::Func f = output->funcs().at(0);
    user_assert(f.defined()) << "Output " << n << " was not defined.\n";
    return f;
}

std::vector<Halide::Func> GeneratorBase::get_array_output(const std::string &n) {
    check_min_phase(GenerateCalled);
    auto *output = find_output_by_name(n);
    // Call for the side-effect of asserting if the value isn't defined.
    (void)output->array_size();
    for (const auto &f : output->funcs()) {
        user_assert(f.defined()) << "Output " << n << " was not fully defined.\n";
    }
    return output->funcs();
}

// Find output by name. If not found, assert-fail. Never returns null.
GeneratorOutputBase *GeneratorBase::find_output_by_name(const std::string &name) {
    // There usually are very few outputs, so a linear search is fine
    GeneratorParamInfo &pi = param_info();
    for (GeneratorOutputBase *output : pi.outputs()) {
        if (output->name() == name) {
            return output;
        }
    }
    internal_error << "Output " << name << " not found.";
    return nullptr;  // not reached
}

void GeneratorBase::set_generator_param_values(const GeneratorParamsMap &params) {
    GeneratorParamInfo &pi = param_info();

    std::unordered_map<std::string, Internal::GeneratorParamBase *> generator_params_by_name;
    for (auto *g : pi.generator_params()) {
        generator_params_by_name[g->name] = g;
    }

    for (auto &key_value : params) {
        auto gp = generator_params_by_name.find(key_value.first);
        user_assert(gp != generator_params_by_name.end())
            << "Generator " << generator_registered_name << " has no GeneratorParam named: " << key_value.first << "\n";
        if (gp->second->is_looplevel_param()) {
            if (!key_value.second.string_value.empty()) {
                gp->second->set_from_string(key_value.second.string_value);
            } else {
                gp->second->set(key_value.second.loop_level);
            }
        } else {
            gp->second->set_from_string(key_value.second.string_value);
        }
    }
}

void GeneratorBase::init_from_context(const ion::GeneratorContext &context) {
    ion::GeneratorContext::init_from_context(context);
    internal_assert(param_info_ptr == nullptr);
    // pre-emptively build our param_info now
    param_info_ptr.reset(new GeneratorParamInfo(this, size));
}

void GeneratorBase::set_generator_names(const std::string &registered_name, const std::string &stub_name) {
    user_assert(is_valid_name(registered_name)) << "Invalid Generator name: " << registered_name;
    internal_assert(!registered_name.empty() && !stub_name.empty());
    internal_assert(generator_registered_name.empty() && generator_stub_name.empty());
    generator_registered_name = registered_name;
    generator_stub_name = stub_name;
}

void GeneratorBase::set_inputs_vector(const std::vector<std::vector<StubInput>> &inputs) {
    advance_phase(InputsSet);
    internal_assert(!inputs_set) << "set_inputs_vector() must be called at most once per Generator instance.\n";
    GeneratorParamInfo &pi = param_info();
    user_assert(inputs.size() == pi.inputs().size())
        << "Expected exactly " << pi.inputs().size()
        << " inputs but got " << inputs.size() << "\n";
    for (size_t i = 0; i < pi.inputs().size(); ++i) {
        pi.inputs()[i]->set_inputs(inputs[i]);
    }
    inputs_set = true;
}

void GeneratorBase::track_parameter_values(bool include_outputs) {
    GeneratorParamInfo &pi = param_info();
    for (auto input : pi.inputs()) {
        if (input->kind() == IOKind::Buffer) {
            internal_assert(!input->parameters_.empty());
            for (auto &p : input->parameters_) {
                // This must use p.name(), *not* input->name()
                get_value_tracker()->track_values(p.name(), ion::Internal::parameter_constraints(p));
            }
        }
    }
    if (include_outputs) {
        for (auto output : pi.outputs()) {
            if (output->kind() == IOKind::Buffer) {
                internal_assert(!output->funcs().empty());
                for (auto &f : output->funcs()) {
                    user_assert(f.defined()) << "Output " << output->name() << " is not fully defined.";
                    auto output_buffers = f.output_buffers();
                    for (auto &o : output_buffers) {
                        Halide::Internal::Parameter p = o.parameter();
                        // This must use p.name(), *not* output->name()
                        get_value_tracker()->track_values(p.name(), ion::Internal::parameter_constraints(p));
                    }
                }
            }
        }
    }
}

void GeneratorBase::check_min_phase(Phase expected_phase) const {
    user_assert(phase >= expected_phase) << "You may not do this operation at this phase.";
}

void GeneratorBase::check_exact_phase(Phase expected_phase) const {
    user_assert(phase == expected_phase) << "You may not do this operation at this phase.";
}

void GeneratorBase::advance_phase(Phase new_phase) {
    switch (new_phase) {
    case Created:
        internal_error << "Impossible";
        break;
    case ConfigureCalled:
        internal_assert(phase == Created) << "pase is " << phase;
        break;
    case InputsSet:
        internal_assert(phase == Created || phase == ConfigureCalled);
        break;
    case GenerateCalled:
        // It's OK to advance directly to GenerateCalled.
        internal_assert(phase == Created || phase == ConfigureCalled || phase == InputsSet);
        break;
    case ScheduleCalled:
        internal_assert(phase == GenerateCalled);
        break;
    }
    phase = new_phase;
}

void GeneratorBase::pre_configure() {
    advance_phase(ConfigureCalled);
}

void GeneratorBase::post_configure() {
}

void GeneratorBase::pre_generate() {
    advance_phase(GenerateCalled);
    GeneratorParamInfo &pi = param_info();
    user_assert(!pi.outputs().empty()) << "Must use Output<> with generate() method.";
    user_assert(get_target() != Halide::Target()) << "The Generator target has not been set.";

    if (!inputs_set) {
        for (auto *input : pi.inputs()) {
            input->init_internals();
        }
        inputs_set = true;
    }
    for (auto *output : pi.outputs()) {
        output->init_internals();
    }
    track_parameter_values(false);
}

void GeneratorBase::post_generate() {
    track_parameter_values(true);
}

void GeneratorBase::pre_schedule() {
    advance_phase(ScheduleCalled);
    track_parameter_values(true);
}

void GeneratorBase::post_schedule() {
    track_parameter_values(true);
}

void GeneratorBase::pre_build() {
    advance_phase(GenerateCalled);
    advance_phase(ScheduleCalled);
    GeneratorParamInfo &pi = param_info();
    user_assert(pi.outputs().empty()) << "May not use build() method with Output<>.";
    if (!inputs_set) {
        for (auto *input : pi.inputs()) {
            input->init_internals();
        }
        inputs_set = true;
    }
    track_parameter_values(false);
}

void GeneratorBase::post_build() {
    track_parameter_values(true);
}

Halide::Pipeline GeneratorBase::get_pipeline() {
    check_min_phase(GenerateCalled);
    if (!pipeline.defined()) {
        GeneratorParamInfo &pi = param_info();
        user_assert(!pi.outputs().empty()) << "Must use get_pipeline<> with Output<>.";
        std::vector<Halide::Func> funcs;
        for (auto *output : pi.outputs()) {
            for (const auto &f : output->funcs()) {
                user_assert(f.defined()) << "Output \"" << f.name() << "\" was not defined.\n";
                if (output->dims_defined()) {
                    user_assert(f.dimensions() == output->dims()) << "Output \"" << f.name()
                                                                  << "\" requires dimensions=" << output->dims()
                                                                  << " but was defined as dimensions=" << f.dimensions() << ".\n";
                }
                if (output->types_defined()) {
                    user_assert((int)f.outputs() == (int)output->types().size()) << "Output \"" << f.name()
                                                                                 << "\" requires a Tuple of size " << output->types().size()
                                                                                 << " but was defined as Tuple of size " << f.outputs() << ".\n";
                    for (size_t i = 0; i < f.output_types().size(); ++i) {
                        Halide::Type expected = output->types().at(i);
                        Halide::Type actual = f.output_types()[i];
                        user_assert(expected == actual) << "Output \"" << f.name()
                                                        << "\" requires type " << expected
                                                        << " but was defined as type " << actual << ".\n";
                    }
                }
                funcs.push_back(f);
            }
        }
        pipeline = Halide::Pipeline(funcs);
    }
    return pipeline;
}

Halide::Module GeneratorBase::build_module(const std::string &function_name,
                                   const Halide::LinkageType linkage_type) {
    Halide::AutoSchedulerResults auto_schedule_results;
    call_configure();
    Halide::Pipeline pipeline = build_pipeline();
    if (get_auto_schedule()) {
        auto_schedule_results = pipeline.auto_schedule(get_target(), get_machine_params());
    }

    const GeneratorParamInfo &pi = param_info();
    std::vector<Halide::Argument> filter_arguments;
    for (const auto *input : pi.inputs()) {
        for (const auto &p : input->parameters_) {
            filter_arguments.push_back(to_argument(p, p.is_buffer() ? Halide::Expr() : input->get_def_expr()));
        }
    }

    Halide::Module result = pipeline.compile_to_module(filter_arguments, function_name, get_target(), linkage_type);
    std::shared_ptr<ExternsMap> externs_map = get_externs_map();
    for (const auto &map_entry : *externs_map) {
        result.append(map_entry.second);
    }

    for (const auto *output : pi.outputs()) {
        for (size_t i = 0; i < output->funcs().size(); ++i) {
            auto from = output->funcs()[i].name();
            auto to = output->array_name(i);
            size_t tuple_size = output->types_defined() ? output->types().size() : 1;
            for (size_t t = 0; t < tuple_size; ++t) {
                std::string suffix = (tuple_size > 1) ? ("." + std::to_string(t)) : "";
                result.remap_metadata_name(from + suffix, to + suffix);
            }
        }
    }

    result.set_auto_scheduler_results(auto_schedule_results);

    return result;
}

Halide::Module GeneratorBase::build_gradient_module(const std::string &function_name) {
    constexpr int DBG = 1;

    // I doubt these ever need customizing; if they do, we can make them arguments to this function.
    const std::string grad_input_pattern = "_grad_loss_for_$OUT$";
    const std::string grad_output_pattern = "_grad_loss_$OUT$_wrt_$IN$";
    const Halide::LinkageType linkage_type = Halide::LinkageType::ExternalPlusMetadata;

    user_assert(!function_name.empty()) << "build_gradient_module(): function_name cannot be empty\n";

    call_configure();
    Halide::Pipeline original_pipeline = build_pipeline();
    std::vector<Halide::Func> original_outputs = original_pipeline.outputs();

    // Construct the adjoint pipeline, which has:
    // - All the same inputs as the original, in the same order
    // - Followed by one grad-input for each original output
    // - Followed by one output for each unique pairing of original-output + original-input.

    const GeneratorParamInfo &pi = param_info();

    // Even though propagate_adjoints() supports Funcs-of-Tuples just fine,
    // we aren't going to support them here (yet); AFAICT, neither PyTorch nor
    // TF support Tensors with Tuples-as-values, so we'd have to split the
    // tuples up into separate Halide inputs and outputs anyway; since Generator
    // doesn't support Tuple-valued Inputs at all, and Tuple-valued Outputs
    // are quite rare, we're going to just fail up front, with the assumption
    // that the coder will explicitly adapt their code as needed. (Note that
    // support for Tupled outputs could be added with some effort, so if this
    // is somehow deemed critical, go for it)
    for (const auto *input : pi.inputs()) {
        const size_t tuple_size = input->types_defined() ? input->types().size() : 1;
        // Note: this should never happen
        internal_assert(tuple_size == 1) << "Tuple Inputs are not yet supported by build_gradient_module()";
    }
    for (const auto *output : pi.outputs()) {
        const size_t tuple_size = output->types_defined() ? output->types().size() : 1;
        internal_assert(tuple_size == 1) << "Tuple Outputs are not yet supported by build_gradient_module";
    }

    std::vector<Halide::Argument> gradient_inputs;

    // First: the original inputs. Note that scalar inputs remain scalar,
    // rather being promoted into zero-dimensional buffers.
    for (const auto *input : pi.inputs()) {
        // There can be multiple Funcs/Parameters per input if the input is an Array
        internal_assert(input->parameters_.size() == input->funcs_.size());
        for (const auto &p : input->parameters_) {
            gradient_inputs.push_back(to_argument(p, p.is_buffer() ? Halide::Expr() : input->get_def_expr()));
            Halide::Internal::debug(DBG) << "    gradient copied input is: " << gradient_inputs.back().name << "\n";
        }
    }

    // Next: add a grad-input for each *original* output; these will
    // be the same shape as the output (so we should copy estimates from
    // those outputs onto these estimates).
    // - If an output is an Array, we'll have a separate input for each array element.

    std::vector<ImageParam> d_output_imageparams;
    for (const auto *output : pi.outputs()) {
        for (size_t i = 0; i < output->funcs().size(); ++i) {
            const Halide::Func &f = output->funcs()[i];
            const std::string output_name = output->array_name(i);
            // output_name is something like "funcname_i"
            const std::string grad_in_name = Halide::Internal::replace_all(grad_input_pattern, "$OUT$", output_name);
            // TODO(srj): does it make sense for gradient to be a non-float type?
            // For now, assume it's always float32 (unless the output is already some float).
            const Halide::Type grad_in_type = output->type().is_float() ? output->type() : Float(32);
            const int grad_in_dimensions = f.dimensions();
            const Halide::ArgumentEstimates grad_in_estimates = f.output_buffer().parameter().get_argument_estimates();
            internal_assert((int)grad_in_estimates.buffer_estimates.size() == grad_in_dimensions);

            ImageParam d_im(grad_in_type, grad_in_dimensions, grad_in_name);
            for (int d = 0; d < grad_in_dimensions; d++) {
                d_im.parameter().set_min_constraint_estimate(d, grad_in_estimates.buffer_estimates[i].min);
                d_im.parameter().set_extent_constraint_estimate(d, grad_in_estimates.buffer_estimates[i].extent);
            }
            d_output_imageparams.push_back(d_im);
            gradient_inputs.push_back(to_argument(d_im.parameter(), Halide::Expr()));

            Halide::Internal::debug(DBG) << "    gradient synthesized input is: " << gradient_inputs.back().name << "\n";
        }
    }

    // Finally: define the output Func(s), one for each unique output/input pair.
    // Note that original_outputs.size() != pi.outputs().size() if any outputs are arrays.
    internal_assert(original_outputs.size() == d_output_imageparams.size());
    std::vector<Halide::Func> gradient_outputs;
    for (size_t i = 0; i < original_outputs.size(); ++i) {
        const Halide::Func &original_output = original_outputs.at(i);
        const ImageParam &d_output = d_output_imageparams.at(i);
        Halide::Region bounds;
        for (int i = 0; i < d_output.dimensions(); i++) {
            bounds.emplace_back(d_output.dim(i).min(), d_output.dim(i).extent());
        }
        Halide::Func adjoint_func = Halide::BoundaryConditions::constant_exterior(d_output, Halide::Internal::make_zero(d_output.type()));
        Halide::Derivative d = propagate_adjoints(original_output, adjoint_func, bounds);

        const std::string &output_name = original_output.name();
        for (const auto *input : pi.inputs()) {
            for (size_t i = 0; i < input->funcs_.size(); ++i) {
                const std::string input_name = input->array_name(i);
                const auto &f = input->funcs_[i];
                const auto &p = input->parameters_[i];

                Halide::Func d_f = d(f);

                std::string grad_out_name = Halide::Internal::replace_all(Halide::Internal::replace_all(grad_output_pattern, "$OUT$", output_name), "$IN$", input_name);
                if (!d_f.defined()) {
                    grad_out_name = "_dummy" + grad_out_name;
                }

                Halide::Func d_out_wrt_in(grad_out_name);
                if (d_f.defined()) {
                    d_out_wrt_in(Halide::_) = d_f(Halide::_);
                } else {
                    Halide::Internal::debug(DBG) << "    No Derivative found for output " << output_name << " wrt input " << input_name << "\n";
                    // If there was no Derivative found, don't skip the output;
                    // just replace with a dummy Func that is all zeros. This ensures
                    // that the signature of the Halide::Pipeline we produce is always predictable.
                    std::vector<Halide::Var> vars;
                    for (int i = 0; i < d_output.dimensions(); i++) {
                        vars.push_back(Halide::Var::implicit(i));
                    }
                    d_out_wrt_in(vars) = Halide::Internal::make_zero(d_output.type());
                }

                d_out_wrt_in.set_estimates(p.get_argument_estimates().buffer_estimates);

                // Useful for debugging; ordinarily better to leave out
                // Halide::Internal::debug(0) << "\n\n"
                //          << "output:\n" << FuncWithDependencies(original_output) << "\n"
                //          << "d_output:\n" << FuncWithDependencies(adjoint_func) << "\n"
                //          << "input:\n" << FuncWithDependencies(f) << "\n"
                //          << "d_out_wrt_in:\n" << FuncWithDependencies(d_out_wrt_in) << "\n";

                gradient_outputs.push_back(d_out_wrt_in);
                Halide::Internal::debug(DBG) << "    gradient output is: " << d_out_wrt_in.name() << "\n";
            }
        }
    }

    Halide::Pipeline grad_pipeline = Halide::Pipeline(gradient_outputs);

    Halide::AutoSchedulerResults auto_schedule_results;
    if (get_auto_schedule()) {
        auto_schedule_results = grad_pipeline.auto_schedule(get_target(), get_machine_params());
    } else {
        user_warning << "Autoscheduling is not enabled in build_gradient_module(), so the resulting "
                        "gradient module will be unscheduled; this is very unlikely to be what you want.\n";
    }

    Halide::Module result = grad_pipeline.compile_to_module(gradient_inputs, function_name, get_target(), linkage_type);
    user_assert(get_externs_map()->empty())
        << "Building a gradient-descent module for a Generator with ExternalCode is not supported.\n";

    result.set_auto_scheduler_results(auto_schedule_results);

    return result;
}

void GeneratorBase::emit_cpp_stub(const std::string &stub_file_path) {
    user_assert(!generator_registered_name.empty() && !generator_stub_name.empty()) << "Generator has no name.\n";
    // Make sure we call configure() so that extra inputs/outputs are added as necessary.
    call_configure();
    // StubEmitter will want to access the GP/SP values, so advance the phase to avoid assert-fails.
    advance_phase(GenerateCalled);
    advance_phase(ScheduleCalled);
    GeneratorParamInfo &pi = param_info();
    std::ofstream file(stub_file_path);
    StubEmitter emit(file, generator_registered_name, generator_stub_name, pi.generator_params(), pi.inputs(), pi.outputs());
    emit.emit();
}

void GeneratorBase::check_scheduled(const char *m) const {
    check_min_phase(ScheduleCalled);
}

void GeneratorBase::check_input_is_singular(Internal::GeneratorInputBase *in) {
    user_assert(!in->is_array())
        << "Input " << in->name() << " is an array, and must be set with a vector type.";
}

void GeneratorBase::check_input_is_array(Internal::GeneratorInputBase *in) {
    user_assert(in->is_array())
        << "Input " << in->name() << " is not an array, and must not be set with a vector type.";
}

void GeneratorBase::check_input_kind(Internal::GeneratorInputBase *in, Internal::IOKind kind) {
    user_assert(in->kind() == kind)
        << "Input " << in->name() << " cannot be set with the type specified.";
}

GIOBase::GIOBase(size_t array_size,
                 const std::string &name,
                 IOKind kind,
                 const std::vector<Halide::Type> &types,
                 int dims)
    : array_size_(array_size), name_(name), kind_(kind), types_(types), dims_(dims) {
}

GIOBase::~GIOBase() {
    // nothing
}

bool GIOBase::array_size_defined() const {
    return array_size_ != -1;
}

size_t GIOBase::array_size() const {
    user_assert(array_size_defined()) << "ArraySize is unspecified for " << input_or_output() << "'" << name() << "'; you need to explicitly set it via the resize() method or by setting '"
                                      << name() << ".size' in your build rules.";
    return (size_t)array_size_;
}

bool GIOBase::is_array() const {
    internal_error << "Unimplemented";
    return false;
}

const std::string &GIOBase::name() const {
    return name_;
}

IOKind GIOBase::kind() const {
    return kind_;
}

bool GIOBase::types_defined() const {
    return !types_.empty();
}

const std::vector<Halide::Type> &GIOBase::types() const {
    // If types aren't defined, but we have one Func that is,
    // we probably just set an Output<Halide::Func> and should propagate the types.
    if (!types_defined()) {
        // use funcs_, not funcs(): the latter could give a much-less-helpful error message
        // in this case.
        const auto &f = funcs_;
        if (f.size() == 1 && f.at(0).defined()) {
            check_matching_types(f.at(0).output_types());
        }
    }
    user_assert(types_defined()) << "Halide::Type is not defined for " << input_or_output() << " '" << name() << "'; you may need to specify '" << name() << ".type' as a GeneratorParam.\n";
    return types_;
}

Halide::Type GIOBase::type() const {
    const auto &t = types();
    internal_assert(t.size() == 1) << "Expected types_.size() == 1, saw " << t.size() << " for " << name() << "\n";
    return t.at(0);
}

bool GIOBase::dims_defined() const {
    return dims_ != -1;
}

int GIOBase::dims() const {
    // If types aren't defined, but we have one Func that is,
    // we probably just set an Output<Halide::Func> and should propagate the types.
    if (!dims_defined()) {
        // use funcs_, not funcs(): the latter could give a much-less-helpful error message
        // in this case.
        const auto &f = funcs_;
        if (f.size() == 1 && f.at(0).defined()) {
            check_matching_dims(funcs().at(0).dimensions());
        }
    }
    user_assert(dims_defined()) << "Dimensions are not defined for " << input_or_output() << " '" << name() << "'; you may need to specify '" << name() << ".dim' as a GeneratorParam.\n";
    return dims_;
}

const std::vector<Halide::Func> &GIOBase::funcs() const {
    internal_assert(funcs_.size() == array_size() && exprs_.empty());
    return funcs_;
}

const std::vector<Halide::Expr> &GIOBase::exprs() const {
    internal_assert(exprs_.size() == array_size() && funcs_.empty());
    return exprs_;
}

void GIOBase::verify_internals() {
    user_assert(dims_ >= 0) << "Generator Input/Output Dimensions must have positive values";

    if (kind() != IOKind::Scalar) {
        for (const Halide::Func &f : funcs()) {
            user_assert(f.defined()) << "Input/Output " << name() << " is not defined.\n";
            user_assert(f.dimensions() == dims())
                << "Expected dimensions " << dims()
                << " but got " << f.dimensions()
                << " for " << name() << "\n";
            user_assert(f.outputs() == 1)
                << "Expected outputs() == " << 1
                << " but got " << f.outputs()
                << " for " << name() << "\n";
            user_assert(f.output_types().size() == 1)
                << "Expected output_types().size() == " << 1
                << " but got " << f.outputs()
                << " for " << name() << "\n";
            user_assert(f.output_types()[0] == type())
                << "Expected type " << type()
                << " but got " << f.output_types()[0]
                << " for " << name() << "\n";
        }
    } else {
        for (const Halide::Expr &e : exprs()) {
            user_assert(e.defined()) << "Input/Ouput " << name() << " is not defined.\n";
            user_assert(e.type() == type())
                << "Expected type " << type()
                << " but got " << e.type()
                << " for " << name() << "\n";
        }
    }
}

std::string GIOBase::array_name(size_t i) const {
    std::string n = name();
    if (is_array()) {
        n += "_" + std::to_string(i);
    }
    return n;
}

// If our type(s) are defined, ensure it matches the ones passed in, asserting if not.
// If our type(s) are not defined, just set to the ones passed in.
void GIOBase::check_matching_types(const std::vector<Halide::Type> &t) const {
    if (types_defined()) {
        user_assert(types().size() == t.size()) << "Halide::Type mismatch for " << name() << ": expected " << types().size() << " types but saw " << t.size();
        for (size_t i = 0; i < t.size(); ++i) {
            user_assert(types().at(i) == t.at(i)) << "Halide::Type mismatch for " << name() << ": expected " << types().at(i) << " saw " << t.at(i);
        }
    } else {
        types_ = t;
    }
}

void GIOBase::check_gio_access() const {
    // // Allow reading when no Generator is set, to avoid having to special-case ctor initing code
    if (!generator) return;
    user_assert(generator->phase > GeneratorBase::InputsSet)
        << "The " << input_or_output() << " \"" << name() << "\" cannot be examined before build() or generate() is called.\n";
}

// If our dims are defined, ensure it matches the one passed in, asserting if not.
// If our dims are not defined, just set to the one passed in.
void GIOBase::check_matching_dims(int d) const {
    internal_assert(d >= 0);
    if (dims_defined()) {
        user_assert(dims() == d) << "Dimensions mismatch for " << name() << ": expected " << dims() << " saw " << d;
    } else {
        dims_ = d;
    }
}

void GIOBase::check_matching_array_size(size_t size) const {
    if (array_size_defined()) {
        user_assert(array_size() == size) << "ArraySize mismatch for " << name() << ": expected " << array_size() << " saw " << size;
    } else {
        array_size_ = size;
    }
}

GeneratorInputBase::GeneratorInputBase(size_t array_size,
                                       const std::string &name,
                                       IOKind kind,
                                       const std::vector<Halide::Type> &t,
                                       int d)
    : GIOBase(array_size, name, kind, t, d) {
    Halide::Internal::ObjectInstanceRegistry::register_instance(this, 0, Halide::Internal::ObjectInstanceRegistry::GeneratorInput, this, nullptr);
}

GeneratorInputBase::GeneratorInputBase(const std::string &name, IOKind kind, const std::vector<Halide::Type> &t, int d)
    : GeneratorInputBase(1, name, kind, t, d) {
    // nothing
}

GeneratorInputBase::~GeneratorInputBase() {
    Halide::Internal::ObjectInstanceRegistry::unregister_instance(this);
}

void GeneratorInputBase::check_value_writable() const {
    user_assert(generator && generator->phase == GeneratorBase::InputsSet)
        << "The Input " << name() << " cannot be set at this point.\n";
}

void GeneratorInputBase::set_def_min_max() {
    // nothing
}

Halide::Expr GeneratorInputBase::get_def_expr() const {
    return Halide::Expr();
}

Halide::Internal::Parameter GeneratorInputBase::parameter() const {
    user_assert(!this->is_array()) << "Cannot call the parameter() method on Input<[]> " << name() << "; use an explicit subscript operator instead.";
    return parameters_.at(0);
}

void GeneratorInputBase::verify_internals() {
    GIOBase::verify_internals();

    const size_t expected = (kind() != IOKind::Scalar) ? funcs().size() : exprs().size();
    user_assert(parameters_.size() == expected) << "Expected parameters_.size() == "
                                                << expected << ", saw " << parameters_.size() << " for " << name() << "\n";
}

void GeneratorInputBase::init_internals() {
    // Call these for the side-effect of asserting if the values aren't defined.
    (void)array_size();
    (void)types();
    (void)dims();

    parameters_.clear();
    exprs_.clear();
    funcs_.clear();
    for (size_t i = 0; i < array_size(); ++i) {
        auto name = array_name(i);
        parameters_.emplace_back(type(), kind() != IOKind::Scalar, dims(), name);
        auto &p = parameters_[i];
        if (kind() != IOKind::Scalar) {
            internal_assert(dims() == p.dimensions());
            funcs_.push_back(make_param_func(p, name));
        } else {
            Halide::Expr e = Halide::Internal::Variable::make(type(), name, p);
            exprs_.push_back(e);
        }
    }

    set_def_min_max();
    verify_internals();
}

void GeneratorInputBase::set_inputs(const std::vector<StubInput> &inputs) {
    generator->check_exact_phase(GeneratorBase::InputsSet);
    parameters_.clear();
    exprs_.clear();
    funcs_.clear();
    check_matching_array_size(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const StubInput &in = inputs.at(i);
        user_assert(in.kind() == kind()) << "An input for " << name() << " is not of the expected kind.\n";
        if (kind() == IOKind::Function) {
            auto f = in.func();
            user_assert(f.defined()) << "The input for " << name() << " is an undefined Func. Please define it.\n";
            check_matching_types(f.output_types());
            check_matching_dims(f.dimensions());
            funcs_.push_back(f);
            parameters_.emplace_back(f.output_types().at(0), true, f.dimensions(), array_name(i));
        } else if (kind() == IOKind::Buffer) {
            auto p = in.parameter();
            user_assert(p.defined()) << "The input for " << name() << " is an undefined Buffer. Please define it.\n";
            check_matching_types({p.type()});
            check_matching_dims(p.dimensions());
            funcs_.push_back(make_param_func(p, name()));
            parameters_.push_back(p);
        } else {
            auto e = in.expr();
            user_assert(e.defined()) << "The input for " << name() << " is an undefined Halide::Expr. Please define it.\n";
            check_matching_types({e.type()});
            check_matching_dims(0);
            exprs_.push_back(e);
            parameters_.emplace_back(e.type(), false, 0, array_name(i));
        }
    }

    set_def_min_max();
    verify_internals();
}

void GeneratorInputBase::set_estimate_impl(const Halide::Var &var, const Halide::Expr &min, const Halide::Expr &extent) {
    internal_assert(exprs_.empty() && !funcs_.empty() && parameters_.size() == funcs_.size());
    for (size_t i = 0; i < funcs_.size(); ++i) {
        Halide::Func &f = funcs_[i];
        f.set_estimate(var, min, extent);
        // Propagate the estimate into the Parameter as well, just in case
        // we end up compiling this for toplevel.
        std::vector<Halide::Var> args = f.args();
        int dim = -1;
        for (size_t a = 0; a < args.size(); ++a) {
            if (args[a].same_as(var)) {
                dim = a;
                break;
            }
        }
        internal_assert(dim >= 0);
        Halide::Internal::Parameter &p = parameters_[i];
        p.set_min_constraint_estimate(dim, min);
        p.set_extent_constraint_estimate(dim, extent);
    }
}

void GeneratorInputBase::set_estimates_impl(const Halide::Region &estimates) {
    internal_assert(exprs_.empty() && !funcs_.empty() && parameters_.size() == funcs_.size());
    for (size_t i = 0; i < funcs_.size(); ++i) {
        Halide::Func &f = funcs_[i];
        f.set_estimates(estimates);
        // Propagate the estimate into the Parameter as well, just in case
        // we end up compiling this for toplevel.
        for (size_t dim = 0; dim < estimates.size(); ++dim) {
            Halide::Internal::Parameter &p = parameters_[i];
            const Halide::Range &r = estimates[dim];
            p.set_min_constraint_estimate(dim, r.min);
            p.set_extent_constraint_estimate(dim, r.extent);
        }
    }
}

GeneratorOutputBase::GeneratorOutputBase(size_t array_size, const std::string &name, IOKind kind, const std::vector<Halide::Type> &t, int d)
    : GIOBase(array_size, name, kind, t, d) {
    internal_assert(kind != IOKind::Scalar);
    Halide::Internal::ObjectInstanceRegistry::register_instance(this, 0, Halide::Internal::ObjectInstanceRegistry::GeneratorOutput,
                                              this, nullptr);
}

GeneratorOutputBase::GeneratorOutputBase(const std::string &name, IOKind kind, const std::vector<Halide::Type> &t, int d)
    : GeneratorOutputBase(1, name, kind, t, d) {
    // nothing
}

GeneratorOutputBase::~GeneratorOutputBase() {
    Halide::Internal::ObjectInstanceRegistry::unregister_instance(this);
}

void GeneratorOutputBase::check_value_writable() const {
    user_assert(generator && generator->phase == GeneratorBase::GenerateCalled)
        << "The Output " << name() << " can only be set inside generate().\n";
}

void GeneratorOutputBase::init_internals() {
    exprs_.clear();
    funcs_.clear();
    if (array_size_defined()) {
        for (size_t i = 0; i < array_size(); ++i) {
            funcs_.emplace_back(array_name(i));
        }
    }
}

void GeneratorOutputBase::resize(size_t size) {
    internal_assert(is_array());
    internal_assert(!array_size_defined()) << "You may only call " << name()
                                           << ".resize() when then size is undefined\n";
    array_size_ = (int)size;
    init_internals();
}

void StubOutputBufferBase::check_scheduled(const char *m) const {
    generator->check_scheduled(m);
}

Halide::Target StubOutputBufferBase::get_target() const {
    return generator->get_target();
}

}  // namespace Internal
}  // namespace ion
