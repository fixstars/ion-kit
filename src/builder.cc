#include <algorithm>
#include <fstream>

#include <Halide.h>

#include "ion/builder.h"
#include "ion/util.h"

#include "json/json.hpp"
#include "uuid/sole.hpp"

#include "dynamic_module.h"
#include "log.h"
#include "lower.h"
#include "metadata.h"
#include "serializer.h"

namespace ion {

namespace {

std::map<Halide::OutputFileType, std::string> compute_output_files(const Halide::Target &target,
                                                                   const std::string &base_path,
                                                                   const std::set<Halide::OutputFileType> &outputs) {
    std::map<Halide::OutputFileType, const Halide::Internal::OutputInfo> output_info = Halide::Internal::get_output_info(target);

    std::map<Halide::OutputFileType, std::string> output_files;
    for (auto o : outputs) {
        output_files[o] = base_path + output_info.at(o).extension;
    }
    return output_files;
}

std::string to_string(Halide::Argument::Kind kind) {
    switch (kind) {
    case Halide::Argument::Kind::InputScalar: return "InputScalar";
    case Halide::Argument::Kind::InputBuffer: return "InputBuffer";
    case Halide::Argument::Kind::OutputBuffer: return "OutputBuffer";
    default: return "Unknown";
    }
}

} // anonymous

using json = nlohmann::json;

struct Builder::Impl {
    // Essential
    Halide::Target target;
    std::unordered_map<std::string, std::shared_ptr<DynamicModule>> bb_modules;
    std::map<std::string, Halide::JITExtern> jit_externs;
    std::vector<Graph> graphs;
    std::vector<Node> nodes;
    std::vector<std::tuple<std::string, std::function<void(const char*)>>> disposers;

    // Cacheable
    Halide::Pipeline pipeline;
    Halide::Callable callable;
    std::unique_ptr<Halide::JITUserContext> jit_ctx;
    Halide::JITUserContext* jit_ctx_ptr;
    std::vector<const void*> args;

    Impl() : jit_ctx(new Halide::JITUserContext), jit_ctx_ptr(jit_ctx.get()) {
    }
    ~Impl();
};

Builder::Builder()
    : impl_(new Impl)
{
}

Builder::~Builder()
{
}

Builder::Impl::~Impl()
{
    for (auto [bb_id, disposer] : disposers) {
        disposer(bb_id.c_str());
    }
}

Node Builder::add(const std::string& name)
{
    Node n(sole::uuid4().str(), name, impl_->target);
    impl_->nodes.push_back(n);
    return n;
}

Node Builder::add(const std::string& name, const std::string& graph_id)
{
    Node n(sole::uuid4().str(), name, impl_->target, graph_id);
    impl_->nodes.push_back(n);
    return n;
}

Graph Builder::add_graph(const std::string& name) {
    Graph g(*this, name);
    impl_->graphs.push_back(g);
    return g;
}

Builder& Builder::set_target(const Halide::Target& target) {
    impl_->target = target;
    return *this;
}

Builder& Builder::with_bb_module(const std::string& module_name_or_path) {
    auto bb_module = std::make_shared<DynamicModule>(module_name_or_path);
    auto register_extern = bb_module->get_symbol<void (*)(std::map<std::string, Halide::JITExtern>&)>("register_externs");
    if (register_extern) {
        register_extern(impl_->jit_externs);

    }
    impl_->bb_modules[module_name_or_path] = bb_module;
    return *this;
}

void Builder::save(const std::string& file_name) {
    determine_and_validate(impl_->nodes);
    std::ofstream ofs(file_name);
    json j;
    j["target"] = impl_->target.to_string();
    j["nodes"] = impl_->nodes;
    ofs << j;
    return;
}

void Builder::load(const std::string& file_name) {
    std::ifstream ifs(file_name);
    json j;
    ifs >> j;
    impl_->target = Halide::Target(j["target"].get<std::string>());
    impl_->nodes = j["nodes"].get<std::vector<Node>>();
    return;
}

void Builder::compile(const std::string& function_name, const CompileOption& option) {
    using namespace Halide;

    // Build pipeline and module first
    Pipeline p = lower(*this, impl_->nodes, true);
    if (!p.defined()) {
        log::warn("This pipeline doesn't produce any outputs. Please bind a buffer with output port.");
        return;
    }

    Module m = p.compile_to_module(p.infer_arguments(), function_name, impl_->target);

    // Tailor prefix
    auto output_prefix = option.output_directory.empty() ? "." : option.output_directory + "/";
    output_prefix += "/" + function_name;

    std::set<OutputFileType> outputs;

#ifdef HALIDE_FOR_FPGA
    if (impl_->target.has_fpga_feature()) {
        outputs.insert(OutputFileType::hls_package);
    } else {
#endif
        outputs.insert(OutputFileType::c_header);
        outputs.insert(OutputFileType::static_library);
#ifdef HALIDE_FOR_FPGA
    }
#endif

    const auto output_files = compute_output_files(impl_->target, output_prefix, outputs);
    m.compile(output_files);

#ifdef HALIDE_FOR_FPGA
#ifdef __linux__
    if (impl_->target.has_fpga_feature()) {
        std::string hls_dir = output_files.at(OutputFileType::hls_package);
        chdir(hls_dir.c_str());
        int ret = std::getenv("ION_CSIM") ? system("make -f Makefile.csim.static") : system("make -f Makefile.ultra96v2");
        std::string lib_name = std::getenv("ION_CSIM") ? function_name + "_sim.a" : function_name + ".a";
        internal_assert(ret == 0) << "Building hls package is failed.\n";
        std::string cmd = "cp " + lib_name + " ../" + function_name + ".a && cp " + function_name + ".h ../";
        ret = system(cmd.c_str());
        internal_assert(ret == 0) << "Building hls package is failed.\n";
        chdir("..");
    }
#endif
#endif

    return;
}

void Builder::run(const ion::PortMap&) {
     if (!impl_->pipeline.defined()) {
        impl_->pipeline = lower(*this, impl_->nodes, false);
        if (!impl_->pipeline.defined()) {
            log::warn("This pipeline doesn't produce any outputs. Please bind a buffer with output port.");
            return;
        }
    }

    if (!impl_->callable.defined()) {
        std::map<std::string, Halide::JITExtern> jit_externs;
        for (auto bb : impl_->bb_modules) {
            auto register_extern = bb.second->get_symbol<void (*)(std::map<std::string, Halide::JITExtern>&)>("register_externs");
            if (register_extern) {
                register_extern(jit_externs);

            }
        }
        impl_->pipeline.set_jit_externs(jit_externs);

        auto inferred_args = impl_->pipeline.infer_arguments();
        // auto inferred_args = generate_arguments_stub(impl_->nodes);

        impl_->callable = impl_->pipeline.compile_to_callable(inferred_args, impl_->target);

        impl_->args.clear();
        impl_->args.push_back(&impl_->jit_ctx_ptr);

        const auto& args(generate_arguments_instance(inferred_args, impl_->nodes));
        impl_->args.insert(impl_->args.end(), args.begin(), args.end());
    }

    impl_->callable.call_argv_fast(impl_->args.size(), impl_->args.data());
}

std::vector<std::string> Builder::bb_names(void) {
    std::vector<std::string> names;
    for (auto n : Halide::Internal::GeneratorRegistry::enumerate()) {
         names.push_back(n);
    }
    return names;
}

std::vector<ArgInfo> Builder::bb_arginfos(const std::string& name) {
    auto generator_names = Halide::Internal::GeneratorRegistry::enumerate();

    if (std::find(generator_names.begin(), generator_names.end(), name) == generator_names.end()) {
        throw std::runtime_error(fmt::format("Cannot find generator : {}", name));
    }

    auto bb(Halide::Internal::GeneratorRegistry::create(name, Halide::GeneratorContext(get_host_target())));

    // TODO: Arginfos with parameters
    // for (const auto& p : n.params()) {
    //     try {
    //         bb->set_generatorparam_value(p.key(), p.val());
    //     } catch (const Halide::CompileError& e) {
    //         auto msg = fmt::format("BuildingBlock \"{}\" has no parameter \"{}\"", n.name(), p.key());
    //         log::error(msg);
    //         throw std::runtime_error(msg);
    //     }
    // }

    try {
        bb->build_pipeline();
    } catch (const Halide::CompileError& e) {
        log::error(e.what());
        throw std::runtime_error(e.what());
    }

    return bb->arginfos();
}

std::string Builder::bb_metadata(void) {

    std::vector<Metadata> md;
    for (auto n : Halide::Internal::GeneratorRegistry::enumerate()) {
         md.push_back(Metadata(n));
    }

    json j(md);

    return j.dump();
}

Target Builder::target() const {
    return impl_->target;
}

const std::vector<Node>& Builder::nodes() const {
    return impl_->nodes;
}

std::vector<Node>& Builder::nodes() {
    return impl_->nodes;
}

const std::map<std::string, Halide::JITExtern>& Builder::jit_externs() const {
    return impl_->jit_externs;
}

void Builder::register_disposer(Impl *impl, const std::string& bb_id, const std::string& disposer_symbol) {
    log::info("Builder::register_disposer");
    for (const auto& kv : impl->bb_modules) {
        const auto& dm(kv.second);
        auto disposer_ptr = dm->get_symbol<void (*)(const char*)>(disposer_symbol);
        if (disposer_ptr) {
            impl->disposers.push_back(std::make_tuple(bb_id, disposer_ptr));
        }
    }
}


const Builder::Impl* Builder::impl_ptr() const {
    return impl_.get();
}


} //namespace ion
