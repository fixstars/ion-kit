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

Halide::Internal::AbstractGenerator::ArgInfo make_arginfo(const std::string& name,
                                                          Halide::Internal::ArgInfoDirection dir,
                                                          Halide::Internal::ArgInfoKind kind,
                                                          const std::vector<Halide::Type>& types,
                                                          int dimensions) {
    return Halide::Internal::AbstractGenerator::ArgInfo {
        name, dir, kind, types, dimensions
    };
}

} // anonymous

using json = nlohmann::json;

Builder::Builder()
    : jit_ctx_(new Halide::JITUserContext), jit_ctx_ptr_(jit_ctx_.get())
{
}

Builder::~Builder()
{
    for (auto [bb_id, disposer] : disposers_) {
        disposer(bb_id.c_str());
    }
}

Node Builder::add(const std::string& k)
{
    // TODO: Validate bb is existing
    Node n(sole::uuid4().str(), k, target_);
    nodes_.push_back(n);
    return n;
}

Builder& Builder::set_target(const Halide::Target& target) {
    target_ = target;
    return *this;
}

Builder& Builder::with_bb_module(const std::string& module_name_or_path) {
    bb_modules_[module_name_or_path] = std::make_shared<DynamicModule>(module_name_or_path);
    return *this;
}


void Builder::save(const std::string& file_name) {
    determine_and_validate(nodes_);
    std::ofstream ofs(file_name);
    json j;
    j["target"] = target_.to_string();
    j["nodes"] = nodes_;
    ofs << j;
    return;
}

void Builder::load(const std::string& file_name) {
    std::ifstream ifs(file_name);
    json j;
    ifs >> j;
    target_ = Halide::Target(j["target"].get<std::string>());
    nodes_ = j["nodes"].get<std::vector<Node>>();
    return;
}

void Builder::compile(const std::string& function_name, const CompileOption& option) {
    using namespace Halide;

    // Build pipeline and module first
    Pipeline p = lower(this, nodes_, true);
    if (!p.defined()) {
        log::warn("This pipeline doesn't produce any outputs. Please bind a buffer with output port.");
        return;
    }

    Module m = p.compile_to_module(p.infer_arguments(), function_name, target_);

    // Tailor prefix
    auto output_prefix = option.output_directory.empty() ? "." : option.output_directory + "/";
    output_prefix += "/" + function_name;

    std::set<OutputFileType> outputs;

#ifdef HALIDE_FOR_FPGA
    if (target_.has_fpga_feature()) {
        outputs.insert(OutputFileType::hls_package);
    } else {
#endif
        outputs.insert(OutputFileType::c_header);
        outputs.insert(OutputFileType::static_library);
#ifdef HALIDE_FOR_FPGA
    }
#endif

    const auto output_files = compute_output_files(target_, output_prefix, outputs);
    m.compile(output_files);

#ifdef HALIDE_FOR_FPGA
#ifdef __linux__
    if (target_.has_fpga_feature()) {
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
     if (!pipeline_.defined()) {
        pipeline_ = lower(this, nodes_, false);
        if (!pipeline_.defined()) {
            log::warn("This pipeline doesn't produce any outputs. Please bind a buffer with output port.");
            return;
        }
    }

    if (!callable_.defined()) {
        std::map<std::string, Halide::JITExtern> jit_externs;
        for (auto bb : bb_modules_) {
            auto register_extern = bb.second->get_symbol<void (*)(std::map<std::string, Halide::JITExtern>&)>("register_externs");
            if (register_extern) {
                register_extern(jit_externs);

            }
        }
        pipeline_.set_jit_externs(jit_externs);

        // TODO: Validate argument list
        // pipeline_.infer_arguments()) {

        callable_ = pipeline_.compile_to_callable(get_arguments_stub(), target_);

        args_.clear();
        args_.push_back(&jit_ctx_ptr_);

        const auto& args(get_arguments_instance());
        args_.insert(args_.end(), args.begin(), args.end());
    }

    callable_.call_argv_fast(args_.size(), args_.data());
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


void Builder::register_disposer(const std::string& bb_id, const std::string& disposer_symbol) {
    log::info("Builder::register_disposer");
    for (const auto& kv : bb_modules_) {
        const auto& dm(kv.second);
        auto disposer_ptr = dm->get_symbol<void (*)(const char*)>(disposer_symbol);
        if (disposer_ptr) {
            disposers_.push_back(std::make_tuple(bb_id, disposer_ptr));
        }
    }
}

std::vector<Halide::Argument> Builder::get_arguments_stub() const {
    std::set<Port::Channel> added_ports;
    std::vector<Halide::Argument> args;
    for (const auto& node : nodes_) {
        for (const auto& [pn, port] : node.iports()) {
            if (port.has_pred()) {
                continue;
            }

            if (added_ports.count(port.impl_->pred_chan)) {
                continue;
            }
            added_ports.insert(port.impl_->pred_chan);

            const auto& port_args(port.as_argument());
            args.insert(args.end(), port_args.begin(), port_args.end());
        }
    }
    return args;
}

std::vector<const void*> Builder::get_arguments_instance() const {
    std::set<Port::Channel> added_args;
    std::vector<const void*> instances;

    // Input
    for (const auto& node : nodes_) {
        for (const auto& [pn, port] : node.iports()) {
            if (port.has_pred()) {
                continue;
            }

            if (added_args.count(port.impl_->pred_chan)) {
                continue;
            }
            added_args.insert(port.impl_->pred_chan);

            const auto& port_instances(port.as_instance());
            instances.insert(instances.end(), port_instances.begin(), port_instances.end());
        }
    }

    // Output
    for (const auto& node : nodes_) {
        for (const auto& [pn, port] : node.oports()) {
            const auto& port_instances(port.as_instance());
            instances.insert(instances.end(), port_instances.begin(), port_instances.end());
        }
    }

    return instances;
}

} //namespace ion
