#include <algorithm>
#include <fstream>

#ifdef __linux__
#include <unistd.h>
#endif

#include <Halide.h>

#include "ion/builder.h"
#include "ion/util.h"

#include "json/json.hpp"
#include "uuid/sole.hpp"

#include "dynamic_module.h"
#include "log.h"
#include "metadata.h"
#include "serializer.h"

#define SW 1

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

bool is_ready(const std::vector<Node>& sorted, const Node& n) {
    bool ready = true;
    for (auto port : n.iports()) {
        // This port has predecessor dependency. Always ready to add.
        if (!port.has_pred()) {
            continue;
        }

        // Check port dependent node is already added
        ready &= std::find_if(sorted.begin(), sorted.end(),
                              [&port](const Node& n) {
                                return n.id() == port.pred_id();
                              }) != sorted.end();
    }
    return ready;
}

std::vector<Node> topological_sort(std::vector<Node> nodes) {
    std::vector<Node> sorted;
    if (nodes.empty()) {
        return sorted;
    }

    auto it = nodes.begin();
    while (!nodes.empty()) {
        if (is_ready(sorted, *it)) {
            sorted.push_back(*it);
            nodes.erase(it);
            it = nodes.begin();
        } else {
            it++;
            if (it == nodes.end()) {
                it = nodes.begin();
            }
        }
    }

    return sorted;
}

} // anonymous

using json = nlohmann::json;

Builder::Builder()
    : jit_ctx_(new Halide::JITUserContext), jit_ctx_ptr_(jit_ctx_.get())
{
    args_.push_back(&jit_ctx_ptr_);
}

Builder::~Builder()
{
    for (auto kv : disposers_) {
        auto bb_id(std::get<0>(kv));
        auto disposer(std::get<1>(kv));
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

Builder& Builder::with_bb_module(const std::string& module_path) {
    bb_modules_[module_path] = std::make_shared<DynamicModule>(module_path);
    return *this;
}


void Builder::save(const std::string& file_name) {
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
    Pipeline p = build(true);
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

void Builder::run(void) {
     PortMap pm;
     run(pm);
}

void Builder::run(ion::PortMap& pm) {
     if (!pipeline_.defined()) {
        pipeline_ = build();
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
    }

    callable_.call_argv_fast(args_.size(), args_.data());
}

Halide::Pipeline Builder::build(bool implicit_output) {

    log::info("Start building pipeline");

    // Sort nodes prior to build.
    // This operation is required especially for the graph which is loaded from JSON definition.
    nodes_ = topological_sort(nodes_);

    auto generator_names = Halide::Internal::GeneratorRegistry::enumerate();

    // Constructing Generator object and setting static parameters
    std::unordered_map<std::string, Halide::Internal::AbstractGeneratorPtr> bbs;
    for (auto n : nodes_) {

        if (std::find(generator_names.begin(), generator_names.end(), n.name()) == generator_names.end()) {
            throw std::runtime_error("Cannot find generator : " + n.name());
        }

        auto bb(Halide::Internal::GeneratorRegistry::create(n.name(), Halide::GeneratorContext(n.target())));
        Halide::GeneratorParamsMap params;
        params["builder_ptr"] = std::to_string(reinterpret_cast<uint64_t>(this));
        params["bb_id"] = n.id();
        for (const auto& p : n.params()) {
            params[p.key()] = p.val();
        }
        bb->set_generatorparam_values(params);
        bbs[n.id()] = std::move(bb);
    }

    // Assigning ports
    std::set<Port::Channel> added_args;
    for (size_t i=0; i<nodes_.size(); ++i) {
        auto n = nodes_[i];
        const auto& bb = bbs[n.id()];
        auto arginfos = bb->arginfos();
        for (size_t j=0; j<n.iports().size(); ++j) {
            auto port = n.iports()[j];
            auto index = port.index();
            // Unbounded parameter
            const auto& arginfo = arginfos[j];
            if (port.has_pred()) {
                auto fs = bbs[port.pred_id()]->output_func(port.pred_name());
                if (arginfo.kind == Halide::Internal::ArgInfoKind::Scalar) {
                    bb->bind_input(arginfo.name, fs);
                } else if (arginfo.kind == Halide::Internal::ArgInfoKind::Function) {
                    auto fs = bbs[port.pred_id()]->output_func(port.pred_name());
                    // no specific index provided, direct output Port
                    if (index == -1) {
                        bb->bind_input(arginfo.name, fs);
                    } else {
                        // access to Port[index]
                        if (index>=fs.size()){
                            throw std::runtime_error("Port index out of range: " + port.pred_name());
                        }
                        bb->bind_input(arginfo.name, {fs[index]});
                    }
                } else {
                    throw std::runtime_error("fixme");
                }
            } else {
                if (arginfo.kind == Halide::Internal::ArgInfoKind::Scalar) {
                    bb->bind_input(arginfo.name, port.as_expr());
                } else if (arginfo.kind == Halide::Internal::ArgInfoKind::Function) {
                    bb->bind_input(arginfo.name, port.as_func());
                } else {
                    throw std::runtime_error("fixme");
                }

                // Adding input args
                if (added_args.count(port.impl_->pred_chan)) {
                    continue;
                }
                added_args.insert(port.impl_->pred_chan);

                const auto& port_instances(port.as_instance());
                args_.insert(args_.end(), port_instances.begin(), port_instances.end());
            }
        }
        bb->build_pipeline();
    }

    std::vector<Halide::Func> output_funcs;

    if (implicit_output) {
        // Collects all output which is never referenced.
        // This mode is used for AOT compilation
        std::unordered_map<std::string, std::vector<std::string>> referenced;
        for (const auto& n : nodes_) {
            for (const auto& port : n.iports()) {
                if (port.has_pred()) {
                    for (const auto &f : bbs[port.pred_id()]->output_func(port.pred_name())) {
                        referenced[port.pred_id()].emplace_back(f.name());
                    }
                }
            }
        }

        for (const auto& node : nodes_) {
            auto node_id = node.id();
            for (auto arginfo : bbs[node_id]->arginfos()) {
                if (arginfo.dir != Halide::Internal::ArgInfoDirection::Output) {
                    continue;
                }

                // This is not output
                // It is not dereferenced, then treat as outputs
                const auto& dv = referenced[node_id];

                for (auto f : bbs[node_id]->output_func(arginfo.name)) {
                    auto it = std::find(dv.begin(), dv.end(), f.name());
                    if (it == dv.end()) {
                        auto fs = bbs[node_id]->output_func(arginfo.name);
                        output_funcs.insert(output_funcs.end(), fs.begin(), fs.end());
                    }
                }
            }
        }
    } else {
        // Collects all output which is bound with buffer.
        // This mode is used for JIT
        for (const auto& node : nodes_) {
            for (const auto& port : node.oports()) {
                const auto& port_instances(port.as_instance());
                if (port_instances.empty()) {
                    continue;
                }

                auto fs(bbs[port.pred_id()]->output_func(port.pred_name()));
                output_funcs.insert(output_funcs.end(), fs.begin(), fs.end());
                args_.insert(args_.end(), port_instances.begin(), port_instances.end());
            }
        }
    }

    if (output_funcs.empty()) {
        return Halide::Pipeline();
    }

    return Halide::Pipeline(output_funcs);
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

} //namespace ion
