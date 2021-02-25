#include <fstream>

#ifdef __linux__
#include <unistd.h>
#endif

#include "ion/builder.h"
#include "ion/generator.h"
#include "ion/json.hpp"
#include "ion/util.h"

#include "dynamic_module.h"
#include "metadata.h"
#include "sole.hpp"

namespace ion {

namespace {

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

bool is_ready(const std::vector<Node>& sorted, const Node& n) {
    bool ready = true;
    for (auto p : n.ports()) {
        // This port has external dependency. Always ready to add.
        if (p.node_id().empty()) {
            continue;
        }

        // Check port dependent node is already added
        ready &= std::find_if(sorted.begin(), sorted.end(),
                              [&p](const Node& n) {
                                  return n.id() == p.node_id();
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
{
}

Node Builder::add(const std::string& k)
{
    Node n(sole::uuid4().str(), k, target_);
    nodes_.push_back(n);
    return n;
}

Builder Builder::set_target(const Halide::Target& target) {
    target_ = target;
    return *this;
}

Builder Builder::with_bb_module(const std::string& module_path) {
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
    Pipeline p = build();
    Module m = p.compile_to_module(p.infer_arguments(), function_name, target_);

    // Tailor prefix
    auto output_prefix = option.output_directory.empty() ? "." : option.output_directory + "/";
    output_prefix += "/" + function_name;

    std::set<Output> outputs;

#ifdef HALIDE_FOR_FPGA
    if (target_.has_fpga_feature()) {
        outputs.insert(Output::hls_package);
    } else {
#endif
        outputs.insert(Output::c_header);
        outputs.insert(Output::static_library);
#ifdef HALIDE_FOR_FPGA
    }
#endif

    const auto output_files = compute_output_files(target_, output_prefix, outputs);
    m.compile(output_files);

#ifdef HALIDE_FOR_FPGA
#ifdef __linux__
    if (target_.has_fpga_feature()) {
        std::string hls_dir = output_files.at(Output::hls_package);
        chdir(hls_dir.c_str());
        int ret = std::getenv("ION_CSIM") ? system("make -f Makefile.csim.static") : system("make -f Makefile.ultra96v2");
        std::string lib_name = std::getenv("ION_CSIM") ? function_name + "_sim.a" : function_name + ".a";
        internal_assert(ret == 0) << "Building hls package is failed.\n";
        std::string cmd = "mv " + lib_name + " ../" + function_name + ".a && mv " + function_name + ".h ../";
        ret = system(cmd.c_str());
        internal_assert(ret == 0) << "Building hls package is failed.\n";
        chdir("..");
    }
#endif
#endif

    return;
}

// NOTE: This function is deprecated
// Halide::Realization Builder::run(const std::vector<int32_t>& sizes, const ion::PortMap& pm) {
//     return build(pm).realize(sizes, target_, pm.get_param_map());
// }

void Builder::run(const ion::PortMap& pm) {
    auto p = build(pm, &outputs_);
    return p.realize(Halide::Realization(outputs_), target_, pm.get_param_map());
}

bool is_dereferenced(const std::vector<Node>& nodes, const std::string node_id, const std::string& func_name) {
    for (const auto& node : nodes) {
        if (node.id() != node_id) {
            continue;
        }

        for (const auto& port : node.ports()) {
            if (port.key() == func_name) {
                return true;
            }
        }

        return false;
    }

    throw std::runtime_error("Unreachable");
}

std::vector<std::tuple<std::string, Halide::Func>> collect_unbound_outputs(const std::vector<Node>& nodes,
                                                                           const std::unordered_map<std::string, std::shared_ptr<Internal::BuildingBlockBase>>& bbs) {
    std::vector<std::tuple<std::string, Halide::Func>> unbounds;
    for (const auto& kv : bbs) {
        auto node_id = kv.first;
        auto bb = kv.second;
        for (const auto& output : bb->param_info().outputs()) {
            if (output->is_array()) {
                throw std::runtime_error("Unreachable");
                // for (const auto &f : bb->get_array_output(p.key())) {
                //     const auto key = f.name();
                //     dereferenced[p.node_id()].emplace_back(key.substr(0, key.find('$')));
                // }
            } else {
                if (!is_dereferenced(nodes, node_id, output->name())) {
                    unbounds.push_back(std::make_tuple(output_name(node_id, output->name()), bb->get_output(output->name())));
                }
            }
        }
    }

    return unbounds;
}

Halide::Pipeline Builder::build(const ion::PortMap& pm, std::vector<Halide::Buffer<>> *outputs) {

    if (pipeline_.defined()) {
        return pipeline_;
    }

    // Sort nodes prior to build.
    // This operation is required especially for the graph which is loaded from JSON definition.
    nodes_ = topological_sort(nodes_);

    auto generator_names = Internal::GeneratorRegistry::enumerate();

    // Constructing Generator object and setting static parameters
    std::unordered_map<std::string, std::shared_ptr<Internal::BuildingBlockBase>> bbs;
    for (auto n : nodes_) {

        if (std::find(generator_names.begin(), generator_names.end(), n.name()) == generator_names.end()) {
            throw std::runtime_error("Cannot find generator : " + n.name());
        }

        std::shared_ptr<Internal::BuildingBlockBase> bb(Internal::GeneratorRegistry::create(n.name(), GeneratorContext(n.target())));
        Internal::GeneratorParamsMap gpm;
        for (const auto& p : n.params()) {
            gpm[p.key()] = p.val();
        }
        bb->set_generator_param_values(gpm);
        bbs[n.id()] = bb;
    }

    // Assigning ports
    for (size_t i=0; i<nodes_.size(); ++i) {
        auto n = nodes_[i];
        auto bb = bbs[n.id()];
        std::vector<std::vector<Internal::StubInput>> args;
        for (size_t j=0; j<n.ports().size(); ++j) {
            auto p = n.ports()[j];
            // Unbounded parameter
            auto *in = bb->param_info().inputs().at(j);
            if (p.node_id().empty()) {
                if (in->is_array()) {
                    throw std::runtime_error(
                        "Unbounded port (" + in->name() + ") corresponding to an array of Inputs is not supported");
                }
                const auto k = in->kind();
                if (k == Internal::IOKind::Scalar) {
                    Halide::Expr e;
                    if (pm.mapped(p.key())) {
                        // This block should be executed when g.run is called with appropriate PortMap.
                        e = pm.get_param_expr(p.key());
                    } else {
                        e = Halide::Param<void>(p.type(), p.key());
                    }
                    args.push_back(bb->build_input(j, e));
                } else if (k == Internal::IOKind::Function) {
                    Halide::Func f;
                    if (pm.mapped(p.key())) {
                        // This block should be executed when g.run is called with appropriate PortMap.
                        f = pm.get_param_func(p.key());
                    } else {
                        f = Halide::ImageParam(in->type(), in->dims(), p.key());
                    }
                    args.push_back(bb->build_input(j, f));
                } else {
                    throw std::runtime_error("fixme");
                }
            } else {
                if (in->is_array()) {
                    auto f_array = bbs[p.node_id()]->get_array_output(p.key());
                    args.push_back(bb->build_input(j, f_array));
                } else {
                    Halide::Func f = bbs[p.node_id()]->get_output(p.key());
                    args.push_back(bb->build_input(j, f));
                }
            }
        }
        bb->apply(args);
    }

    // Traverse bbs and bundling all outputs
    std::unordered_map<std::string, std::vector<std::string>> dereferenced;
    for (size_t i=0; i<nodes_.size(); ++i) {
        auto n = nodes_[i];
        for (size_t j=0; j<n.ports().size(); ++j) {
            auto p = n.ports()[j];

            if (!p.node_id().empty() && bbs[n.id()]->param_info().inputs().at(j)->is_array()) {
                for (const auto &f : bbs[p.node_id()]->get_array_output(p.key())) {
                    const auto key = f.name();
                    dereferenced[p.node_id()].emplace_back(key.substr(0, key.find('$')));
                }
            } else {
                dereferenced[p.node_id()].push_back(p.key());
            }
        }
    }

    std::vector<Halide::Func> output_funcs;

    if (outputs) {
        // This is explicit mode. Make output list based on specified port map.
        for (auto kv : pm.get_output_buffer()) {
           auto node_id = std::get<0>(kv.first);
           auto port_key = std::get<1>(kv.first);

           bool found = false;
           for (auto info : bbs[node_id]->param_info().outputs()) {
               if (info->name() == port_key) {
                   if (info->is_array()) {
                       auto fs = bbs[node_id]->get_array_output(port_key);
                       if (fs.size() != kv.second.size()) {
                           throw std::runtime_error("Invalid size of array : " + node_id + ", " + port_key);
                       }
                       for (size_t i=0; i<fs.size(); ++i) {
                           output_funcs.push_back(fs[i]);
                           outputs->push_back(kv.second[i]);
                       }
                   } else {
                       auto f = bbs[node_id]->get_output(port_key);
                       if (1 != kv.second.size()) {
                           throw std::runtime_error("Invalid size of array : " + node_id + ", " + port_key);
                       }
                       output_funcs.push_back(f);
                       outputs->push_back(kv.second.front());
                   }
                   found = true;
               }
           }
           if (!found) {
               throw std::runtime_error("Invalid output port: " + node_id + ", " + port_key);
           }
        }
    } else {
        // This is implicit mode. Make output list based on unbound output in the graph.
        for (int i=0; i<nodes_.size(); ++i) {
            auto node_id = nodes_[i].id();
            auto p = bbs[node_id]->get_pipeline();
            for (auto f : p.outputs()) {

                // It is not dereferenced, then treat as outputs
                const auto& dv = dereferenced[node_id];
                std::string key = f.name();
                key = key.substr(0, key.find('$'));
                auto it = std::find(dv.begin(), dv.end(), key);
                if (it == dv.end()) {
                    output_funcs.push_back(f);
                }
            }
        }
    }

    pipeline_ = Halide::Pipeline(output_funcs);
    return pipeline_;
}

std::string Builder::bb_metadata(void) {

    std::vector<Metadata> md;
    for (auto n : Internal::GeneratorRegistry::enumerate()) {
        md.push_back(Metadata(n));
    }

    json j(md);

    return j.dump();
}


const std::vector<Node>& Builder::get_nodes() const {
    return nodes_;
}

} //namespace ion
