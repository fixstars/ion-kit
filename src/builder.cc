#include <fstream>

#include "ion/builder.h"
#include "ion/json.hpp"
#include "ion/util.h"

#include "dynamic_module.h"
#include "metadata.h"
#include "sole.hpp"

namespace ion {

namespace {

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
    auto output_prefix = option.output_directory.empty() ? "." : option.output_directory;
    output_prefix += "/" + function_name;

    // Generate header
    Outputs output_files = Outputs().c_header(output_prefix + ".h");

    // Generate library
    if (target_.os == Target::Windows && !target_.has_feature(Target::MinGW)) {
        output_files = output_files.static_library(output_prefix + ".lib");
    } else {
        output_files = output_files.static_library(output_prefix + ".a");
    }

    m.compile(output_files);
    return;
}

Halide::Realization Builder::run(const std::vector<int32_t>& sizes, const ion::PortMap& pm) {
    return build(pm).realize(sizes, target_, pm.get_param_map());
}

void Builder::run(const ion::PortMap& pm) {
    auto p = build(pm, &outputs_);
    return p.realize(Halide::Realization(outputs_), target_, pm.get_param_map());
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

    std::vector<std::string> output_names;
    std::vector<Halide::Func> output_funcs;
    for (int i=0; i<nodes_.size(); ++i) {
        auto node_id = nodes_[i].id();
        auto p = bbs[node_id]->get_pipeline();
        for (auto f : p.outputs()) {
            const auto& dv = dereferenced[node_id];
            std::string key = f.name();
            key = key.substr(0, key.find('$'));
            auto it = std::find(dv.begin(), dv.end(), key);
            if (it == dv.end()) {
                output_names.push_back(output_name(node_id, key));
                output_funcs.push_back(f);
            }
        }
    }

    if (outputs) {
        // Push buffer based on funcs
        for (auto n : output_names) {
            outputs->push_back(pm.get_output_buffer(n));
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

} //namespace ion
