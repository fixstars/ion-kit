#include <Halide.h>

#include "ion/node.h"
#include "log.h"
#include "lower.h"

namespace ion {

namespace {

bool is_free(const std::string& pn) {
    return pn.find("_ion_iport_") != std::string::npos;
}

std::tuple<Halide::Internal::AbstractGenerator::ArgInfo, bool> find_ith_input(const std::vector<Halide::Internal::AbstractGenerator::ArgInfo>& arginfos, int i) {
    int j = 0;
    for (const auto& arginfo : arginfos) {
        if (arginfo.dir != Halide::Internal::ArgInfoDirection::Input) {
            continue;
        }

        if (i == j) {
            return std::make_tuple(arginfo, true);
        }

        j++;
    }

    return std::make_tuple(Halide::Internal::AbstractGenerator::ArgInfo(), false);
}

bool is_ready(const std::vector<Node>& sorted, const Node& n) {
    bool ready = true;
    for (const auto& [pn, port] : n.iports()) {
        // This port has predecessor dependency. Always ready to add.
        if (!port.has_pred()) {
            continue;
        }

        const auto& port_(port); // This is workaround for Clang-14 (MacOS)

        // Check port dependent node is already added
        ready &= std::find_if(sorted.begin(), sorted.end(),
                              [&](const Node& n) {
                                return n.id() == port_.pred_id();
                              }) != sorted.end();
    }
    return ready;
}

} // anonymous

void determine_and_validate(std::vector<Node>& nodes) {

    auto generator_names = Halide::Internal::GeneratorRegistry::enumerate();

    for (auto n : nodes) {
        if (std::find(generator_names.begin(), generator_names.end(), n.name()) == generator_names.end()) {
            throw std::runtime_error("Cannot find generator : " + n.name());
        }

        auto bb(Halide::Internal::GeneratorRegistry::create(n.name(), Halide::GeneratorContext(n.target())));

        // Validate and set parameters
        for (const auto& p : n.params()) {
            try {
                bb->set_generatorparam_value(p.key(), p.val());
            } catch (const Halide::CompileError& e) {
                auto msg = fmt::format("BuildingBlock \"{}\" has no parameter \"{}\"", n.name(), p.key());
                log::error(msg);
                throw std::runtime_error(msg);
            }
        }

        try {
            bb->build_pipeline();
        } catch (const Halide::CompileError& e) {
            log::error(e.what());
            throw std::runtime_error(e.what());
        }

        const auto& arginfos(bb->arginfos());

        // validate input port
        auto i = 0;
        for (auto& [pn, port] : n.iports()) {
            if (is_free(pn)) {
                const auto& [arginfo, found] = find_ith_input(arginfos, i);
                if (!found) {
                    auto msg = fmt::format("BuildingBlock \"{}\" has no input #{}", n.name(), i);
                    log::error(msg);
                    throw std::runtime_error(msg);
                }

                port.determine_succ(n.id(), pn, arginfo.name);
                pn = arginfo.name;
            }

            const auto& pn_(pn); // This is workaround for Clang-14 (MacOS)
            if (!std::count_if(arginfos.begin(), arginfos.end(),
                               [&](Halide::Internal::AbstractGenerator::ArgInfo arginfo){ return pn_ == arginfo.name && Halide::Internal::ArgInfoDirection::Input == arginfo.dir; })) {
                auto msg = fmt::format("BuildingBlock \"{}\" has no input \"{}\"", n.name(), pn);
                log::error(msg);
                throw std::runtime_error(msg);
            }

            i++;
        }

        // validate output
        for (const auto& [pn, port] : n.oports()) {
            const auto& pn_(pn); // This is workaround for Clang-14 (MacOS)
            if (!std::count_if(arginfos.begin(), arginfos.end(),
                               [&](Halide::Internal::AbstractGenerator::ArgInfo arginfo){ return pn_ == arginfo.name && Halide::Internal::ArgInfoDirection::Output == arginfo.dir; })) {
                auto msg = fmt::format("BuildingBlock \"{}\" has no output \"{}\"", n.name(), pn);
                log::error(msg);
                throw std::runtime_error(msg);
            }
        }
    }
}

void topological_sort(std::vector<Node>& nodes) {
    std::vector<Node> sorted;
    if (nodes.empty()) {
        return;
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

    nodes.swap(sorted);
}

Halide::Pipeline lower(const Builder* builder_ptr, std::vector<Node>& nodes, bool implicit_output) {
#if 0
    log::info("Start building pipeline");

    determine_and_validate(nodes);

    // Sort nodes prior to build.
    // This operation is required especially for the graph which is loaded from JSON definition.
    topological_sort(nodes);

    // Constructing Generator object and setting static parameters
    std::unordered_map<std::string, Halide::Internal::AbstractGeneratorPtr> bbs;
    for (auto n : nodes) {
        auto bb(Halide::Internal::GeneratorRegistry::create(n.name(), Halide::GeneratorContext(n.target())));

        // Default parameter
        Halide::GeneratorParamsMap params;
        params["builder_ptr"] = std::to_string(reinterpret_cast<uint64_t>(builder_ptr));
        params["bb_id"] = n.id();

        // User defined parameter
        for (const auto& p : n.params()) {
            params[p.key()] =  p.val();
        }
        bb->set_generatorparam_values(params);
        bbs[n.id()] = std::move(bb);
    }

    // Assigning ports and build pipeline
    for (size_t i=0; i<nodes.size(); ++i) {
        auto n = nodes[i];
        const auto& bb = bbs[n.id()];
        auto arginfos = bb->arginfos();
        const auto& iports(n.iports());
        for (size_t j=0; j<iports.size(); ++j) {
            const auto& [pn, port] = iports[j];
            auto index = port.index();
            const auto& arginfo = arginfos[j];
            if (port.has_pred()) {

                const auto& pred_bb(bbs[port.pred_id()]);

                auto fs = pred_bb->output_func(port.pred_name());
                if (arginfo.kind == Halide::Internal::ArgInfoKind::Scalar) {
                    bb->bind_input(arginfo.name, fs);
                } else if (arginfo.kind == Halide::Internal::ArgInfoKind::Function) {
                    // no specific index provided, direct output Port
                    if (index == -1) {
                        bb->bind_input(arginfo.name, fs);
                    } else {
                        // access to Port[index]
                        if (index>=static_cast<decltype(index)>(fs.size())){
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
            }
        }
        bb->build_pipeline();
    }

    std::vector<Halide::Func> output_funcs;

    if (implicit_output) {
        // Collects all output which is never referenced.
        // This mode is used for AOT compilation
        std::unordered_map<std::string, std::vector<std::string>> referenced;
        for (const auto& n : nodes) {
            for (const auto& [pn, port] : n.iports()) {
                if (port.has_pred()) {
                    for (const auto &f : bbs[port.pred_id()]->output_func(port.pred_name())) {
                        referenced[port.pred_id()].emplace_back(f.name());
                    }
                }
            }
        }

        for (const auto& node : nodes) {
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
        for (const auto& node : nodes) {
            for (const auto& [pn, port] : node.oports()) {
                const auto& port_instances(port.as_instance());
                if (port_instances.empty()) {
                    continue;
                }

                const auto& pred_bb(bbs[port.pred_id()]);

                // Validate port exists
                const auto& port_(port); // This is workaround for Clang-14 (MacOS)
                const auto& pred_arginfos(pred_bb->arginfos());
                if (!std::count_if(pred_arginfos.begin(), pred_arginfos.end(),
                                   [&](Halide::Internal::AbstractGenerator::ArgInfo arginfo){ return port_.pred_name() == arginfo.name && Halide::Internal::ArgInfoDirection::Output == arginfo.dir; })) {
                    auto msg = fmt::format("BuildingBlock \"{}\" has no output \"{}\"", pred_bb->name(), port.pred_name());
                    log::error(msg);
                    throw std::runtime_error(msg);
                }


                auto fs(bbs[port.pred_id()]->output_func(port.pred_name()));
                output_funcs.insert(output_funcs.end(), fs.begin(), fs.end());
            }
        }
    }

    if (output_funcs.empty()) {
        return Halide::Pipeline();
    }

    return Halide::Pipeline(output_funcs);
#else
    return Halide::Pipeline();
#endif
}

} // namespace ion
