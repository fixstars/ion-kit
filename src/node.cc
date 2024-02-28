#include "ion/node.h"

#include "log.h"

namespace ion {

Node::Impl::Impl(const std::string& id_, const std::string& name_, const Halide::Target& target_)
    : id(id_), name(name_), target(target_), params(), ports()
{
    auto bb(Halide::Internal::GeneratorRegistry::create(name_, Halide::GeneratorContext(target_)));
    if (!bb) {
        log::error("BuildingBlock {} is not found", name_);
        throw std::runtime_error("Failed to create building block object");
    }

    arginfos = bb->arginfos();
}

Node::Impl::Impl(const std::string& id_, const std::string& name_, const Halide::Target& target_, const std::string& graph_id)
    : id(id_), name(name_), target(target_), params(), ports(), graph_id(graph_id)
{
    auto bb(Halide::Internal::GeneratorRegistry::create(name_, Halide::GeneratorContext(target_)));
    if (!bb) {
        log::error("BuildingBlock {} is not found", name_);
        throw std::runtime_error("Failed to create building block object");
    }

    arginfos = bb->arginfos();
    std::cout<<graph_id<<std::endl;
}

void Node::set_iport(const std::vector<Port>& ports) {

    impl_->ports.erase(std::remove_if(impl_->ports.begin(), impl_->ports.end(),
                                      [&](const Port &p) { return p.has_succ_by_nid(this->id()); }),
                       impl_->ports.end());

    size_t i = 0;
    for (auto& port : ports) {
        // TODO: Validation is better to be done lazily after BuildingBlock::configure
        //
        // if (info.dir == Halide::Internal::ArgInfoDirection::Output) {
        //     continue;
        // }

        // if (i >= ports.size()) {
        //     log::error("Port {} is out of range", i);
        //     throw std::runtime_error("Failed to validate input port");
        // }

        // NOTE: Is succ_chans name OK to be just leave as it is?
        port.impl_->succ_chans.insert({id(), "_ion_iport_" + std::to_string(i)});
        port.impl_ ->graph_id = impl_->graph_id;
        impl_->ports.push_back(port);

        i++;
    }
}

void Node::set_iport(Port port) {
    port.impl_ ->graph_id = impl_->graph_id;
    port.impl_->succ_chans.insert({id(), port.pred_name()});
    impl_->ports.push_back(port);
}

void Node::set_iport(const std::string& name, Port port) {
    port.impl_ ->graph_id = impl_->graph_id;
    port.impl_->succ_chans.insert({id(), name});
    impl_->ports.push_back(port);
}

Port Node::operator[](const std::string& name) {
    auto it = std::find_if(impl_->ports.begin(), impl_->ports.end(),
                           [&](const Port& p){ return p.pred_id() == impl_->id && p.pred_name() == name; });
    if (it == impl_->ports.end()) {
        // This is output port which is never referenced.
        // Bind myself as a predecessor and register
        Port port(impl_->id, name);
        impl_->ports.push_back(port);
        return port;
    } else {
        // Port is already registered
        return *it;
    }
}

Port Node::iport(const std::string& pn) {
    for (const auto& p: impl_->ports) {
        auto it = std::find_if(p.impl_->succ_chans.begin(), p.impl_->succ_chans.end(),
                               [&](const Port::Channel& c) { return std::get<0>(c) == impl_->id && std::get<1>(c) == pn; });
        if (it != p.impl_->succ_chans.end()) {
            return p;
        }
    }

    auto msg = fmt::format("BuildingBlock \"{}\" has no input \"{}\"", name(), pn);
    log::error(msg);
    throw std::runtime_error(msg);
}

std::vector<std::tuple<std::string, Port>> Node::iports() const {
    std::vector<std::tuple<std::string, Port>> iports;
    for (const auto& p: impl_->ports) {
        auto it = std::find_if(p.impl_->succ_chans.begin(), p.impl_->succ_chans.end(),
                               [&](const Port::Channel& c) { return std::get<0>(c) == impl_->id; });
        if (it != p.impl_->succ_chans.end()) {
            iports.push_back(std::make_tuple(std::get<1>(*it), p));
        }
    }
    return iports;
}

Port Node::oport(const std::string& pn) {
    return this->operator[](pn);

    // TODO: It is better to just return exisitng output port?
    //
    // auto it = std::find_if(impl_->ports.begin(), impl_->ports.end(),
    //                        [&](const Port& p) { return p.pred_id() == id() && p.pred_name() == pn; });

    // if (it != impl_->ports.end()) {
    //     return *it;
    // }

    // auto msg = fmt::format("BuildingBlock \"{}\" has no output \"{}\"", name(), pn);
    // log::error(msg);
    // throw std::runtime_error(msg);
}

std::vector<std::tuple<std::string, Port>> Node::oports() const {
    std::vector<std::tuple<std::string, Port>> oports;
    for (const auto& p: impl_->ports) {
        if (id() == p.pred_id()) {
            oports.push_back(std::make_tuple(p.pred_name(), p));
        }
    }
    return oports;
}

} // namespace ion
