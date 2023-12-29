#include "ion/node.h"

#include "log.h"

namespace ion {

Node::Impl::Impl(const std::string& id_, const std::string& name_, const Halide::Target& target_)
    : id(id_), name(name_), target(target_), params(), ports()
{
    auto bb(Halide::Internal::GeneratorRegistry::create(name_, Halide::GeneratorContext(target_)));
    if (!bb) {
        log::error("BuildingBlock {} is not found", name_);
        throw std::runtime_error("Failed to create building block");
    }

    arginfos = bb->arginfos();
}

void Node::set_iports(const std::vector<Port>& ports) {

    size_t i = 0;
    for (const auto& info : impl_->arginfos) {
        if (info.dir == Halide::Internal::ArgInfoDirection::Output) {
            continue;
        }

        if (i >= ports.size()) {
            log::error("Port {} is out of range", i);
            throw std::runtime_error("Failed to validate input port");
        }

        auto& port(ports[i]);

        port.impl_->succ_chans.insert({.node_id=id(), .name=info.name});

        impl_->ports.push_back(port);

        i++;
    }
}

} // namespace ion
