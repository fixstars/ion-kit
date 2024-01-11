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

void Node::set_iport(const std::vector<Port>& ports) {

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
        port.impl_->succ_chans.insert({id(), "_ion_iport_" + i});

        impl_->ports.push_back(port);

        i++;
    }
}

Port Node::operator[](const std::string& name) {
        // TODO: Validation is better to be done lazily after BuildingBlock::configure
        //
        // if (std::find_if(impl_->arginfos.begin(), impl_->arginfos.end(),
        //                  [&](const Halide::Internal::AbstractGenerator::ArgInfo& info) { return info.name == name; }) == impl_->arginfos.end()) {
        //     log::error("Port {} is not found", name);
        //     throw std::runtime_error("Failed to find port");
        // }

        auto it = std::find_if(impl_->ports.begin(), impl_->ports.end(),
                               [&](const Port& p){ return (p.pred_name() == name && p.pred_id() == impl_->id) || p.has_succ({impl_->id, name}); });
        if (it == impl_->ports.end()) {
            // This is output port which is never referenced.
            // Bind myself as a predecessor and register

            // TODO: Validate with arginfo
            Port port(impl_->id, name);
            impl_->ports.push_back(port);
            return port;
        } else {
            // Port is already registered
            return *it;
        }
    }

} // namespace ion
