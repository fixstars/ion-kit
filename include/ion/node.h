#ifndef ION_NODE_H
#define ION_NODE_H

#include <memory>
#include <string>
#include <vector>

#include <Halide.h>

#include "param.h"
#include "port.h"

namespace ion {

/**
 * Node class is used to manage node which consists graph structure.
 */
class Node {
    struct Impl {
        std::string id;
        std::string name;
        Halide::Target target;
        std::vector<Param> params;
        std::vector<Port> ports;

        Impl(): id(), name(), target(), params(), ports() {}

        Impl(const std::string& id_, const std::string& name_, const Halide::Target& target_)
            : id(id_), name(name_), target(target_), params(), ports() {
        }
    };

public:
    friend class Builder;

    Node() : impl_(new Impl) {};

    /**
     * Set the target of the node.
     * @arg target: The target ofject which consists of OS, Architecture, and sets of Features.
     * See https://halide-lang.org/docs/struct_halide_1_1_target.html for more details.
     * This target object can be retrieved by calling BuildingBlock::get_target from BuildingBlock::generate and BuildingBlock::schedule.
     * @return Node object whose target is set.
     */
    Node set_target(const Halide::Target& target) {
        impl_->target = target;
        return *this;
    }

    /**
     * Set the static parameters of the node.
     * @arg args: Variadic arguments of ion::Param.
     * Each of the arguments is applied and evaluated at compile time as a GeneratorParam declared in user-defined class deriving BuildingBlock.
     * @return Node object whose parameter is set.
     */
    template<typename... Args>
    Node set_param(Args ...args) {
        impl_->params = std::vector<Param>{args...};
        return *this;
    }

    void set_param(const std::vector<Param>& params) {
        impl_->params = params;
    }

    /**
     * Set the dynamic port of the node.
     * @arg args: Variadic arguments of ion::Port.
     * Each of the arguments is connected to each of the Input of this Node.
     * Each of the arguemnts should be one of the following:
     * 1. input port of the pipeline (Created by calling constructor of ion::Port).
     * 2. output port of another node (Retrieved by calling Node::operator[]).
     * @return Node object whose port is set.
     */
    template<typename... Args>
    Node operator()(Args ...args) {
        impl_->ports = std::vector<Port>{args...};
        return *this;
    }

    void set_port(std::vector<Port>& ports) {
        impl_->ports = ports;
    }

    /**
     * Retrieve output port of the node.
     * @arg name: The name of port name which is matched with first argument of Output declared in user-defined class deriving BuildingBlock.
     * @return Port object which is specified by name.
     */
    Port operator[](const std::string& name) {
        auto it = std::find_if(impl_->ports.begin(), impl_->ports.end(), [&name](const Port& p){ return p.name() == name; });
        if (it != impl_->ports.end()) {
            // This is input port, bind myself and create new Port instance
            return *it;
        } else {
            // This is output port, bind myself and create new Port instance
            return Port(name, impl_->id);
        }
    }

    const std::string& id() const {
        return impl_->id;
    }

    std::string& id() {
        return impl_->id;
    }

    const std::string& name() const {
        return impl_->name;
    }

    std::string& name(){
        return impl_->name;
    }

    const Halide::Target& target() const {
        return impl_->target;
    }

    Halide::Target& target() {
        return impl_->target;
    }

    const std::vector<Param>& params() const {
        return impl_->params;
    }

    std::vector<Param>& params() {
        return impl_->params;
    }

    const std::vector<Port>& ports() const {
        return impl_->ports;
    }

    std::vector<Port>& ports() {
        return impl_->ports;
    }

private:
    Node(const std::string& id, const std::string& name, const Halide::Target& target)
        : impl_(new Impl{id, name, target})
    {
    }

    std::shared_ptr<Impl> impl_;
};

} // namespace ion

#endif // ION_NODE_H
