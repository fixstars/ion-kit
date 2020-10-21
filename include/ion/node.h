#ifndef ION_NODE_H
#define ION_NODE_H

#include <memory>
#include <string>
#include <vector>

#include <Halide.h>

#include "json.hpp"
#include "param.h"
#include "port.h"

namespace ion {

using json = nlohmann::json;

/**
 * Node class is used to manage node which consists graph structure.
 */
class Node {
    struct Node_ {
        std::string id;
        std::string name;
        Halide::Target target;
        std::vector<Param> params;
        std::vector<Port> ports;

        Node_(): id(), name(), target(), params(), ports() {}

        Node_(const std::string& id_, const std::string& name_, const Halide::Target& target_)
            : id(id_), name(name_), target(target_), params(), ports() {
        }
    };

public:
    friend class Builder;

    friend void to_json(json&, const Node&);
    friend void from_json(const json&, Node&);

    Node() : impl_(new Node_) {};

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

    void operator()(const std::vector<Port>& ports) {
        impl_->ports = ports;
    }

    /**
     * Retrieve output port of the node.
     * @arg key: The key of port name which is matched with first argument of Output declared in user-defined class deriving BuildingBlock.
     * @return Port object which is specified by key.
     */
    Port operator[](const std::string& key) {
        return Port(key, impl_->id);
    }

    std::string id() const {
        return impl_->id;
    }

    std::string name() const {
        return impl_->name;
    }

    Halide::Target target() const {
        return impl_->target;
    }

    std::vector<Param> params() const {
        return impl_->params;
    }

    std::vector<Port> ports() const {
        return impl_->ports;
    }

private:
    Node(const std::string& id, const std::string& name, const Halide::Target& target)
        : impl_(new Node_{id, name, target})
    {
    }

    std::shared_ptr<Node_> impl_;
};

} // namespace ion

#endif // ION_NODE_H
