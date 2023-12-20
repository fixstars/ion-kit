#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Halide.h>

#include "util.h"

namespace ion {

/**
 * Port class is used to create dynamic i/o for each node.
 */
class Port {

    struct Port_ {
        std::string name;
        Halide::Type type;
        int32_t dimensions;
        int32_t index;
        std::string node_id;
        std::vector<Halide::Internal::Parameter> params;

        Port_() : name(), type(), dimensions(0), index(-1), node_id(), params() {}
    };

 public:
     friend class Node;

     Port() : impl_(new Port_) {};

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& n, Halide::Type t) : impl_(new Port_) {
         impl_->name = n;
         impl_->type = t;
     }

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& n, Halide::Type t, int32_t d) : impl_(new Port_) {
        impl_->name = n;
        impl_->type = t;
        impl_->dimensions = d;
     }

     std::string name() const { return impl_->name; }
     std::string& name() { return impl_->name; }

     Halide::Type type() const { return impl_->type; }
     Halide::Type& type() { return impl_->type; }

     int32_t dimensions() const { return impl_->dimensions; }
     int32_t& dimensions() { return impl_->dimensions; }

     int32_t index() const { return impl_->index; }
     int32_t& index() { return impl_->index; }

     std::string node_id() const { return impl_->node_id; }
     std::string& node_id() { return impl_->node_id; }

     std::vector<Halide::Internal::Parameter>& params() {
         if (index() == -1) {
             impl_->params.resize(1, Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name())});
         } else {
             impl_->params.resize(index()+1, Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name())});
         }
         return impl_->params;
     }

     bool is_bound() const {
         return !node_id().empty();
     }


     void set_index(int index) {
         impl_->index = index;
     }

    /**
     * Overloaded operator to set the port index and return a reference to the current port. eg. port[0]
     */
     Port operator[](int index) {
         set_index(index);
         return *this;
     }

private:
    /**
     * This port is bound with some node.
     */
     Port(const std::string& n, const std::string& ni) : impl_(new Port_) {
         impl_->name = n;
         impl_->node_id = ni;
     }

     std::shared_ptr<Port_> impl_;
};

} // namespace ion

#endif // ION_PORT_H
