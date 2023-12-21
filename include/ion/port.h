#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Halide.h>

#include "json/json.hpp"

#include "util.h"

namespace ion {

/**
 * Port class is used to create dynamic i/o for each node.
 */
class Port {

    struct Impl {
        std::string name;
        Halide::Type type;
        int32_t dimensions;
        std::string node_id;
        std::vector<Halide::Internal::Parameter> params;

        Impl() : name(), type(), dimensions(0), node_id(), params() {}
    };

 public:
     friend class Node;
     friend class nlohmann::adl_serializer<Port>;

     Port() : impl_(new Impl), index_(-1) {};
     Port(const std::shared_ptr<Impl>& impl) : impl_(impl), index_(-1) {};

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& n, Halide::Type t) : impl_(new Impl), index_(-1) {
         impl_->name = n;
         impl_->type = t;
     }

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& n, Halide::Type t, int32_t d) : impl_(new Impl), index_(-1) {
        impl_->name = n;
        impl_->type = t;
        impl_->dimensions = d;
     }

     const std::string& name() const { return impl_->name; }
     std::string& name() { return impl_->name; }

     const Halide::Type& type() const { return impl_->type; }
     Halide::Type& type() { return impl_->type; }

     int32_t dimensions() const { return impl_->dimensions; }
     int32_t& dimensions() { return impl_->dimensions; }

     const std::string& node_id() const { return impl_->node_id; }
     std::string& node_id() { return impl_->node_id; }

     const std::vector<Halide::Internal::Parameter>& params() const { return impl_->params; }
     std::vector<Halide::Internal::Parameter>& params() { return impl_->params; }

     int32_t index() const { return index_; }
     int32_t& index() { return index_; }

     bool is_bound() const {
         return !node_id().empty();
     }

     void set_index(int index) {
         index_ = index;
     }

    /**
     * Overloaded operator to set the port index and return a reference to the current port. eg. port[0]
     */
     Port operator[](int index) {
         Port port(*this);
         port.index_ = index;
         return port;
     }

     template<typename T>
     void bind(T v) {
         auto i = index_ == -1 ? 0 : index_;
         impl_->params.resize(i+1, Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name())});
         impl_->params[i].set_scalar(v);
     }

     template<typename T>
     void bind(const Halide::Buffer<T>& buf) {
         auto i = index_ == -1 ? 0 : index_;
         impl_->params.resize(i+1, Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name())});
         impl_->params[i].set_buffer(buf);
     }

     template<typename T>
     void bind(const std::vector<Halide::Buffer<T>>& bufs) {
         impl_->params.resize(bufs.size(), Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name())});
         for (size_t i=0; i<bufs.size(); ++i) {
             impl_->params[i].set_buffer(bufs[i]);
         }
     }

     static std::shared_ptr<Impl> find_impl(uintptr_t ptr) {
         static std::unordered_map<uintptr_t, std::shared_ptr<Impl>> impls;
         static std::mutex mutex;
         std::scoped_lock lock(mutex);
         if (!impls.count(ptr)) {
             impls[ptr] = std::make_shared<Impl>();
         }
         return impls[ptr];
     }

private:
    /**
     * This port is bound with some node.
     */
     Port(const std::string& n, const std::string& ni) : impl_(new Impl), index_(-1) {
         impl_->name = n;
         impl_->node_id = ni;
     }

     std::shared_ptr<Impl> impl_;

     // NOTE:
     // The reasons why index sits outside of the impl_ is because
     // index is tentatively used to hold index of params.
     int32_t index_;
};

} // namespace ion

#endif // ION_PORT_H
