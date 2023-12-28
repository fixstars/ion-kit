#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <variant>
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
        std::unordered_map<int32_t, Halide::Internal::Parameter> params;
        std::unordered_map<int32_t, const void *> instances;

        Impl() {}

        Impl(const std::string& n, const Halide::Type& t, int32_t d, const std::string& nid)
            : name(n), type(t), dimensions(d), node_id(nid)
        {
            params[0] = Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(node_id, name, 0));
        }
    };

 public:
     friend class Builder;
     friend class Node;
     friend class nlohmann::adl_serializer<Port>;

     Port() : impl_(new Impl("", Halide::Type(), 0, "")), index_(-1) {}
     Port(const std::shared_ptr<Impl>& impl) : impl_(impl), index_(-1) {}

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& n, Halide::Type t) : impl_(new Impl(n, t, 0, "")), index_(-1) {}

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& n, Halide::Type t, int32_t d) : impl_(new Impl(n, t, d, "")), index_(-1) {}

     const std::string& name() const { return impl_->name; }

     const Halide::Type& type() const { return impl_->type; }

     int32_t dimensions() const { return impl_->dimensions; }

     const std::string& node_id() const { return impl_->node_id; }

     int32_t size() const { return impl_->params.size(); }

     int32_t index() const { return index_; }

     bool has_source() const {
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
     void bind(T *v) {
         auto i = index_ == -1 ? 0 : index_;
         if (has_source()) {
             impl_->params[i] = Halide::Internal::Parameter{Halide::type_of<T>(), false, 0, argument_name(node_id(), name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), false, dimensions(), argument_name(node_id(), name(), i)};
         }

         impl_->instances[i] = v;
     }


     template<typename T>
     void bind(const Halide::Buffer<T>& buf) {
         auto i = index_ == -1 ? 0 : index_;
         if (has_source()) {
             impl_->params[i] = Halide::Internal::Parameter{buf.type(), true, buf.dimensions(), argument_name(node_id(), name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(node_id(), name(), i)};
         }

         impl_->instances[i] = buf.raw_buffer();
     }

     template<typename T>
     void bind(const std::vector<Halide::Buffer<T>>& bufs) {
         for (size_t i=0; i<bufs.size(); ++i) {
             if (has_source()) {
                 impl_->params[i] = Halide::Internal::Parameter{bufs[i].type(), true, bufs[i].dimensions(), argument_name(node_id(), name(), i)};
             } else {
                 impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(node_id(), name(), i)};
             }

             impl_->instances[i] = bufs[i].raw_buffer();
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
     Port(const std::string& n, const std::string& nid) : impl_(new Impl), index_(-1) {
         impl_->name = n;
         impl_->node_id = nid;
     }

     std::vector<Halide::Argument> as_argument() const {
         std::vector<Halide::Argument> args;
         for (const auto& [i, param] : impl_->params) {
             if (args.size() <= i) {
                 args.resize(i+1, Halide::Argument());
             }
             auto kind = impl_->dimensions == 0 ? Halide::Argument::InputScalar : Halide::Argument::InputBuffer;
             args[i] = Halide::Argument(argument_name(impl_->node_id, impl_->name, i),  kind, impl_->type, impl_->dimensions, Halide::ArgumentEstimates());
         }
         return args;
     }

     std::vector<const void *> as_instance() const {
         std::vector<const void *> instances;
        for (const auto& [i, instance] : impl_->instances) {
             if (instances.size() <= i) {
                 instances.resize(i+1, nullptr);
             }
             instances[i] = instance;
        }
         return instances;
     }

     std::vector<Halide::Expr> as_expr() const {
         if (impl_->dimensions != 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Expr> es;
         for (const auto& [i, param] : impl_->params) {
             if (es.size() <= i) {
                 es.resize(i+1, Halide::Expr());
             }
             es[i] = Halide::Internal::Variable::make(impl_->type, argument_name(impl_->node_id, impl_->name, i), param);
         }
         return es;
     }

     std::vector<Halide::Func> as_func() const {
         if (impl_->dimensions == 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Func> fs;
         for (const auto& [i, param] : impl_->params ) {
             if (fs.size() <= i) {
                 fs.resize(i+1, Halide::Func());
             }
             std::vector<Halide::Var> args;
             std::vector<Halide::Expr> args_expr;
             for (int i = 0; i < impl_->dimensions; ++i) {
                 args.push_back(Halide::Var::implicit(i));
                 args_expr.push_back(Halide::Var::implicit(i));
             }
             Halide::Func f(param.type(), param.dimensions(), argument_name(impl_->node_id, impl_->name, i) + "_im");
             f(args) = Halide::Internal::Call::make(param, args_expr);
             fs[i] = f;
         }
         return fs;
     }

     std::shared_ptr<Impl> impl_;

     // NOTE:
     // The reasons why index sits outside of the impl_ is because
     // index is tentatively used to hold index of params.
     int32_t index_;
};

} // namespace ion

#endif // ION_PORT_H
