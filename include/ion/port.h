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
        std::string pred_id;
        std::string pred_name;

        std::string succ_id;
        std::string succ_name;

        Halide::Type type;
        int32_t dimensions;

        std::unordered_map<int32_t, Halide::Internal::Parameter> params;
        std::unordered_map<int32_t, const void *> instances;

        Impl() {}

        Impl(const std::string& pid, const std::string& pn, const std::string& sid, const std::string& sn, const Halide::Type& t, int32_t d)
            : pred_id(pid), pred_name(pn), succ_id(sid), succ_name(sn), type(t), dimensions(d)
        {
            params[0] = Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(pid, pn, sid, sn, 0));
        }
    };

 public:
     friend class Builder;
     friend class Node;
     friend class nlohmann::adl_serializer<Port>;

     Port() : impl_(new Impl("", "", "", "", Halide::Type(), 0)), index_(-1) {}
     Port(const std::shared_ptr<Impl>& impl) : impl_(impl), index_(-1) {}

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& n, Halide::Type t) : impl_(new Impl("", "", "", n, t, 0)), index_(-1) {}

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& n, Halide::Type t, int32_t d) : impl_(new Impl("", "", "", n, t, d)), index_(-1) {}

     const std::string& pred_name() const { return impl_->pred_name; }
     const std::string& succ_name() const { return impl_->succ_name; }

     const Halide::Type& type() const { return impl_->type; }

     int32_t dimensions() const { return impl_->dimensions; }

     const std::string& pred_id() const { return impl_->pred_id; }

     const std::string& succ_id() const { return impl_->succ_id; }

     int32_t size() const { return impl_->params.size(); }

     int32_t index() const { return index_; }

     bool has_pred() const { return !pred_id().empty(); }

     bool has_succ() const { return !succ_id().empty(); }

     void set_index(int index) { index_ = index; }

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
         if (has_pred()) {
             impl_->params[i] = Halide::Internal::Parameter{Halide::type_of<T>(), false, 0, argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), false, dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
         }

         impl_->instances[i] = v;
     }


     template<typename T>
     void bind(const Halide::Buffer<T>& buf) {
         auto i = index_ == -1 ? 0 : index_;
         if (has_pred()) {
             impl_->params[i] = Halide::Internal::Parameter{buf.type(), true, buf.dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
         }

         impl_->instances[i] = buf.raw_buffer();
     }

     template<typename T>
     void bind(const std::vector<Halide::Buffer<T>>& bufs) {
         for (size_t i=0; i<bufs.size(); ++i) {
             if (has_pred()) {
                 impl_->params[i] = Halide::Internal::Parameter{bufs[i].type(), true, bufs[i].dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
             } else {
                 impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i)};
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
     * This port is created from another node
     */
     Port(const std::string& pid, const std::string& pn, const std::string& sid, const std::string& sn) : impl_(new Impl(pid, pn, sid, sn, Halide::Type(), 0)), index_(-1) {}


     std::vector<Halide::Argument> as_argument() const {
         std::vector<Halide::Argument> args;
         for (const auto& [i, param] : impl_->params) {
             if (args.size() <= i) {
                 args.resize(i+1, Halide::Argument());
             }
             auto kind = dimensions() == 0 ? Halide::Argument::InputScalar : Halide::Argument::InputBuffer;
             args[i] = Halide::Argument(argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i),  kind, type(), dimensions(), Halide::ArgumentEstimates());
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
         if (dimensions() != 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Expr> es;
         for (const auto& [i, param] : impl_->params) {
             if (es.size() <= i) {
                 es.resize(i+1, Halide::Expr());
             }
             es[i] = Halide::Internal::Variable::make(type(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i), param);
         }
         return es;
     }

     std::vector<Halide::Func> as_func() const {
         if (dimensions() == 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Func> fs;
         for (const auto& [i, param] : impl_->params ) {
             if (fs.size() <= i) {
                 fs.resize(i+1, Halide::Func());
             }
             std::vector<Halide::Var> args;
             std::vector<Halide::Expr> args_expr;
             for (int i = 0; i < dimensions(); ++i) {
                 args.push_back(Halide::Var::implicit(i));
                 args_expr.push_back(Halide::Var::implicit(i));
             }
             Halide::Func f(param.type(), param.dimensions(), argument_name(pred_id(), pred_name(), succ_id(), succ_name(), i) + "_im");
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
