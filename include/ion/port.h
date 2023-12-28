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

#define SW 0

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
        std::vector<Halide::ImageParam> fparams;

        std::unordered_map<int32_t, std::variant<Halide::Internal::Parameter, Halide::ImageParam>> vparams;

        Impl() {}

        Impl(const std::string& n, const Halide::Type& t, int32_t d, const std::string& nid)
            : name(n), type(t), dimensions(d), node_id(nid)
        {
            if (dimensions == 0) {
                params = { Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(node_id, name)) };

                vparams[0] = Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(node_id, name, 0));
            } else {
                fparams = { Halide::ImageParam(type, dimensions, argument_name(node_id, name)) };

                vparams[0] = Halide::ImageParam(type, dimensions, argument_name(node_id, name, 0));
            }
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

     int32_t size() const { return (impl_->dimensions == 0) ? impl_->params.size() : impl_->fparams.size(); }

     int32_t index() const { return index_; }

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

         // Old
         if (impl_->params.size() <= i) {
             impl_->params.resize(i+1);
             impl_->params[i] = Halide::Internal::Parameter{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name(), i)};
         }
         impl_->params[i].set_scalar(v);

         // New
         Halide::Internal::Parameter param{type(), dimensions() != 0, dimensions(), argument_name(node_id(), name(), i)};
         param.set_scalar(v);
         impl_->vparams[i] = param;
     }

     template<typename T>
     void bind(const Halide::Buffer<T>& buf) {
         auto i = index_ == -1 ? 0 : index_;

         // Old
         if (impl_->fparams.size() <= i) {
             impl_->fparams.resize(i+1);
             impl_->fparams[i] = Halide::ImageParam{type(), dimensions(), argument_name(node_id(), name(), i)};
         }
         impl_->fparams[i].set(buf);

         // New
         Halide::ImageParam param{type(), dimensions(), argument_name(node_id(), name(), i)};
         param.set(buf);
         impl_->vparams[i] = param;
     }

     template<typename T>
     void bind(const std::vector<Halide::Buffer<T>>& bufs) {
         // Old
         if (impl_->fparams.size() != bufs.size()) {
             impl_->fparams.resize(bufs.size());
             for (size_t i=0; i<bufs.size(); ++i) {
                 impl_->fparams[i] = Halide::ImageParam{type(), dimensions(), argument_name(node_id(), name(), i)};
             }
         }
         for (size_t i=0; i<bufs.size(); ++i) {
             impl_->fparams[i].set(bufs[i]);
         }

         // New
         for (size_t i=0; i<bufs.size(); ++i) {
             Halide::ImageParam param{type(), dimensions(), argument_name(node_id(), name(), i)};
             param.set(bufs[i]);
             impl_->vparams[i] = param;
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
     Port(const std::string& n, const std::string& nid) : impl_(new Impl(n, Halide::Type(), 0, nid)), index_(-1) {}

     std::vector<Halide::Argument> as_argument() const {
         std::vector<Halide::Argument> args;
#if SW
         if (dimensions() == 0) {
             for (auto i = 0; i<impl_->params.size(); ++i) {
                 args.push_back(Halide::Argument(argument_name(node_id(), name(), i),  Halide::Argument::InputScalar, type(), dimensions(), Halide::ArgumentEstimates()));
             }
         } else {
             for (auto i = 0; i<impl_->fparams.size(); ++i) {
                 args.push_back(Halide::Argument(argument_name(node_id(), name(), i),  Halide::Argument::InputBuffer, type(), dimensions(), Halide::ArgumentEstimates()));
             }
         }
#else
         for (const auto& [i, param] : impl_->vparams) {
             if (args.size() <= i) {
                 args.resize(i+1, Halide::Argument());
             }
             auto kind = impl_->dimensions == 0 ? Halide::Argument::InputScalar : Halide::Argument::InputBuffer;
             args[i] = Halide::Argument(argument_name(impl_->node_id, impl_->name, i),  kind, impl_->type, impl_->dimensions, Halide::ArgumentEstimates());
         }

#endif
         return args;
     }

     std::vector<const void *> as_instance() const {
         std::vector<const void *> instances;
#if SW
         if (dimensions() == 0) {
             for (const auto& param : impl_->params) {
                 instances.push_back(param.scalar_address());
             }
         } else {
             for (const auto& fparam : impl_->fparams) {
                 instances.push_back(fparam.get().raw_buffer());
             }
         }
#else
         for (const auto& [i, param] : impl_->vparams) {
             if (instances.size() <= i) {
                 instances.resize(i+1, nullptr);
             }
             if (auto p = std::get_if<Halide::Internal::Parameter>(&param)) {
                 instances[i] = p->scalar_address();
             } else if (auto p = std::get_if<Halide::ImageParam>(&param)) {
                 instances[i] = p->get().raw_buffer();
             }
         }
#endif

         return instances;
     }

     std::vector<Halide::Expr> as_expr() const {
         if (impl_->dimensions != 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Expr> es;
#if SW
         int32_t i = 0;
         for (const auto& p : impl_->params) {
             es.push_back(Halide::Internal::Variable::make(impl_->type, argument_name(impl_->node_id, impl_->name, i++), p));
         }
#else
         for (const auto& [i, param] : impl_->vparams) {
             if (es.size() <= i) {
                 es.resize(i+1, Halide::Expr());
             }
             es[i] = Halide::Internal::Variable::make(impl_->type, argument_name(impl_->node_id, impl_->name, i),
                                                      *std::get_if<Halide::Internal::Parameter>(&param));
         }
#endif
         return es;
     }

     std::vector<Halide::Func> as_func() const {
         if (impl_->dimensions == 0) {
            throw std::runtime_error("Unreachable");
         }

         std::vector<Halide::Func> fs;
#if SW
         for (const auto& p : impl_->fparams ) {
             fs.push_back(p);
         }
#else
         for (const auto& [i, param] : impl_->vparams ) {
             if (fs.size() <= i) {
                 fs.resize(i+1, Halide::Func());
             }
             fs[i] = *std::get_if<Halide::ImageParam>(&param);
         }
#endif
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
