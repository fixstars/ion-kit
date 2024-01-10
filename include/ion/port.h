#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <Halide.h>

#include "util.h"

namespace ion {

template<typename T>
std::string unify_name(const std::vector<Halide::Buffer<T>>& bufs) {
    std::stringstream ss;
    for (auto i=0; i<bufs.size(); ++i) {
        ss << bufs[i].name();
        if (i < bufs.size()) {
            ss << "_";
        }
    }
    return ss.str();
}

template<typename T>
int32_t unify_dimension(const std::vector<Halide::Buffer<T>>& bufs) {
    int32_t dimension = 0;
    for (auto i=0; i<bufs.size(); ++i) {
        if (i == 0) {
            dimension = bufs[i].dimensions();
        } else if (dimension != bufs[i].dimensions()) {
            throw std::runtime_error("Buffer dimensions should be same");
        }
    }
    return dimension;
}

/**
 * Port class is used to create dynamic i/o for each node.
 */
class Port {
    friend class Builder;
    friend class Node;

public:
    using Channel = std::tuple<std::string, std::string>;

private:
    struct Impl {
        // std::string pred_id;
        // std::string pred_name;

        // std::string succ_id;
        // std::string succ_name;

        Channel pred_chan;
        std::set<Channel> succ_chans;

        Halide::Type type;
        int32_t dimensions;

        std::unordered_map<int32_t, Halide::Internal::Parameter> params;
        std::unordered_map<int32_t, const void *> instances;

        Impl() {}

        Impl(const std::string& pid, const std::string& pn, const Halide::Type& t, int32_t d)
            : pred_chan{pid, pn}, succ_chans{}, type(t), dimensions(d)
        {
            params[0] = Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(pid, pn, 0));
        }
    };

public:

    Port() : impl_(new Impl("", "", Halide::Type(), 0)), index_(-1) {}

    Port(const std::shared_ptr<Impl>& impl, int32_t index) : impl_(impl), index_(index) {}

    /**
     * Construct new port for scalar value.
     * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
     * @arg t: The type of the value.
     */
    Port(const std::string& n, Halide::Type t) : impl_(new Impl("", n, t, 0)), index_(-1) {}

    /**
     * Construct new port for vector value.
     * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
     * @arg t: The type of the element value.
     * @arg d: The dimension of the port. The range is 1 to 4.
     */
    Port(const std::string& n, Halide::Type t, int32_t d) : impl_(new Impl("", n, t, d)), index_(-1) {}

    /**
     * Construct new port from scalar pointer
     */
    template<typename T,
             typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    Port(T *vptr) : impl_(new Impl("", Halide::Internal::unique_name("ion_port"), Halide::type_of<T>(), 0)), index_(-1) {
        this->bind(vptr);
    }

    /**
     * Construct new port from buffer
     */
    template<typename T>
    Port(const Halide::Buffer<T>& buf) : impl_(new Impl("", buf.name(), Halide::type_of<T>(), buf.dimensions())), index_(-1) {
        this->bind(buf);
    }

    /**
     * Construct new port from array of buffer
     */
    template<typename T>
    Port(const std::vector<Halide::Buffer<T>>& bufs) : impl_(new Impl("", unify_name(bufs), Halide::type_of<T>(), unify_dimension(bufs))), index_(-1) {
        this->bind(bufs);
    }

    // Getter
    const Channel& pred_chan() const { return impl_->pred_chan; }
    const std::string& pred_id() const { return std::get<0>(impl_->pred_chan); }
    const std::string& pred_name() const { return std::get<1>(impl_->pred_chan); }
    const std::set<Channel>& succ_chans() const { return impl_->succ_chans; }
    const Halide::Type& type() const { return impl_->type; }
    int32_t dimensions() const { return impl_->dimensions; }
    int32_t size() const { return impl_->params.size(); }
    int32_t index() const { return index_; }
    uintptr_t impl_ptr() const { return reinterpret_cast<uintptr_t>(impl_.get()); }

    // Setter
    void set_index(int index) { index_ = index; }

    // Util
    bool has_pred() const { return !std::get<0>(impl_->pred_chan).empty(); }
    bool has_succ() const { return !impl_->succ_chans.empty(); }
    bool has_succ(const Channel& c) const { return impl_->succ_chans.count(c); }

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
             impl_->params[i] = Halide::Internal::Parameter{Halide::type_of<T>(), false, 0, argument_name(pred_id(), pred_name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), false, dimensions(), argument_name(pred_id(), pred_name(), i)};
         }

         impl_->instances[i] = v;
     }


     template<typename T>
     void bind(const Halide::Buffer<T>& buf) {
         auto i = index_ == -1 ? 0 : index_;
         if (has_pred()) {
             impl_->params[i] = Halide::Internal::Parameter{buf.type(), true, buf.dimensions(), argument_name(pred_id(), pred_name(), i)};
         } else {
             impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(pred_id(), pred_name(), i)};
         }

         impl_->instances[i] = buf.raw_buffer();
     }

     template<typename T>
     void bind(const std::vector<Halide::Buffer<T>>& bufs) {
         for (int i=0; i<static_cast<int>(bufs.size()); ++i) {
             if (has_pred()) {
                 impl_->params[i] = Halide::Internal::Parameter{bufs[i].type(), true, bufs[i].dimensions(), argument_name(pred_id(), pred_name(), i)};
             } else {
                 impl_->params[i] = Halide::Internal::Parameter{type(), true, dimensions(), argument_name(pred_id(), pred_name(), i)};
             }

             impl_->instances[i] = bufs[i].raw_buffer();
         }
     }

     static std::tuple<std::shared_ptr<Impl>, bool> find_impl(uintptr_t ptr) {
         static std::unordered_map<uintptr_t, std::shared_ptr<Impl>> impls;
         static std::mutex mutex;
         std::scoped_lock lock(mutex);
         bool found = true;
         if (!impls.count(ptr)) {
             impls[ptr] = std::make_shared<Impl>();
             found = false;
         }
         return std::make_tuple(impls[ptr], found);
     }

private:
    /**
     * This port is created from another node
     */
     Port(const std::string& pid, const std::string& pn) : impl_(new Impl(pid, pn, Halide::Type(), 0)), index_(-1) {}


     std::vector<Halide::Argument> as_argument() const {
         std::vector<Halide::Argument> args;
         for (const auto& [i, param] : impl_->params) {
             if (args.size() <= i) {
                 args.resize(i+1, Halide::Argument());
             }
             auto kind = dimensions() == 0 ? Halide::Argument::InputScalar : Halide::Argument::InputBuffer;
             args[i] = Halide::Argument(argument_name(pred_id(), pred_name(), i),  kind, type(), dimensions(), Halide::ArgumentEstimates());
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
             es[i] = Halide::Internal::Variable::make(type(), argument_name(pred_id(), pred_name(), i), param);
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
             Halide::Func f(param.type(), param.dimensions(), argument_name(pred_id(), pred_name(), i) + "_im");
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
