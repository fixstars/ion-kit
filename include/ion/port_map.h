#ifndef ION_PORT_MAP_H
#define ION_PORT_MAP_H

#include <functional>
#include <string>
#include <tuple>

#include <Halide.h>

#include "ion/util.h"

namespace std
{

template<>
struct hash<tuple<string, string>>
{
    std::size_t operator()(const tuple<string, string>& k) const noexcept
    {
        return std::hash<std::string>{}(std::get<0>(k)) ^ std::hash<std::string>{}(std::get<1>(k));
    }
};

template<>
struct equal_to<tuple<string, string>>
{
    bool operator()(const tuple<string, string>& v0, const tuple<string, string>& v1) const
    {
        return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1));
    }
};

} // std

namespace ion {

/**
 * PortMap is used to assign actual value to the port input.
 */
class PortMap {

public:

    template<typename T>
    void set(const Halide::Param<T>& p, T v) {
        param_expr_[p.name()] = p;
        param_map_.set(p, v);

        auto & vs(param_expr_instance_[p.name()]);
        vs.resize(sizeof(v));
        std::memcpy(vs.data(), &v, sizeof(v));
    }

    template<typename T>
    void set(const Halide::ImageParam& p, Halide::Buffer<T>& buf) {
        param_func_[p.name()] = p;
        param_map_.set(p, buf);

        param_func_instance_[p.name()] = buf.raw_buffer();
    }

    /**
     * Set the scalar value against to the port.
     * Template type T is allowed to be one of the following.
     * - bool
     * - uint8_t
     * - uint16_t
     * - uint32_t
     * - uint64_t
     * - int8_t
     * - int16_t
     * - int32_t
     * - int64_t
     * - float
     * - double
     * @arg p: The port object which value is assigned.
     * @arg v: Actual value to be mapped to the port.
     */
    template<typename T>
    void set(Port p, T v) {
        param_expr_[p.key()] = p.expr();
        p.set_to_param_map(param_map_, v);

        auto & vs(param_expr_instance_[p.key()]);
        vs.resize(sizeof(v));
        std::memcpy(vs.data(), &v, sizeof(v));
    }

    /**
     * Set the vector value against to the port.
     * Following type value is allowed to specified:
     * - bool
     * - uint8_t
     * - uint16_t
     * - uint32_t
     * - uint64_t
     * - int8_t
     * - int16_t
     * - int32_t
     * - int64_t
     * - float
     * - double
     * @arg p: The port object which value is assigned.
     * @arg buf: Actual value to be mapped to the port.
     * Buffer dimension should be matched with port's one.
     */
    template<typename T>
    void set(Port p, Halide::Buffer<T>& buf) {
        if (p.bound()) {
            // This is just an output.
            output_buffer_[std::make_tuple(p.node_id(), p.key())] = { buf };
            output_buffer_instance_[std::make_tuple(p.node_id(), p.key())] = { buf.raw_buffer() };
        } else {
            param_func_[p.key()] = p.func();
            p.set_to_param_map(param_map_, buf);
            param_func_instance_[p.key()] = buf.raw_buffer();
        }
    }

    /**
     * Set the vector of the vector values against to the port.
     * Following type value is allowed to specified:
     * - bool
     * - uint8_t
     * - uint16_t
     * - uint32_t
     * - uint64_t
     * - int8_t
     * - int16_t
     * - int32_t
     * - int64_t
     * - float
     * - double
     * @arg p: The port object which value is assigned.
     * @arg bufs: Actual value to be mapped to the port.
     * Buffer dimension should be matched with port's one.
     */
    template<typename T>
    void set(Port p, const std::vector<Halide::Buffer<T>> &bufs) {
        if (p.bound()) {
            // This is just an output.
            for (size_t i=0; i<bufs.size(); ++i) {
                auto buf = bufs[i];
                output_buffer_[std::make_tuple(p.node_id(), p.key())].push_back(buf);
                output_buffer_instance_[std::make_tuple(p.node_id(), p.key())].push_back(buf.raw_buffer());
            }
        } else {
            throw std::invalid_argument(
                "Unbounded port (" + p.key() + ") corresponding to an array of Inputs is not supported");
        }
    }

    bool mapped(const std::string& k) const {
        return param_expr_.count(k) != 0 || param_func_.count(k) != 0;
    }

    Halide::Expr get_param_expr(const std::string& k) const {
        return param_expr_.at(k);
    }

    Halide::Func get_param_func(const std::string& k) const {
        return param_func_.at(k);
    }

    std::unordered_map<std::tuple<std::string, std::string>, std::vector<Halide::Buffer<>>> get_output_buffer() const {
        return output_buffer_;
    }

    Halide::ParamMap get_param_map() const {
        return param_map_;
    }

    std::vector<Halide::Argument> get_arguments_stub() const {
        std::vector<Halide::Argument> args;
        for (auto kv : param_func_) {
            args.push_back(Halide::Argument(kv.first, Halide::Argument::InputBuffer, kv.second.type(), 0, Halide::ArgumentEstimates()));
        }
        for (auto kv : param_expr_) {
            args.push_back(Halide::Argument(kv.first, Halide::Argument::InputScalar, kv.second.type(), 0, Halide::ArgumentEstimates()));
        }
        for (auto kv : output_buffer_) {
            for (auto f : kv.second) {
                args.push_back(Halide::Argument(std::get<0>(kv.first) + "_" + std::get<1>(kv.first), Halide::Argument::OutputBuffer, f.type(), 0, Halide::ArgumentEstimates()));
            }
        }
        return args;
    }

    std::vector<void*> get_arguments_instance() const {
        std::vector<void*> args;
        for (auto kv : param_func_instance_) {
            args.push_back(kv.second);
        }
        for (auto kv : param_expr_instance_) {
            args.push_back(reinterpret_cast<void*>(kv.second.data()));
        }
        for (auto kv : output_buffer_instance_) {
            for (auto f : kv.second) {
                args.push_back(f);
            }
        }
        return args;
    }

 private:

    std::unordered_map<std::string, Halide::Expr> param_expr_;
    std::unordered_map<std::string, std::vector<uint8_t>> param_expr_instance_;
    std::unordered_map<std::string, Halide::Func> param_func_;
    std::unordered_map<std::string, halide_buffer_t*> param_func_instance_;
    std::unordered_map<std::tuple<std::string, std::string>, std::vector<Halide::Buffer<>>> output_buffer_;
    std::unordered_map<std::tuple<std::string, std::string>, std::vector<halide_buffer_t*>> output_buffer_instance_;
    Halide::ParamMap param_map_;
};


} // namespace ion

#endif // ION_PORT_MAP_H
