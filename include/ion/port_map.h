#ifndef ION_PORT_MAP_H
#define ION_PORT_MAP_H

#include <string>

#include <Halide.h>

#include "ion/util.h"

namespace ion {

/**
 * PortMap is used to assign actual value to the port input.
 */
class PortMap {

    using key_t = std::tuple<std::string, std::string>;

    struct key_hash : public std::unary_function<key_t, std::size_t>
    {
        std::size_t operator()(const key_t& k) const
        {
            return std::hash<std::string>{}(std::get<0>(k)) ^ std::hash<std::string>{}(std::get<1>(k));
        }
    };

    struct key_equal : public std::binary_function<key_t, key_t, bool>
    {
        bool operator()(const key_t& v0, const key_t& v1) const
        {
            return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1));
        }
    };

public:

    template<typename T>
    void set(const Halide::Param<T>& p, T v) {
        param_expr_[p.name()] = p;
        param_map_.set(p, v);
    }

    template<typename T>
    void set(const Halide::ImageParam& p, Halide::Buffer<T> &buf) {
        param_func_[p.name()] = p;
        param_map_.set(p, buf);
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
    void set(Port p, Halide::Buffer<T> &buf) {
        if (p.bound()) {
            // This is just an output.
            output_buffer_[std::make_tuple(p.node_id(), p.key())] = { buf };
        } else {
            param_func_[p.key()] = p.func();
            p.set_to_param_map(param_map_, buf);
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
                output_buffer_[std::make_tuple(p.node_id(), p.key())].push_back(bufs[i]);
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

    std::unordered_map<key_t, std::vector<Halide::Buffer<>>, key_hash, key_equal> get_output_buffer() const {
        return output_buffer_;
    }

    Halide::ParamMap get_param_map() const {
        return param_map_;
    }

 private:

    std::unordered_map<std::string, Halide::Expr> param_expr_;
    std::unordered_map<std::string, Halide::Func> param_func_;
    std::unordered_map<key_t, std::vector<Halide::Buffer<>>, key_hash, key_equal> output_buffer_;
    Halide::ParamMap param_map_;
};


} // namespace ion

#endif // ION_PORT_MAP_H
