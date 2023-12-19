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
struct hash<tuple<string, string, int>>
{
    std::size_t operator()(const tuple<string, string, int>& k) const noexcept
    {
        return std::hash<std::string>{}(std::get<0>(k)) ^ std::hash<std::string>{}(std::get<1>(k)) ^ std::hash<int>{}(std::get<2>(k));
    }
};

template<>
struct equal_to<tuple<string, string, int>>
{
    bool operator()(const tuple<string, string, int>& v0, const tuple<string, string, int>& v1) const
    {
        return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1) && std::get<2>(v0) == std::get<2>(v1));
    }
};

} // std


namespace ion {

/**
 * PortMap is used to assign actual value to the port input.
 */
class PortMap {

public:

    PortMap() : dirty_(false)
    {
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
        auto param = p.param();
        param.set_scalar(v);
        param_[(p.key())] = param;
        dirty_ = true;
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
        if (p.is_bound()) {
            // This is just an output.
            output_buffer_[std::make_tuple(p.node_id(), p.key(), p.index())] = { buf };
        } else {
            auto param = p.param();
            param.set_buffer(buf);
            param_[p.key()] = param;
        }

        dirty_ = true;
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
    // template<typename T>
    // void set(Port p, const std::vector<Halide::Buffer<T>> &bufs) {
    //     if (p.is_bound()) {
    //         // This is just an output.
    //         for (size_t i=0; i<bufs.size(); ++i) {
    //             auto buf = bufs[i];
    //             output_buffer_[std::make_tuple(p.node_id(), p.key(), p.index())].push_back(buf);
    //         }
    //     } else {
    //         throw std::invalid_argument(
    //             "Unbounded port (" + p.key() + ") corresponding to an array of Inputs is not supported");
    //     }

    //     dirty_ = true;
    // }

    bool is_mapped(const std::string& k) const {
        return param_.count(k);
    }

    Halide::Internal::Parameter get_param(const std::string& k) const {
        return param_.at(k);
    }

    std::unordered_map<std::tuple<std::string, std::string, int>, std::vector<Halide::Buffer<>>> get_output_buffer() const {
        return output_buffer_;
    }

    std::vector<Halide::Argument> get_arguments_stub() const {
        std::vector<Halide::Argument> args;
        for (auto kv : param_) {
            auto kind = kv.second.is_buffer() ? Halide::Argument::InputBuffer : Halide::Argument::InputScalar;
            args.push_back(Halide::Argument(kv.first,  kind, kv.second.type(), kv.second.dimensions(), Halide::ArgumentEstimates()));
        }
        return args;
    }

    std::vector<const void*> get_arguments_instance() const {
        std::vector<const void*> args;
        for (auto kv : param_) {
            if (kv.second.is_buffer()) {
                args.push_back(kv.second.raw_buffer());
            } else {
                args.push_back(kv.second.scalar_address());
            }
        }
        return args;
    }

    void updated() {
        dirty_ = false;
    }

    bool dirty() const {
        return dirty_;
    }

 private:
    bool dirty_;
    std::unordered_map<std::string, Halide::Internal::Parameter> param_;
    std::unordered_map<std::tuple<std::string, std::string, int>, std::vector<Halide::Buffer<>>> output_buffer_;
};

} // namespace ion

#endif // ION_PORT_MAP_H
