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
    void set(Port port, T v) {
        auto& buf(scalar_buffer_[argument_name(port.pred_id(), port.pred_name(), port.index())]);
        buf.resize(sizeof(v));
        std::memcpy(buf.data(), &v, sizeof(v));
        port.bind(reinterpret_cast<T*>(buf.data()));
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
    void set(Port port, Halide::Buffer<T>& buf) {
        if (port.has_pred()) {
            // This is just an output.
            output_buffer_[std::make_tuple(port.pred_id(), port.pred_name(), port.index())] = { buf };
        } else {
            port.bind(buf);
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
    template<typename T>
    void set(Port port, const std::vector<Halide::Buffer<T>> &bufs) {
        if (port.has_pred()) {
            // This is just an output.
            for (auto buf : bufs) {
                output_buffer_[std::make_tuple(port.pred_id(), port.pred_name(), port.index())].push_back(buf);
            }
        } else {
            port.bind(bufs);
        }

        dirty_ = true;
    }

    std::unordered_map<std::tuple<std::string, std::string, int>, std::vector<Halide::Buffer<>>> get_output_buffer() const {
        return output_buffer_;
    }

    void updated() {
        dirty_ = false;
    }

    bool dirty() const {
        return dirty_;
    }

 private:
    bool dirty_;
    std::unordered_map<std::tuple<std::string, std::string, int>, std::vector<Halide::Buffer<>>> output_buffer_;

    std::unordered_map<std::string, std::vector<uint8_t>> scalar_buffer_;
};

} // namespace ion

#endif // ION_PORT_MAP_H
