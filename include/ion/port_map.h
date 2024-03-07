#ifndef ION_PORT_MAP_H
#define ION_PORT_MAP_H

#include <string>

#include <Halide.h>

namespace ion {

/**
 * PortMap is used to assign actual value to the port input.
 */
class PortMap {

public:
    template<typename T>
    [[deprecated("Port::bind can be used instead of PortMap.")]]
    void set(Port port, T v) {
        auto& buf(scalar_buffer_[argument_name(port.pred_id(), port.pred_name(), port.index(), port.graph_id())]);
        buf.resize(sizeof(v));
        std::memcpy(buf.data(), &v, sizeof(v));
        port.bind(reinterpret_cast<T*>(buf.data()));
    }

    template<typename T>
    [[deprecated("Port::bind can be used instead of PortMap.")]]
    void set(Port port, Halide::Buffer<T>& buf) {
        port.bind(buf);
    }

    template<typename T>
    [[deprecated("Port::bind can be used instead of PortMap.")]]
    void set(Port port, const std::vector<Halide::Buffer<T>> &bufs) {
        port.bind(bufs);
    }

private:
    std::unordered_map<std::string, std::vector<uint8_t>> scalar_buffer_;
};

} // namespace ion

#endif // ION_PORT_MAP_H
