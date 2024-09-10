#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <string>
namespace ion {

class Port;

std::string array_name(const std::string &port_name, size_t i);

// a string-like identifier that is typed on a tag type
template<class Tag>
struct StringID {
    using tag_type = Tag;

    // needs to be default-constructable because of use in map[] below
    StringID(std::string s)
        : _value(std::move(s)) {
    }
    StringID()
        : _value() {
    }
    // provide access to the underlying string value
    const std::string &value() const {
        return _value;
    }

    struct StringIDHash {
        // Use hash of string as hash function.
        size_t operator()(const StringID &id) const {
            return std::hash<std::string>()(id.value());
        }
    };

private:
    std::string _value;

    // will only compare against same type of id.
    friend bool operator<(const StringID &l, const StringID &r) {
        return l._value < r._value;
    }

    friend bool operator==(const StringID &l, const StringID &r) {
        return l._value == r._value;
    }

    // and let's go ahead and provide expected free functions
    friend auto to_string(const StringID &r)
        -> const std::string & {
        return r._value;
    }
};

struct node_tag {};
struct graph_tag {};
struct port_tag {};

using NodeID = StringID<node_tag>;
using GraphID = StringID<graph_tag>;
using PortID = StringID<port_tag>;

std::string argument_name(const NodeID &node_id, const PortID &portId, const std::string &name, int32_t index, const GraphID &graph_id);

}  // namespace ion

#endif
