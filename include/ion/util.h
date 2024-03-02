#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <string>
#include <typeinfo>
namespace ion {

class Port;

std::string argument_name(const std::string& node_id, const std::string& name, int32_t index, const std::string& graph_id);



std::string array_name(const std::string& port_name, size_t i);

 struct portal_tag {
    };
    struct cake_tag {
    };

// a string-like identifier that is typed on a tag type
    template<class Tag>
    struct string_id {
        using tag_type = Tag;

        // needs to be default-constuctable because of use in map[] below
        string_id(std::string s) : _value(std::move(s)) {}

        string_id() : _value() {}

        // provide access to the underlying string value
        const std::string &value() const { return _value; }

    private:
        std::string _value;

        // will only compare against same type of id.
        friend bool operator<(const string_id &l, const string_id &r) {
            return l._value < r._value;
        }

        friend bool operator==(const string_id &l, const string_id &r) {
            return l._value == r._value;
        }

        // and let's go ahead and provide expected free functions
        friend
        auto to_string(const string_id &r)
        -> const std::string & {
            return r._value;
        }

    };

    struct node_tag {};
    struct graph_tag {};
    struct port_tag {};

    using NodeID = string_id<node_tag>;
    using GraphID = string_id<graph_tag>;
    using PortID = string_id<port_tag>;


} // namespace ion

#endif
