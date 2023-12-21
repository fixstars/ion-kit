#ifndef ION_SERIALIZER_H
#define ION_SERIALIZER_H

#include "ion/node.h"
#include "ion/param.h"
#include "ion/port.h"
#include "ion/util.h"

#include "json/json.hpp"

namespace nlohmann {
template <>
class adl_serializer<halide_type_t> {
public:
    static void to_json(json& j, const halide_type_t& v) {
        j["code"] = v.code;
        j["bits"] = v.bits;
        j["lanes"] = v.lanes;
    }

    static void from_json(const json& j, halide_type_t& v) {
        v.code = j["code"];
        v.bits = j["bits"];
        v.lanes = j["lanes"];
    }
};

template <>
class adl_serializer<ion::Param> {
public:
static void to_json(json& j, const ion::Param& v) {
    j["key"] = v.key();
    j["val"] = v.val();
}

static void from_json(const json& j, ion::Param& v) {
    v.key() = j["key"].get<std::string>();
    v.val() = j["val"].get<std::string>();
}
};

template<>
class adl_serializer<ion::Port> {
 public:
     static void to_json(json& j, const ion::Port& v) {
         j["name"] = v.name();
         j["type"] = static_cast<halide_type_t>(v.type());
         j["dimensions"] = v.dimensions();
         j["index"] = v.index();
         j["node_id"] = v.node_id();
         j["array_size"] = v.params().size();
         j["impl"] = reinterpret_cast<uintptr_t>(v.impl_.get());
     }

     static void from_json(const json& j, ion::Port& v) {
         v = ion::Port(ion::Port::find_impl(j["impl"].get<uintptr_t>()));
         v.name() = j["name"].get<std::string>();
         v.type() = j["type"].get<halide_type_t>();
         v.dimensions() = j["dimensions"];
         v.node_id() = j["node_id"].get<std::string>();
         v.params() = std::vector<Halide::Internal::Parameter>(
             j["array_size"],
             Halide::Internal::Parameter(v.type(), v.dimensions() != 0, v.dimensions(), ion::argument_name(v.node_id(), v.name()))
         );
         v.index() = j["index"];
     }
};

template <>
class adl_serializer<ion::Node> {
 public:
     static void to_json(json& j, const ion::Node& v) {
         j["id"] = v.id();
         j["name"] = v.name();
         j["target"] = v.target().to_string();
         j["params"] = v.params();
         j["ports"] = v.ports();
     }

     static void from_json(const json& j, ion::Node& v) {
         v.id() = j["id"].get<std::string>();
         v.name() = j["name"].get<std::string>();
         v.target() = Halide::Target(j["target"].get<std::string>());
         v.params() = j["params"].get<std::vector<ion::Param>>();
         v.ports() = j["ports"].get<std::vector<ion::Port>>();
     }
};
}

#endif
