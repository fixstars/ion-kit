#ifndef ION_SERIALIZER_H
#define ION_SERIALIZER_H

#include "ion/node.h"
#include "ion/param.h"
#include "ion/port.h"

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
    j["key_"] = v.key();
    j["val_"] = v.val();
}

static void from_json(const json& j, ion::Param& v) {
    v.key() = j["key_"].get<std::string>();
    v.val() = j["val_"].get<std::string>();
}
};

template<>
class adl_serializer<ion::Port> {
 public:
     static void to_json(json& j, const ion::Port& v) {
         j["name_"] = v.name();
         j["type_"] = static_cast<halide_type_t>(v.type());
         j["dimensions_"] = v.dimensions();
         j["index_"] = v.index();
         j["node_id_"] = v.node_id();
     }

     static void from_json(const json& j, ion::Port& v) {
         v.name() = j["name_"].get<std::string>();
         v.type() = j["type_"].get<halide_type_t>();
         v.dimensions() = j["dimensions_"];
         v.index() = j["index_"];
         v.node_id() = j["node_id_"].get<std::string>();
         if (v.node_id().empty()) {
             if (v.index() == -1) {
                 v.params() = { Halide::Internal::Parameter(v.type(), v.dimensions() != 0, v.dimensions(), v.name()) };
             } else {
                 v.params() = std::vector<Halide::Internal::Parameter>(v.index()+1, Halide::Internal::Parameter{v.type(), v.dimensions() != 0, v.dimensions(), v.name()});
             }
         }
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
