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
         j["pred_id"] = v.impl_->pred_id;
         j["pred_name"] = v.impl_->pred_name;
         j["succ_id"] = v.impl_->succ_id;
         j["succ_name"] = v.impl_->succ_name;
         j["type"] = static_cast<halide_type_t>(v.impl_->type);
         j["dimensions"] = v.impl_->dimensions;
         j["size"] = v.impl_->params.size();
         j["impl_ptr"] = reinterpret_cast<uintptr_t>(v.impl_.get());
         j["index"] = v.index_;
     }

     static void from_json(const json& j, ion::Port& v) {
         v = ion::Port(ion::Port::find_impl(j["impl_ptr"].get<uintptr_t>()));
         v.impl_->pred_id = j["pred_id"].get<std::string>();
         v.impl_->pred_name = j["pred_name"].get<std::string>();
         v.impl_->succ_id = j["succ_id"].get<std::string>();
         v.impl_->succ_name = j["succ_name"].get<std::string>();
         v.impl_->type = j["type"].get<halide_type_t>();
         v.impl_->dimensions = j["dimensions"];
         for (auto i=0; i<j["size"]; ++i) {
             v.impl_->params[i] = Halide::Internal::Parameter(v.impl_->type, v.impl_->dimensions != 0, v.impl_->dimensions, ion::argument_name(v.impl_->pred_id, v.impl_->pred_name, v.impl_->succ_id, v.impl_->succ_name, i));
         }
         v.index_ = j["index"];
     }
};

template <>
class adl_serializer<ion::Node> {
 public:
     static void to_json(json& j, const ion::Node& v) {
         j["id"] = v.impl_->id;
         j["name"] = v.impl_->name;
         j["target"] = v.impl_->target.to_string();
         j["params"] = v.impl_->params;
         j["ports"] = v.impl_->ports;
     }

     static void from_json(const json& j, ion::Node& v) {
         v.impl_->id = j["id"].get<std::string>();
         v.impl_->name = j["name"].get<std::string>();
         v.impl_->target = Halide::Target(j["target"].get<std::string>());
         v.impl_->params = j["params"].get<std::vector<ion::Param>>();
         v.impl_->ports = j["ports"].get<std::vector<ion::Port>>();
     }
};
}

#endif
