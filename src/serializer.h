#ifndef ION_SERIALIZER_H
#define ION_SERIALIZER_H

#include "ion/node.h"
#include "ion/param.h"
#include "ion/port.h"
#include "ion/util.h"

#include "json/json.hpp"

#include "log.h"

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
         j["pred_chan"] = v.impl_->pred_chan;
         j["succ_chans"] = v.impl_->succ_chans;
         j["type"] = static_cast<halide_type_t>(v.impl_->type);
         j["dimensions"] = v.impl_->dimensions;
         j["size"] = v.impl_->params.size();
         j["impl_ptr"] = reinterpret_cast<uintptr_t>(v.impl_.get());
         j["index"] = v.index_;
     }

     static void from_json(const json& j, ion::Port& v) {
         v = ion::Port(ion::Port::find_impl(j["impl_ptr"].get<uintptr_t>()));
         v.impl_->pred_chan = j["pred_chan"].get<ion::Port::Channel>();
         v.impl_->succ_chans = j["succ_chans"].get<std::set<ion::Port::Channel>>();
         v.impl_->type = j["type"].get<halide_type_t>();
         v.impl_->dimensions = j["dimensions"];
         for (auto i=0; i<j["size"]; ++i) {
             v.impl_->params[i] = Halide::Internal::Parameter(v.impl_->type, v.impl_->dimensions != 0, v.impl_->dimensions, ion::argument_name(v.pred_id(), v.pred_name(), i));
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
         auto bb(Halide::Internal::GeneratorRegistry::create(v.impl_->name, Halide::GeneratorContext(v.impl_->target)));
         if (!bb) {
             ion::log::error("BuildingBlock {} is not found", v.impl_->name);
             throw std::runtime_error("Failed to create building block");
         }
         v.impl_->arginfos = bb->arginfos();
     }
};
}

#endif
