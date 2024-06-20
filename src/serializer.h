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
struct adl_serializer<halide_type_t> {
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
struct adl_serializer<ion::Param> {
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
struct adl_serializer<ion::Port> {
     static void to_json(json& j, const ion::Port& v) {
         j["id"] = to_string(v.id());
         std::map<std::string, std::string> stringMap;
         j["pred_chan"] = std::make_tuple(to_string(std::get<0>(v.pred_chan())), std::get<1>(v.pred_chan()));
         std::set<std::tuple<std::string, std::string>> succ_chans;
         for (auto& c:v.succ_chans()){
             succ_chans.insert(std::make_tuple(to_string(std::get<0>(c)), std::get<1>(c)));
         }
         j["succ_chans"] = succ_chans;
         j["type"] = static_cast<halide_type_t>(v.type());
         j["dimensions"] = v.dimensions();
         j["size"] = v.size();
         j["index"] = v.index();
     }

     static void from_json(const json& j, ion::Port& v) {
         auto [impl, found] = ion::Port::find_impl(j["id"].get<std::string>());
         if (!found) {
             impl->pred_chan = j["pred_chan"].get<std::tuple<std::string, std::string>>();
             std::set<ion::Port::Channel> succ_chans;
             for (auto & p : j["succ_chans"]){
                 succ_chans.insert(p.get<std::tuple<std::string, std::string>>());
             }
             impl->succ_chans = succ_chans;
             impl->type = j["type"].get<halide_type_t>();
             impl->dimensions = j["dimensions"];
             for (auto i=0; i<j["size"]; ++i) {
                 impl->params[i] = Halide::Parameter(impl->type, impl->dimensions != 0, impl->dimensions,
                                                               ion::argument_name(std::get<0>(impl->pred_chan), impl->id,  std::get<1>(impl->pred_chan), i, impl->graph_id.value()));
             }
         }
         v = ion::Port(impl, j["index"]);
     }
};

template <>
struct adl_serializer<ion::Node> {
     static void to_json(json& j, const ion::Node& v) {
         j["id"] = to_string(v.id());
         j["name"] = v.name();
         j["target"] = v.target().to_string();
         j["params"] = v.params();
         j["ports"] = v.ports();
     }

     static void from_json(const json& j, ion::Node& v) {
         auto impl = std::make_shared<ion::Node::Impl>();
         impl->id = j["id"].get<std::string>();
         impl->name = j["name"].get<std::string>();
         impl->target = Halide::Target(j["target"].get<std::string>());
         impl->params = j["params"].get<std::vector<ion::Param>>();
         impl->ports = j["ports"].get<std::vector<ion::Port>>();
         auto bb(Halide::Internal::GeneratorRegistry::create(impl->name, Halide::GeneratorContext(impl->target)));
         if (!bb) {
             ion::log::error("BuildingBlock {} is not found", impl->name);
             throw std::runtime_error("Failed to create building block object");
         }
         impl->arginfos = bb->arginfos();
         v = ion::Node(impl);
     }
};
}

#endif
