#include <sstream>

#include "ion/building_block.h"
#include "ion/port.h"

#include "json/json.hpp"
#include "metadata.h"
#include "serializer.h"

namespace {

std::string unquote(const std::string &s) {
    if (s.size() < 2) {
        return s;
    }

    size_t epos = s.size() - 1;

    if (s[0] == '"' && s[epos] == '"') {
        return s.substr(1, s.size() - 2);
    } else {
        return s;
    }
}

}  // namespace

namespace ion {

using json = nlohmann::json;

PortMD::PortMD(const std::string &n, const std::vector<Halide::Type> &ts, int d)
    : name(n), types(ts), dimension(d) {
}

void to_json(json &j, const PortMD &v) {
    j["name"] = v.name;
    std::vector<halide_type_t> types;
    for (auto t : v.types) {
        types.push_back(t);
    }
    j["types"] = types;
    j["dimension"] = v.dimension;
}

void from_json(const json &j, PortMD &v) {
    v.name = j["name"].get<std::string>();
    auto types = j["types"].get<std::vector<halide_type_t>>();
    for (auto t : types) {
        v.types.push_back(t);
    }
    v.dimension = j["dimension"];
}

ParamMD::ParamMD(const std::string &n, const std::string &dv, const std::string &ct, const std::string &td)
    : name(n), default_value(dv), c_type(ct), type_decls(td) {
}

void to_json(json &j, const ParamMD &v) {
    j["name"] = v.name;
    j["c_type"] = v.c_type;
    j["type_decls"] = v.type_decls;

    if (v.c_type.find("uint8_t") == 0) {
        j["default_value"] = std::to_string(*reinterpret_cast<const int8_t *>(v.default_value.c_str()));
    } else if (v.c_type.find("uint8_t") == 0) {
        j["default_value"] = std::to_string(*reinterpret_cast<const uint8_t *>(v.default_value.c_str()));
    } else {
        j["default_value"] = v.default_value;
    }
}

void from_json(const json &j, ParamMD &v) {
    v.name = j["name"].get<std::string>();
    v.c_type = j["c_type"];
    v.type_decls = j["type_decls"];

    if (v.c_type.find("uint8_t") == 0) {
        std::ostringstream ss;
        ss << static_cast<uint8_t>(std::stoi(j["default_value"].get<std::string>()));
        v.default_value = ss.str();
    } else if (v.c_type.find("uint8_t") == 0) {
        std::ostringstream ss;
        ss << static_cast<uint8_t>(std::stoi(j["default_value"].get<std::string>()));
        v.default_value = ss.str();
    } else {
        v.default_value = j["default_value"].get<std::string>();
    }
}

Metadata::Metadata(const std::string &n)
    : name(n) {
    auto bb = Halide::Internal::GeneratorRegistry::create(n, Halide::GeneratorContext(Halide::get_host_target()));

    for (auto arginfo : bb->arginfos()) {
        if (arginfo.dir == Halide::Internal::ArgInfoDirection::Input) {
            inputs.push_back(PortMD(arginfo.name, arginfo.types, arginfo.dimensions));
        } else if (arginfo.dir == Halide::Internal::ArgInfoDirection::Output) {
            outputs.push_back(PortMD(arginfo.name, arginfo.types, arginfo.dimensions));
        } else {
            throw std::runtime_error("fixme");
        }
    }

    // TBD: Handle parameter
    // for (auto info : bb->param_info().building_block_params()) {
    //     auto dv = info->is_synthetic_param() ? "" : unquote(info->get_default_value());
    //     auto ctv = info->is_synthetic_param() ? "" : info->get_c_type();
    //     auto tdv = info->is_synthetic_param() ? "" : info->get_type_decls();
    //     params.push_back(ParamMD(info->name(), dv, ctv, tdv));
    // }
}

void to_json(json &j, const Metadata &v) {
    j["name"] = v.name;
    j["inputs"] = v.inputs;
    j["outputs"] = v.outputs;
    j["params"] = v.params;
}

void from_json(const json &j, Metadata &v) {
    v.name = j["name"].get<std::string>();
    v.inputs = j["inputs"].get<std::vector<PortMD>>();
    v.outputs = j["outputs"].get<std::vector<PortMD>>();
    v.params = j["params"].get<std::vector<ParamMD>>();
}

}  // namespace ion
