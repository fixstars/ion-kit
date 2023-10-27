#include <string>

#include <Halide.h>

#include "ion/json.hpp"
#include "ion/node.h"
#include "ion/param.h"
#include "ion/port.h"

namespace ion {
void to_json(json& j, const Node& v) {
    j["id"] = v.impl_->id;
    j["name"] = v.impl_->name;
    j["target"] = v.impl_->target.to_string();
    j["params"] = v.impl_->params;
    j["ports"] = v.impl_->ports;
}

void from_json(const json& j, Node& v) {
    v.impl_->id = j["id"].get<std::string>();
    v.impl_->name = j["name"].get<std::string>();
    v.impl_->target = Halide::Target(j["target"].get<std::string>());
    v.impl_->params = j["params"].get<std::vector<Param>>();
    v.impl_->ports = j["ports"].get<std::vector<Port>>();
}

} // namespace ion
