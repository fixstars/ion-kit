#include "ion/param.h"

namespace ion {

void to_json(json& j, const Param& v) {
    j["key_"] = v.key_;
    j["val_"] = v.val_;
}

void from_json(const json& j, Param& v) {
    v.key_ = j["key_"].get<std::string>();
    v.val_ = j["val_"].get<std::string>();
}

} // namespace ion
