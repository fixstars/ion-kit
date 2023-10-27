#include <Halide.h>

#include "ion/json.hpp"
#include "ion/port.h"

namespace ion {

#define DEFINE_SET_TO_PARAM(TYPE, MEMBER_V)                                 \
template<>                                                                  \
void ParamContainer::set_to_param_map<TYPE>(Halide::ParamMap& pm, TYPE v) { \
    if (dim_ != 0) {                                                        \
        throw std::runtime_error("Invalid port type");                      \
    }                                                                       \
    pm.set(MEMBER_V, v);                                                    \
}
DEFINE_SET_TO_PARAM(bool, b_);
DEFINE_SET_TO_PARAM(int8_t, i8_);
DEFINE_SET_TO_PARAM(int16_t, i16_);
DEFINE_SET_TO_PARAM(int32_t, i32_);
DEFINE_SET_TO_PARAM(int64_t, i64_);
DEFINE_SET_TO_PARAM(uint8_t, u8_);
DEFINE_SET_TO_PARAM(uint16_t, u16_);
DEFINE_SET_TO_PARAM(uint32_t, u32_);
DEFINE_SET_TO_PARAM(uint64_t, u64_);
DEFINE_SET_TO_PARAM(float, f32_);
DEFINE_SET_TO_PARAM(double, f64_);
#undef DEFINE_SET_TO_PARAM

void to_json(json& j, const Port& v) {
    j["key_"] = v.key_;
    j["type_"] = static_cast<halide_type_t>(v.type_);
    j["dimensions_"] = v.dimensions_;
    j["node_id_"] = v.node_id_;
}

void from_json(const json& j, Port& v) {
    v.key_ = j["key_"].get<std::string>();
    v.type_ = j["type_"].get<halide_type_t>();
    v.dimensions_ = j["dimensions_"];
    v.node_id_ = j["node_id_"].get<std::string>();
    if (v.node_id_.empty()) {
        v.param_info_ = std::make_shared<ParamContainer>(v.key_, v.type_, v.dimensions_);
    } else {
        v.param_info_ = nullptr;
    }
}

} // namespace ion
