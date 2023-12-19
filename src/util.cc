#include <algorithm>

#include "ion/util.h"

namespace ion {

std::string argument_name(const std::string& node_id, const std::string& port_key) {
    std::string s("_" + node_id + "_" + port_key);
    std::replace(s.begin(), s.end(), '-', '_');
    return s;
}

std::string array_name(const std::string& port_key, size_t i) {
    return port_key + "_" + std::to_string(i);
}

} // namespace ion

