#include <algorithm>

#include "ion/util.h"

namespace ion {

std::string argument_name(const std::string& node_id, const std::string& port_name, int32_t index) {
    if (index == -1) {
        index = 0;
    }

    std::string s = "_" + node_id + "_" + port_name + "_" + std::to_string(index);;
    std::replace(s.begin(), s.end(), '-', '_');

    return s;
}

std::string array_name(const std::string& port_name, size_t i) {
    return port_name + "_" + std::to_string(i);
}

} // namespace ion

