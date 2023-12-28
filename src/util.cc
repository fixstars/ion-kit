#include <algorithm>

#include "ion/port.h"
#include "ion/util.h"

namespace ion {

std::string argument_name(const std::string& pred_id, const std::string& pred_name, const std::string& succ_id, const std::string& succ_name, int32_t index) {
    if (index == -1) {
        index = 0;
    }

    std::string s = "_" + pred_id + "_" + pred_name + "_" + succ_id + "_" + succ_name + "_" + std::to_string(index);;
    std::replace(s.begin(), s.end(), '-', '_');

    return s;
}

std::string array_name(const std::string& port_name, size_t i) {
    return port_name + "_" + std::to_string(i);
}

} // namespace ion

