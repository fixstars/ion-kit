#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <string>

namespace ion {

std::string argument_name(const std::string& node_id, const std::string& port_name, int32_t index = -1);

std::string array_name(const std::string& port_name, size_t i);

} // namespace ion

#endif
