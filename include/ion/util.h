#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <string>

namespace ion {

class Port;

std::string argument_name(const std::string& node_id, const std::string& name, int32_t index);

std::string array_name(const std::string& port_name, size_t i);

} // namespace ion

#endif
