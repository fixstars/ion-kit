#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <string>

namespace ion {

std::string output_name(const std::string& node_id, const std::string& port_key);

std::string array_name(const std::string& port_key, size_t i);

} // namespace ion

#endif
