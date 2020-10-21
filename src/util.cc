#include "ion/util.h"

#include <sstream>

namespace ion {

std::string output_name(const std::string& node_id, const std::string& port_key) {
    std::stringstream ss;

    // Make sure to start from legal character;
    ss << "_";

    // Rpleace '-' by '_'
    for (auto c : node_id + "_" + port_key) {
        if (c == '-') {
            ss << '_';
        } else {
            ss << c;
        }
    }

    return ss.str();
}

std::string array_name(const std::string& port_key, size_t i) {
    return port_key + "_" + std::to_string(i);
}

} // namespace ion

