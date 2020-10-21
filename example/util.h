#ifndef ION_KIT_EXAMPLE_UTIL_H
#define ION_KIT_EXAMPLE_UTIL_H

#include <string>

namespace ion {

std::string get_target_from_cmdline(int argc, char *argv[]) {
    if (argc < 2) {
        return "host";
    }

    return argv[1];
}

}  // namespace ion

#endif  // ION_KIT_EXAMPLE_UTIL_H
