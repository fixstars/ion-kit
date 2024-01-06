#ifndef ION_KIT_EXAMPLE_UTIL_H
#define ION_KIT_EXAMPLE_UTIL_H

#include <ion/ion.h>

namespace ion {

Target get_target_from_cmdline(int argc, char *argv[]) {
    if (argc < 2) {
        return Target("host");
    }

    return Target(argv[1]);
}

}  // namespace ion

#endif  // ION_KIT_EXAMPLE_UTIL_H
