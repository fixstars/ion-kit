#ifndef ION_TARGET_H
#define ION_TARGET_H

#include <Halide.h>

namespace ion {

using Target = Halide::Target;

Target get_host_target();

Target get_target_from_environment();

} // ion

#endif // ION_TARGET_H
