#include <Halide.h>

#include "ion/type.h"
#include "ion/target.h"

namespace ion {

Target get_host_target() {
    return Halide::get_host_target();
}

Target get_target_from_environment() {
    return Halide::get_target_from_environment();
}

} // ion

