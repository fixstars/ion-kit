#ifndef ION_BUFFER_H
#define ION_BUFFER_H

#include <Halide.h>

namespace ion {

template<typename T = void>
using Buffer = Halide::Buffer<T>;

}  // namespace ion

#endif  // ION_BUFFER_H
