#ifndef ION_TYPE_H
#define ION_TYPE_H

#include <Halide.h>

namespace ion {

using Type = Halide::Type;

template<typename T>
Type type_of() {
    return Halide::type_of<T>();
}

} // ion

#endif // ION_TYPE_H
