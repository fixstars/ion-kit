#ifndef ION_BLOCK_H
#define ION_BLOCK_H

#include "generator.h"

namespace ion {

namespace Internal {

using BuildingBlockBase = ::ion::Internal::GeneratorBase;

}  // namespace Internal

template<typename T>
using BuildingBlock = Generator<T>;

} // namespace ion

#define ION_REGISTER_BUILDING_BLOCK(...) ION_REGISTER_GENERATOR(__VA_ARGS__)

#endif
