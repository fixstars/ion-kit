#ifndef ION_BUILDING_BLOCK_H
#define ION_BUILDING_BLOCK_H

#include <vector>
#include <Halide.h>

// #include "generator.h"

namespace ion {

template<typename T>
class  BuildingBlock : public Halide::Generator<T> {
};

template<typename T>
using GeneratorParam = Halide::GeneratorParam<T>;

template<typename T>
using GeneratorInput = Halide::GeneratorInput<T>;

template<typename T>
using GeneratorOutput = Halide::GeneratorOutput<T>;

// template<typename T>
// using Param = Halide::GeneratorParam<T>;

template<typename T>
using Input = Halide::GeneratorInput<T>;

template<typename T>
using Output = Halide::GeneratorOutput<T>;

} // namespace ion

#define ION_REGISTER_BUILDING_BLOCK(...) HALIDE_REGISTER_GENERATOR(__VA_ARGS__)

#endif
