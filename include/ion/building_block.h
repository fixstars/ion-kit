#ifndef ION_BUILDING_BLOCK_H
#define ION_BUILDING_BLOCK_H

#include <vector>

#include <Halide.h>

#include "builder.h"

namespace ion {

template<typename T>
using BuildingBlockParam = Halide::GeneratorParam<T>;

template<typename T>
using BuildingBlockInput = Halide::GeneratorInput<T>;

template<typename T>
using BuildingBlockOutput = Halide::GeneratorOutput<T>;

template<typename T>
using Input = Halide::GeneratorInput<T>;

template<typename T>
using Output = Halide::GeneratorOutput<T>;

template<typename T>
class BuildingBlock : public Halide::Generator<T> {

    BuildingBlockParam<uint64_t> builder_ptr{"builder_ptr", 0};
    BuildingBlockParam<std::string> bb_id{"bb_id", ""};

 protected:

     template<typename... Ts>
     void register_disposer(const std::string& n) {
         auto bb(reinterpret_cast<Builder*>(static_cast<uint64_t>(builder_ptr)));
         if (bb) {
             bb->register_disposer(bb_id, n);
         }
     }

     ion::Buffer<uint8_t> get_id() {
         std::string bb_id_s(bb_id);
         Buffer<uint8_t> buf(static_cast<int>(bb_id_s.size() + 1));
         buf.fill(0);
         std::memcpy(buf.data(), bb_id_s.c_str(), bb_id_s.size());
         return buf;
     }
};

} // namespace ion

#define ION_REGISTER_BUILDING_BLOCK(...) HALIDE_REGISTER_GENERATOR(__VA_ARGS__)

#endif
