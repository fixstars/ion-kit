#ifndef ION_LOWER_H
#define ION_LOWER_H

#include <vector>

#include "ion/builder.h"

namespace Halide {
class Argument;
class Pipeline;
}  // namespace Halide

namespace ion {
class Node;

void determine_and_validate(std::vector<Node> &nodes);

std::vector<const void *> generate_arguments_instance(const std::vector<Halide::Argument> &inferred_args, const std::vector<Node> &nodes);

Halide::Pipeline lower(Builder builder, std::vector<Node> &nodes, bool implicit_output);

}  // namespace ion

#endif  // ION_LOWER_H
