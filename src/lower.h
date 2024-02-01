#ifndef ION_LOWER_H
#define ION_LOWER_H

#include <vector>

#include "ion/builder.h"

namespace Halide {
class Pipeline;
}

namespace ion {

class Node;

void determine_and_validate(std::vector<Node>& nodes);

void topological_sort(std::vector<Node>& nodes);

std::vector<Halide::Argument> get_arguments_stub(const std::vector<Node>& nodes);

std::vector<const void*> get_arguments_instance(const std::vector<Node>& nodes);

Halide::Pipeline lower(Builder builder, std::vector<Node>& nodes, bool implicit_output);

} // namespace ion

#endif // ION_LOWER_H
