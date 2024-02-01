#ifndef ION_LOWER_H
#define ION_LOWER_H

#include <vector>

namespace Halide {
class Pipeline;
}

namespace ion {

class Builder;
class Node;

void determine_and_validate(std::vector<Node>& nodes);

void topological_sort(std::vector<Node>& nodes);

Halide::Pipeline lower(const Builder *builder_ptr, std::vector<Node>& nodes, bool implicit_output);

} // namespace ion

#endif // ION_LOWER_H
