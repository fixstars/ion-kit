#ifndef ION_GRAPH_H
#define ION_GRAPH_H

#include <vector>

#include "node.h"

namespace ion {

class Graph {
public:

    /**
     * Adding new node to the graph.
     * @arg k: The key of the node which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     */
    Node add(const std::string& k);

    void run();

private:
    std::vector<Node> nodes_;
};

} // namespace ion

#endif // ION_GRAPH_H
