#ifndef ION_GRAPH_H
#define ION_GRAPH_H

#include <vector>

#include "node.h"

namespace ion {

class Builder;

class Graph {

    struct Impl;

public:

    Graph& operator+=(const Graph& rhs);

    friend Graph operator+(const Graph& lhs, const Graph& rhs);

    // TODO: Naming
    Graph(Builder builder, const std::string& name = "");

    /**
     * Adding new node to the graph.
     * @arg n: The name of the building block which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     */
    Node add(const std::string& name);

    /**
     * Run the pipeline immediately.
     */
    void run();

    /**
     * Get the node list.
     */
    const std::vector<Node>& nodes() const;
    std::vector<Node>& nodes();

private:
    std::shared_ptr<Impl> impl_;
};

} // namespace ion

#endif // ION_GRAPH_H
