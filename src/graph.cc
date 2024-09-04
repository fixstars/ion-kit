#include "ion/graph.h"
#include "ion/builder.h"

#include "log.h"
#include "lower.h"
#include "uuid/sole.hpp"
namespace ion {

struct Graph::Impl {
    Builder &builder;
    std::string name;
    GraphID id;
    std::vector<Node> nodes;
    // Cacheable
    Halide::Pipeline pipeline;
    Halide::Callable callable;
    std::unique_ptr<Halide::JITUserContext> jit_ctx;
    Halide::JITUserContext *jit_ctx_ptr;
    std::vector<const void *> args;

    Impl(Builder &b, const std::string &n)
        : id(sole::uuid4().str()), builder(b), name(n), jit_ctx(new Halide::JITUserContext), jit_ctx_ptr(jit_ctx.get()) {
    }
};

Graph::Graph() {
}

Graph::Graph(Builder &builder, const std::string &name)
    : impl_(new Impl(builder, name)) {
}

Graph &Graph::operator+=(const Graph &rhs) {
    impl_->nodes.insert(impl_->nodes.end(), rhs.impl_->nodes.begin(), rhs.impl_->nodes.end());
    return *this;
}

Graph operator+(const Graph &lhs, const Graph &rhs) {
    Graph g(lhs.impl_->builder);
    g += lhs;
    g += rhs;
    return g;
}

Node Graph::add(const std::string &name) {
    auto n = impl_->builder.add(name, impl_->id);
    impl_->nodes.push_back(n);
    return n;
}

Graph &Graph::set_jit_context(Halide::JITUserContext *user_context_ptr) {
    impl_->jit_ctx_ptr = user_context_ptr;
    return *this;
}

void Graph::run() {
    if (!impl_->pipeline.defined()) {
        impl_->pipeline = lower(impl_->builder, impl_->nodes, false);
        if (!impl_->pipeline.defined()) {
            log::warn("This pipeline doesn't produce any outputs. Please bind a buffer with output port.");
            return;
        }
    }

    if (!impl_->callable.defined()) {

        impl_->pipeline.set_jit_externs(impl_->builder.jit_externs());

        auto inferred_args = impl_->pipeline.infer_arguments();

        impl_->callable = impl_->pipeline.compile_to_callable(inferred_args, impl_->builder.target());

        impl_->args.clear();
        impl_->args.push_back(&impl_->jit_ctx_ptr);

        const auto &args(generate_arguments_instance(inferred_args, impl_->nodes));
        impl_->args.insert(impl_->args.end(), args.begin(), args.end());
    }

    impl_->callable.call_argv_fast(impl_->args.size(), impl_->args.data());
}

const std::vector<Node> &Graph::nodes() const {
    return impl_->nodes;
}

std::vector<Node> &Graph::nodes() {
    return impl_->nodes;
}

}  // namespace ion
