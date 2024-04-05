#ifndef ION_BUILDER_H
#define ION_BUILDER_H

#include <string>
#include <vector>
#include <unordered_map>

#include <Halide.h>

#include "def.h"
#include "buffer.h"
#include "graph.h"
#include "node.h"
#include "target.h"

namespace ion {

using ArgInfo = Halide::Internal::AbstractGenerator::ArgInfo;

class DynamicModule;

/**
 * Builder class is used to build graph, compile, run, save and load it.
 */
class Builder {
public:

    struct Impl;

    /**
     * CompileOption class holds option field for compilation.
     */
    struct CompileOption {
        std::string output_directory;
    };

    Builder();
    ~Builder();

    /**
     * Adding new node to the builder.
     * @arg k: The key of the node which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     */
    Node add(const std::string& name);

    /**
     * Adding new node to the specific graph.
     * @arg k: The key of the node which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     * @arg id: graph unique identifier
     */
    Node add(const std::string& name, const GraphID& graph_id);

    /**
     * Adding new node to the graph.
     * @arg k: The key of the node which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     */
    Graph add_graph(const std::string& name);

    /**
     * Set the target of the pipeline built with this builder.
     * @arg target: The target ofject which consists of OS, Architecture, and sets of Features.
     * See https://halide-lang.org/docs/struct_halide_1_1_target.html for more details.
     */
    Builder& set_target(const Target& target);

    /**
     * Load bb module dynamically and enable it to compile your pipeline.
     * @arg module_path: DSO path on your filesystem.
     * @note This API is expected to be used from external process.
     * This information is not stored in graph definition exported by Builder::save because it is not portable.
     */
    Builder& with_bb_module(const std::string& path);

    /**
     * Save the pipeline as a file in JSON format.
     * @arg file_name: The file path to be written.
     */
    void save(const std::string& file_name);

    /**
     * Load the pipeline from a file which is written by Builder::save.
     * @arg file_name: The file path to be read.
     */
    void load(const std::string& file_name);

    /**
     * Compile the pipeline into static library and header.
     * @arg function_name: The symbol name of the entry point in the static library.
     * This name is also used as prefix of the static library and header.
     */
    void compile(const std::string& function_name, const CompileOption& option = CompileOption{});

    /**
     * Run the pipeline immediately.
     * @arg pm: This remains just for backward compatibility. Port::bind can be used instead of PortMap.
     * This argument will be removed in coming major release.
     */
    void run();

    /**
     * Retrieve names of BBs
     */
    std::vector<std::string> bb_names(void);

    /**
     * Retrieve arginfo of specific bb
     */
    std::vector<ArgInfo> bb_arginfos(const std::string& name);

    /**
     * Retrieve metadata of Building Block in json format.
     */
    std::string bb_metadata(void);


    /**
     * Get target
     */
    Target target() const;

    /**
     * Get the node list.
     */
    const std::vector<Node>& nodes() const;
    std::vector<Node>& nodes();

    /**
     * Get registered externs
     */
    const std::map<std::string, Halide::JITExtern>& jit_externs() const;

    /**
     * Register disposer hook which will be called from Builder destructor.
     * This is available only for JIT mode.
     */
    static void register_disposer(Impl* impl, const std::string& bb_id, const std::string& disposer_symbol);

    /**
     * Retrieve impl pointer for lowering
     */
    const Impl *impl_ptr() const;

private:

    std::shared_ptr<Impl> impl_;
};

} // namespace ion

#endif // ION_BUILDER_H
