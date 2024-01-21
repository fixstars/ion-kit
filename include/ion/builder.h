#ifndef ION_BUILDER_H
#define ION_BUILDER_H

#include <string>
#include <vector>
#include <unordered_map>

#include <Halide.h>

#include "def.h"
#include "buffer.h"
#include "node.h"
#include "target.h"
#include "port_map.h"

namespace ion {

class DynamicModule;

/**
 * Builder class is used to build graph, compile, run, save and load it.
 */
class Builder {
public:
    /**
     * CompileOption class holds option field for compilation.
     */
    struct CompileOption {
        std::string output_directory;
    };

    Builder();

    ~Builder();

    /**
     * Adding new node to the graph.
     * @arg k: The key of the node which should be matched with second argument of ION_REGISTER_BUILDING_BLOCK().
     */
    Node add(const std::string& k);

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

    void run();

    void run(ion::PortMap& ports);


    /**
     * Retrieve metadata of Building Block in json format.
     */
    std::string bb_metadata(void);

    /**
     * Get the node list.
     */
    const std::vector<Node>& nodes() const { return nodes_; }
    std::vector<Node>& nodes() { return nodes_; }


    /**
     * Register disposer hook which will be called from Builder destructor.
     * This is available only for JIT mode.
     */
    void register_disposer(const std::string& bb_id, const std::string& disposer_symbol);

private:

    Halide::Pipeline build(bool implicit_output = false);

    void determine_and_validate();

    std::vector<Halide::Argument> get_arguments_stub() const;
    std::vector<const void*> get_arguments_instance() const;

    void set_jit_externs(const std::map<std::string, Halide::JITExtern> &externs) {
        pipeline_.set_jit_externs(externs);
    }

    Halide::Target target_;
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::shared_ptr<DynamicModule>> bb_modules_;
    Halide::Pipeline pipeline_;
    Halide::Callable callable_;
    std::unique_ptr<Halide::JITUserContext> jit_ctx_;
    Halide::JITUserContext* jit_ctx_ptr_;
    std::vector<const void*> args_;
    std::vector<std::tuple<std::string, std::function<void(const char*)>>> disposers_;
};

} // namespace ion

#endif // ION_BUILDER_H
