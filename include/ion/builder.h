#ifndef ION_BUILDER_H
#define ION_BUILDER_H

#include <string>
#include <vector>
#include <unordered_map>

#include <Halide.h>

#include "block.h"
#include "json.hpp"
#include "node.h"
#include "port_map.h"

namespace ion {

using json = nlohmann::json;

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
    Builder set_target(const Halide::Target& target);

    /**
     * Load bb module dynamically and enable it to compile your pipeline.
     * @arg module_path: DSO path on your filesystem.
     * @note This API is expected to be used from external process.
     * This information is not stored in graph definition exported by Builder::save because it is not portable.
     */
    Builder with_bb_module(const std::string& path);

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
     * Compile and execute the pipeline.
     * @arg sizes: The expected output port extent.
     * @arg ports: The mapping of the port and actual value.
     * @return Execution result of the pipeline.
     * See https://halide-lang.org/docs/class_halide_1_1_realization.html for more details.
     */
    Halide::Realization run(const std::vector<int32_t>& sizes, const ion::PortMap& ports);

    /**
     * Compile and execute the pipeline.
     * @arg r: The list of output.
     * @arg ports: The mapping of the port and actual value.
     * @return Execution result of the pipeline.
     * See https://halide-lang.org/docs/class_halide_1_1_realization.html for more details.
     */
    void run(const ion::PortMap& ports);

    /**
     * Retrieve metadata of Building Block in json format.
     */
    std::string bb_metadata(void);

private:
    Halide::Pipeline build(const ion::PortMap& ports = ion::PortMap(), std::vector<Halide::Buffer<>> *outputs = nullptr);

    Halide::Target target_;
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::shared_ptr<DynamicModule>> bb_modules_;
    Halide::Pipeline pipeline_;
    std::vector<Halide::Buffer<>> outputs_;
};

} // namespace ion

#endif // ION_BUILDER_H
