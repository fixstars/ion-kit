#ifndef ION_METADATA_H
#define ION_METADATA_H

#include <string>
#include <vector>

#include <Halide.h>

#include "ion/json.hpp"

namespace ion {

using json = nlohmann::json;

struct PortMD {
    friend void to_json(json&, const PortMD&);
    friend void from_json(const json&, PortMD&);

    std::string name;
    Halide::Type type;
    int dimension;

    PortMD() {}
    PortMD(const std::string& n, Halide::Type t, int d);
};

struct ParamMD {
    friend void to_json(json&, const ParamMD&);
    friend void from_json(const json&, ParamMD&);

    std::string name;
    std::string default_value;
    std::string c_type;
    std::string type_decls;

    ParamMD() {}
    ParamMD(const std::string& n, const std::string& dv, const std::string& ct, const std::string& td);
};

struct Metadata {
    friend void to_json(json&, const Metadata&);
    friend void from_json(const json&, Metadata&);

    std::string name;
    std::vector<PortMD> inputs;
    std::vector<PortMD> outputs;
    std::vector<ParamMD> params;

    Metadata(const std::string& n);
};

} //namespace ion

#endif
