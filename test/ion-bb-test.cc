#include "test-bb.h"
#include "test-rt.h"


extern "C" void register_externs(std::map<std::string, Halide::JITExtern>& externs) {
    externs.insert({"consume", Halide::JITExtern(consume)});
    externs.insert({"branch", Halide::JITExtern(branch)});
    externs.insert({"inc", Halide::JITExtern(inc)});
}
