#include <map>
#include <string>
#include <Halide.h>

#if defined(ION_ENABLE_BB_BASE)
#include "base/bb.h"
#include "base/rt.h"
#endif

#if defined(ION_ENABLE_BB_DNN)
#include "dnn/bb.h"
#include "dnn/rt.h"
#endif

#if defined(ION_ENABLE_BB_IMAGE_IO)
#include "image-io/bb.h"
#include "image-io/rt.h"
#endif

#if defined(ION_ENABLE_BB_IMAGE_PROCESSING)
#include "image-processing/bb.h"
#include "image-processing/rt.h"
#endif

#if defined(ION_ENABLE_BB_OPENCV)
#include "opencv/bb.h"
#include "opencv/rt.h"
#endif

#if defined(ION_ENABLE_BB_SGM)
#include "sgm/bb.h"
#include "sgm/rt.h"
#endif

extern "C" void register_externs(std::map<std::string, Halide::JITExtern>& externs) {
#if defined(ION_ENABLE_BB_BASE)
  for (auto kv : ion::bb::base::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
#if defined(ION_ENABLE_BB_DNN)
  for (auto kv : ion::bb::dnn::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
#if defined(ION_ENABLE_BB_IMAGE_IO)
  for (auto kv : ion::bb::image_io::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
#if defined(ION_ENABLE_BB_IMAGE_PROCESSING)
  for (auto kv : ion::bb::image_processing::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
#if defined(ION_ENABLE_BB_OPENCV)
  for (auto kv : ion::bb::opencv::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
#if defined(ION_ENABLE_BB_SGM)
  for (auto kv : ion::bb::sgm::extern_functions) {
    externs.insert({kv.first, Halide::JITExtern(kv.second)});
  }
#endif
}
