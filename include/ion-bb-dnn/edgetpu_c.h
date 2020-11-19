#ifndef ION_BB_DNN_EDGETPU_C_H
#define ION_BB_DNN_EDGETPU_C_H

#include "tensorflowlite_c.h"
#include "rt_util.h"

#ifdef __cplusplus
extern "C" {
#endif

enum edgetpu_device_type {
  EDGETPU_APEX_PCI = 0,
  EDGETPU_APEX_USB = 1,
};

struct edgetpu_device {
  enum edgetpu_device_type type;
  const char* path;
};

struct edgetpu_option {
  const char* name;
  const char* value;
};

using edgetpu_list_devices_t = struct edgetpu_device* (*)(size_t* num_devices);
using edgetpu_free_devices_t = void (*)(struct edgetpu_device* dev);
using edgetpu_create_delegate_t = TfLiteDelegate* (*)(enum edgetpu_device_type type, const char* name, const struct edgetpu_option* options, size_t num_options);
using edgetpu_free_delegate_t = void (*)(TfLiteDelegate* delegate);
using edgetpu_verbosity_t = void (*)(int verbosity);
using edgetpu_version_t = const char* (*)();

edgetpu_list_devices_t    edgetpu_list_devices;
edgetpu_free_devices_t    edgetpu_free_devices;
edgetpu_create_delegate_t edgetpu_create_delegate;
edgetpu_free_delegate_t   edgetpu_free_delegate;
edgetpu_verbosity_t       edgetpu_verbosity;
edgetpu_version_t         edgetpu_version;

bool edgetpu_init() {
    static ion::bb::dnn::DynamicModule dm("libedgetpu.so.1", true);
    if (!dm.is_available()) {
        return false;
    }

#define RESOLVE_SYMBOL(SYM_NAME)                                 \
        SYM_NAME = dm.get_symbol<SYM_NAME ## _t>(#SYM_NAME);     \
        if (SYM_NAME == nullptr) {                               \
            throw std::runtime_error(                            \
                #SYM_NAME " is unavailable on your edgetpu DSO"); \
        }

    RESOLVE_SYMBOL(edgetpu_list_devices);
    RESOLVE_SYMBOL(edgetpu_free_devices);
    RESOLVE_SYMBOL(edgetpu_create_delegate);
    RESOLVE_SYMBOL(edgetpu_free_delegate);
    RESOLVE_SYMBOL(edgetpu_verbosity);
    RESOLVE_SYMBOL(edgetpu_version);

#undef RESOLVE_SYMBOL

    return true;
}
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ION_BB_DNN_EDGETPU_C_H
