#include "edgetpu_c.h"
#include "rt_common.h"

#define RESOLVE_SYMBOL(SYM_NAME)                                 \
        SYM_NAME = dm.get_symbol<SYM_NAME ## _t>(#SYM_NAME);     \
        if (SYM_NAME == nullptr) {                               \
            throw std::runtime_error(                            \
                #SYM_NAME " is unavailable on your edgetpu DSO"); \
        }

// using edgetpu_list_devices_t = struct edgetpu_device* (*)(size_t* num_devices);
// using edgetpu_free_devices_t = void (*)(struct edgetpu_device* dev);
// using edgetpu_create_delegate_t = TfLiteDelegate* (*)(enum edgetpu_device_type type, const char* name, const struct edgetpu_option* options, size_t num_options);
// using edgetpu_free_delegate_t = void (*)(TfLiteDelegate* delegate);
// using edgetpu_verbosity_t = void (*)(int verbosity);
// using edgetpu_version_t = const char* (*)();

edgetpu_list_devices_t    edgetpu_list_devices;
edgetpu_free_devices_t    edgetpu_free_devices;
edgetpu_create_delegate_t edgetpu_create_delegate;
edgetpu_free_delegate_t   edgetpu_free_delegate;
edgetpu_verbosity_t       edgetpu_verbosity;
edgetpu_version_t         edgetpu_version;

void edgetpu_init() {
    static ion::bb::dnn::DynamicModule dm("libedgetpu.so.1", true);
    if (!dm.is_available()) {
        return;
    }

    RESOLVE_SYMBOL(edgetpu_list_devices);
    RESOLVE_SYMBOL(edgetpu_free_devices);
    RESOLVE_SYMBOL(edgetpu_create_delegate);
    RESOLVE_SYMBOL(edgetpu_free_delegate);
    RESOLVE_SYMBOL(edgetpu_verbosity);
    RESOLVE_SYMBOL(edgetpu_version);

    return;
}
