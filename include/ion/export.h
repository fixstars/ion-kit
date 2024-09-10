#ifndef ION_EXPORT_H
#define ION_EXPORT_H

#ifdef _MSC_VER
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT __attribute__((visibility("default")))
#endif

#endif  // ION_EXPORT_H
