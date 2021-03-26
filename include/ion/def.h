#ifndef ION_DEF_H
#define ION_DEF_H

#ifndef ION_ATTRIBUTE_DEPRECATED
#ifdef ION_ALLOW_DEPRECATED
#define ION_ATTRIBUTE_DEPRECATED(x)
#else
#ifdef _MSC_VER
#define ION_ATTRIBUTE_DEPRECATED(x) __declspec(deprecated(x))
#else
#define ION_ATTRIBUTE_DEPRECATED(x) __attribute__((deprecated(x)))
#endif
#endif
#endif

#endif
