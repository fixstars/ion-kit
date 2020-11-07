#ifndef ION_BB_GENESIS_CLOUD_RT_COMMON_H
#define ION_BB_GENESIS_CLOUD_RT_COMMON_H

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

#include <cstdio>
#include <string>

namespace {

template<typename... Rest>
std::string format(const char *fmt, const Rest&... rest)
{
    int length = snprintf(NULL, 0, fmt, rest...) + 1; // Explicit place for null termination
    std::vector<char> buf(length, 0);
    snprintf(&buf[0], length, fmt, rest...);
    std::string s(buf.begin(), std::find(buf.begin(), buf.end(), '\0'));
    return s;
}

}

#endif
