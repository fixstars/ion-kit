#ifndef ION_LOG_H
#define ION_LOG_H

#include "spdlog/spdlog.h"

namespace ion {
namespace log {
std::shared_ptr<spdlog::logger> get();
template<class... Types> static void critical(Types... args) { get()->critical(args...); }
template<class... Types> static void error   (Types... args) { get()->error   (args...); }
template<class... Types> static void warn    (Types... args) { get()->warn    (args...); }
template<class... Types> static void info    (Types... args) { get()->info    (args...); }
template<class... Types> static void debug   (Types... args) { get()->debug   (args...); }
template<class... Types> static void trace   (Types... args) { get()->trace   (args...); }
} // log
} // ion

#endif
