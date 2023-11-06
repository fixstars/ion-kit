#ifndef ION_LOG_H
#define ION_LOG_H

#include "spdlog/spdlog.h"

namespace ion {
namespace log {
template<class... Types> static void critical(Types... args) { spdlog::get("ion")->critical(args...); }
template<class... Types> static void error   (Types... args) { spdlog::get("ion")->error   (args...); }
template<class... Types> static void warn    (Types... args) { spdlog::get("ion")->warn    (args...); }
template<class... Types> static void info    (Types... args) { spdlog::get("ion")->info    (args...); }
template<class... Types> static void debug   (Types... args) { spdlog::get("ion")->debug   (args...); }
template<class... Types> static void trace   (Types... args) { spdlog::get("ion")->trace   (args...); }
} // log
} // ion

#endif
