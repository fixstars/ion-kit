#ifndef ION_LOG_H
#define ION_LOG_H

#ifndef FMT_CONSTEVAL
#define FMT_CONSTEVAL  // To prevent format string is evaluated as constexpr
#endif
#include "spdlog/spdlog.h"

namespace ion {
namespace log {

namespace level {
enum level_enum : int {
    trace = spdlog::level::trace,
    debug = spdlog::level::debug,
    info = spdlog::level::info,
    warn = spdlog::level::warn,
    err = spdlog::level::err,
    critical = spdlog::level::critical,
    off = spdlog::level::off,
    n_levels
};
}

std::shared_ptr<spdlog::logger> get();
bool should_log(level::level_enum level);

template<typename... Args>
inline void critical(Args... args) {
    if (get()) get()->critical(args...);
}
template<typename... Args>
inline void error(Args... args) {
    if (get()) get()->error(args...);
}
template<typename... Args>
inline void warn(Args... args) {
    if (get()) get()->warn(args...);
}
template<typename... Args>
inline void info(Args... args) {
    if (get()) get()->info(args...);
}
template<typename... Args>
inline void debug(Args... args) {
    if (get()) get()->debug(args...);
}
template<typename... Args>
inline void trace(Args... args) {
    if (get()) get()->trace(args...);
}

}  // namespace log
}  // namespace ion

#endif
