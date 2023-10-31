#ifndef ION_LOG_H
#define ION_LOG_H

#include "spdlog/spdlog.h"
#include "spdlog/cfg/helpers.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace ion {

struct log {

    log() {
        static auto logger = spdlog::stderr_color_mt("ion");
        auto env_val = spdlog::details::os::getenv("ION_LOG_LEVEL");
        if (env_val.empty()) {
            spdlog::set_level(spdlog::level::critical);
        } else {
            spdlog::cfg::helpers::load_levels(env_val);
        }
    }

    template<class... Types> static void critical(Types... args) { spdlog::get("ion")->critical(args...); }
    template<class... Types> static void error   (Types... args) { spdlog::get("ion")->error   (args...); }
    template<class... Types> static void warn    (Types... args) { spdlog::get("ion")->warn    (args...); }
    template<class... Types> static void info    (Types... args) { spdlog::get("ion")->info    (args...); }
    template<class... Types> static void debug   (Types... args) { spdlog::get("ion")->debug   (args...); }

};

} // ion

#endif
