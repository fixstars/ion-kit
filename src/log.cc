#ifndef FMT_CONSTEVAL
#define FMT_CONSTEVAL // To prevent format string is evaluated as constexpr
#endif
#include "spdlog/cfg/helpers.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "log.h"

namespace ion {
namespace log {

std::shared_ptr<spdlog::logger> get() {
    return spdlog::get("ion");
}

bool should_log(level::level_enum level) {
    if (get()) {
        return get()->should_log(static_cast<spdlog::level::level_enum>(level));
    } else {
        return false;
    }
}

} // log
} // ion

namespace {

struct Logger {
     Logger()
     {
         auto log_level = spdlog::level::off;
         auto env_val = spdlog::details::os::getenv("ION_LOG_LEVEL");
         if (env_val.empty()) {
             return;
         }

         log_level = spdlog::level::from_str(env_val);

         auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
         console_sink->set_level(log_level);

         auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/ion.log", false);
         file_sink->set_level(log_level);

         auto logger = std::make_shared<spdlog::logger>("ion", spdlog::sinks_init_list{console_sink, file_sink});
         logger->set_level(log_level);

         logger->debug("ion-kit version is {}", ION_KIT_VERSION);

         spdlog::register_logger(logger);
     }
} logger;

} // anonymous
