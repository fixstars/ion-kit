#include "spdlog/cfg/helpers.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "log.h"

namespace {

struct Logger {
     Logger()
     {
         auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
         auto env_val = spdlog::details::os::getenv("ION_LOG_LEVEL");
         if (env_val.empty()) {
             console_sink->set_level(spdlog::level::critical);
         } else {
             console_sink->set_level(spdlog::level::from_str(env_val));
         }

         auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/ion.log", false);
         file_sink->set_level(spdlog::level::trace);

         auto logger = std::make_shared<spdlog::logger>("ion", spdlog::sinks_init_list{console_sink, file_sink});
         logger->set_level(spdlog::level::trace);

         logger->debug("ion-kit version is {}", ION_KIT_VERSION);

         spdlog::register_logger(logger);
     }
} logger;

} // anonymous
