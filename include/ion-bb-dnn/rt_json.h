#ifndef ION_BB_DNN_RT_JSON_H
#define ION_BB_DNN_RT_JSON_H

#include <chrono>
#include <exception>
#include <sstream>
#include <thread>
#include <tuple>
#include <queue>

#include "json.hpp"

namespace ion {
namespace bb {
namespace dnn {
namespace json {

class DictAverageRegurator {
 public:
     static DictAverageRegurator& get_instance(const std::string& session_id, uint32_t period_in_sec) {
         static std::unordered_map<std::string, std::unique_ptr<DictAverageRegurator>> instances;
         if (instances.count(session_id) == 0) {
             instances[session_id] = std::unique_ptr<DictAverageRegurator>(new DictAverageRegurator(period_in_sec));
         }
         return *instances[session_id].get();
     }

     nlohmann::json process(nlohmann::json in) {
         using js = nlohmann::json;

         if (!in.is_object()) {
             throw std::runtime_error("Unexpected data format: input is not an object");
         }

         for (js::iterator it = in.begin(); it != in.end(); ++it) {
             if (!it.value().is_number()) {
                 throw std::runtime_error("Unexpected format: value is not a number");
             }

             if (data_.count(it.key()) == 0) {
                 data_[it.key()] = 0.0f;
             }

             data_[it.key()] = data_[it.key()] + static_cast<float>(it.value());
         }
         count_++;

         auto now = std::chrono::system_clock::now();
         if (std::chrono::duration_cast<std::chrono::seconds>(now - tp_).count() >= period_in_sec_) {
             js j;
             for (auto& d : data_) {
                 std::stringstream ss;
                 ss << std::fixed << std::setprecision(2) << d.second / static_cast<float>(count_);
                 j[d.first] = ss.str();
             }
             data_.clear();
             tp_ = now;
             count_ = 0;
             return j;
         } else {
             return js();
         }

         return in;
     }

 private:
     DictAverageRegurator(uint32_t period_in_sec) : period_in_sec_(period_in_sec), tp_(std::chrono::system_clock::now()), count_(0) {}
     uint32_t period_in_sec_;
     std::unordered_map<std::string, float> data_;
     std::chrono::time_point<std::chrono::system_clock> tp_;
     uint32_t count_;
};

class WebHookUploader {
 public:
     static WebHookUploader& get_instance(const std::string& session_id, const std::string& url) {
         static std::unordered_map<std::string, std::unique_ptr<WebHookUploader>> instances;
         if (instances.count(session_id) == 0) {
             instances[session_id] = std::unique_ptr<WebHookUploader>(new WebHookUploader(url));
         }
         return *instances[session_id].get();
     }

     void upload(nlohmann::json in) {
         using js = nlohmann::json;
         if (in.is_null()) {
             return;
         }

         js j;
         j["value1"] = in.dump();

         std::unique_lock<std::mutex> lock(mutex_);
         if (ep_) {
             std::rethrow_exception(ep_);
         }

         queue_.push(j.dump());
         cv_.notify_one();
     }

     ~WebHookUploader() {
         if (thread_->joinable()) {
             keep_running_ = false;
             cv_.notify_one();
             thread_->join();
         }
     }

 private:
     WebHookUploader(const std::string& url)
        : keep_running_(true) {
         std::string host_name;
         std::tie(host_name, path_name_) = parse_url(url);
         if (host_name.empty() || path_name_.empty()) {
             throw std::runtime_error("Invalid URL : " + url);
         }

         cli_ = std::unique_ptr<httplib::Client>(new httplib::Client(host_name.c_str()));
         if (!cli_->is_valid()) {
             throw std::runtime_error("Failed to create HTTP client : " + url);
         }

         thread_ = std::unique_ptr<std::thread>(new std::thread(entry_point, this));
     };

     static void entry_point(WebHookUploader* obj) {
         try {
             obj->thread_main();
         }
         catch (...) {
             std::unique_lock<std::mutex> lock(obj->mutex_);
             obj->ep_ = std::current_exception();
         }
     }

     void thread_main() {
         while (true) {
             std::string body;
             {
                 std::unique_lock<std::mutex> lock(mutex_);
                 cv_.wait(lock, [&] { return !queue_.empty() || !keep_running_; });
                 if (!keep_running_) {
                     break;
                 }
                 body = queue_.front();
                 queue_.pop();
             }

             auto res = cli_->Post(path_name_.c_str(), body, "application/json");
             if (!res || res->status != 200) {
                 throw std::runtime_error("Failed to upload data");
             }
         }
     }

     std::unique_ptr<httplib::Client> cli_;
     std::string path_name_;

     std::unique_ptr<std::thread> thread_;
     std::queue<std::string> queue_;
     std::mutex mutex_;
     std::condition_variable cv_;
     bool keep_running_;
     std::exception_ptr ep_;
};

} // json
} // dnn
} // bb
} // ion

#endif
