#ifndef ION_BB_IMAGE_IO_RT_FILE_H
#define ION_BB_IMAGE_IO_RT_FILE_H

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "json/json.hpp"

#include "log.h"

#include "rt_common.h"
#include "httplib.h"

#include "opencv_loader.h"


extern "C" int ION_EXPORT ion_bb_image_io_color_data_loader(halide_buffer_t *session_id_buf, halide_buffer_t *url_buf, int32_t width, int32_t height, halide_buffer_t *out) {

    using namespace ion::bb::image_io;

    try {

        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width,
            out->dim[1].min = 0;
            out->dim[1].extent = height;
            out->dim[2].min = 0;
            out->dim[2].extent = 3;
        } else {
            const std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            const std::string url = reinterpret_cast<const char *>(url_buf->host);
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence<uint8_t>>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence<uint8_t>>(new ImageSequence<uint8_t>(session_id, url));
            }

            Halide::Runtime::Buffer<uint8_t> obuf(*out);
            seqs[session_id]->get(width, height, IMREAD_COLOR,  obuf);

        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}
ION_REGISTER_EXTERN(ion_bb_image_io_color_data_loader);

extern "C" int ION_EXPORT ion_bb_image_io_grayscale_data_loader(halide_buffer_t *session_id_buf, halide_buffer_t *url_buf, int32_t width, int32_t height, int32_t dynamic_range, halide_buffer_t *out) {

    using namespace ion::bb::image_io;

    try {
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width;
            out->dim[1].min = 0;
            out->dim[1].extent = height;
        } else {
            const std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            const std::string url = reinterpret_cast<const char *>(url_buf->host);
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence<uint16_t>>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence<uint16_t>>(new ImageSequence<uint16_t>(session_id, url));
            }
            Halide::Runtime::Buffer<uint16_t> obuf(*out);
            seqs[session_id]->get(width, height, IMREAD_GRAYSCALE, obuf);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}
ION_REGISTER_EXTERN(ion_bb_image_io_grayscale_data_loader);

extern "C" int ION_EXPORT ion_bb_image_io_image_saver(halide_buffer_t *in, int32_t width, int32_t height, halide_buffer_t *path, halide_buffer_t *out) {
    try {
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = 3;
            in->dim[1].min = 0;
            in->dim[1].extent = width;
            in->dim[2].min = 0;
            in->dim[2].extent = height;
        } else {
            Halide::Runtime::Buffer<uint8_t> obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);
            std::memcpy(obuf.data(), in->host, 3* width*height*sizeof(uint8_t));
            Halide::Tools::save_image(obuf, reinterpret_cast<const char *>(path->host));
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}
ION_REGISTER_EXTERN(ion_bb_image_io_image_saver);


namespace {

class Writer {
public:
    static Writer& get_instance(const std::string& id, std::vector<int32_t>& payload_size, const std::string& output_directory, bool write_framecount, const std::string& prefix = "raw-")
    {

        if (instances.count(id) == 0) {
            instances[id] = std::unique_ptr<Writer>(new Writer(payload_size, output_directory, write_framecount, prefix ));
        }
        return *instances[id];
    }

    ~Writer() {
        if (!disposed_){
            ion::log::debug("Trying to call dispose from distructor since disposed_ is {}", disposed_);
            dispose();
        }

    }

    void post_image(std::vector<void *>& outs, std::vector<size_t>& size,
                    ion::bb::image_io::rawHeader& header_info, void* framecounts)
    {
        if (with_header_){
            write_config_file(header_info);
        }
        std::unique_lock<std::mutex> lock(mutex_);
        buf_cv_.wait(lock, [&] { return !buf_queue_.empty() || ep_; });
        if (ep_) {
            std::rethrow_exception(ep_);
        }
        uint8_t* buffer = buf_queue_.front();
        buf_queue_.pop();
        size_t offset = 0;
        for (int i = 0; i < outs.size(); ++i){
            std::memcpy(buffer + offset, reinterpret_cast<int32_t*>(framecounts) + i, sizeof(int32_t));
            offset += sizeof(int32_t);
            std::memcpy(buffer + offset, outs[i], size[i]);
            offset += size[i];
        }
        task_queue_.push(std::make_tuple(0, buffer, offset));
        task_cv_.notify_one();
    }

    void post_gendc(std::vector<void *>& outs, std::vector<size_t>& size, ion::bb::image_io::rawHeader& header_info)
    {
        if (with_header_){
            write_config_file(header_info);
        }
        std::unique_lock<std::mutex> lock(mutex_);
        buf_cv_.wait(lock, [&] { return !buf_queue_.empty() || ep_; });
        if (ep_) {
            std::rethrow_exception(ep_);
        }
        uint8_t* buffer = buf_queue_.front();
        buf_queue_.pop();
        size_t offset = 0;
        for (int i = 0; i < outs.size(); ++i){
            std::memcpy(buffer + offset, outs[i], size[i]);
            offset += size[i];
        }
        task_queue_.push(std::make_tuple(0, buffer, offset));
        task_cv_.notify_one();
    }

    void dispose() {
         ion::log::debug("Writer::dispose() :: is called");
        // Already disposed if thread is not joinable
        if (thread_ && thread_->joinable()) {
            keep_running_ = false;
            task_cv_.notify_one();
            thread_->join();
            thread_ = nullptr;
        }
         ion::log::debug("Writer::dispose() :: is finished");
         disposed_ = true;
    }


    static void release_instance(const char * id) {
        ion::log::debug("Writer::release_instance() :: is called");
        if (instances.count(id) == 0) {
             return;
        }

        Writer & writer = *instances[id].get();
        writer.dispose();
        instances.erase(id);
        ion::log::debug("Writer::release_instance() :: Instance is delete");

       }

    void write_config_file(ion::bb::image_io::rawHeader& header_info){
        nlohmann::json j_sensor;
        j_sensor["prefix"] = prefix_;
        j_sensor["framerate"] = header_info.fps_;
        j_sensor["width"] = header_info.width_;
        j_sensor["height"] = header_info.height_;
        j_sensor["pfnc_pixelformat"] = header_info.pfnc_pixelformat;

        auto filename = prefix_ + "config.json";
        std::ofstream config(output_directory_ / filename);
        config << std::setw(4) << j_sensor << std::endl;
        config.close();
        with_header_ = false;
    }

private:
    Writer(std::vector<int32_t>& payload_size, const std::string& output_directory, bool write_framecount, const std::string& prefix)
        : keep_running_(true), output_directory_(output_directory), with_header_(true), disposed_(false), prefix_(prefix)
    {
        int total_payload_size = 0;
        for (auto s : payload_size){
            total_payload_size += s;
            if (write_framecount){
                total_payload_size += sizeof(int32_t);
            }
        }
        int buffer_num = get_buffer_num(total_payload_size);

        for (int i = 0; i < buffer_num; ++i) {
            buffers_.emplace_back(total_payload_size);
            buf_queue_.push(buffers_[i].data());
        }
        thread_ = std::make_shared<std::thread>(entry_point, this);
        auto filename = prefix_ + std::to_string(0) + ".bin";
        ofs_ = std::ofstream(output_directory_ / filename, std::ios::binary);
    }

    int get_buffer_num(int width, int height, int num_sensor = 2, int data_in_byte = 2) {
        // fix the memory size 2GB
        const double memory_size_in_MB = 2048.0;
        return static_cast<int>( memory_size_in_MB * 1024 * 1024 / width / height / num_sensor / data_in_byte);
    }

    int get_buffer_num(int32_t payloadsize) {
        // fix the memory size 2GB
        const double memory_size_in_MB = 2048.0;
        return static_cast<int>( memory_size_in_MB * 1024 * 1024 / payloadsize);
    }

    static void entry_point(Writer* obj) {
        try {
            obj->thread_main();
        }
        catch (...) {
            std::unique_lock<std::mutex> lock(obj->mutex_);
            obj->ep_ = std::current_exception();
        }
    }

    void thread_main() {
        uint32_t frame_count;
        uint8_t* buffer;
        size_t size;

        // Main loop
        const uint32_t rotate_limit = 60;
        uint32_t file_idx = 1;
        uint32_t i = 0;

        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_cv_.wait(lock, [&] { return !task_queue_.empty() || !keep_running_; });
                if (!keep_running_) {
                    break;
                }
                std::tie(frame_count, buffer, size) = task_queue_.front();
                task_queue_.pop();
            }

            if (i == rotate_limit) {
                i = 0;
                ofs_ = std::ofstream(output_directory_ / (prefix_ + std::to_string(file_idx++) + ".bin"), std::ios::binary);
            }

            ofs_.write(reinterpret_cast<const char*>(buffer), size);

            {
                std::unique_lock<std::mutex> lock(mutex_);
                buf_queue_.push(buffer);
                buf_cv_.notify_one();
            }

            ++i;
        }

        // Flush everything
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (task_queue_.empty()) {
                    break;
                }
                std::tie(frame_count, buffer, size) = task_queue_.front();
                task_queue_.pop();
            }

            if (i == rotate_limit) {
                i = 0;
                ofs_ = std::ofstream(output_directory_ / (prefix_ + std::to_string(file_idx++) + ".bin"), std::ios::binary);
            }


            ofs_.write(reinterpret_cast<const char*>(buffer), size);

            {
                std::unique_lock<std::mutex> lock(mutex_);
                buf_queue_.push(buffer);
                buf_cv_.notify_one();
            }

            ++i;
        }

        ofs_.close();
    }

    static std::unordered_map < std::string, std::unique_ptr<Writer>> instances; // declares Writer::instance
    std::shared_ptr<std::thread> thread_;
    std::vector<std::vector<uint8_t>> buffers_;
    std::mutex mutex_;
    std::condition_variable buf_cv_;
    std::condition_variable task_cv_;
    std::queue<uint8_t*> buf_queue_;
    std::queue<std::tuple<uint32_t, uint8_t*, size_t>> task_queue_;
    bool keep_running_;
    std::exception_ptr ep_;
    std::ofstream ofs_;
    uint32_t width_;
    uint32_t height_;
    std::filesystem::path output_directory_;
    std::string prefix_;
    bool disposed_;

    bool with_header_;
};

std::unordered_map< std::string, std::unique_ptr<Writer>> Writer::instances; // defines Writer::instance
} // namespace


extern "C"
int ION_EXPORT writer_dispose(const char *id) {
    Writer::release_instance(id);
    return 0;
}


extern "C" ION_EXPORT
int ion_bb_image_io_binary_gendc_saver( halide_buffer_t * id_buf, halide_buffer_t * gendc, halide_buffer_t * deviceinfo,
    int payloadsize, halide_buffer_t*  output_directory_buf, halide_buffer_t*  prefix_buf,
    halide_buffer_t * out)
    {
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
        std::vector<int32_t>payloadsize_list{payloadsize};
        const std::string prefix(reinterpret_cast<const char*>(prefix_buf->host));
        auto& w(Writer::get_instance(id,payloadsize_list, output_directory, false, prefix));
        if (gendc->is_bounds_query() || deviceinfo->is_bounds_query()) {
            if (gendc->is_bounds_query()) {
                gendc->dim[0].min = 0;
                gendc->dim[0].extent = payloadsize;
            }
            if (deviceinfo->is_bounds_query()) {
                deviceinfo->dim[0].min = 0;
                deviceinfo->dim[0].extent = sizeof(ion::bb::image_io::rawHeader);
            }
            return 0;
        }
        else {
            ion::bb::image_io::rawHeader header_info;
            ::memcpy(&header_info, deviceinfo->host, sizeof(ion::bb::image_io::rawHeader));

            std::vector<void *> obufs{gendc->host};
            std::vector<size_t> size_in_bytes{gendc->size_in_bytes()};
            w.post_gendc(obufs, size_in_bytes, header_info);


        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

ION_REGISTER_EXTERN(ion_bb_image_io_binary_gendc_saver);

extern "C" ION_EXPORT
int ion_bb_image_io_binary_image_saver(
    halide_buffer_t * id_buf,
    halide_buffer_t * image, halide_buffer_t * deviceinfo, halide_buffer_t * frame_count,
    int width, int height, int dim, int byte_depth, halide_buffer_t*  output_directory_buf,
    halide_buffer_t*  prefix_buf,
    halide_buffer_t * out)
    {
    try {
        int num_output = 1;
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        int32_t frame_size = dim == 2 ? width * height * byte_depth : width * height * 3 * byte_depth;
        std::vector<int32_t>frame_size_list{frame_size};
        const std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
        const std::string prefix(reinterpret_cast<const char*>(prefix_buf->host));
        auto& w(Writer::get_instance(id, frame_size_list, output_directory, true, prefix));

        if (image->is_bounds_query() || deviceinfo->is_bounds_query() || frame_count->is_bounds_query()) {
            if (image->is_bounds_query()) {
                image->dim[0].min = 0;
                image->dim[0].extent = width;
                image->dim[1].min = 0;
                image->dim[1].extent = height;
                if (dim == 3){
                    image->dim[2].min = 0;
                    image->dim[2].extent = 3;
                }
            }
            if (deviceinfo->is_bounds_query()) {
                deviceinfo->dim[0].min = 0;
                deviceinfo->dim[0].extent = sizeof(ion::bb::image_io::rawHeader);
            }
            if (frame_count->is_bounds_query()) {
                frame_count->dim[0].min = 0;
                frame_count->dim[0].extent = num_output;
            }
            return 0;
        }
        else {


            ion::bb::image_io::rawHeader header_info;
            memcpy(&header_info, deviceinfo->host, sizeof(ion::bb::image_io::rawHeader));
            std::vector<ion::bb::image_io::rawHeader> header_infos{header_info};

            std::vector<void *> obufs{image->host};
            std::vector<size_t> size_in_bytes{image->size_in_bytes()};
            w.post_image(obufs, size_in_bytes, header_info, frame_count->host);
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}

ION_REGISTER_EXTERN(ion_bb_image_io_binary_image_saver);

namespace {

class Reader {
public:
    static Reader &get_instance(const std::string &id, int width, int height, const std::string &output_directory, const std::string & prefix) {
        auto it = instances.find(id);
        if (it == instances.end()) {
            instances[id] = std::unique_ptr<Reader>(new Reader(width, height, output_directory, prefix));
        }
        return *instances[id];
    }

    void get(uint8_t *ptr,  size_t size) {
        current_idx_ = file_idx_;

        if (finished_) {
            return;
        }

        uint32_t frame_count = 0;
        ifs_.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
        ifs_.read(reinterpret_cast<char *>(latest_frame_.data()), size);
        std::memcpy(ptr, latest_frame_.data(), size);

        // rotate
        ifs_.peek();
        if (ifs_.eof()) {
            open_and_check(output_directory_, file_idx_, ifs_, &finished_);
        }
        latest_frame_count_ = frame_count;
    }

    ~Reader() {
        if (!finished_) {
            ion::log::debug("Trying to call dispose from destructor since finished_ is {}", finished_);
            dispose();
        }
    }

    bool get_finished() const {
        return finished_;
    }

    uint32_t get_index() const {
        return current_idx_;
    }

    uint32_t get_frame_count() const {
        return latest_frame_count_;
    }

    void dispose() {
        ion::log::debug("Reader::dispose() :: is called");
        ifs_.close();
        finished_= true;
        ion::log::debug("Reader::dispose() :: is finished");
    }

    static void release_instance(const char *id) {
        ion::log::debug("Reader::release_instance() :: is called");
        if (instances.count(id) == 0) {
            return;
        }
        Reader &r = *instances[id].get();
        r.dispose();
        instances.erase(id);

        ion::log::debug("Reader::release_instance() :: Instance is delete");
    }

private:
    Reader(int width, int height, const std::string &output_directory, const std::string &prefix)
        : width_(width), height_(height), output_directory_(output_directory), prefix_(prefix),
          file_idx_(0), latest_frame_(width * height),
          latest_frame_count_(0), finished_(false) {

        open_and_check(output_directory_, file_idx_, ifs_, &finished_);
        if (finished_) {
            return;
        }
        current_idx_ = file_idx_;
    }

    void open_and_check(const std::filesystem::path &output_directory, uint32_t &file_idx, std::ifstream &ifs, bool *finished) {
        auto file_path = output_directory / (prefix_ + std::to_string(file_idx++) + ".bin");
        ifs.close();  // ensure the previous stream is closed before reopening
        ifs.clear();  // clear any error flags
        ifs.open(file_path, std::ios::binary);
        if (ifs.fail()) {
            *finished = true;
            return;
        }
    }

    uint32_t width_;
    uint32_t height_;
    std::filesystem::path output_directory_;
    const std::string prefix_;
    uint32_t file_idx_;
    std::vector<uint16_t> latest_frame_;
    uint32_t latest_frame_count_;
    std::ifstream ifs_;
    bool finished_;

    uint32_t current_idx_;
    static std::unordered_map<std::string, std::unique_ptr<Reader>> instances;  // declares Reader::instance
};

std::unordered_map<std::string, std::unique_ptr<Reader>> Reader::instances;  // defines Reader::instance
}  // namespace


extern "C" int ION_EXPORT reader_dispose(const char *id) {
    Reader::release_instance(id);
    return 0;
}

extern "C" ION_EXPORT int binaryloader(halide_buffer_t *id_buf, int width, int height, halide_buffer_t *output_directory_buf, halide_buffer_t *prefix_buf,
                                       halide_buffer_t *out) {
    try {

        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string output_directory(reinterpret_cast<const char *>(output_directory_buf->host));
        const std::string prefix(reinterpret_cast<const char *>(prefix_buf->host));
        auto &r(Reader::get_instance(id, width, height, output_directory,prefix));
        if (out->is_bounds_query()) {

            if (out->is_bounds_query()) {
                out->dim[0].min = 0;
                out->dim[0].extent = width;
                out->dim[1].min = 0;
                out->dim[1].extent = height;
            }
        } else {
            r.get(out->host,  out->size_in_bytes());
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(binaryloader);

extern "C" ION_EXPORT int binaryloader_finished(halide_buffer_t *in, halide_buffer_t *id_buf, int width, int height,
                                                halide_buffer_t *output_directory_buf, halide_buffer_t *prefix_buf,
                                                halide_buffer_t *finished, halide_buffer_t *bin_idx,halide_buffer_t *frame_count) {

    try {
        if (in->is_bounds_query() ) {
            if (in->is_bounds_query()) {
                in->dim[0].min = 0;
                in->dim[0].extent = width;
                in->dim[1].min = 0;
                in->dim[1].extent = height;
            }
        } else {
            const std::string id(reinterpret_cast<const char *>(id_buf->host));
            const std::string prefix(reinterpret_cast<const char *>(prefix_buf->host));
            const std::string output_directory(reinterpret_cast<const char *>(output_directory_buf->host));
            auto &r(Reader::get_instance(id, width, height, output_directory,prefix));
            *reinterpret_cast<bool *>(finished->host) = r.get_finished();
            *reinterpret_cast<uint32_t *>(bin_idx->host) = r.get_index();
            *reinterpret_cast<uint32_t *>(frame_count->host) = r.get_frame_count();
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(binaryloader_finished);

#endif
