#ifndef ION_BB_IMAGE_IO_RT_FILE_H
#define ION_BB_IMAGE_IO_RT_FILE_H

#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <queue>

#include "rt_common.h"
#include "ghc/filesystem.hpp"
#include "nlohmann/json.hpp"

#include "httplib.h"

#ifndef _WIN32
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#ifndef _WIN32
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
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence>(new ImageSequence(session_id, url));
            }
            auto frame = seqs[session_id]->get(width, height, cv::IMREAD_COLOR);

            // Resize to desired width/height
            cv::resize(frame, frame, cv::Size(width, height), 0, 0);

            // Convert to RGB from BGR
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            // Reshape interleaved to planar
            frame = frame.reshape(1, width*height).t();

            std::memcpy(out->host, frame.data, width * height * 3 * sizeof(uint8_t));
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
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence>(new ImageSequence(session_id, url));
            }
            auto frame = seqs[session_id]->get(width, height, cv::IMREAD_GRAYSCALE);

            // Normalize value range from 0-255 into 0-dynamic_range
            cv::normalize(frame, frame, 0, dynamic_range, cv::NORM_MINMAX, CV_16UC1);

            std::memcpy(out->host, frame.data, width * height * sizeof(uint16_t));
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

extern "C" int ION_EXPORT ion_bb_image_io_saver(halide_buffer_t *in, int32_t in_extent_1, int32_t in_extent_2, halide_buffer_t *path, halide_buffer_t *out) {
    try {
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = 3;
            in->dim[1].min = 0;
            in->dim[1].extent = in_extent_1;
            in->dim[2].min = 0;
            in->dim[2].extent = in_extent_2;
        } else {
            cv::Mat img(std::vector<int>{in_extent_2, in_extent_1}, CV_8UC3, in->host);
            cv::imwrite(reinterpret_cast<const char *>(path->host), img);
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
ION_REGISTER_EXTERN(ion_bb_image_io_saver);
#endif

namespace {

class Writer {
public:
    static Writer& get_instance(int width, int height, const ::std::string& output_directory, ion::bb::image_io::rawHeader header_info)
    {
        auto itr = instances.find(output_directory);
        if (itr == instances.end()) {
            instances[output_directory] = std::unique_ptr<Writer>(new Writer(width, height, output_directory, header_info));
        }
        return *instances[output_directory];
    }

    static Writer& get_instance(int total_payload_size, const ::std::string& output_directory, bool config_file)
    {
        auto itr = instances.find(output_directory);
        if (itr == instances.end()) {
            instances[output_directory] = std::unique_ptr<Writer>(new Writer(total_payload_size, output_directory, config_file));
        }
        return *instances[output_directory];
    }

    ~Writer() {
        dispose();
    }

    void post(uint32_t frame_count, const uint8_t* ptr0, const uint8_t* ptr1, size_t size)
    {
        ::std::unique_lock<::std::mutex> lock(mutex_);
        buf_cv_.wait(lock, [&] { return !buf_queue_.empty() || ep_; });
        if (ep_) {
            ::std::rethrow_exception(ep_);
        }
        uint8_t* buffer = buf_queue_.front();
        buf_queue_.pop();
        ::std::memcpy(buffer, ptr0, size);
        ::std::memcpy(buffer + size, ptr1, size);
        task_queue_.push(::std::make_tuple(frame_count, buffer, 2 * size));
        task_cv_.notify_one();
    }


    void post_gendc(std::vector<void *>& outs, std::vector<size_t>& size, std::vector<ion::bb::image_io::rawHeader>& header_infos)
    {
        if (with_header_){
            write_config_file(header_infos);
        }
        ::std::unique_lock<::std::mutex> lock(mutex_);
        buf_cv_.wait(lock, [&] { return !buf_queue_.empty() || ep_; });
        if (ep_) {
            ::std::rethrow_exception(ep_);
        }
        uint8_t* buffer = buf_queue_.front();
        buf_queue_.pop();
        size_t offset = 0;
        for (int i = 0; i < outs.size(); ++i){
            ::std::memcpy(buffer + offset, outs[i], size[i]);
            offset += size[i];
        }
        task_queue_.push(::std::make_tuple(0, buffer, offset));
        task_cv_.notify_one();
    }

    void dispose() {
        // Already disposed if thread is not joinable
        if (thread_ && thread_->joinable()) {
            keep_running_ = false;
            task_cv_.notify_one();
            thread_->join();
            thread_ = nullptr;
        }
    }

    void release_instance(const ::std::string& output_directory) {
        instances.erase(output_directory);
    }

    void write_config_file(std::vector<ion::bb::image_io::rawHeader>& header_infos){
        nlohmann::json j;
        j["num_device"] = header_infos.size();
        for (int i = 0; i < header_infos.size(); ++i){
            nlohmann::json j_ith_sensor;
            j_ith_sensor["framerate"] = header_infos[i].fps_;
            j_ith_sensor["width"] = header_infos[i].width_;
            j_ith_sensor["height"] = header_infos[i].height_;
            j_ith_sensor["pfnc_picelformat"] = header_infos[i].pfnc_picelformat;
            j["sensor" + std::to_string(i+1)] = j_ith_sensor;
        }

        ::std::ofstream config(output_directory_ / "config.json");
        config << std::setw(4) << j << std::endl;
        config.close();

        with_header_ = false;
    }

private:
    Writer(int width, int height, const ::std::string& output_directory,
        ion::bb::image_io::rawHeader header_info)
        : keep_running_(true), width_(width), height_(height), output_directory_(output_directory),
        header_info_(header_info), with_header_(true), with_framecount_(true)
    {
        int buffer_num = get_buffer_num(width, height);
        for (int i = 0; i < buffer_num; ++i) {
            buffers_.emplace_back(2 * width * height * sizeof(uint16_t));
            buf_queue_.push(buffers_[i].data());
        }
        thread_ = ::std::make_shared<::std::thread>(entry_point, this);
        ofs_ = ::std::ofstream(output_directory_ / "raw-0.bin", ::std::ios::binary);

        // write header (size is 512)
        ofs_.write(reinterpret_cast<const char*>(&header_info_), sizeof(header_info_));
        char padding_item = '0';
        for (int i = 0; i < 512 - sizeof(header_info_); ++i) {
            ofs_.write(reinterpret_cast<const char*>(&padding_item), sizeof(padding_item));
        }
    }

    Writer(int total_payload_size, const ::std::string& output_directory, bool config_file)
        : keep_running_(true), output_directory_(output_directory), with_header_(!config_file), with_framecount_(false)
    {
        int buffer_num = get_buffer_num(total_payload_size);
        for (int i = 0; i < buffer_num; ++i) {
            buffers_.emplace_back(total_payload_size);
            buf_queue_.push(buffers_[i].data());
        }
        thread_ = ::std::make_shared<::std::thread>(entry_point, this);

        // nlohmann::json j;
        // j["num_device"] = header_infos.size();
        // for (int i = 0; i < header_infos.size(); ++i){
        //     nlohmann::json j_ith_sensor;
        //     j_ith_sensor["framerate"] = header_infos[i].fps_;
        //     j_ith_sensor["width"] = header_infos[i].width_;
        //     j_ith_sensor["height"] = header_infos[i].height_;
        //     j_ith_sensor["pfnc_picelformat"] = header_infos[i].pfnc_picelformat;
        //     j["sensor" + std::to_string(i+1)] = j_ith_sensor;
        // }

        // ::std::ofstream config(output_directory_ / "config.json");
        // config << std::setw(4) << j << std::endl;
        // config.close();
        ofs_ = ::std::ofstream(output_directory_ / "raw-0.bin", ::std::ios::binary);
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
            ::std::unique_lock<::std::mutex> lock(obj->mutex_);
            obj->ep_ = ::std::current_exception();
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
                ::std::unique_lock<::std::mutex> lock(mutex_);
                task_cv_.wait(lock, [&] { return !task_queue_.empty() || !keep_running_; });
                if (!keep_running_) {
                    break;
                }
                ::std::tie(frame_count, buffer, size) = task_queue_.front();
                task_queue_.pop();
            }

            if (i == rotate_limit) {
                i = 0;
                ofs_ = ::std::ofstream(output_directory_ / ("raw-" + ::std::to_string(file_idx++) + ".bin"), ::std::ios::binary);

                if (with_header_){
                    // write header (size is 512)
                    ofs_.write(reinterpret_cast<const char*>(&header_info_), sizeof(header_info_));
                    char padding_item = '0';
                    for (int i = 0; i < 512 - sizeof(header_info_); ++i) {
                        ofs_.write(reinterpret_cast<const char*>(&padding_item), sizeof(padding_item));
                    }
                }
            }

            if (with_framecount_){
              ofs_.write(reinterpret_cast<const char*>(&frame_count), sizeof(frame_count));
            }
            ofs_.write(reinterpret_cast<const char*>(buffer), size);

            {
                ::std::unique_lock<::std::mutex> lock(mutex_);
                buf_queue_.push(buffer);
                buf_cv_.notify_one();
            }

            ++i;
        }

        // Flush everything
        while (true) {
            {
                ::std::unique_lock<::std::mutex> lock(mutex_);
                if (task_queue_.empty()) {
                    break;
                }
                ::std::tie(frame_count, buffer, size) = task_queue_.front();
                task_queue_.pop();
            }

            if (i == rotate_limit) {
                i = 0;
                ofs_ = ::std::ofstream(output_directory_ / ("raw-" + ::std::to_string(file_idx++) + ".bin"), ::std::ios::binary);

                if (with_header_){
                    // write header (size is 512)
                    ofs_.write(reinterpret_cast<const char*>(&header_info_), sizeof(header_info_));
                    char padding_item = '0';
                    for (int i = 0; i < 512 - sizeof(header_info_); ++i) {
                        ofs_.write(reinterpret_cast<const char*>(&padding_item), sizeof(padding_item));
                    }
                }
            }

            if (with_framecount_){
              ofs_.write(reinterpret_cast<const char*>(&frame_count), sizeof(frame_count));
            }
            ofs_.write(reinterpret_cast<const char*>(buffer), size);

            {
                ::std::unique_lock<::std::mutex> lock(mutex_);
                buf_queue_.push(buffer);
                buf_cv_.notify_one();
            }

            ++i;
        }

        ofs_.close();
    }

    static ::std::unordered_map < ::std::string, std::unique_ptr<Writer>> instances; // declares Writer::instance
    ::std::shared_ptr<::std::thread> thread_;
    ::std::vector<::std::vector<uint8_t>> buffers_;
    ::std::mutex mutex_;
    ::std::condition_variable buf_cv_;
    ::std::condition_variable task_cv_;
    ::std::queue<uint8_t*> buf_queue_;
    ::std::queue<::std::tuple<uint32_t, uint8_t*, size_t>> task_queue_;
    bool keep_running_;
    ::std::exception_ptr ep_;
    ::std::ofstream ofs_;
    uint32_t width_;
    uint32_t height_;
    ghc::filesystem::path output_directory_;

    ion::bb::image_io::rawHeader header_info_;
    bool with_header_;
    bool with_framecount_;
};

::std::unordered_map< ::std::string, std::unique_ptr<Writer>> Writer::instances; // defines Writer::instance
} // namespace


extern "C" ION_EXPORT
int binarysaver(halide_buffer_t * in0, halide_buffer_t * in1, halide_buffer_t * fc,
    bool dispose, int width, int height, halide_buffer_t*  output_directory_buf,
    float r_gain0, float g_gain0, float b_gain0, float r_gain1, float g_gain1, float b_gain1,
    int offset0_x, int offset0_y, int offset1_x, int offset1_y,
    int outputsize0_x, int outputsize0_y, int outputsize1_x, int outputsize1_y, float fps,
    halide_buffer_t * out)
    {
    try {
        const ::std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
        ion::bb::image_io::rawHeader header_info = {
            1, width, height, r_gain0, g_gain0, b_gain0, r_gain1, g_gain1, b_gain1,
            offset0_x, offset0_y, offset1_x, offset1_y,
            outputsize0_x, outputsize0_y, outputsize1_x, outputsize1_y, fps, PFNC_Mono12};

        auto& w(Writer::get_instance(width, height, output_directory, header_info));
        if (in0->is_bounds_query() || in1->is_bounds_query()) {
            if (in0->is_bounds_query()) {
                in0->dim[0].min = 0;
                in0->dim[0].extent = width;
                in0->dim[1].min = 0;
                in0->dim[1].extent = height;
            }
            if (in1->is_bounds_query()) {
                in1->dim[0].min = 0;
                in1->dim[0].extent = width;
                in1->dim[1].min = 0;
                in1->dim[1].extent = height;
            }
            if (fc->is_bounds_query()) {
                if (fc->is_bounds_query()) {
                    fc->dim[0].min = 0;
                    fc->dim[0].extent = 1;
                }

            }
        }
        else {
            const uint32_t frame_count = reinterpret_cast<uint32_t*>(fc->host)[0];
            w.post(frame_count, in0->host, in1->host, in0->size_in_bytes());

            if (dispose) {
                w.dispose();
                w.release_instance(output_directory);
                return 0;
            }
        }

        return 0;
    }
    catch (const ::std::exception& e) {
        ::std::cerr << e.what() << ::std::endl;
        return -1;
    }
    catch (...) {
        ::std::cerr << "Unknown error" << ::std::endl;
        return -1;
    }
}

ION_REGISTER_EXTERN(binarysaver);

extern "C" ION_EXPORT
int ion_bb_image_io_binary_2gendc_saver(halide_buffer_t * in0, halide_buffer_t * in1, halide_buffer_t * in2, halide_buffer_t * in3,
    bool dispose, int payloadsize0, int payloadsize1, halide_buffer_t*  output_directory_buf,
    halide_buffer_t * out)
    {
    try {
        const ::std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
        auto& w(Writer::get_instance(payloadsize0 + payloadsize1, output_directory, false));
        if (in0->is_bounds_query() || in1->is_bounds_query() || in2->is_bounds_query() || in3->is_bounds_query()) {
            if (in0->is_bounds_query()) {
                in0->dim[0].min = 0;
                in0->dim[0].extent = payloadsize0;
            }
            if (in1->is_bounds_query()) {
                in1->dim[0].min = 0;
                in1->dim[0].extent = payloadsize1;
            }
            if (in2->is_bounds_query()) {
                in2->dim[0].min = 0;
                in2->dim[0].extent = 76;
            }
            if (in3->is_bounds_query()) {
                in3->dim[0].min = 0;
                in3->dim[0].extent = 76;
            }
        }
        else {
            ion::bb::image_io::rawHeader header_info0, header_info1;
            ::memcpy(&header_info0, in2->host, sizeof(ion::bb::image_io::rawHeader));
            ::memcpy(&header_info1, in3->host, sizeof(ion::bb::image_io::rawHeader));
            std::vector<ion::bb::image_io::rawHeader> header_infos{header_info0, header_info1};

            std::vector<void *> obufs{in0->host, in1->host};
            std::vector<size_t> size_in_bytes{in0->size_in_bytes(), in1->size_in_bytes()};
            w.post_gendc(obufs, size_in_bytes, header_infos);

            if (dispose) {
                w.dispose();
                w.release_instance(output_directory);
                return 0;
            }
        }

        return 0;
    }
    catch (const ::std::exception& e) {
        ::std::cerr << e.what() << ::std::endl;
        return -1;
    }
    catch (...) {
        ::std::cerr << "Unknown error" << ::std::endl;
        return -1;
    }
}

ION_REGISTER_EXTERN(ion_bb_image_io_binary_2gendc_saver);

extern "C" ION_EXPORT
int ion_bb_image_io_binary_1gendc_saver(halide_buffer_t * in0, halide_buffer_t * in1,
    bool dispose, int payloadsize, halide_buffer_t*  output_directory_buf,
    halide_buffer_t * out)
    {
    try {
        const ::std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
        auto& w(Writer::get_instance(payloadsize, output_directory, false));
        if (in0->is_bounds_query() || in1->is_bounds_query()) {
            if (in0->is_bounds_query()) {
                in0->dim[0].min = 0;
                in0->dim[0].extent = payloadsize;
            }
            if (in1->is_bounds_query()) {
                in1->dim[0].min = 0;
                in1->dim[0].extent = 76;
            }
        }
        else {
            ion::bb::image_io::rawHeader header_info0;
            ::memcpy(&header_info0, in1->host, sizeof(ion::bb::image_io::rawHeader));
            std::vector<ion::bb::image_io::rawHeader> header_infos{header_info0};

            std::vector<void *> obufs{in0->host};
            std::vector<size_t> size_in_bytes{in0->size_in_bytes()};
            w.post_gendc(obufs, size_in_bytes, header_infos);

            if (dispose) {
                w.dispose();
                w.release_instance(output_directory);
                return 0;
            }
        }

        return 0;
    }
    catch (const ::std::exception& e) {
        ::std::cerr << e.what() << ::std::endl;
        return -1;
    }
    catch (...) {
        ::std::cerr << "Unknown error" << ::std::endl;
        return -1;
    }
}

ION_REGISTER_EXTERN(ion_bb_image_io_binary_1gendc_saver);

namespace {

    class Reader {
    public:
        static Reader& get_instance(::std::string session_id, int width, int height, const ::std::string& output_directory) {
            auto it = instances.find(session_id);
            if (it == instances.end()) {
                instances[session_id] = std::unique_ptr<Reader>(new Reader(width, height, output_directory));
            }

            return *instances[session_id];
        }

        void get(uint8_t* ptr0, uint8_t* ptr1, size_t size) {

            current_idx_ = file_idx_;

            if (finished_) {
                return;
            }

            if (read_count_ < offset_frame_count_) {
                ::std::memset(ptr0, 0, size);
                ::std::memset(ptr1, 0, size);
            }
            else {
                uint32_t frame_count = 0;
                ifs_.read(reinterpret_cast<char*>(&frame_count), sizeof(frame_count));

                if (frame_count != (latest_frame_count_ + 1)) {
                    ifs_.seekg(-static_cast<int>(sizeof(frame_count)), ::std::ios::cur);
                }
                else {
                    ifs_.read(reinterpret_cast<char*>(latest_frame0_.data()), size);
                    ifs_.read(reinterpret_cast<char*>(latest_frame1_.data()), size);
                }

                ::std::memcpy(ptr0, latest_frame0_.data(), size);
                ::std::memcpy(ptr1, latest_frame1_.data(), size);

                latest_frame_count_++;

                // rotate
                ifs_.peek();
                if (ifs_.eof()) {
                    open_and_check(width_, height_, output_directory_, file_idx_, ifs_, &finished_);
                    if (finished_) {
                        ifs_ = ::std::ifstream();
                    }
                }
            }
            read_count_++;
        }

        void close() {
            ifs_.close();
        }

        bool get_finished() const {
            return finished_;
        }

        uint32_t get_index() {
            return current_idx_;
        }

        void release_instance(const ::std::string& session_id) {
            instances.erase(session_id);
        }

    private:
        Reader(int width, int height, const ::std::string& output_directory)
            : width_(width), height_(height), output_directory_(output_directory),
            file_idx_(0), latest_frame0_(width* height), latest_frame1_(width* height),
            latest_frame_count_((::std::numeric_limits<uint32_t>::max)()), read_count_(0), finished_(false)
        {

            open_and_check(width_, height_, output_directory_, file_idx_, ifs_, &finished_);
            if (finished_) {
                return;
            }

            // Determine counter might be reset to zero (may dropped first few frames)
            uint32_t prev_frame_count = 0;
            const size_t size = static_cast<size_t>(width * height * sizeof(uint16_t));
            while (true) {
                uint32_t frame_count = 0;
                ifs_.read(reinterpret_cast<char*>(&frame_count), sizeof(frame_count));
                ifs_.seekg(2 * size, ::std::ios::cur);
                if (prev_frame_count > frame_count) {
                    ifs_.seekg(-static_cast<int>(sizeof(frame_count)) - 2 * size, ::std::ios::cur);
                    offset_frame_count_ = frame_count;
                    break;
                }
                prev_frame_count = frame_count;
                ifs_.peek();

                if (ifs_.eof()) {
                    open_and_check(width_, height_, output_directory_, file_idx_, ifs_, &finished_);
                    if (finished_) {
                        // Seek to first file and set offset when We cannot find base frame.
                        file_idx_ = 0;
                        open_and_check(width_, height_, output_directory_, file_idx_, ifs_, &finished_);
                        ifs_.read(reinterpret_cast<char*>(&offset_frame_count_), sizeof(offset_frame_count_));
                        ifs_.seekg(-static_cast<int>(sizeof(offset_frame_count_)), ::std::ios::cur);
                        finished_ = false;
                        read_count_ = offset_frame_count_;
                        latest_frame_count_ = read_count_ - 1;
                        break;
                    }
                }
            }
            current_idx_ = file_idx_;
        }

        void open_and_check(uint32_t width, uint32_t height, const ghc::filesystem::path output_directory, uint32_t& file_idx, ::std::ifstream& ifs, bool* finished) {
            auto file_path = output_directory / ("raw-" + ::std::to_string(file_idx++) + ".bin");

            ifs = ::std::ifstream(file_path, ::std::ios::binary);
            if (ifs.fail()) {
                *finished = true;
                return;
            }

            // skip header (size is 512)
            ifs.seekg(512, ::std::ios_base::beg);
        }

        uint32_t width_;
        uint32_t height_;
        ghc::filesystem::path output_directory_;
        uint32_t file_idx_;
        ::std::vector<uint16_t> latest_frame0_;
        ::std::vector<uint16_t> latest_frame1_;
        uint32_t latest_frame_count_;
        uint32_t offset_frame_count_;
        uint32_t read_count_;
        ::std::ifstream ifs_;
        bool finished_;

        uint32_t current_idx_;
        static ::std::unordered_map < ::std::string, std::unique_ptr<Reader>> instances; // declares Writer::instance
    };

    ::std::unordered_map< ::std::string, std::unique_ptr<Reader>> Reader::instances; // defines Writer::instance
}

extern "C" ION_EXPORT
int binaryloader(halide_buffer_t *session_id_buf, int width, int height, halide_buffer_t * output_directory_buf,
        halide_buffer_t * out0, halide_buffer_t * out1) {
    try {

        const ::std::string session_id(reinterpret_cast<const char*>(session_id_buf->host));
        const ::std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));

        auto& r(Reader::get_instance(session_id, width, height, output_directory));
        if (out0->is_bounds_query() || out1->is_bounds_query()) {

            if (out0->is_bounds_query()) {
                out0->dim[0].min = 0;
                out0->dim[0].extent = width;
                out0->dim[1].min = 0;
                out0->dim[1].extent = height;
            }
            if (out1->is_bounds_query()) {
                out1->dim[0].min = 0;
                out1->dim[0].extent = width;
                out1->dim[1].min = 0;
                out1->dim[1].extent = height;
            }
        }
        else {
           r.get(out0->host, out1->host, out0->size_in_bytes());
        }
        return 0;
    }
    catch (const ::std::exception& e) {
        ::std::cerr << e.what() << ::std::endl;
        return -1;
    }
    catch (...) {
        ::std::cerr << "Unknown error" << ::std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(binaryloader);

extern "C" ION_EXPORT
int binaryloader_finished(halide_buffer_t* in0, halide_buffer_t* in1, halide_buffer_t *session_id_buf, int width, int height,
        halide_buffer_t * output_directory_buf,
        halide_buffer_t * finished, halide_buffer_t* bin_idx) {

    try {
        if (in0->is_bounds_query() || in1->is_bounds_query()) {
            if (in0->is_bounds_query()) {
                in0->dim[0].min = 0;
                in0->dim[0].extent = width;
                in0->dim[1].min = 0;
                in0->dim[1].extent = height;
            }
            if (in1->is_bounds_query()) {
                in1->dim[0].min = 0;
                in1->dim[0].extent = width;
                in1->dim[1].min = 0;
                in1->dim[1].extent = height;
            }
		}
		else {
            const ::std::string session_id(reinterpret_cast<const char*>(session_id_buf->host));
            const ::std::string output_directory(reinterpret_cast<const char*>(output_directory_buf->host));
            auto& r(Reader::get_instance(session_id, width, height, output_directory));
            auto finished_flag = r.get_finished();
           *reinterpret_cast<bool*>(finished->host) = finished_flag;
           *reinterpret_cast<uint8_t*>(bin_idx->host) = r.get_index();
           if (finished_flag) {
               r.close();
               r.release_instance(session_id);
           }
        }

        return 0;
    }
    catch (const ::std::exception& e) {
        ::std::cerr << e.what() << ::std::endl;
        return -1;
    }
    catch (...) {
        ::std::cerr << "Unknown error" << ::std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(binaryloader_finished);

#endif
