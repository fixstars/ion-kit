#ifndef ION_BB_IMAGE_IO_RT_U3V_H
#define ION_BB_IMAGE_IO_RT_U3V_H

#include <chrono>
#include <cstring>
#include <iostream>

#include <HalideBuffer.h>

#include "log.h"

#include "rt_common.h"
#include "gendc_separator/ContainerHeader.h"
#include "gendc_separator/tools.h"

#ifdef _WIN32
    #define GOBJECT_FILE "gobject-2.0-0"
    #define ARAVIS_FILE "aravis-0.8-0"
#else
    #define GOBJECT_FILE "gobject-2.0"
    #define ARAVIS_FILE "aravis-0.8"
#endif

namespace ion {
namespace bb {
namespace image_io {

class U3V {

    struct GError
    {
        uint32_t       domain;
        int32_t         code;
        const char     *message;
    };

    enum OperationMode
    {   Came2USB1,
        Came1USB1,
        Came2USB2,
        Came1USB2
    };

    using gpointer = struct gpointer_*;

    using g_object_unref_t = void (*)(gpointer);

    typedef enum ArvAcquisitionMode{
        ARV_ACQUISITION_MODE_CONTINUOUS,
        ARV_ACQUISITION_MODE_SINGLE_FRAME
    } ArvAcquisitionMode_t;

    typedef enum ArvBufferStatus{
        ARV_BUFFER_STATUS_UNKNOWN,
        ARV_BUFFER_STATUS_SUCCESS,
        ARV_BUFFER_STATUS_CLEARED,
        ARV_BUFFER_STATUS_TIMEOUT,
        ARV_BUFFER_STATUS_MISSING_PACKETS,
        ARV_BUFFER_STATUS_WRONG_PACKET_ID,
        ARV_BUFFER_STATUS_SIZE_MISMATCH,
        ARV_BUFFER_STATUS_FILLING,
        ARV_BUFFER_STATUS_ABORTED
    } ArvBufferStatus_t;

    typedef enum ArvBufferPayloadType{
        ARV_BUFFER_PAYLOAD_TYPE_UNKNOWN,
        ARV_BUFFER_PAYLOAD_TYPE_IMAGE,
        ARV_BUFFER_PAYLOAD_TYPE_RAWDATA,
        ARV_BUFFER_PAYLOAD_TYPE_FILE,
        ARV_BUFFER_PAYLOAD_TYPE_CHUNK_DATA,
        ARV_BUFFER_PAYLOAD_TYPE_EXTENDED_CHUNK_DATA,
        ARV_BUFFER_PAYLOAD_TYPE_JPEG,
        ARV_BUFFER_PAYLOAD_TYPE_JPEG2000,
        ARV_BUFFER_PAYLOAD_TYPE_H264,
        ARV_BUFFER_PAYLOAD_TYPE_MULTIZONE_IMAGE
    }ArvBufferPayloadType_t;

    typedef enum ArvDeviceStatus{
        ARV_DEVICE_STATUS_UNKNOWN,
        ARV_DEVICE_STATUS_SUCCESS,
        ARV_DEVICE_STATUS_TIMEOUT,
        ARV_DEVICE_STATUS_WRITE_ERROR
    }ArvDeviceStatus_t;

    using ArvDevice_t = struct ArvDevice*;
    using ArvFakeDevice_t = struct ArvFakeDevice*;
    using ArvStream_t = struct ArvStream*;
    using ArvStreamCallback_t = struct ArvStreamCallback*;
    using ArvBuffer_t = struct ArvBuffer*;
    using ArvGcNode_t = struct ArvGcNode*;
    using ArvCamera_t = struct ArvCamera*;

    using arv_get_major_version_t = uint32_t(*)();
    using arv_get_minor_version_t = uint32_t(*)();
    using arv_get_micro_version_t = uint32_t(*)();

    using arv_update_device_list_t = void(*)();
    using arv_get_n_devices_t = unsigned int(*)();

    using arv_get_device_id_t = const char*(*)(unsigned int);
    using arv_get_device_model_t = const char*(*)(unsigned int);
    using arv_get_device_serial_nbr_t = const char*(*)(unsigned int);
    using arv_open_device_t = ArvDevice*(*)(const char*, GError**);

    using arv_device_set_string_feature_value_t = void(*)(ArvDevice*, const char*, const char*, GError**);
    using arv_device_set_float_feature_value_t = void(*)(ArvDevice*, const char*, double, GError**);
    using arv_device_set_integer_feature_value_t = void(*)(ArvDevice*, const char*, int64_t, GError**);

    using arv_device_get_string_feature_value_t = const char *(*)(ArvDevice*, const char*, GError**);
    using arv_device_get_integer_feature_value_t = int(*)(ArvDevice*, const char*, GError**);
    using arv_device_get_float_feature_value_t = double(*)(ArvDevice*, const char*, GError**);

    using arv_device_get_integer_feature_bounds_t = void(*)(ArvDevice*, const char*, int64_t*, int64_t*, GError**);
    using arv_device_get_float_feature_bounds_t = void(*)(ArvDevice*, const char*, double*, double*, GError**);

    using arv_device_is_feature_available_t = bool(*)(ArvDevice*, const char*, GError**);

    using arv_device_get_register_feature_length_t = uint64_t(*)(ArvDevice*, const char*, GError**);
    using arv_device_get_register_feature_value_t =	void(*)(ArvDevice*, const char*, uint64_t, void*, GError**);

    using arv_device_create_stream_t = ArvStream*(*)(ArvDevice*, ArvStreamCallback*, void*, GError**);

    using arv_buffer_new_allocate_t = ArvBuffer*(*)(size_t);
    using arv_stream_push_buffer_t = void(*)(ArvStream*, ArvBuffer*);

    using arv_acquisition_mode_to_string_t = const char*(*)(ArvAcquisitionMode);
    using arv_device_execute_command_t = void(*)(ArvDevice*, const char*, GError**);
    using arv_stream_timeout_pop_buffer_t = ArvBuffer*(*)(ArvStream*, uint64_t);
    using arv_stream_get_n_buffers_t = void(*)(ArvStream*, int32_t*, int32_t*);
    using arv_buffer_get_status_t = ArvBufferStatus(*)(ArvBuffer*);
    using arv_buffer_get_payload_type_t = ArvBufferPayloadType(*)(ArvBuffer*);
    using arv_buffer_get_data_t = void*(*)(ArvBuffer*, size_t*);
    using arv_buffer_get_part_data_t = void*(*)(ArvBuffer*, uint_fast32_t, size_t*);
    using arv_buffer_get_timestamp_t = uint64_t(*)(ArvBuffer*);
    using arv_device_get_feature_t = ArvGcNode*(*)(ArvDevice*, const char*);

    using arv_buffer_has_gendc_t = bool*(*)(ArvBuffer*);
    using arv_buffer_get_gendc_descriptor_t = void*(*)(ArvBuffer*, size_t*);

    using arv_shutdown_t = void(*)(void);

    using arv_camera_new_t = ArvCamera*(*)(const char*, GError**);
    using arv_camera_get_device_t = ArvDevice*(*)(ArvCamera *);
    using arv_fake_device_new_t = ArvDevice*(*)(const char*, GError**);
    using arv_set_fake_camera_genicam_filename_t = void(*)(const char*);
    using arv_enable_interface_t = void(*)(const char*);
    using arv_camera_create_stream_t = ArvStream*(*)(ArvCamera*, ArvStreamCallback*, void*, GError**);
    using arv_fake_device_get_fake_camera_t = ArvCamera*(*)(ArvFakeDevice*);

    struct DeviceInfo {
        const char* dev_id_;
        ArvDevice* device_;
        ArvCamera* camera_;

        int32_t u3v_payload_size_;
        int32_t image_payload_size_;
        uint32_t frame_count_;

        float gain_ =-1;
        float exposure_ =-1;

        int32_t int_gain_ = -1;
        int32_t int_exposure_ = -1;

        float exposure_range_[2];

        ArvStream* stream_;

        // genDC
        int64_t data_offset_;
        std::tuple<int32_t, int32_t> available_comp_part;
        int32_t framecount_offset_;
        bool is_data_image_;

        rawHeader header_info_;
    };

    public:
    ~U3V(){
        if (!disposed_){
            log::debug("Trying to call dispose from distructor since disposed_ is {}", disposed_);
            dispose();
        }
    }

    void dispose(){
        log::debug("U3V::dispose() :: is called");
        for (auto i=0; i<devices_.size(); ++i) {
            auto d = devices_[i];
            arv_device_execute_command(d.device_, "AcquisitionStop", &err_);
            log::debug("U3V::dispose() :: AcquisitionStop");
            /*
            Note:
            unref stream also unref the buffers pushed to stream
            all buffers are in stream so do not undef buffres separately
            */
            auto start = std::chrono::system_clock::now();
            g_object_unref(reinterpret_cast<gpointer>(d.stream_));
            auto end = std::chrono::system_clock::now();
            log::debug("U3V::dispose() :: g_object_unref took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());

            start = std::chrono::system_clock::now();
            g_object_unref(reinterpret_cast<gpointer>(d.device_));
            end = std::chrono::system_clock::now();
            log::debug("U3V::dispose() :: g_object_unref took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        }

        devices_.clear();

        // arv_shutdown();
        disposed_ = true;

        log::debug("U3V::dispose() :: Instance is deleted");
    }

      static void release_instance(const char * id) {
        log::debug("U3V::release_instance() :: is called");
        if (instances_.count(id) == 0) {
             return;
        }

        U3V & u3v = *instances_[id].get();
        u3v.dispose();
        instances_.erase(id);
        log::debug("U3V::release_instance() :: is finished");

       }

    void SetGain(int32_t sensor_idx, const std::string key, double v) {

        if (is_param_integer_){
            SetGain(sensor_idx, key, static_cast<int32_t>(v));
            return;
        }

        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].gain_ != v){
                err_ =  Set(devices_[sensor_idx].device_, key.c_str(), v);
                devices_[sensor_idx].gain_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

    void SetGain(int32_t sensor_idx, const std::string key, int32_t v) {
        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].int_gain_ != v){
                err_ =  Set(devices_[sensor_idx].device_, key.c_str(), static_cast<int64_t>(v));
                devices_[sensor_idx].int_gain_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

    void SetExposure(int32_t sensor_idx, const std::string key, double v) {

        if (is_param_integer_){
            SetExposure(sensor_idx, key, static_cast<int32_t>(v));
            return;
        }

        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].exposure_ != v){
                err_ = Set(devices_[sensor_idx].device_, key.c_str(), v);
                devices_[sensor_idx].exposure_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

    void SetExposure(int32_t sensor_idx, const std::string key, int32_t v) {
        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].int_exposure_ != v){
                err_ = Set(devices_[sensor_idx].device_, key.c_str(), static_cast<int64_t>(v));
                devices_[sensor_idx].int_exposure_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

     void setFrameSync(std::vector<ArvBuffer *> &bufs, int timeout_us){
        uint32_t max_cnt = 0;
        while (true) {
            // Update max_cnt
            for (int i = 0; i < num_sensor_; ++i) {
                if (max_cnt < devices_[i].frame_count_) {
                    max_cnt = devices_[i].frame_count_;
                }
            }

            // Check all count is same as max_cnt;
            bool synchronized = true;
            for (int i = 0; i < num_sensor_; ++i) {
                synchronized &= devices_[i].frame_count_ == max_cnt;
            }

            // If it is synchronized, break the loop
            if (synchronized) {
                break;
            }

            // Acquire buffer until cnt is at least max_cnt
            for (int i = 0; i < devices_.size(); ++i) {
                while (devices_[i].frame_count_ < max_cnt) {
                    arv_stream_push_buffer(devices_[i].stream_, bufs[i]);
                    bufs[i] = arv_stream_timeout_pop_buffer(devices_[i].stream_, timeout_us);

                    if (bufs[i] == nullptr) {
                        log::error("pop_buffer failed  when sync frame due to timeout ({}s)", timeout_us * 1e-6f);
                        throw ::std::runtime_error("buffer is null");
                    }
                    devices_[i].frame_count_ = is_gendc_
                                               ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(
                                    bufs[i], devices_[i]))
                                               : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[i]) &
                                                                       0x00000000FFFFFFFF);

                    i == 0 ?
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[i].frame_count_,
                               "") :
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "",
                               devices_[i].frame_count_);
                }
            }
        }

    }

    void setRealTime(std::vector<ArvBuffer *> &bufs, int timeout_us = 3 * 1000 * 1000){
        std::vector<int32_t> N_output_buffers(num_sensor_);
        for (auto i = 0; i < num_sensor_; ++i) {
            int32_t num_input_buffer;
            arv_stream_get_n_buffers(devices_[i].stream_, &num_input_buffer, &(N_output_buffers[i]));
        }
        // if all stream has N output buffers, discard N-1 of them
        for (auto i = 0; i < num_sensor_; ++i) {
            for (auto j = 0; j < N_output_buffers[i] - 1; ++j) {
                bufs[i] = arv_stream_timeout_pop_buffer(devices_[i].stream_, timeout_us);
                if (bufs[i] == nullptr) {
                    log::error("pop_buffer(L2) failed due to timeout ({}s)", timeout_us * 1e-6f);
                    throw ::std::runtime_error("buffer is null");
                }
                devices_[i].frame_count_ = is_gendc_
                                           ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(
                                bufs[i], devices_[i]))
                                           : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[i]) &
                                                                   0x00000000FFFFFFFF);
                i == 0 ?
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20}) [skipped for realtime display]",
                           devices_[i].frame_count_, "") :
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20}) [skipped for realtime display]",
                           "", devices_[i].frame_count_);
                arv_stream_push_buffer(devices_[i].stream_, bufs[i]);

            }
        }
    }

    void get(std::vector<Halide::Buffer<>>& outs) {
        auto timeout_us = 30 * 1000 * 1000;
        if (sim_mode_){
            std::vector<ArvBuffer *> bufs(num_sensor_);
            for (int i = 0;i< num_sensor_;i++){
                    auto size = devices_[i].u3v_payload_size_;
                    arv_stream_push_buffer (devices_[i].stream_,  arv_buffer_new_allocate (size));
                    bufs[i] = arv_stream_timeout_pop_buffer (devices_[i].stream_, timeout_us);
                    if (bufs[i] == nullptr){
                        log::error("pop_buffer(L1) failed due to timeout ({}s)", timeout_us*1e-6f);
                        throw ::std::runtime_error("Buffer is null");
                    }
                    devices_[i].frame_count_ += 1;
                    memcpy(outs[i].data(), arv_buffer_get_part_data(bufs[i], 0, nullptr), size);
                }
        }else {
            int32_t num_device = num_sensor_;
            std::vector<ArvBuffer *> bufs(num_device);

            // default is OperationMode::Came1USB1
            if (operation_mode_ == OperationMode::Came2USB2 || operation_mode_ == OperationMode::Came1USB1) {

                // if aravis output queue length is more than N (where N > 1) for all devices, pop all N-1 buffer
                if (realtime_display_mode_) {
                    setRealTime(bufs,timeout_us);
                }

                // get the first buffer for each stream
                for (auto i = 0; i < devices_.size(); ++i) {
                    bufs[i] = arv_stream_timeout_pop_buffer(devices_[i].stream_, timeout_us);
                    if (bufs[i] == nullptr) {
                        log::error("pop_buffer(L1) failed due to timeout ({}s)", timeout_us * 1e-6f);
                        throw ::std::runtime_error("Buffer is null");
                    }
                    devices_[i].frame_count_ = is_gendc_
                                               ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(bufs[i],
                                                                                                             devices_[i]))
                                               : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[i]) &
                                                                       0x00000000FFFFFFFF);
                    i == 0 ?
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[i].frame_count_, "") :
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[i].frame_count_);
                }

                if (frame_sync_) {
                    setFrameSync(bufs,timeout_us);
                }

                for (int i = 0; i < num_sensor_; ++i) {
                    auto sz = (std::min)(devices_[i].image_payload_size_, static_cast<int32_t>(outs[i].size_in_bytes()));
                    ::memcpy(outs[i].data(), arv_buffer_get_part_data(bufs[i], 0, nullptr), sz);
                    arv_stream_push_buffer(devices_[i].stream_, bufs[i]);
                    log::trace("Obtained Frame from USB{}: {}", i, devices_[i].frame_count_);
                }

            } else if (operation_mode_ == OperationMode::Came1USB2) {

                uint32_t latest_cnt = 0;
                int32_t min_frame_device_idx = 0;

                // if aravis output queue length is more than N (where N > 1) for all devices, pop all N-1 buffer
                if (realtime_display_mode_) {
                    setRealTime(bufs,timeout_us);
                }

                //first buffer
                cameN_idx_ = (cameN_idx_ + 1) >= num_device ? 0 : cameN_idx_ + 1;
                bufs[cameN_idx_] = arv_stream_timeout_pop_buffer(devices_[cameN_idx_].stream_, 30 * 1000 * 1000);
                if (bufs[cameN_idx_] == nullptr) {
                    log::error("pop_buffer(L4) failed due to timeout ({}s)", timeout_us * 1e-6f);
                    throw ::std::runtime_error("buffer is null");
                }
                devices_[cameN_idx_].frame_count_ = is_gendc_
                                                    ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(
                                bufs[cameN_idx_], devices_[cameN_idx_]))
                                                    : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[cameN_idx_]) &
                                                                            0x00000000FFFFFFFF);
                latest_cnt = devices_[cameN_idx_].frame_count_;
                cameN_idx_ == 0 ?
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[cameN_idx_].frame_count_, "") :
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[cameN_idx_].frame_count_);

            int internal_count = 0;
            int max_internal_count = 1000;

            while (frame_cnt_ >= latest_cnt) {
                arv_stream_push_buffer(devices_[cameN_idx_].stream_, bufs[cameN_idx_]);
                bufs[cameN_idx_] = arv_stream_timeout_pop_buffer (devices_[cameN_idx_].stream_, 30 * 1000 * 1000);
                if (bufs[cameN_idx_] == nullptr){
                    log::error("pop_buffer(L4) failed due to timeout ({}s)", timeout_us*1e-6f);
                    throw ::std::runtime_error("buffer is null");
                }
                devices_[cameN_idx_].frame_count_ = is_gendc_
                        ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(bufs[cameN_idx_], devices_[cameN_idx_]))
                        : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[cameN_idx_]) & 0x00000000FFFFFFFF);
                latest_cnt = devices_[cameN_idx_].frame_count_;

                cameN_idx_ == 0 ?
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[cameN_idx_].frame_count_, "") :
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[cameN_idx_].frame_count_);

                if (internal_count++ > max_internal_count){
                    log::error("pop_buffer(L9) The sequential invalid buffer is more than {}; Stop the pipeline.", max_internal_count);
                    throw ::std::runtime_error("Invalid framecount");
                }
            }

            frame_cnt_ = latest_cnt;
            auto sz = (std::min)(devices_[cameN_idx_].image_payload_size_, static_cast<int32_t>(outs[0].size_in_bytes()));
            ::memcpy(outs[0].data(), arv_buffer_get_part_data(bufs[cameN_idx_], 0, nullptr), sz);
            arv_stream_push_buffer(devices_[cameN_idx_].stream_, bufs[cameN_idx_]);

            log::trace("Obtained Frame from USB{}: {}", cameN_idx_, frame_cnt_);
            }
        }
    }

    void get_frame_count(uint32_t * out){
        if (num_sensor_ != devices_.size()){
            ::memcpy(out, &frame_cnt_, sizeof(uint32_t));
        }else{
            for (int nd = 0; nd < num_sensor_; nd++){
                ::memcpy(out + nd, &devices_[nd].frame_count_, sizeof(uint32_t));
            }
        }
    }

    void get_gendc(std::vector<void *>& outs) {

        // TODO: Is 3 second fine?
        auto timeout_us = 3 * 1000 * 1000;

        int32_t num_device = devices_.size();
        std::vector<ArvBuffer *> bufs(num_device);

        if (operation_mode_ == OperationMode::Came2USB2 || operation_mode_ == OperationMode::Came1USB1){

            if (realtime_display_mode_){
                setRealTime(bufs,timeout_us);
            }

            // get the first buffer for each stream
            for (auto i = 0; i< devices_.size(); ++i) {
                bufs[i] = arv_stream_timeout_pop_buffer (devices_[i].stream_, timeout_us);
                if (bufs[i] == nullptr){
                    log::error("pop_buffer(L5) failed due to timeout ({}s)", timeout_us*1e-6f);
                    throw ::std::runtime_error("buffer is null");
                }
                devices_[i].frame_count_ = is_gendc_
                    ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(bufs[i], devices_[i]))
                    : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[i]) & 0x00000000FFFFFFFF);

                i == 0 ?
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[i].frame_count_, "") :
                    log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[i].frame_count_);
            }

            if (frame_sync_) {
                setFrameSync(bufs,timeout_us);
            }

            for (int i = 0; i < num_sensor_; ++i){
                ::memcpy(outs[i], arv_buffer_get_data(bufs[i], nullptr), devices_[i].u3v_payload_size_);
                // ::memcpy(outs[i*num_sensor_+1], &(devices_[i].header_info_), sizeof(ion::bb::image_io::rawHeader));
                arv_stream_push_buffer(devices_[i].stream_, bufs[i]);
                log::trace("Obtained Frame from USB{}: {}", i, devices_[i].frame_count_);
            }
        } else if (operation_mode_ == OperationMode::Came1USB2) {
            uint32_t latest_cnt = 0;
            int32_t min_frame_device_idx = 0;

            // if aravis output queue length is more than N (where N > 1) for all devices, pop all N-1 buffer
            if (realtime_display_mode_){
                setRealTime(bufs,timeout_us);
            }

            //first buffer
            cameN_idx_ = (cameN_idx_+1) >= num_device ? 0 : cameN_idx_+1;
            bufs[cameN_idx_] = arv_stream_timeout_pop_buffer (devices_[cameN_idx_].stream_, 30 * 1000 * 1000);
            if (bufs[cameN_idx_] == nullptr){
                log::error("pop_buffer(L4) failed due to timeout ({}s)", timeout_us*1e-6f);
                throw ::std::runtime_error("buffer is null");
            }
            devices_[cameN_idx_].frame_count_ = is_gendc_
                    ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(bufs[cameN_idx_], devices_[cameN_idx_]))
                    : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[cameN_idx_]) & 0x00000000FFFFFFFF);
            latest_cnt = devices_[cameN_idx_].frame_count_;
            cameN_idx_ == 0 ?
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[cameN_idx_].frame_count_, "") :
                log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[cameN_idx_].frame_count_);

            int internal_count = 0;
            int max_internal_count = 1000;

                while (frame_cnt_ >= latest_cnt) {
                    arv_stream_push_buffer(devices_[cameN_idx_].stream_, bufs[cameN_idx_]);
                    auto timeout2_us = 30 * 1000 * 1000;
                    bufs[cameN_idx_] = arv_stream_timeout_pop_buffer (devices_[cameN_idx_].stream_, timeout2_us);
                    if (bufs[cameN_idx_] == nullptr){
                        log::error("pop_buffer(L8) failed due to timeout ({}s)", timeout2_us*1e-6f);
                            throw ::std::runtime_error("buffer is null");
                    }
                    devices_[cameN_idx_].frame_count_ = is_gendc_
                            ? static_cast<uint32_t>(get_frame_count_from_genDC_descriptor(bufs[cameN_idx_], devices_[cameN_idx_]))
                            : static_cast<uint32_t>(arv_buffer_get_timestamp(bufs[cameN_idx_]) & 0x00000000FFFFFFFF);
                    cameN_idx_ == 0 ?
                        log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", devices_[cameN_idx_].frame_count_, "") :
                        log::trace("All-Popped Frames (USB0, USB1)=({:20}, {:20})", "", devices_[cameN_idx_].frame_count_);
                    latest_cnt = devices_[cameN_idx_].frame_count_;
                    if (internal_count++ > max_internal_count){
                        log::error("pop_buffer(L10) The sequential invalid buffer is more than {}; Stop the pipeline.", max_internal_count);
                        throw ::std::runtime_error("Invalid framecount");
                    }
                }

            frame_cnt_ = latest_cnt;
            ::memcpy(outs[0], arv_buffer_get_data(bufs[cameN_idx_], nullptr), devices_[cameN_idx_].u3v_payload_size_);
            // ::memcpy(outs[1], &(devices_[cameN_idx_].header_info_), sizeof(ion::bb::image_io::rawHeader));
            arv_stream_push_buffer(devices_[cameN_idx_].stream_, bufs[cameN_idx_]);
            log::trace("Obtained Frame from USB{}: {}", cameN_idx_, frame_cnt_);
        }
    }

    void get_device_info(std::vector<void *>& outs){
        if (sim_mode_||operation_mode_ == OperationMode::Came2USB2 || operation_mode_ == OperationMode::Came1USB1){
            for (int i = 0; i < num_sensor_; ++i){
                ::memcpy(outs[i], &(devices_[i].header_info_), sizeof(ion::bb::image_io::rawHeader));
                log::trace("Obtained Device info USB{}", i);
            }
        } else if (operation_mode_ == OperationMode::Came1USB2) {
            ::memcpy(outs[0], &(devices_[cameN_idx_].header_info_), sizeof(ion::bb::image_io::rawHeader));
            log::trace("Obtained Device info (OperationMode::Came1USB2)");
        }
    }

protected:
    U3V(int32_t num_sensor, bool frame_sync, bool realtime_display_mode, bool sim_mode, int32_t width, int32_t height , float_t fps, const std::string & pixel_format,  char* dev_id = nullptr)
    : gobject_(GOBJECT_FILE, true), aravis_(ARAVIS_FILE, true),
        num_sensor_(num_sensor),
        frame_sync_(frame_sync), realtime_display_mode_(realtime_display_mode), is_gendc_(false), is_param_integer_(false),
        devices_(num_sensor), buffers_(num_sensor), operation_mode_(OperationMode::Came1USB1), frame_cnt_(0), cameN_idx_(-1), disposed_(false), sim_mode_(sim_mode)
    {
        init_symbols();
        log::debug("U3V:: 23-11-18 : updating obtain and write");
        log::info("Using aravis-{}.{}.{}", arv_get_major_version(), arv_get_minor_version(), arv_get_micro_version());

        if (!sim_mode_){
            arv_update_device_list();
            auto n_devices = arv_get_n_devices ();
            if (n_devices == 0){
                log::warn("Fallback to simulation mode: Could not find camera");
                sim_mode_ = true;
            }
        }
    }

    void start_stream_sim(int32_t width, int32_t height, float_t fps, const std::string& pixel_format){
            auto path = std::getenv("GENICAM_FILENAME");
            if (path == nullptr){
                throw std::runtime_error("Please define GENICAM_FILENAME by `set GENICAM_FILENAME=` or `export GENICAM_FILENAME=`");

            }
            pixel_format_ = pixel_format;
            arv_set_fake_camera_genicam_filename (path);

            arv_enable_interface ("Fake");
            log::info("Creating U3V instance with {} fake devices...", num_sensor_);

            auto fake_camera0 = arv_camera_new ("Fake_1", &err_);
            auto fake_device0 = arv_camera_get_device(fake_camera0);
            devices_[0].device_ = fake_device0;
            devices_[0].dev_id_=  "fake_0";
            devices_[0].camera_ = fake_camera0;
            if (num_sensor_==2){
                // aravis only provide on ARV_FAKE_DEVICE_ID https://github.com/Sensing-Dev/aravis/blob/main/src/arvfakeinterface.c
                auto fake_camera1 = arv_camera_new ("Fake_1", &err_);
                auto fake_device1 = arv_camera_get_device(fake_camera1);
                devices_[1].device_ = fake_device1;
                devices_[1].dev_id_=  "fake_1";
                devices_[1].camera_ = fake_camera1;
            }
            // Config fake cameras
            for (int i = 0;i< num_sensor_;i++){
                // setting the params if it is not zero
                log::info("Width {}, Height {} PixelFormat {}...", width, height, pixel_format_);
                arv_device_set_integer_feature_value (devices_[i].device_, "Width", width, &err_);
                arv_device_set_integer_feature_value (devices_[i].device_, "Height", height, &err_);
                arv_device_set_float_feature_value (devices_[i].device_, "AcquisitionFrameRate",fps, &err_);
                if (pixel_format_ != "Mono8")
                    arv_device_set_string_feature_value(devices_[i].device_, "PixelFormat", pixel_format.c_str(), &err_);
                devices_[i].u3v_payload_size_ =  arv_device_get_integer_feature_value (devices_[i].device_, "PayloadSize", &err_);
                auto px =arv_device_get_integer_feature_value(devices_[i].device_, "PixelFormat", &err_);
                auto fps = arv_device_get_float_feature_value(devices_[i].device_, "AcquisitionFrameRate", &err_);
                struct rawHeader header=  { 1, width, height,
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    width, height, width, height, static_cast<float>(fps), px};
                devices_[i].header_info_ = header;
                devices_[i].image_payload_size_ = devices_[i].u3v_payload_size_;
                devices_[i].frame_count_  = 0;

            }

            // Start streaming and start acquisition
            devices_[0].stream_ = arv_device_create_stream (devices_[0].device_, NULL, NULL, &err_);
            if (num_sensor_==2){
                devices_[1].stream_ = arv_device_create_stream (devices_[1].device_, NULL, NULL, &err_);
            }

            for (auto i=0; i<devices_.size(); ++i) {
                arv_device_execute_command(devices_[i].device_, "AcquisitionStart", &err_);
                log::info("\tFake Device {}::{} : {}", i, "Command", "AcquisitionStart");
            }
    }

    void start_stream_no_sim (int n_devices, char* dev_id = nullptr){
        if (n_devices < num_sensor_){
                log::info("{} device is found; but the num_device is set to {}", n_devices, num_sensor_);
                throw std::runtime_error("Device number is not match, please set num_devices again");
        }
        frame_sync_ = num_sensor_ > 1 ? frame_sync_ : false;
        unsigned int target_device_idx;
        if (n_devices != num_sensor_ && dev_id == nullptr) {
            n_devices = num_sensor_;
            log::info("Multiple devices are found; The first device is selected");
        }

        log::info("Creating U3V instance with {} devices...", num_sensor_);
        log::info("Acquisition option::{} is {}", "frame_sync_", frame_sync_);
        log::info("Acquisition option::{} is {}", "realtime_display_mode_", realtime_display_mode_);

        for (int i = 0; i < n_devices; ++i){
            if (dev_id == arv_get_device_id (i) && dev_id != nullptr){
                /* if device id is specified
                TODO: dev_id may be more than 1
                */
                devices_[i].dev_id_ = dev_id;
            }
            else{
                /* if device id is not specified */
                devices_[i].dev_id_ = arv_get_device_id (i);
            }
            log::info("\tDevice/USB {}::{} : {}", i, "DeviceID", devices_[i].dev_id_);

            devices_[i].device_ = arv_open_device(devices_[i].dev_id_, &err_);
            if (err_ ) {
                throw std::runtime_error(err_->message);
            }

            if (devices_[i].device_ == nullptr) {
                throw std::runtime_error("device is null");
            }

            pixel_format_ = arv_device_get_string_feature_value(devices_[i].device_, "PixelFormat", &err_);
            if (err_ ) {
                throw std::runtime_error(err_->message);
            }
            log::info("\tDevice/USB {}::{} : {}", i, "PixelFormat", pixel_format_);

            // Some type of U3V Camera has integer type on gain and exposure
            const char* device_vender_name;
            device_vender_name = arv_device_get_string_feature_value(devices_[i].device_, "DeviceVendorName", &err_);
            if (strcmp(device_vender_name, "Sony Semiconductor Solutions Corporation")==0){
                const char* device_model_name;
                device_model_name = arv_device_get_string_feature_value(devices_[i].device_, "DeviceModelName", &err_);
                if (strcmp(device_model_name, "    ")==0){
                    is_param_integer_ = true;
                }
            }

            // Here PayloadSize is the one for U3V data
            devices_[i].u3v_payload_size_ = arv_device_get_integer_feature_value(devices_[i].device_, "PayloadSize", &err_);
            log::info("\tDevice/USB {}::{} : {}", i, "PayloadSize", devices_[i].u3v_payload_size_);
            if (err_ ) {
                throw std::runtime_error(err_->message);
            }

            devices_[i].stream_ = arv_device_create_stream(devices_[i].device_, nullptr, nullptr, &err_);
            if (err_ ) {
                throw std::runtime_error(err_->message);
            }
            if (devices_[i].stream_ == nullptr) {
                throw std::runtime_error("stream is null");
            }

            // check it the device has gendc mode ==============================
            is_gendc_ = arv_device_is_feature_available(devices_[i].device_, "GenDCDescriptor", &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
            is_gendc_ &= arv_device_is_feature_available(devices_[i].device_, "GenDCStreamingMode", &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }

            // check it the device is gendc mode ===============================
            if (is_gendc_){
                const char * streaming_mode;
                streaming_mode = arv_device_get_string_feature_value(devices_[i].device_, "GenDCStreamingMode", &err_);
                if (err_) {
                    throw std::runtime_error(err_->message);
                }
                is_gendc_ &= (strcmp(streaming_mode, "On")==0);
            }

            // Check each parameters for GenDC device ==========================
            if (is_gendc_){
                log::info("\tDevice/USB {}::{} : {}", i, "GenDC", "Available");
                uint64_t gendc_desc_size = arv_device_get_register_feature_length(devices_[i].device_, "GenDCDescriptor", &err_);
                if (err_) {
                    throw std::runtime_error(err_->message);
                }

                char* buffer;
                buffer = (char*) malloc(gendc_desc_size);
                arv_device_get_register_feature_value(devices_[i].device_, "GenDCDescriptor", gendc_desc_size, (void*)buffer, &err_);
                if (err_) {
                    throw std::runtime_error(err_->message);
                }
                if(isGenDC(buffer)){
                    gendc_descriptor_= ContainerHeader(buffer);
                    std::tuple<int32_t, int32_t> data_comp_and_part = gendc_descriptor_.getFirstAvailableDataOffset(true);
                    if (std::get<0>(data_comp_and_part) == -1){
                        devices_[i].is_data_image_ = false;
                        data_comp_and_part = gendc_descriptor_.getFirstAvailableDataOffset(false);
                        if (std::get<0>(data_comp_and_part) == -1){
                            throw std::runtime_error("None of the data in GenDC is available\n");
                        }
                    }else{
                        devices_[i].is_data_image_ = true;
                    }
                    devices_[i].data_offset_ = gendc_descriptor_.getDataOffset(std::get<0>(data_comp_and_part), std::get<1>(data_comp_and_part));
                    devices_[i].image_payload_size_ = gendc_descriptor_.getDataSize(std::get<0>(data_comp_and_part), std::get<1>(data_comp_and_part));
                    devices_[i].framecount_offset_ = gendc_descriptor_.getOffsetFromTypeSpecific(std::get<0>(data_comp_and_part), std::get<1>(data_comp_and_part), 3, 0);
                }
                free(buffer);
            }else{
                devices_[i].data_offset_ = 0;
                devices_[i].image_payload_size_ = devices_[i].u3v_payload_size_;
                log::info("\tDevice/USB {}::{} : {}", i, "GenDC", "Not Supported");
            }


            // Set Device Info =================================================
            {
                int32_t wi = arv_device_get_integer_feature_value(devices_[i].device_, "Width", &err_);
                int32_t hi = arv_device_get_integer_feature_value(devices_[i].device_, "Height", &err_);
                double fps = 0.0;
                if (arv_device_is_feature_available(devices_[i].device_, "AcquisitionFrameRate", &err_)){
                    fps = arv_device_get_float_feature_value(devices_[i].device_, "AcquisitionFrameRate", &err_);
                }
                log::info("\tDevice/USB {}::{} : {}", i, "Width", wi);
                log::info("\tDevice/USB {}::{} : {}", i, "Height", hi);

                int32_t px =
                    pixel_format_ == "RGB8" ? PFNC_RGB8 :
                    pixel_format_ == "GBR8" ? PFNC_BGR8 :
                    pixel_format_ == "Mono8" ? PFNC_Mono8 :
                    pixel_format_ == "Mono10" ? PFNC_Mono10 :
                    pixel_format_ == "Mono12" ? PFNC_Mono12 :
                    pixel_format_ == "BayerBG8" ? PFNC_BayerBG8 :
                    pixel_format_ == "BayerBG10" ? PFNC_BayerBG10 :
                    pixel_format_ == "BayerBG12" ? PFNC_BayerBG12 :
                    pixel_format_ == "BayerGR8" ? PFNC_BayerGR8 :
                    pixel_format_ == "BayerGR12" ? PFNC_BayerGR12 :
                    pixel_format_ == "YCbCr422_8" ? PFNC_YCbCr422_8 : 0;
                if (px == 0){
                    log::info("The pixel format is not supported for header info");
                }


                devices_[i].header_info_ = { 1, wi, hi,
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    wi, hi, wi, hi, static_cast<float>(fps), px
                };
            }

            if (arv_device_is_feature_available(devices_[0].device_, "OperationMode", &err_)){
                const char* operation_mode_in_string;
                operation_mode_in_string = arv_device_get_string_feature_value(devices_[0].device_, "OperationMode", &err_);
                if (strcmp(operation_mode_in_string, "Came2USB1")==0){
                    operation_mode_ = OperationMode::Came2USB1;
                }else if (strcmp(operation_mode_in_string, "Came1USB1")==0){
                    operation_mode_ = OperationMode::Came1USB1;
                }else if (strcmp(operation_mode_in_string, "Came2USB2")==0){
                    operation_mode_ = OperationMode::Came2USB2;
                }else if (strcmp(operation_mode_in_string, "Came1USB2")==0){
                    operation_mode_ = OperationMode::Came1USB2;
                    n_devices = 2;
                    devices_.resize(n_devices);
                    buffers_.resize(n_devices);
                }
                log::info("\tDevice/USB {}::{} : {}", i, "OperationMode", operation_mode_in_string);
            }
        }

        for (auto i=0; i<devices_.size(); ++i) {
            const size_t buffer_size = 1 * 1024 * 1024 * 1024; // 1GiB for each
            auto n = (buffer_size + devices_[i].u3v_payload_size_ - 1) / devices_[i].u3v_payload_size_;
            for (auto j=0; j<n; ++j) {
                auto b = arv_buffer_new_allocate(devices_[i].u3v_payload_size_);
                buffers_[i].push_back(b);
                arv_stream_push_buffer(devices_[i].stream_, b);
            }
            log::info("\tDevice/USB {}::{} : {}", i, "Buffer Size", buffer_size);
            log::info("\tDevice/USB {}::{} : {}", i, "Number of Buffers", n);

        }

        for (auto i=0; i<devices_.size(); ++i) {
            arv_device_set_string_feature_value(devices_[i].device_, "AcquisitionMode", arv_acquisition_mode_to_string(ARV_ACQUISITION_MODE_CONTINUOUS), &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
            log::info("\tDevice/USB {}::{} : {}", i, "Command", "AcquisitionMode");
        }

        for (auto i=0; i<devices_.size(); ++i) {
            arv_device_execute_command(devices_[i].device_, "AcquisitionStart", &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
            log::info("\tDevice/USB {}::{} : {}", i, "Command", "AcquisitionStart");
        }


    }


    void init_symbols_gobject() {
	if (!gobject_.is_available()) {
            throw ::std::runtime_error("libgobject-2.0 is unavailable on your system.");
        }

        #define GET_SYMBOL(LOCAL_VAR, TARGET_SYMBOL)                    \
            LOCAL_VAR = gobject_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL);   \
            if (LOCAL_VAR == nullptr) {                                 \
                throw ::std::runtime_error(                             \
                    TARGET_SYMBOL " is unavailable on gobject-2.0");     \
            }

        GET_SYMBOL(g_object_unref, "g_object_unref");

        #undef GET_SYMBOL
    }

    void init_symbols_aravis() {
	if (!aravis_.is_available()) {
            throw ::std::runtime_error("libaravis-0.8 is unavailable on your system.");
        }

        #define GET_SYMBOL(LOCAL_VAR, TARGET_SYMBOL)                    \
            LOCAL_VAR = aravis_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL);   \
            if (LOCAL_VAR == nullptr) {                                 \
                throw ::std::runtime_error(                             \
                    TARGET_SYMBOL " is unavailable on aravis-0.8");     \
            }

        GET_SYMBOL(arv_get_major_version, "arv_get_major_version");
        GET_SYMBOL(arv_get_minor_version, "arv_get_minor_version");
        GET_SYMBOL(arv_get_micro_version, "arv_get_micro_version");

        GET_SYMBOL(arv_update_device_list, "arv_update_device_list");
        GET_SYMBOL(arv_get_n_devices, "arv_get_n_devices");

        GET_SYMBOL(arv_get_device_id, "arv_get_device_id");
        GET_SYMBOL(arv_get_device_model, "arv_get_device_model");
        GET_SYMBOL(arv_get_device_serial_nbr, "arv_get_device_serial_nbr");

        GET_SYMBOL(arv_open_device, "arv_open_device");
        GET_SYMBOL(arv_device_set_string_feature_value, "arv_device_set_string_feature_value");
        GET_SYMBOL(arv_device_set_float_feature_value, "arv_device_set_float_feature_value");
        GET_SYMBOL(arv_device_set_integer_feature_value, "arv_device_set_integer_feature_value");
        GET_SYMBOL(arv_device_get_string_feature_value, "arv_device_get_string_feature_value");
        GET_SYMBOL(arv_device_get_integer_feature_value, "arv_device_get_integer_feature_value");
        GET_SYMBOL(arv_device_get_float_feature_value, "arv_device_get_float_feature_value");
        GET_SYMBOL(arv_device_get_integer_feature_bounds, "arv_device_get_integer_feature_bounds");
        GET_SYMBOL(arv_device_get_float_feature_bounds, "arv_device_get_float_feature_bounds");

        GET_SYMBOL(arv_device_get_register_feature_length, "arv_device_get_register_feature_length");
        GET_SYMBOL(arv_device_get_register_feature_value, "arv_device_get_register_feature_value");

        GET_SYMBOL(arv_device_create_stream, "arv_device_create_stream");
        GET_SYMBOL(arv_buffer_new_allocate, "arv_buffer_new_allocate");
        GET_SYMBOL(arv_stream_push_buffer, "arv_stream_push_buffer");
        GET_SYMBOL(arv_stream_get_n_buffers, "arv_stream_get_n_buffers");
        GET_SYMBOL(arv_acquisition_mode_to_string, "arv_acquisition_mode_to_string");
        GET_SYMBOL(arv_device_is_feature_available, "arv_device_is_feature_available");
        GET_SYMBOL(arv_device_execute_command, "arv_device_execute_command");
        GET_SYMBOL(arv_stream_timeout_pop_buffer, "arv_stream_timeout_pop_buffer");
        GET_SYMBOL(arv_buffer_get_status, "arv_buffer_get_status");
        GET_SYMBOL(arv_buffer_get_payload_type, "arv_buffer_get_payload_type");
        GET_SYMBOL(arv_buffer_get_data, "arv_buffer_get_data");
        GET_SYMBOL(arv_buffer_get_part_data, "arv_buffer_get_part_data");
        GET_SYMBOL(arv_buffer_get_timestamp, "arv_buffer_get_timestamp");
        GET_SYMBOL(arv_device_get_feature, "arv_device_get_feature");

        GET_SYMBOL(arv_buffer_has_gendc, "arv_buffer_has_gendc");
        GET_SYMBOL(arv_buffer_get_gendc_descriptor, "arv_buffer_get_gendc_descriptor");

        GET_SYMBOL(arv_shutdown, "arv_shutdown");

        GET_SYMBOL(arv_camera_create_stream, "arv_camera_create_stream");
        GET_SYMBOL(arv_camera_new, "arv_camera_new");
        GET_SYMBOL(arv_camera_get_device, "arv_camera_get_device");
        GET_SYMBOL(arv_fake_device_new, "arv_fake_device_new");
        GET_SYMBOL(arv_set_fake_camera_genicam_filename, "arv_set_fake_camera_genicam_filename");
        GET_SYMBOL(arv_enable_interface, "arv_enable_interface");
        GET_SYMBOL(arv_fake_device_get_fake_camera, "arv_fake_device_get_fake_camera");
        #undef GET_SYMBOL
    }

    void init_symbols() {
        init_symbols_gobject();
        init_symbols_aravis();
    }

    int32_t get_frame_count_from_genDC_descriptor(ArvBuffer * buf, DeviceInfo& d){
        int32_t frame_count = 0;;
        memcpy (&frame_count, ((char *) arv_buffer_get_data(buf, nullptr) + d.framecount_offset_), sizeof(int32_t));
        return frame_count;
    }

    template<typename T>
    GError* Set(ArvDevice* dev_handle, const char* key, T v) {
        return SetFeatureValue(dev_handle, key, v);
    }

    template<typename T>
    GError* Get(ArvDevice* dev_handle, const char* key, T* v) {
        T vp;
        err_ = GetFeatureValue(dev_handle, key, vp);
        *v = vp;
        return err_;
    }

    GError* SetFeatureValue(ArvDevice *device, const char *feature, const char *value){
        arv_device_set_string_feature_value (device, feature, value, &err_);
        return err_;
    }

    GError* SetFeatureValue(ArvDevice *device, const char *feature, double value){
        double min_v, max_v;
        arv_device_get_float_feature_bounds (device, feature, &min_v, &max_v, &err_);
        if (err_ != nullptr) {
            return err_;
        }
        value = (std::max)(min_v, value);
        value = (std::min)(max_v, value);

        arv_device_set_float_feature_value (device, feature, value, &err_);

        return err_;
    }

    GError* SetFeatureValue(ArvDevice *device, const char *feature, int64_t value){
        int64_t min_v, max_v;
        arv_device_get_integer_feature_bounds(device, feature, &min_v, &max_v, &err_);
        if (err_ != nullptr) {
            return err_;
        }
        value = (std::max)(min_v, value);
        value = (std::min)(max_v, value);

        arv_device_set_integer_feature_value (device, feature, value, &err_);
        return err_;
    }

    GError* GetFeatureValue(ArvDevice *device, const char *feature, int64_t& value){
        value = arv_device_get_integer_feature_value(device, feature, &err_);
        return err_;
    }

    static std::map<std::string, std::shared_ptr<U3V>> instances_;
    bool disposed_;
    bool sim_mode_;

private:
    g_object_unref_t g_object_unref;

    arv_get_major_version_t arv_get_major_version;
    arv_get_minor_version_t arv_get_minor_version;
    arv_get_micro_version_t arv_get_micro_version;

    arv_update_device_list_t arv_update_device_list;
    arv_get_n_devices_t arv_get_n_devices;

    arv_get_device_id_t arv_get_device_id;
    arv_get_device_model_t arv_get_device_model;
    arv_get_device_serial_nbr_t arv_get_device_serial_nbr;

    arv_open_device_t arv_open_device;
    arv_device_set_string_feature_value_t arv_device_set_string_feature_value;
    arv_device_set_float_feature_value_t arv_device_set_float_feature_value;
    arv_device_set_integer_feature_value_t arv_device_set_integer_feature_value;

    arv_device_get_string_feature_value_t arv_device_get_string_feature_value;
    arv_device_get_integer_feature_value_t arv_device_get_integer_feature_value;
    arv_device_get_float_feature_value_t arv_device_get_float_feature_value;

    arv_device_get_integer_feature_bounds_t arv_device_get_integer_feature_bounds;
    arv_device_get_float_feature_bounds_t arv_device_get_float_feature_bounds;

    arv_device_is_feature_available_t arv_device_is_feature_available;

    arv_device_get_register_feature_length_t arv_device_get_register_feature_length;
    arv_device_get_register_feature_value_t arv_device_get_register_feature_value;

    arv_device_create_stream_t arv_device_create_stream;

    arv_buffer_new_allocate_t arv_buffer_new_allocate;
    arv_stream_push_buffer_t arv_stream_push_buffer;
    arv_stream_get_n_buffers_t arv_stream_get_n_buffers;
    arv_acquisition_mode_to_string_t arv_acquisition_mode_to_string;
    arv_device_execute_command_t arv_device_execute_command;
    arv_stream_timeout_pop_buffer_t arv_stream_timeout_pop_buffer;
    arv_buffer_get_status_t arv_buffer_get_status;
    arv_buffer_get_payload_type_t arv_buffer_get_payload_type;
    arv_buffer_get_data_t arv_buffer_get_data;
    arv_buffer_get_part_data_t arv_buffer_get_part_data;
    arv_buffer_get_timestamp_t arv_buffer_get_timestamp;
    arv_device_get_feature_t arv_device_get_feature;

    arv_buffer_has_gendc_t arv_buffer_has_gendc;
    arv_buffer_get_gendc_descriptor_t arv_buffer_get_gendc_descriptor;

    arv_shutdown_t arv_shutdown;

    arv_camera_new_t  arv_camera_new;
    arv_camera_get_device_t arv_camera_get_device;
    arv_camera_create_stream_t arv_camera_create_stream;

    arv_fake_device_new_t arv_fake_device_new;
    arv_enable_interface_t arv_enable_interface;
    arv_set_fake_camera_genicam_filename_t   arv_set_fake_camera_genicam_filename;
    arv_fake_device_get_fake_camera_t arv_fake_device_get_fake_camera;


    int32_t num_sensor_;

    DynamicModule gobject_;
    DynamicModule aravis_;
    GError *err_ = nullptr;

    bool frame_sync_;
    bool realtime_display_mode_;
    bool is_gendc_;
    bool is_param_integer_;
    int32_t operation_mode_;

    uint32_t frame_cnt_;
    int32_t cameN_idx_;

    // genDC
    ContainerHeader gendc_descriptor_;

    std::string pixel_format_;

    std::vector<DeviceInfo> devices_;

    std::vector<std::vector<ArvBuffer*> > buffers_;

}; // class U3V

std::map<std::string, std::shared_ptr<U3V>>  U3V::instances_;


class U3VFakeCam : public U3V{
public:
     static U3V & get_instance(const std::string& id,
                              int32_t num_sensor,
                              int32_t width = 640,
                              int32_t height = 480,
                              float_t fps = 25.0,
                              const std::string& pixel_format = "Mono8"
                              )
    {
        if (instances_.count(id) == 0) {
            ion::log::info("Create U3VFakeCam U3V instance: {}", id);
            instances_[id] = std::unique_ptr<U3V>(new U3VFakeCam(num_sensor,  width, height, fps, pixel_format));
        }

        return *instances_[id].get();
}
private:
    U3VFakeCam(int32_t num_sensor, int32_t width, int32_t height , float_t fps, const std::string & pixel_format,  char* dev_id = nullptr)
     : U3V(num_sensor,  false, false, true,  width, height , fps, pixel_format,  nullptr){
         start_stream_sim(width, height, fps, pixel_format);
    };

};


class U3VRealCam: public U3V{
public:
    static U3V & get_instance(const std::string& id,
                              int32_t num_sensor,
                              bool frame_sync,
                              bool realtime_display_mode,
                              bool sim_mode = false,
                              int32_t width = 640,
                              int32_t height = 480,
                              float_t fps = 25.0,
                              const std::string& pixel_format = "Mono8"
                              )
    {
        if (instances_.count(id) == 0) {
            ion::log::info("Create U3VRealCam instance: {}", id);
            instances_[id] = std::unique_ptr<U3V>(new U3VRealCam(num_sensor, frame_sync, realtime_display_mode, sim_mode, width, height, fps, pixel_format));
        }

        return *instances_[id].get();
    }


private:
    U3VRealCam(int32_t num_sensor, bool frame_sync, bool realtime_display_mode, bool sim_mode, int32_t width, int32_t height , float_t fps, const std::string & pixel_format,  char* dev_id = nullptr)
     : U3V(num_sensor,  frame_sync, realtime_display_mode, sim_mode,  width, height , fps, pixel_format,  nullptr){
        // check if the camera is available
        if (sim_mode_){
            start_stream_sim(width, height, fps, pixel_format);
        }else{
            start_stream_no_sim(num_sensor, dev_id);
        }
    };
};

}  // namespace image_io
}  // namespace bb
}  // namespace ion

extern "C"
int ION_EXPORT u3v_dispose(const char *id) {
    ion::bb::image_io::U3V::release_instance(id);
    return 0;
}

int u3v_camera_frame_count(
    const std::string& id, int32_t num_sensor, bool frame_sync, bool realtime_display_mode,
    halide_buffer_t* out)
{
    try {
        auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, num_sensor, frame_sync, realtime_display_mode));
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = num_sensor;
            return 0;
        }
        else {
            u3v.get_frame_count(reinterpret_cast<uint32_t*>(out->host));
        }

        return 0;
    } catch (const std::exception &e) {
        ion::log::error("frame_count");
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera1(
    bool frame_sync, bool realtime_display_mode, double gain0, double exposure0,
    halide_buffer_t * id_buf, halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    halide_buffer_t * out0)
{
    using namespace Halide;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));

        auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 1, frame_sync, realtime_display_mode));
        if (out0->is_bounds_query()) {
            //bounds query
            return 0;
        }else{
            // set gain & exposure
            u3v.SetGain(0, gain_key, gain0);
            u3v.SetExposure(0, exposure_key, exposure0);

            std::vector<Halide::Buffer<> > obufs{Halide::Buffer<>(*out0)};
            u3v.get(obufs);
        }

        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera2(
    bool frame_sync, bool realtime_display_mode, double gain0, double gain1, double exposure0, double exposure1,
    halide_buffer_t * id_buf, halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    halide_buffer_t * out0, halide_buffer_t * out1)
{
    using namespace Halide;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 2, frame_sync, realtime_display_mode));
        if (out0->is_bounds_query() || out1->is_bounds_query()) {
            //bounds query
            return 0;
        }else{
            // set gain & exposure
            u3v.SetGain(0, gain_key, gain0);
            u3v.SetGain(1, gain_key, gain1);
            u3v.SetExposure(0, exposure_key, exposure0);
            u3v.SetExposure(1, exposure_key, exposure1);

            std::vector<Halide::Buffer<> > obufs{Halide::Buffer<>(*out0), Halide::Buffer<>(*out1)};
            u3v.get(obufs);
        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera2);


extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera1_frame_count(
    halide_buffer_t *,
    int32_t num_sensor, bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * id_buf, halide_buffer_t* out)
{
    const std::string id(reinterpret_cast<const char *>(id_buf->host));
    return u3v_camera_frame_count(id, num_sensor, frame_sync, realtime_display_mode, out);
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera1_frame_count);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera2_frame_count(
    halide_buffer_t *,
    halide_buffer_t *,
    int32_t num_sensor, bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * id_buf, halide_buffer_t* out)
{    const std::string id(reinterpret_cast<const char *>(id_buf->host));
    return u3v_camera_frame_count(id, num_sensor, frame_sync, realtime_display_mode, out);
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera2_frame_count);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_gendc_camera1(
    halide_buffer_t * id_buf,
    bool frame_sync, bool realtime_display_mode, bool enable_control,
    halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    double gain0, double exposure0,
    halide_buffer_t * out_gendc
    )
{
    using namespace Halide;
    int num_output = 1;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, num_output, frame_sync, realtime_display_mode));
        if (out_gendc->is_bounds_query()) {
            return 0;
        }
        // set gain & exposure
        if (enable_control){
            ion::log::debug("Setting gain0:{} exposure0:{}", gain0, exposure0);
            u3v.SetGain(0, gain_key, gain0);
            u3v.SetExposure(0, exposure_key, exposure0);
        }
        std::vector<void *> obufs{out_gendc->host};
        u3v.get_gendc(obufs);

        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_gendc_camera1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_gendc_camera2(
    halide_buffer_t * id_buf,
    bool frame_sync, bool realtime_display_mode, bool enable_control,
    halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    double gain0, double exposure0,
    double gain1, double exposure1,
    halide_buffer_t * out_gendc0, halide_buffer_t * out_gendc1
    )
{
    using namespace Halide;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 2, frame_sync, realtime_display_mode));
        if (out_gendc0->is_bounds_query() || out_gendc1->is_bounds_query() ) {
            return 0;
        }else{
            // set gain & exposure
            if (enable_control) {
                ion::log::debug("Setting gain0:{} exposure0:{}", gain0, exposure0);
                u3v.SetGain(0, gain_key, gain0);
                u3v.SetExposure(0, exposure_key, exposure0);

                ion::log::debug("Setting gain1:{} exposure1:{}", gain1, exposure1);
                u3v.SetGain(1, gain_key, gain1);
                u3v.SetExposure(1, exposure_key, exposure1);
            }

            std::vector<void *> obufs{out_gendc0->host, out_gendc1->host};
            u3v.get_gendc(obufs);

        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_gendc_camera2);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_multiple_camera1(
    halide_buffer_t * id_buf,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode,
    bool enable_control,
    halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf, halide_buffer_t * pixel_format_buf,
    double gain0, double exposure0,
    halide_buffer_t * out0)
{
    using namespace Halide;
    int num_output = 1;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        const std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));
        std::vector<Halide::Buffer<>> obufs{Halide::Buffer<>(*out0)};
        if (out0->is_bounds_query()) {
            return 0;
        }
        if(force_sim_mode){
             auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, num_output, width, height, fps, pixel_format));
             u3v.get(obufs);
        }else{
             auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, num_output, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
             if (enable_control) {
                 // set gain & exposure
                ion::log::debug("Setting gain0:{} exposure0:{}", gain0, exposure0);
                u3v.SetGain(0, gain_key, gain0);
                u3v.SetExposure(0, exposure_key, exposure0);
             }
             u3v.get(obufs);
        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_multiple_camera1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_multiple_camera2(
    halide_buffer_t * id_buf,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode, bool enable_control,
    halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf, halide_buffer_t * pixel_format_buf,
    double gain0, double exposure0,
    double gain1, double exposure1,
    halide_buffer_t * out0, halide_buffer_t * out1)
{
    using namespace Halide;
    int num_output = 2;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));
        std::vector<Halide::Buffer<>> obufs{Halide::Buffer<>(*out0), Halide::Buffer<>(*out1)};
        if (out0->is_bounds_query() || out1->is_bounds_query()) {
            return 0;
        }
        if(force_sim_mode){
             auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, num_output, width, height, fps, pixel_format));
             u3v.get(obufs);
        }else{
             auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, num_output, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
             if (enable_control) {
                 // set gain & exposure
                ion::log::debug("Setting gain0:{} exposure0:{}", gain0, exposure0);
                u3v.SetGain(0, gain_key, gain0);
                u3v.SetExposure(0, exposure_key, exposure0);
                ion::log::debug("Setting gain1:{} exposure1:{}", gain1, exposure1);
                u3v.SetGain(1, gain_key, gain1);
                u3v.SetExposure(1, exposure_key, exposure1);
            }
             u3v.get(obufs);
        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_multiple_camera2);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_multiple_camera_frame_count1(
    halide_buffer_t *,
    halide_buffer_t * id_buf, int32_t num_sensor,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * pixel_format_buf,
    halide_buffer_t* out)
{

    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = num_sensor;
            return 0;
        }
        if(force_sim_mode){
              auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, 1, width, height, fps, pixel_format));
              u3v.get_frame_count(reinterpret_cast<uint32_t*>(out->host));
        }else{
             auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 1, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
             u3v.get_frame_count(reinterpret_cast<uint32_t*>(out->host));
        }

        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown when get frame count: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_multiple_camera_frame_count1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_multiple_camera_frame_count2(
    halide_buffer_t *,
    halide_buffer_t *,
    halide_buffer_t * id_buf, int32_t num_sensor,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * pixel_format_buf,
    halide_buffer_t* out)
{
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = num_sensor;
            return 0;
        }
        if(force_sim_mode){
            auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, 2, width, height, fps, pixel_format));
            u3v.get_frame_count(reinterpret_cast<uint32_t*>(out->host));
        }else{
            auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 2, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
            u3v.get_frame_count(reinterpret_cast<uint32_t*>(out->host));
        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown when get frame count: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }

}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_multiple_camera_frame_count2);


extern "C"
int ION_EXPORT ion_bb_image_io_u3v_device_info1(
    halide_buffer_t *,
    halide_buffer_t * id_buf, int32_t num_sensor,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * pixel_format_buf,
    halide_buffer_t * out_deviceinfo
    )
{
    using namespace Halide;
    int num_output = 1;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        const std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));

        if (out_deviceinfo->is_bounds_query()){
            out_deviceinfo->dim[0].min = 0;
            out_deviceinfo->dim[0].extent = sizeof(ion::bb::image_io::rawHeader);
            return 0;
        }
        std::vector<void *> obufs{out_deviceinfo->host};
        if(force_sim_mode){
            auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, 2, width, height, fps, pixel_format));
            u3v.get_device_info(obufs);
        }else{
            auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 2, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
            u3v.get_device_info(obufs);
        }

        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_device_info1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_device_info2(
    halide_buffer_t *, halide_buffer_t *,
    halide_buffer_t * id_buf, int32_t num_sensor,
    bool force_sim_mode,
    int32_t width, int32_t height, float_t fps,
    bool frame_sync, bool realtime_display_mode,
    halide_buffer_t * pixel_format_buf,
    halide_buffer_t * deviceinfo0, halide_buffer_t * deviceinfo1
    )
{

    using namespace Halide;
    try {
        const std::string id(reinterpret_cast<const char *>(id_buf->host));
        int num_output = 2;
        const std::string pixel_format(reinterpret_cast<const char *>(pixel_format_buf->host));

        if (deviceinfo0->is_bounds_query() || deviceinfo1->is_bounds_query()) {
            if (deviceinfo0->is_bounds_query()){
                deviceinfo0->dim[0].min = 0;
                deviceinfo0->dim[0].extent = sizeof(ion::bb::image_io::rawHeader);
            }
            if (deviceinfo1->is_bounds_query()){
                deviceinfo1->dim[0].min = 0;
                deviceinfo1->dim[0].extent = sizeof(ion::bb::image_io::rawHeader);
            }
            return 0;
        }
        std::vector<void *> obufs{deviceinfo0->host, deviceinfo1->host};
        if(force_sim_mode){
            auto &u3v(ion::bb::image_io::U3VFakeCam::get_instance(id, 2, width, height, fps, pixel_format));
            u3v.get_device_info(obufs);
        }else{
            auto &u3v(ion::bb::image_io::U3VRealCam::get_instance(id, 2, frame_sync, realtime_display_mode, force_sim_mode, width, height, fps, pixel_format));
            u3v.get_device_info(obufs);
        }
        return 0;
    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_device_info2);
#endif
