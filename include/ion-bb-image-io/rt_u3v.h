#ifndef ION_BB_IMAGE_IO_RT_U3V_H
#define ION_BB_IMAGE_IO_RT_U3V_H

#include <cstring>
#include <iostream>
#include "rt_common.h"

#include <HalideBuffer.h>

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

    using ArvInterface_t = struct ArvInterface*;
    using ArvDevice_t = struct ArvDevice*;
    using ArvStream_t = struct ArvStream*;
    using ArvStreamCallback_t = struct ArvStreamCallback*;
    using ArvBuffer_t = struct ArvBuffer*;
    using ArvGcNode_t = struct ArvGcNode*;

    using arv_update_device_list_t = void(*)();
    using arv_get_n_devices_t = unsigned int(*)();

    using arv_get_device_id_t = const char*(*)(unsigned int);
    using arv_get_device_model_t = const char*(*)(unsigned int);
    using arv_get_device_serial_nbr_t = const char*(*)(unsigned int);
    using arv_open_device_t = ArvDevice*(*)(const char*, GError**);

    using arv_device_set_string_feature_value_t = void(*)(ArvDevice*, const char*, const char*, GError**);
    using arv_device_set_float_feature_value_t = void(*)(ArvDevice*, const char*, float, GError**);
    using arv_device_set_integer_feature_value_t = void(*)(ArvDevice*, const char*, int64_t, GError**);

    using arv_device_get_string_feature_value_t = const char *(*)(ArvDevice*, const char*, GError**);
    using arv_device_get_integer_feature_value_t = int(*)(ArvDevice*, const char*, GError**);
    using arv_device_get_float_feature_value_t = float(*)(ArvDevice*, const char*, GError**);

    using arv_device_get_integer_feature_bounds_t = void(*)(ArvDevice*, const char*, int64_t*, int64_t*, GError**);

    using arv_device_create_stream_t = ArvStream*(*)(ArvDevice*, ArvStreamCallback*, void*, GError**);

    using arv_buffer_new_allocate_t = ArvBuffer*(*)(size_t);
    using arv_stream_push_buffer_t = void(*)(ArvStream*, ArvBuffer*);

    using arv_acquisition_mode_to_string_t = const char*(*)(ArvAcquisitionMode);
    using arv_device_execute_command_t = void(*)(ArvDevice*, const char*, GError**);
    using arv_stream_timeout_pop_buffer_t = ArvBuffer*(*)(ArvStream*, uint64_t);
    using arv_buffer_get_status_t = ArvBufferStatus(*)(ArvBuffer*);
    using arv_buffer_get_payload_type_t = ArvBufferPayloadType(*)(ArvBuffer*);
    using arv_buffer_get_data_t = void*(*)(ArvBuffer*, size_t*);
    using arv_buffer_get_timestamp_t = uint64_t(*)(ArvBuffer*);
    using arv_device_get_feature_t = ArvGcNode*(*)(ArvDevice*, const char*);

    using arv_shutdown_t = void(*)(void);

    typedef struct {
        const char* dev_id_;
        ArvDevice* device_;

        int32_t payload_size_;
        uint64_t frame_count_;

        float gain_;
        float exposure_;

        float exposure_range_[2];

        ArvStream* stream_;
    } DeviceInfo;

    public:
    static U3V & get_instance(std::string pixel_format, int32_t num_sensor, bool frame_sync)
    {
        if (instance_ == nullptr){
            instance_ = std::unique_ptr<U3V>(new U3V(pixel_format, num_sensor, frame_sync));
        }
        return *instance_;
    }

    void dispose(){

        for (auto i=0; i<devices_.size(); ++i) {
            auto d = devices_[i];
            arv_device_execute_command(d.device_, "AcquisitionStop", nullptr);

            /*
                Note:
                unref stream also unref the buffers pushed to stream
                all buffers are in stream so do not undef buffres separately
            */
            g_object_unref(reinterpret_cast<gpointer>(d.stream_));
            g_object_unref(reinterpret_cast<gpointer>(d.device_));
        }

        arv_shutdown();
        instance_.reset(nullptr);
    }

    void SetGain(int32_t sensor_idx, const std::string key, int32_t v) {
        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].gain_ != v){
                err_ =  Set(devices_[sensor_idx].device_, key.c_str(), static_cast<int64_t>(v));
                devices_[sensor_idx].gain_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

    void SetExposure(int32_t sensor_idx, const std::string key, int32_t v) {
        if (sensor_idx < num_sensor_ ){
            if(devices_[sensor_idx].exposure_ != v){
                err_ = Set(devices_[sensor_idx].device_, key.c_str(), static_cast<int64_t>(v));
                devices_[sensor_idx].exposure_ = v;
            }
            return;
        }else{
            throw std::runtime_error("the index number " + std::to_string(sensor_idx) + " exceeds the number of sensor " + std::to_string(num_sensor_));
        }
    }

    uint32_t get_frame_count(){
        /* assume frame_sync is ON */
        return devices_[0].frame_count_;
    }

    void get(std::vector<void *>& outs) {
        std::vector<ArvBuffer *> bufs(devices_.size());
        for (auto i = 0; i< devices_.size(); ++i) {
            bufs[i] = arv_stream_timeout_pop_buffer (devices_[i].stream_, 3 * 1000 * 1000);
            if (bufs[i] == nullptr){
                throw ::std::runtime_error("buffer is null");
            }
            devices_[i].frame_count_ = static_cast<uint64_t>(arv_buffer_get_timestamp(bufs[i]) & 0x00000000FFFFFFFF);
        }

        if (frame_sync_) {
            uint32_t max_cnt = 0;
            while (true) {
                // Update max_cnt
                for (int i=0; i<num_sensor_; ++i) {
                    if (max_cnt < devices_[i].frame_count_) {
                        max_cnt = devices_[i].frame_count_;
                    }
                }

                // Check all count is same as max_cnt;
                bool synchronized = true;
                for (int i=0; i<num_sensor_; ++i) {
                    synchronized &= devices_[i].frame_count_ == max_cnt;
                }

                // If it is synchronized, break the loop
                if (synchronized) {
                    break;
                }

                // Acquire buffer until cnt is at least max_cnt
                for (int i=0; i<devices_.size(); ++i) {
                    while (devices_[i].frame_count_ < max_cnt) {
                        arv_stream_push_buffer(devices_[i].stream_, bufs[i]);
                        bufs[i] = arv_stream_timeout_pop_buffer (devices_[i].stream_, 3 * 1000 * 1000);
                        if (bufs[i] == nullptr){
                            throw ::std::runtime_error("buffer is null");
                        }
                        devices_[i].frame_count_ = static_cast<uint64_t>(arv_buffer_get_timestamp(bufs[i]) & 0x00000000FFFFFFFF);
                    }
                }
            }
        }

        for (int i = 0; i < num_sensor_; ++i){
            ::memcpy(outs[i], arv_buffer_get_data(bufs[i], nullptr), devices_[i].payload_size_);
            arv_stream_push_buffer(devices_[i].stream_, bufs[i]);
        }
    }

    private:
    U3V(std::string pixel_format, int32_t num_sensor, bool frame_sync, char* dev_id = nullptr)
    : gobject_(GOBJECT_FILE, true), aravis_(ARAVIS_FILE, true), pixel_format_(pixel_format), num_sensor_(num_sensor), frame_sync_(frame_sync), devices_(num_sensor), buffers_(num_sensor)
    {
        init_symbols();

        arv_update_device_list();

        unsigned int n_devices = arv_get_n_devices ();

        if (n_devices < num_sensor_){
            throw std::runtime_error("Device not found");
        }
        frame_sync_ = num_sensor_ > 1 ? frame_sync_ : false;

        unsigned int target_device_idx;

        if (n_devices == num_sensor_ || dev_id != nullptr) {
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

                devices_[i].device_ = arv_open_device(devices_[i].dev_id_, &err_);

                // TODO: checking the device status here

                if (err_ ) {
                    throw std::runtime_error(err_->message);
                }
                if (devices_[i].device_ == nullptr) {
                    throw std::runtime_error("device is null");
                }

                arv_device_set_string_feature_value(devices_[i].device_, "PixelFormat", pixel_format_.c_str(), &err_);
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

                devices_[i].payload_size_ = arv_device_get_integer_feature_value(devices_[i].device_, "PayloadSize", &err_);
                if (err_) {
                    throw std::runtime_error(err_->message);
                }
            }
        } else {
            throw std::runtime_error("Multiple devices are found; please set the right Device ID");
        }

        for (auto i=0; i<devices_.size(); ++i) {
            const size_t buffer_size = 1 * 1024 * 1024 * 1024; // 1GiB for each
            auto n = (buffer_size + devices_[i].payload_size_ - 1) / devices_[i].payload_size_;
            for (auto j=0; j<n; ++j) {
                auto b = arv_buffer_new_allocate(devices_[i].payload_size_);
                buffers_[i].push_back(b);
                arv_stream_push_buffer(devices_[i].stream_, b);
            }
        }

        for (auto d : devices_) {
            arv_device_set_string_feature_value(d.device_, "AcquisitionMode", arv_acquisition_mode_to_string(ARV_ACQUISITION_MODE_CONTINUOUS), &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
        }

        for (auto d : devices_) {
            arv_device_execute_command(d.device_, "AcquisitionStop", &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
        }

        for (auto d : devices_) {
            arv_device_execute_command(d.device_, "AcquisitionStart", &err_);
            if (err_) {
                throw std::runtime_error(err_->message);
            }
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
        GET_SYMBOL(arv_device_create_stream, "arv_device_create_stream");
        GET_SYMBOL(arv_buffer_new_allocate, "arv_buffer_new_allocate");
        GET_SYMBOL(arv_stream_push_buffer, "arv_stream_push_buffer");
        GET_SYMBOL(arv_acquisition_mode_to_string, "arv_acquisition_mode_to_string");
        GET_SYMBOL(arv_device_execute_command, "arv_device_execute_command");
        GET_SYMBOL(arv_stream_timeout_pop_buffer, "arv_stream_timeout_pop_buffer");
        GET_SYMBOL(arv_buffer_get_status, "arv_buffer_get_status");
        GET_SYMBOL(arv_buffer_get_payload_type, "arv_buffer_get_payload_type");
        GET_SYMBOL(arv_buffer_get_data, "arv_buffer_get_data");
        GET_SYMBOL(arv_buffer_get_timestamp, "arv_buffer_get_timestamp");
        GET_SYMBOL(arv_device_get_feature, "arv_device_get_feature");

        GET_SYMBOL(arv_shutdown, "arv_shutdown");

        #undef GET_SYMBOL
    }

    void init_symbols(){
        init_symbols_gobject();
        init_symbols_aravis();
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

    g_object_unref_t g_object_unref;

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

    arv_device_create_stream_t arv_device_create_stream;

    arv_buffer_new_allocate_t arv_buffer_new_allocate;
    arv_stream_push_buffer_t arv_stream_push_buffer;
    arv_acquisition_mode_to_string_t arv_acquisition_mode_to_string;
    arv_device_execute_command_t arv_device_execute_command;
    arv_stream_timeout_pop_buffer_t arv_stream_timeout_pop_buffer;
    arv_buffer_get_status_t arv_buffer_get_status;
    arv_buffer_get_payload_type_t arv_buffer_get_payload_type;
    arv_buffer_get_data_t arv_buffer_get_data;
    arv_buffer_get_timestamp_t arv_buffer_get_timestamp;
    arv_device_get_feature_t arv_device_get_feature;

    arv_shutdown_t arv_shutdown;

    static std::unique_ptr<U3V> instance_;
    int32_t num_sensor_;

    DynamicModule gobject_;
    DynamicModule aravis_;
    GError *err_ = nullptr;

    bool frame_sync_;

    std::string pixel_format_;

    std::vector<DeviceInfo> devices_;

    std::vector<std::vector<ArvBuffer*> > buffers_;

}; // class U3V

std::unique_ptr<U3V> U3V::instance_;

int u3v_camera_frame_count(
    bool dispose, int32_t num_sensor, bool frame_sync, halide_buffer_t * pixel_format_buf,
    halide_buffer_t* out)
{
    try {
        const ::std::string pixel_format(reinterpret_cast<const char*>(pixel_format_buf->host));
        auto &u3v(ion::bb::image_io::U3V::get_instance(pixel_format, num_sensor, frame_sync));
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = 1;
        }
        else {
            * reinterpret_cast<uint32_t*>(out->host) = u3v.get_frame_count();
            if(dispose){
                u3v.dispose();
            }
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

}  // namespace image_io
}  // namespace bb
}  // namespace ion

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera1(
    bool frame_sync, int32_t gain0, int32_t exposure0,
    halide_buffer_t* pixel_format_buf, halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    halide_buffer_t * out0)
{
    using namespace Halide;
    try {
        const ::std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const ::std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        const ::std::string pixel_format(reinterpret_cast<const char*>(pixel_format_buf->host));
        auto &u3v(ion::bb::image_io::U3V::get_instance(pixel_format, 1, frame_sync));
        if (out0->is_bounds_query()) {
            //bounds query
            return 0;
        }else{
            // set gain & exposure
            u3v.SetGain(0, gain_key, gain0);
            u3v.SetExposure(0, exposure_key, exposure0);

            std::vector<void *> obufs{out0->host};
            u3v.get(obufs);
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
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera1);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera2(
    bool frame_sync, int32_t gain0, int32_t gain1, int32_t exposure0, int32_t exposure1,
    halide_buffer_t* pixel_format_buf, halide_buffer_t * gain_key_buf, halide_buffer_t * exposure_key_buf,
    halide_buffer_t * out0, halide_buffer_t * out1)
{
    using namespace Halide;
    try {
        const ::std::string gain_key(reinterpret_cast<const char*>(gain_key_buf->host));
        const ::std::string exposure_key(reinterpret_cast<const char*>(exposure_key_buf->host));
        const ::std::string pixel_format(reinterpret_cast<const char*>(pixel_format_buf->host));
        auto &u3v(ion::bb::image_io::U3V::get_instance(pixel_format, 2, frame_sync));
        if (out0->is_bounds_query() || out1->is_bounds_query()) {
            //bounds query
            return 0;
        }else{
            // set gain & exposure
            u3v.SetGain(0, gain_key, gain0);
            u3v.SetGain(1, gain_key, gain1);
            u3v.SetExposure(0, exposure_key, exposure0);
            u3v.SetExposure(1, exposure_key, exposure1);

            std::vector<void *> obufs{out0->host, out1->host};
            u3v.get(obufs);
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
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera2);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera1_frame_count(
    halide_buffer_t *,
    bool dispose, int32_t num_sensor, bool frame_sync, halide_buffer_t * pixel_format_buf,
    halide_buffer_t* out)
{
    return ion::bb::image_io::u3v_camera_frame_count(dispose, num_sensor, frame_sync, pixel_format_buf, out);
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera1_frame_count);

extern "C"
int ION_EXPORT ion_bb_image_io_u3v_camera2_frame_count(
    halide_buffer_t *,
    halide_buffer_t *,
    bool dispose, int32_t num_sensor, bool frame_sync, halide_buffer_t * pixel_format_buf,
    halide_buffer_t* out)
{
    return ion::bb::image_io::u3v_camera_frame_count(dispose, num_sensor, frame_sync, pixel_format_buf, out);
}
ION_REGISTER_EXTERN(ion_bb_image_io_u3v_camera2_frame_count);

#endif
