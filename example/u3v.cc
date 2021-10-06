#include <fstream>
#include <iostream>
#include <string>

// to display
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ion
#include <ion/ion.h>
#include "ion-bb-core/bb.h"
#include "ion-bb-core/rt.h"
#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-io/rt.h"
#include "ion-bb-image-processing/bb.h"
#include "ion-bb-image-processing/rt.h"

#include "ion-bb-image-io/ghc/filesystem.hpp"

using namespace ion;

void display_and_save(int32_t width, int32_t height, std::string directory_path, rawHeader header_info){

    Builder b;
    b.set_target(Halide::get_host_target());

    Port dispose{ "dispose", Halide::type_of<bool>() };
    Port gain0_p{ "gain0", Halide::type_of<int32_t>() };
    Port gain1_p{ "gain1", Halide::type_of<int32_t>() };
    Port exposure0_p{ "exposure0", Halide::type_of<int32_t>() };
    Port exposure1_p{ "exposure1", Halide::type_of<int32_t>() };
    Port wp{ "width", Halide::type_of<int32_t>() };
    Port hp{ "height", Halide::type_of<int32_t>() };

    // obtain sensor images
    auto n = b.add("u3v_camera")(gain0_p, gain1_p, exposure0_p, exposure1_p)
        .set_param(
            Param{"pixel_format_ptr", "Mono12"},
            Param{"num_sensor", "2"},
            Param{"frame_sync", "true"},
            Param{"gain_key", "SensorGain"},
            Param{"exposure_key", "SensorShutter"}
        );
    Port lp = n["output0"];
    Port rp = n["output1"];
    Port fcp = n["frame_count"];

    n = b.add("image_io_binarysaver")(rp, lp, fcp, dispose, wp, hp).set_param(
            Param{"output_directory", directory_path},
            Param{"fps", "60.0"});
    Port terminator = n["output"];

    // TODO replace the following section (bayer2BGR)
    //      with the exisiting BBs
    n = b.add("image_io_bayer2bgr")(lp, wp, hp);
    lp = n["output"];
    n = b.add("image_io_bayer2bgr")(rp, wp, hp);
    rp = n["output"];
    n = b.add("core_reorder_buffer_3d_uint8")(lp).set_param(Param{"dim0", "1"}, Param{"dim1", "2"}, Param{"dim2", "0"});
    lp = n["output"];
    n = b.add("core_reorder_buffer_3d_uint8")(rp).set_param(Param{"dim0", "1"}, Param{"dim1", "2"}, Param{"dim2", "0"});
    rp = n["output"];

    // display images
    n = b.add("image_io_gui_display")(lp).set_param(
      Param{"idx", "0"},
      Param{"width", std::to_string(width)},
      Param{"height", std::to_string(height)}
      );
    Port display_output0_p = n["output"];
    n = b.add("image_io_gui_display")(rp).set_param(
      Param{"idx", "1"},
      Param{"width", std::to_string(width)},
      Param{"height", std::to_string(height)});
    Port display_output1_p = n["output"];

    PortMap pm;
    /* input */
    pm.set(dispose, false);
    pm.set(wp, width);
    pm.set(hp, height);

    /* output */
    Halide::Buffer<int> out0 = Halide::Buffer<int>::make_scalar();
    Halide::Buffer<int> out1 = Halide::Buffer<int>::make_scalar();
    pm.set(display_output0_p, out0);
    pm.set(display_output1_p, out1);

    Halide::Buffer<int32_t> out = Halide::Buffer<int32_t>::make_scalar();
    pm.set(terminator, out);

    int32_t gain0 = 0;
    int32_t gain1 = 480;
    int32_t exposure0 = 1000;
    int32_t exposure1 = 1000;

    for (int i=0; i< 400; ++i) {
        pm.set(gain0_p, gain0++);
        pm.set(gain1_p, gain1--);
        pm.set(exposure0_p, exposure0);
        pm.set(exposure1_p, exposure1);
        b.run(pm);
    }

    pm.set(dispose, true);
    b.run(pm);

    cv::destroyAllWindows();
}

void open_and_check(int32_t& width, int32_t& height, const ghc::filesystem::path output_directory, uint32_t& file_idx, std::ifstream& ifs, bool *finished) {
    auto file_path = output_directory / ("raw-" + ::std::to_string(file_idx++) + ".bin");

    ifs = ::std::ifstream(file_path, ::std::ios::binary);
    if (ifs.fail()) {
        *finished = true;
        return;
    }

    int32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&width), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&height), sizeof(int32_t));

    ifs = ::std::ifstream(file_path, ::std::ios::binary);

    // skip header (size is 512)
    ifs.seekg(512, ::std::ios_base::beg);
}

bool load_header_file(ghc::filesystem::path output_directory, rawHeader header_info)
{
    std::ifstream ifs;
    int  width_, height_;
    bool finished_ = false;
    uint32_t file_idx_ = 0;

    // first look
    open_and_check(width_, height_, output_directory, file_idx_, ifs, &finished_);
    if (finished_) { return false; }

    bool ret = true;
    ret = ret && width_ == header_info.width_;
    ret = ret && height_ == header_info.height_;

    auto log_filename = output_directory / "frame_log.txt";
    std::cout << "log written in " << log_filename << std::endl;
    std::ofstream ofs(log_filename, std::ofstream::out);
    ofs << width_ << "x" << height_ << "\n";

    /* there's no audio recording feature yet so offset != 0 */
    uint32_t offset_frame_count;
    const size_t size = static_cast<size_t>(width_ *height_*sizeof(uint16_t));

    // first frame count
    ifs.read(reinterpret_cast<char *>(&offset_frame_count), sizeof(offset_frame_count));
    ifs.seekg(2*size, std::ios::cur);

    uint32_t frame_index = offset_frame_count;
    ofs << "offset_frame_count: " << offset_frame_count << "\n";

    uint32_t frame_count = frame_index;
    ofs << frame_index++ << " : " << frame_count << "\n";

    uint  skip_count = 0;

    while (!finished_) {
        ifs.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
        while( frame_index < frame_count ){
            ofs << frame_index++ << " : x" << "\n";
            ++skip_count;
        }
        ofs << frame_index++ << " : " << frame_count << "\n";
        ifs.seekg(2 * size, std::ios::cur);

        // rotate
        ifs.peek();
        if (ifs.eof()) {
            open_and_check(width_, height_, output_directory, file_idx_, ifs, &finished_);
        } else if (ifs.fail()) {
            throw std::runtime_error("Invalid to acccess file.");
        }
    }

    uint total_frame = frame_count - offset_frame_count;
	std::cout << (total_frame-skip_count)*1.0 / total_frame << std::endl;
	ofs << (total_frame-skip_count)*1.0 / total_frame << "\n";
    ofs.close();

    ifs.close();
    return ret;
}

int main() {
    int32_t width = 1920;
    int32_t height = 1080;
    float fps = 60.0f;

    ghc::filesystem::path test_directory = "u3v_framerate_test";
    std::string output_directory_prefix = "u3v_framerate_test";

    if(! ghc::filesystem::is_directory(test_directory)){
        bool ret = ghc::filesystem::create_directory(test_directory);
    }

    rawHeader header_info = {
        0, width, height,
        0.5f, 1.0f, 1.5f, 1.2f, 2.2f, 0.2f,
        0, 0, 0, 0, width, height, width, height, fps};

    for (int i = 0; i < 50; ++i){
        ghc::filesystem::path output_directory = test_directory / (output_directory_prefix + std::to_string(i));
        if(! ghc::filesystem::is_directory(output_directory)){
            bool ret = ghc::filesystem::create_directory(output_directory);
        }
        display_and_save(width, height, output_directory, header_info);
        bool ret = load_header_file(output_directory, header_info);

        if (!ret){
            std::runtime_error("header info is incorrect at test " + std::to_string(i) );
        }
    }
    return 0;
}
