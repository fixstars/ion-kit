#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ion/ion.h>

#include <exception>

using namespace ion;

#define FEATURE_GAIN_KEY "Gain"
#define FEATURE_EXPOSURE_KEY "ExposureTime"
#define NUM_BIT_SHIFT 4

// In this tutorial, we will create a simple application that obtains image data from a pair of usb3 vision sensors,
// and adds smoothing processing using OpenCV, and displays the data on the screen.

// Define parameters
//  Resize it according to the resolution of the sensor.
const int32_t width = 640;
const int32_t height = 480;
double gain = 400;
double exposure = 400;

#ifdef _WIN32
    #define MODULE_NAME "ion-bb.dll"
#elif __APPLE__
    #define MODULE_NAME "libion-bb.dylib"
#else
    #define MODULE_NAME "libion-bb.so"
#endif

int positive_pow(int base, int expo){
  if (expo <= 0){
      return 1;
  }
  if (expo == 1){
      return base;
  }else{
      return base * positive_pow(base, expo-1);
  }
}

int main(int argc, char *argv[])
{
  // Define builders to build, compile, and execute pipelines.
  //  Build the pipeline by adding nodes to the Builder.
  Builder b;

  // Set the target hardware. The default is CPU.
  b.set_target(Halide::get_host_target());

  // Load standard building block
  b.with_bb_module(MODULE_NAME);

  // Define the input port
  //  Port class is used to define dynamic I/O for each node.
  Port dispose_p{ "dispose",  Halide::type_of<bool>() };
  Port gain0_p{ "gain0", Halide::type_of<double>() };
  Port gain1_p{ "gain1", Halide::type_of<double>() };
  Port exposure0_p{ "exposure0", Halide::type_of<double>() };
  Port exposure1_p{ "exposure1", Halide::type_of<double>() };

  //  Connect the input port to the Node instance created by b.add().
  Node n = b.add("image_io_u3v_camera1_u16x2")(dispose_p, gain0_p, exposure0_p)
    .set_param(
      Param{"pixel_format_ptr", "Mono12"},
      Param{"frame_sync", "false"},
      Param{"gain_key", FEATURE_GAIN_KEY},
      Param{"exposure_key", FEATURE_EXPOSURE_KEY},
      Param{"realtime_diaplay_mode", "true"}
    );

  // Define output ports and pass each object from Node instance.
  Port rp = n["output0"];
  Port frame_count_p = n["frame_count"];

  // Using PortMap, define output ports that map data to input ports and pass each object from a Node instance.
  PortMap pm;

  // In this sample, the gain value and exposure time of both sensors are set statically.
  pm.set(gain0_p, gain);
  pm.set(exposure0_p, exposure);

  // Map data from output ports by using PortMap.
  // Of the output of "u3v_camera1_u16x2",
  // - rp (output of the obtained video data) to output0,
  // - frame_count_p (output of the frame number of the obtained video) to frame_count,
  // each stored in the buffer.
  std::vector< int > buf_size = std::vector < int >{ width, height };
  Halide::Buffer<uint16_t> output0(buf_size);
  Halide::Buffer<uint32_t> frame_count(1);

  pm.set(rp, output0);
  pm.set(frame_count_p, frame_count);

  // Obtain image data continuously for 100 frames to facilitate operation check.
  int loop_num = 100;
  for (int i = 0; i < loop_num; ++i)
  {
    pm.set(dispose_p, i == loop_num-1);
    // JIT compilation and execution of pipelines with Builder.
    try {
        b.run(pm);
    }catch(std::exception& e){
        // e.what() shows the error message if pipeline build/run was failed.
        std::cerr << "Failed to build pipeline" << std::endl;
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    // Convert the retrieved buffer object to OpenCV buffer format.
    cv::Mat A(height, width, CV_16UC1);

    std::memcpy(A.ptr(), output0.data(), output0.size_in_bytes());

    // Depends on sensor image pixel format, apply bit shift on images
    A = A * positive_pow(2, NUM_BIT_SHIFT);

    // Display the image
    cv::imshow("A", A);

    // Wait for key input
    //   When any key is pressed, close the currently displayed image and proceed to the next frame.
    cv::waitKey(1);
  }

  return 0;
}