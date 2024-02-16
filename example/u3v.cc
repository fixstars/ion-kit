#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ion/ion.h>

#include <exception>

using namespace ion;

#define FEATURE_GAIN_KEY "Gain"
#define FEATURE_EXPOSURE_KEY "ExposureTime"


// In this tutorial, we will create a simple application that obtains image data from a pair of usb3 vision sensors,
// and adds smoothing processing using OpenCV, and displays the data on the screen.

// Define parameters
//  Resize it according to the resolution of the sensor.
const int32_t width = 640;
const int32_t height = 480;
double gain = 400;
double exposure = 400;

int positive_pow(int base, int expo) {
    if (expo <= 0) {
        return 1;
    }
    if (expo == 1) {
        return base;
    } else {
        return base * positive_pow(base, expo-1);
    }
}

int main(int argc, char *argv[])
{
    try {
        // Define builders to build, compile, and execute pipelines.
        //  Build the pipeline by adding nodes to the Builder.
        Builder b;

        // Set the target hardware. The default is CPU.
        b.set_target(ion::get_host_target());

        // Load standard building block
        b.with_bb_module("ion-bb");

        int width = 500;
        int height = 500;
        Node n = b.add("image_io_u3v_camera_fake_u8x2")().set_param(
                Param("num_devices", 1),
                Param("width", width),
                Param("height", height)
                );


        // Map output buffer and ports by using Port::bind.
        // - output: output of the obtained video data
        // - frame_count: output of the frame number of the obtained video
        std::vector< int > buf_size = std::vector < int >{ width, height };
        Buffer<uint8_t> output(buf_size);


        n["output"][0].bind(output);

        // Obtain image data continuously for 100 frames to facilitate operation check.
        int loop_num = 100;

        int user_input = -1;

        while(user_input == -1)
        {
            // JIT compilation and execution of pipelines with Builder.
        b.run();

        // Convert the retrieved buffer object to OpenCV buffer format.
        cv::Mat A(height, width, CV_8UC1, output.data());

        // Depends on sensor image pixel format, apply bit shift on images
        // Display the image
        cv::imshow("Fake Camera", A);
        user_input = cv::waitKeyEx(1);
        }
      cv::destroyAllWindows();


    } catch (const ion::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

  return 0;
}
