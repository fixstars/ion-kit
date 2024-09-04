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

int positive_pow(int base, int expo) {
    if (expo <= 0) {
        return 1;
    }
    if (expo == 1) {
        return base;
    } else {
        return base * positive_pow(base, expo - 1);
    }
}

int main(int argc, char *argv[]) {
    try {
        // Define builders to build, compile, and execute pipelines.
        //  Build the pipeline by adding nodes to the Builder.
        Builder b;

        // Set the target hardware. The default is CPU.
        b.set_target(ion::get_host_target());

        // Load standard building block
        b.with_bb_module("ion-bb");

        //  Connect the input port to the Node instance created by b.add().
        Node n = b.add("image_io_u3v_cameraN_u16x2")(&gain, &exposure)
                     .set_params(
                         Param("num_devices", 1),
                         Param("frame_sync", false),
                         Param("gain_key", FEATURE_GAIN_KEY),
                         Param("exposure_key", FEATURE_EXPOSURE_KEY),
                         Param("realtime_display_mode", true),
                         Param("enable_control", true));

        // Map output buffer and ports by using Port::bind.
        // - output: output of the obtained video data
        // - frame_count: output of the frame number of the obtained video
        std::vector<int> buf_size = std::vector<int>{width, height};
        Buffer<uint16_t> output(buf_size);
        Buffer<uint32_t> frame_count(1);

        n["output"][0].bind(output);
        n["frame_count"][0].bind(frame_count);

        // Obtain image data continuously for 100 frames to facilitate operation check.
        int loop_num = 100;
        int coef = positive_pow(2, NUM_BIT_SHIFT);
        for (int i = 0; i < loop_num; ++i) {
            // JIT compilation and execution of pipelines with Builder.
            b.run();

            // Convert the retrieved buffer object to OpenCV buffer format.
            cv::Mat A(height, width, CV_16UC1, output.data());

            // Depends on sensor image pixel format, apply bit shift on images
            A = A * coef;

            // Display the image
            cv::imshow("A", A);

            // Wait for key input
            //   When any key is pressed, close the currently displayed image and proceed to the next frame.
            cv::waitKey(1);
        }

    } catch (const ion::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
