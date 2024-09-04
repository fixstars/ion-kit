#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ion/ion.h>

#include <exception>

using namespace ion;

// Before you run this script please ensure you `export GENICAM_FILENAME=<path-to-arv-fake-camera.xml>` or`set GENICAM_FILENAME=<path-to-arv-fake-camera.xml>`
// original arv-fake-camera.xml can be download at https://github.com/Sensing-Dev/aravis/blob/main/src/arv-fake-camera.xml
// you can also create your fake-camera.xml by editing original xml file and `export GENICAM_FILENAME=<path-to-your-fake-camera.xml>
int main(int argc, char *argv[]) {
    try {
        // Define builders to build, compile, and execute pipelines.
        //  Build the pipeline by adding nodes to the Builder.
        Builder b;

        // Set the target hardware. The default is CPU.
        b.set_target(ion::get_host_target());
        // Load standard building block
        b.with_bb_module("ion-bb");

        int width = 640;
        int height = 480;
        int num_device = 2;
        // if you don't set width and height, default width is 640 and default height is 480
        Node n = b.add("image_io_u3v_cameraN_u8x2")().set_params(
            Param("num_devices", num_device),
            Param("pixel_format", "Mono8"));

        /******************** force simulation mode*************************/
        //        int width = 960;
        //        int height = 640;
        //        int num_device = 2;
        //        Node n = b.add("image_io_u3v_cameraN_u8x2")().set_params(
        //        Param("num_devices", num_device),
        //        Param("force_sim_mode", true),
        //        Param("width", width),
        //        Param("height", height));

        /********************RGB 8*************************/
        //        Node n = b.add("image_io_u3v_cameraN_u8x3")().set_params(
        //                Param("num_devices", num_device),
        //                Param("pixel_format", "RGB8"));

        /********************Mono16*************************/
        //        Node n = b.add("image_io_u3v_cameraN_u16x2")().set_params(
        //                Param("num_devices", num_device),
        //                Param("pixel_format", "Mono16"));

        std::vector<int> buf_size = std::vector<int>{width, height};

        std::vector<Halide::Buffer<uint8_t>> outputs;
        std::vector<Halide::Buffer<uint32_t>> frame_counts;
        for (int i = 0; i < num_device; ++i) {
            outputs.push_back(Halide::Buffer<uint8_t>(buf_size));
            frame_counts.push_back(Halide::Buffer<uint32_t>(1));
        }
        n["output"].bind(outputs);
        n["frame_count"].bind(frame_counts);

        // Obtain image data
        int user_input = -1;
        while (user_input == -1) {
            // JIT compilation and execution of pipelines with Builder.
            b.run();

            // Convert the retrieved buffer object to OpenCV buffer format.
            // Depends on sensor image pixel format, apply bit shift on images
            // Display the image
            for (int i = 0; i < num_device; i++) {
                cv::Mat img(height, width, CV_8UC1, outputs[i].data());
                cv::imshow("Fake Camera" + std::to_string(i), img);
            }

            user_input = cv::waitKeyEx(1);
        }

        cv::destroyAllWindows();

    } catch (const ion::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
