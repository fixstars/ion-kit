#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ion/ion.h>

#include <exception>

using namespace ion;

// Before you run this script please ensure you `export GENICAM_FILENAME=<path-to-arv-fake-camera.xml>` or`set GENICAM_FILENAME=<path-to-arv-fake-camera.xml>`
// original arv-fake-camera.xml can be download at https://github.com/Sensing-Dev/aravis/blob/main/src/arv-fake-camera.xml
// you can also create your fake-camera.xml by editing original xml file and `export GENICAM_FILENAME=<path-to-your-fake-camera.xml>
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
        int num_device = 2;
        Node n = b.add("image_io_u3v_cameraN_u8x2")().set_param(
                Param("width", width),
                Param("height", height),
                Param("fps", 30)
                );


        // Map output buffer and ports by using Port::bind.
        // - output: output of the obtained video data
        // - frame_count: output of the frame number of the obtained video
        std::vector< int > buf_size = std::vector < int >{ width, height };


        std::vector<Halide::Buffer<uint8_t>> output;
        for (int i = 0; i < num_device; ++i){
          output.push_back(Halide::Buffer<uint8_t>(buf_size));
        }
        n["output"].bind(output);
        Buffer<uint32_t> frame_count(1);
        n["frame_count"].bind(frame_count);
        
        // Obtain image data continuously for 100 frames to facilitate operation check.
        int user_input = -1;
        while(user_input == -1)
        {
            // JIT compilation and execution of pipelines with Builder.
        b.run();

        // Convert the retrieved buffer object to OpenCV buffer format.
        // Depends on sensor image pixel format, apply bit shift on images
        // Display the image
        for (int i = 0;i<num_device;i++){
            cv::Mat img(height, width, CV_8UC1, output[i].data());
            cv::imshow("Fake Camera" + std::to_string(i), img);
        }
        std::cout << frame_count(0) << std::endl;
        user_input = cv::waitKeyEx(1);
        }
      cv::destroyAllWindows();

    } catch (const ion::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

  return 0;
}
