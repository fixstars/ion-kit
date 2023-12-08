#include <iostream>
#include <ion/ion.h>

// to display
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define DISPLAY

using namespace ion;

void display_image_uint8(Halide::Buffer<uint8_t> buffer, std::string filename) {
    int width = buffer.width();
    int height = buffer.height();
    int channels = buffer.channels();
    cv::Mat img_out;
    if (channels == 3) {
        cv::merge(std::vector<cv::Mat>{
                          cv::Mat(height, width, CV_8U, buffer.data() + width * height * 2), //b
                          cv::Mat(height, width, CV_8U, buffer.data() + width * height * 1), //g
                          cv::Mat(height, width, CV_8U, buffer.data())}, //r
                  img_out);

    } else {
        cv::Mat img_out(height, width, CV_8U, buffer.data());
    }
#ifdef DISPLAY
    cv::imshow( "Display window: " + filename, img_out);
    cv::waitKey(3000);
#endif
}


int main(int argc, char *argv[]) {
    try {
        // TODO: Test with FullHD
        const int width = 200;
        const int height = 150;

        Param wparam("width", std::to_string(width));
        Param hparam("height", std::to_string(height));

        Port wport("width", Halide::type_of<int32_t>());
        Port hport("height", Halide::type_of<int32_t>());

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n;
        n = b.add("image_io_cameraN").set_param(
                wparam,
                hparam,
                Param{"num_devices", "2"},
                Param{"urls", "http://optipng.sourceforge.net/pngtech/img/lena.png;http://upload.wikimedia.org/wikipedia/commons/0/05/Cat.png"}
                //input urls split by ';'
        );


        PortMap pm;
        Halide::Buffer<uint8_t> out_buf0( width, height,3);
        Halide::Buffer<uint8_t> out_buf1( width, height,3);





        pm.set(wport, width);
        pm.set(hport, height);

        pm.set(n["output"][0],out_buf0);
        pm.set(n["output"][1],out_buf1);

        b.run(pm);
        display_image_uint8(out_buf0, "display.png");
        display_image_uint8(out_buf1, "display.png");
        std::cout << "Success" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
