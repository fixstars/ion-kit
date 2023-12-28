#include <iostream>
#include <ion/ion.h>

// to display
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace ion;

void display_image_float(Halide::Buffer<float> buffer, std::string filename) {
    int width = buffer.width();
    int height = buffer.height();
    int channels = buffer.channels();
    cv::Mat img_out;
    if (channels == 3) {
        cv::Mat img_float;
        cv::merge(std::vector<cv::Mat>{
                          cv::Mat(height, width, CV_32F, buffer.data() + width * height * 2),
                          cv::Mat(height, width, CV_32F, buffer.data() + width * height * 1),
                          cv::Mat(height, width, CV_32F, buffer.data())},
                  img_float);
        img_float.convertTo(img_out, CV_8U, 255);
    } else {
        cv::Mat img_float(height, width, CV_32F, buffer.data());
        img_float.convertTo(img_out, CV_8U, 255);
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
        n = b.add("image_io_cameraN").set_params(
                wparam,
                hparam,
                Param{"num_devices", "2"},
                Param{"urls", "http://optipng.sourceforge.net/pngtech/img/lena.png;http://upload.wikimedia.org/wikipedia/commons/0/05/Cat.png"}
                //input urls split by ';'
        );
        n = b.add("base_normalize_3d_uint8")(n["output"][1]);  // access only port[1]
        n = b.add("image_processing_resize_nearest_3d")(n["output"]).set_params(
                Param{"width", std::to_string(width)},
                Param{"height", std::to_string(height)},
                Param{"scale", std::to_string(2)});
        Port output = n["output"];

        PortMap pm;

        pm.set(wport, width);
        pm.set(hport, height);
        Halide::Buffer<float> out_buf( width, height,3);
        pm.set(output, out_buf);

        b.run(pm);
        display_image_float(out_buf, "display.png");
        std::cout << "Success" << std::endl;
    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
