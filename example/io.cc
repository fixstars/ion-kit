#include <ion/ion.h>
#include <iostream>

#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-io/rt.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int height = 128;
        const int width = 128;

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n;
        n = b.add("image_io_grayscale_data_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/pedestrian.jpg"});

        Halide::Buffer<uint16_t> out_buf(width, height);

        PortMap pm;
        pm.set(n["output"], out_buf);
        b.run(pm);

        // cv::Mat predicted(height, width, CV_8UC3, out_buf.data());
        // cv::cvtColor(predicted, predicted, cv::COLOR_RGB2BGR);
        // cv::imwrite("predicted.png", predicted);

        std::cout << "yolov4 example done!!!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
