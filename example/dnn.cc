#include <ion/ion.h>
#include <iostream>

#include "ion-bb-dnn/bb.h"
#include "ion-bb-dnn/rt.h"
#include "ion-bb-genesis-cloud/bb.h"
#include "ion-bb-genesis-cloud/rt.h"

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int height = 341;
        const int width = 512;
        const int channel = 3;

        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n;
        n = b.add("genesis_cloud_image_loader").set_param(Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/crosswalk-small.png"});
        n = b.add("genesis_cloud_normalize_u8x3")(n["output"]);
        n = b.add("dnn_tlt_peoplenet")(n["output"]);
        n = b.add("genesis_cloud_denormalize_u8x3")(n["output"]);

        Halide::Buffer<uint8_t> out_buf(channel, width, height);

        PortMap pm;
        pm.set(n["output"], out_buf);
        b.run(pm);

        cv::Mat predicted(height, width, CV_8UC3, out_buf.data());
        cv::cvtColor(predicted, predicted, cv::COLOR_RGB2BGR);
        cv::imwrite("predicted.png", predicted);

        std::cout << "yolov4 example done!!!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
