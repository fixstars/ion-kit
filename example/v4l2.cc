#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <ion/ion.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-io/rt.h"

#include "util.h"

using namespace ion;

Halide::Buffer<float> load_raw(std::string filename) {
    std::ifstream ifs(filename, std::ios_base::binary);

    assert(ifs.is_open());

    uint16_t width, height;
    ifs.read(reinterpret_cast<char *>(&width), sizeof(width));
    ifs.read(reinterpret_cast<char *>(&height), sizeof(height));

    std::vector<uint16_t> data(width * height);

    ifs.read(reinterpret_cast<char *>(data.data()), width * height * sizeof(uint16_t));

    Halide::Buffer<float> buffer(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            buffer(x, y) = static_cast<float>(data[y * width + x]) / 4095.f;
        }
    }

    return buffer;
}

void save_image(Halide::Buffer<float> buffer, std::string filename) {
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

    cv::imwrite(filename, img_out);
}

int main(int argc, char *argv[]) {
    Builder b;
    b.set_target(Halide::Target{get_target_from_cmdline(argc, argv)});

    auto n0 = b.add("image_io_imx219").set_param(Param{"index", "0"});
    auto n1 = b.add("image_io_imx219").set_param(Param{"index", "1"});
    auto n2 = b.add("image_io_imx219").set_param(Param{"index", "2"});
    auto n3 = b.add("image_io_imx219").set_param(Param{"index", "3"});
    auto n4 = b.add("image_io_imx219").set_param(Param{"index", "4"});
    auto n5 = b.add("image_io_imx219").set_param(Param{"index", "5"});

    PortMap pm;
    Halide::Buffer<uint16_t> obuf0(3264, 2464), obuf1(3264, 2464), obuf2(3264, 2464), obuf3(3264, 2464), obuf4(3264, 2464), obuf5(3264, 2464);
    pm.set(n0["output"], obuf0);
    pm.set(n1["output"], obuf1);
    pm.set(n2["output"], obuf2);
    pm.set(n3["output"], obuf3);
    pm.set(n4["output"], obuf4);
    pm.set(n5["output"], obuf5);

    b.run(pm);

    std::ofstream ofs("out.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(obuf0.data()), obuf0.size_in_bytes());
    ofs.write(reinterpret_cast<const char *>(obuf1.data()), obuf1.size_in_bytes());
    ofs.write(reinterpret_cast<const char *>(obuf2.data()), obuf2.size_in_bytes());
    ofs.write(reinterpret_cast<const char *>(obuf3.data()), obuf3.size_in_bytes());
    ofs.write(reinterpret_cast<const char *>(obuf4.data()), obuf4.size_in_bytes());
    ofs.write(reinterpret_cast<const char *>(obuf5.data()), obuf5.size_in_bytes());

    return 0;
}
