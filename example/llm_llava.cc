#include <ion/ion.h>
#include <iostream>

#include <opencv2/highgui.hpp>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int width = 503;
        const int height = 337;

        Buffer<int8_t> prompt{1024};

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        auto n_img = b.add("image_io_color_data_loader").set_param(Param("url", "http://www.onthejob.education/images/4th_level/Road_Worker/Road_Worker_Darwin.jpg"), Param("width", width), Param("height", height));
        auto n_txt = b.add("llm_llava")(n_img["output"], prompt);
#if 0
        auto n = b.add("llm_draw_text")(n_img["output"], n_txt["output"]);
        n = b.add("image_io_gui_display")(n["output"]).set_param(Param{"width", width}, Param{"height", height});

        auto r = ion::Buffer<int>::make_scalar();
        n["output"].bind(r);

        for (int i=0; i<100; ++i) {
            b.run();
            cv::waitKey(1);
        }

        std::cout << "llava example done!!!" << std::endl;
#else
        Buffer<int8_t> txt_output{1024};
        n_txt["output"].bind(txt_output);

        b.run();

        std::cout << reinterpret_cast<const char*>(txt_output.data()) << std::endl;
        #endif
    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        return 1;
    }

    return 0;
}
