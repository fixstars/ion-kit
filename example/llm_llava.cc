#include <ion/ion.h>
#include <iostream>

#include <opencv2/highgui.hpp>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        const int width = 503;
        const int height = 337;

        Buffer<int8_t> prompt{1024};
        prompt.fill(0);
        std::string prompt_s("<image>Explain the image in one sentense");
        for (auto i = 0; i < prompt_s.size(); ++i) {
            prompt(i) = prompt_s[i];
        }

        Builder b;
        b.set_target(Halide::get_target_from_environment().with_feature(Halide::Target::Debug).with_feature(Halide::Target::TracePipeline));
        b.with_bb_module("ion-bb");

        auto n_img = b.add("image_io_color_data_loader").set_param(Param{"url", "http://www.onthejob.education/images/4th_level/Road_Worker/Road_Worker_Darwin.jpg"}, Param{"width", width}, Param{"height", height});
        n_img = b.add("base_reorder_buffer_3d_uint8")(n_img["output"]).set_param(Param{"dim0", 2}, Param{"dim1", 0}, Param{"dim2", 1});
        auto n_txt = b.add("llm_llava")(n_img["output"], prompt).set_param(Param{"width", width}, Param{"height", height});

        Buffer<int8_t> txt_output{1024};
        n_txt["output"].bind(txt_output);

        b.run();

        std::cout << reinterpret_cast<const char*>(txt_output.data()) << std::endl;
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
