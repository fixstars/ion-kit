#include <ion/ion.h>
#include <iostream>

#include <opencv2/highgui.hpp>

using namespace ion;

int main(int argc, char *argv[]) {
    try {
        // const int width = 1280;
        // const int height = 960;
        const int width = 503;
        const int height = 337;

        Buffer<int8_t> prompt{1024};
        prompt.fill(0);
        std::string prompt_s("<image>Explain the image in one sentence.");
        for (auto i = 0; i < prompt_s.size(); ++i) {
            prompt(i) = prompt_s[i];
        }

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        auto n_img_cwh = b.add("image_io_color_data_loader").set_param(Param{"url", "http://www.onthejob.education/images/4th_level/Road_Worker/Road_Worker_Darwin.jpg"}, Param{"width", width}, Param{"height", height});
-       auto n_img_whc = b.add("base_reorder_buffer_3d_uint8")(n_img_cwh["output"]).set_param(Param{"dim0", 2}, Param{"dim1", 0}, Param{"dim2", 1});
        // auto n_img_cwh = b.add("image_io_u3v_cameraN_u8x3").set_param(Param{"num_devices", "1"}, Param{"realtime_diaplay_mode", true});
        // auto n_img_whc = b.add("base_reorder_buffer_3d_uint8")(n_img_cwh["output"]).set_param(Param{"dim0", 1}, Param{"dim1", 2}, Param{"dim2", 0});

        auto n_disp = b.add("image_io_gui_display")(n_img_whc["output"][0]).set_param(Param{"width", width}, Param{"height", height});
        auto n_txt = b.add("llm_llava")(n_img_cwh["output"][0], prompt).set_param(Param{"width", width}, Param{"height", height});

        Buffer<int8_t> txt_output{1024};
        n_txt["output"].bind(txt_output);

        Buffer<int32_t> result = Buffer<int32_t>::make_scalar();
        n_disp["output"].bind(result);

        // for (int i=0; i<1; ++i) {
        while (true) {
            b.run();
            std::cout << reinterpret_cast<const char *>(txt_output.data()) << std::endl;
        }

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
