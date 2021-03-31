#include <ion/ion.h>

#include "ion-bb-core/bb.h"
#include "ion-bb-fpga/bb.h"
#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-processing/bb.h"

using namespace ion;

int main(int argc, char *argv[]) {
    Builder b;
    b.set_target(Halide::get_target_from_environment());

    Node imx219 = b.add("image_io_imx219")
                      .set_param(
                          Param{"force_sim_mode", "true"},
                          Param{"url", "http://ion-archives.s3-us-west-2.amazonaws.com/pedestrian.jpg"});

    Node downscale = b.add("image_processing_bayer_downscale_uint16")
                         .set_param(
                             Param{"input_width", "3264"},
                             Param{"input_height", "2464"},
                             Param{"downscale_factor", "2"})(
                             imx219["output"]);

    Node isp = b.add("fpga_simple_isp_with_unsharp_mask")
                   .set_param(
                       Param{"bayer_pattern", "0"},
                       Param{"width", "1632"},
                       Param{"height", "1232"},
                       Param{"normalize_input_bits", "10"},
                       Param{"normalize_input_shift", "6"},
                       Param{"offset_offset_r", "64"},
                       Param{"offset_offset_g", "64"},
                       Param{"offset_offset_b", "64"},
                       Param{"white_balance_gain_r", "10240"},
                       Param{"white_balance_gain_g", "8192"},
                       Param{"white_balance_gain_b", "13107"},
                       Param{"gamma_gamma", "0.454545"},
                       Param{"unroll_level", "3"})(
                       downscale["output"]);

    Node reorder = b.add("core_reorder_buffer_3d_uint8")
                       .set_param(
                           Param{"dim0", "1"},
                           Param{"dim1", "2"},
                           Param{"dim2", "0"})(
                           isp["output"]);

    Node output = b.add("image_io_image_saver")
                      .set_param(
                          Param{"width", "816"},
                          Param{"height", "616"},
                          Param{"path", "out.png"})(
                          reorder["output"]);

    b.compile("fpga_isp");

    return 0;
}
