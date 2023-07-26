from ionpy import Node, Builder, Buffer, PortMap, Port, Param, Type, TypeCode
import numpy as np
import cv2

import os

feature_gain_key = 'Gain'
feature_exposure_key = 'ExposureTime'
num_bit_shift = 0

if os.name == 'nt':
    module_name = 'ion-bb.dll'
elif os.name == 'posix':
    module_name = 'libion-bb.so'

if __name__ == "__main__":

    # Define parameters
    width = 1920
    height = 1080
    gain = 400
    exposure = 400

    # Build the pipeline by adding nodes to this builder.
    builder = Builder()
    #   Set the target hardware, The default is CPU.
    builder.set_target('host')
    #   Load building block module from the library
    builder.with_bb_module(module_name)


    # Define Input Port
    #    Port class would be  used to define dynamic O/O for each node.
    t = Type(TypeCode.Uint, 1, 1)
    dispose_p = Port('dispose', t, 0)
    t = Type(TypeCode.Int, 32, 1)
    gain0_p = Port('gain0', t, 0)
    exposure0_p = Port('exposure0', t, 0)

    # Params
    pixel_format_ptr = Param('pixel_format_ptr', 'Mono12')
    frame_sync = Param('frame_sync', 'true')
    gain_key = Param('gain_key', feature_gain_key)
    exposure_key = Param('exposure_key', feature_exposure_key)
    realtime_diaplay_mode = Param('realtime_diaplay_mode', 'true')


    #    Add node and connect the input port to the node instance
    node = builder.add('image_io_u3v_camera1_u16x2')\
        .set_port([dispose_p, gain0_p, exposure0_p, ])\
        .set_param([pixel_format_ptr, frame_sync, gain_key, exposure_key, realtime_diaplay_mode, ])


    # Define Output Port
    lp = node.get_port('output0')
    frame_count_p = node.get_port('frame_count')

    # portmap
    port_map = PortMap()

    # input values
    port_map.set_i32(gain0_p, gain)
    port_map.set_i32(exposure0_p, exposure)

    # output values
    buf_size = (width, height, )
    t = Type(TypeCode.Uint, 16, 1)
    output0 = Buffer(t, buf_size)
    t = Type(TypeCode.Uint, 32, 1)
    frame_count = Buffer(t, (1,))

    port_map.set_buffer(lp, output0)
    port_map.set_buffer(frame_count_p, frame_count)

    buf_size_opencv = (height, width)

    loop_num = 100

    for x in range(loop_num):
        port_map.set_u1(dispose_p, x==loop_num-1)

        # running the builder
        builder.run(port_map)

        output0_bytes = output0.read(width*height*2)

        output0_np_HxW = np.frombuffer(output0_bytes, np.uint16).reshape(buf_size_opencv)

        output0_np_HxW *= pow(2, num_bit_shift)

        median_output0_np_HxW = cv2.medianBlur(output0_np_HxW, 5)

        cv2.imshow("A", output0_np_HxW)
        cv2.imshow("C", median_output0_np_HxW)
        cv2.waitKey(0)

    cv2.destroyAllWindows()