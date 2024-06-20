from ionpy import Node, Builder, Buffer, PortMap, Port, Param, Type, TypeCode
import numpy as np
import cv2

from sys import platform

feature_gain_key = 'Gain'
feature_exposure_key = 'ExposureTime'
num_bit_shift = 0

if __name__ == "__main__":

    # Define parameters
    width = 1920
    height = 1080
    gain = 400
    exposure = 400

    # Build the pipeline by adding nodes to this builder.
    builder = Builder()

    # Set the target hardware, The default is CPU.
    builder.set_target('host')

    # Load building block module from the library
    builder.with_bb_module("ion-bb")

    # Define Input Port
    #    Port class would be  used to define dynamic O/O for each node.
    t = Type(TypeCode.Float, 64, 1)
    gain0_p = Port('gain0', t, 0)
    exposure0_p = Port('exposure0', t, 0)

    # Params
    num_devices = Param('num_devices', '1')
    frame_sync = Param('frame_sync', 'false')
    gain_key = Param('gain_key', feature_gain_key)
    exposure_key = Param('exposure_key', feature_exposure_key)
    realtime_diaplay_mode = Param('realtime_diaplay_mode', 'true')
    enable_control = Param('enable_control', 'true')

    #    Add node and connect the input port to the node instance
    node = builder.add('image_io_u3v_cameraN_u16x2')\
        .set_iportss([gain0_p, exposure0_p])\
        .set_params([num_devices, frame_sync, gain_key, exposure_key, realtime_diaplay_mode, enable_control])

    # Define Output Port
    out_p = node.get_port('output')
    frame_count_p = node.get_port('frame_count')

    # input values
    gain0_p.bind(gain)
    exposure0_p.bind(exposure)

    # output values
    odata0 = np.full((height, width), fill_value=0, dtype=np.uint16)
    output0 = Buffer(array=odata0)
    out_p[0].bind(output0)

    fcdata = np.full((1), fill_value=0, dtype=np.uint32)
    frame_count = Buffer(array=fcdata)
    frame_count_p.bind(frame_count)

    loop_num = 100

    for x in range(loop_num):

        # running the builder
        builder.run()

        odata0 *= pow(2, num_bit_shift)

        median_output0_np_HxW = cv2.medianBlur(odata0, 5)

        cv2.imshow("A", odata0)
        cv2.imshow("C", median_output0_np_HxW)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
