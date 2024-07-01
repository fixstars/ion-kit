# https://github.com/fixstars/ion-csharp/blob/master/test/Test.cs
from ionpy import Node, Builder, Buffer, PortMap, Port, Param, Type, TypeCode
import numpy as np
import cv2
import os


def test_portmap_access():

    w = 200
    h = 200
    t = Type(code_=TypeCode.Uint, bits_=8, lanes_=1)
    width = Param(key='width', val=str(w))
    height = Param(key='height', val=str(h))
    urls = Param(key="urls",val="http://optipng.sourceforge.net/pngtech/img/lena.png;http://upload.wikimedia.org/wikipedia/commons/0/05/Cat.png")

    builder = Builder()
    builder.set_target(target='host')
    # make sure path includes libion-bb.so
    builder.with_bb_module(path='ion-bb')

    node = builder.add('image_io_cameraN').set_param(params=[width, height, urls])



    sizes = (w, h, 3)
    obuf1 = Buffer(type=t, sizes=sizes)
    obuf2 = Buffer(type=t, sizes=sizes)
    odata_bytes1 = np.zeros(w*h*3,dtype=np.uint8).tobytes()
    odata_bytes2 = np.zeros(w*h*3,dtype=np.uint8).tobytes()
    obuf1.write(data=odata_bytes1)
    obuf2.write(data=odata_bytes2)

    port_map = PortMap()
    port_map.set_buffer(port=node.get_port(name='output')[0], buffer=obuf1)
    port_map.set_buffer(port=node.get_port(name='output')[1], buffer=obuf2)

    builder.run(port_map=port_map)

    out_byte_arr1 = obuf1.read(len(odata_bytes1))
    out_byte_arr2 = obuf2.read(len(odata_bytes2))

    img_out1 = np.frombuffer(out_byte_arr1, dtype=np.uint8)
    img_out2 = np.frombuffer(out_byte_arr2, dtype=np.uint8)


    b1 = img_out1[:w*h]
    g1 = img_out1[w*h:2*w*h]
    r1 = img_out1[2*w*h:]

    b2 = img_out2[:w*h]
    g2 = img_out2[w*h:2*w*h]
    r2  = img_out2[2*w*h:]
    #
    img_out1 =  cv2.merge((r1,g1,b1)).reshape(w,h,3)
    img_out2 =  cv2.merge((r2,g2,b2)).reshape(w,h,3)

    DISPLAY = os.getenv("DISPLAY","false").lower()=="true"

    if DISPLAY:
        cv2.imshow("display1", img_out1)
        cv2.imshow("display2", img_out2)
        cv2.waitKey(3000)
