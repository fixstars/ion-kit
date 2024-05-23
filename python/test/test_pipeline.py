# https://github.com/fixstars/ion-csharp/blob/master/test/Test.cs
from ionpy import Node, Builder, Buffer, Port, Param, Type, TypeCode
import numpy as np  # TODO: rewrite with pure python
import cv2
import os


def test_pipeline():
    w = 200
    h = 200
    t = Type(code_=TypeCode.Float, bits_=32, lanes_=1)
    width = Param(key='width', val=str(w))
    height = Param(key='height', val=str(h))
    urls = Param(key="urls",
                 val="http://optipng.sourceforge.net/pngtech/img/lena.png;http://upload.wikimedia.org/wikipedia/commons/0/05/Cat.png")

    builder = Builder()
    builder.set_target(target='host')
    # make sure path includes libion-bb.so
    builder.with_bb_module(path='ion-bb')

    node = builder.add('image_io_cameraN').set_param(params=[width, height, urls])
    node1 = builder.add("base_normalize_3d_uint8").set_iport(ports=[node.get_port(name='output')[0], ])
    node2 = builder.add("base_normalize_3d_uint8").set_iport(ports=[node.get_port(name='output')[1], ])

    sizes = (w, h, 3)

    img_out1 = np.full(sizes, fill_value=0, dtype=np.uint8)
    img_out2 = np.full(sizes, fill_value=0, dtype=np.uint8)
    obuf1 = Buffer(array=img_out1)
    obuf2 = Buffer(array=img_out2)

    output_port1 = node1.get_port(name='output')
    output_port1.bind(obuf1)
    output_port2 = node2.get_port(name='output')
    output_port2.bind(obuf2)

    builder.run()

    b1 = img_out1[:w * h]
    g1 = img_out1[w * h:2 * w * h]
    r1 = img_out1[2 * w * h:]

    b2 = img_out2[:w * h]
    g2 = img_out2[w * h:2 * w * h]
    r2 = img_out2[2 * w * h:]

    img_out1 = cv2.merge((r1, g1, b1)).reshape(w, h, 3)
    img_out2 = cv2.merge((r2, g2, b2)).reshape(w, h, 3)

    DISPLAY = os.getenv("DISPLAY", "false").lower() == "true"
    if DISPLAY:
        cv2.imshow("display1", img_out1)
        cv2.imshow("display2", img_out2)
        cv2.waitKey(5000)
