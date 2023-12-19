# https://github.com/fixstars/ion-csharp/blob/master/test/Test.cs
from ionpy import Node, Builder, Buffer, PortMap, Port, Param, Type, TypeCode
import numpy as np # TODO: rewrite with pure python


def test_all():
    t = Type(code_=TypeCode.Int, bits_=32, lanes_=1)
    input_port = Port(key='input', type=t, dim=2)
    value41 = Param(key='v', val='41')

    builder = Builder()
    builder.set_target(target='host')
    # make sure path includes libion-bb-test.so
    builder.with_bb_module(path='ion-bb-test')
    # builder.with_bb_module(path='ion-bb-test.dll') # for Windows

    node = builder.add('test_inc_i32x2').set_port(ports=[ input_port, ]).set_param(params=[ value41, ])

    port_map = PortMap()

    sizes = (4, 4)
    ibuf = Buffer(type=t, sizes=sizes)
    obuf = Buffer(type=t, sizes=sizes)

    idata = np.full((4*4, ), fill_value=1, dtype=np.int32)
    odata = np.full((4*4, ), fill_value=0, dtype=np.int32)

    idata_bytes = idata.tobytes(order='C')
    odata_bytes = odata.tobytes(order='C')

    ibuf.write(data=idata_bytes)
    obuf.write(data=odata_bytes)

    port_map.set_buffer(port=input_port, buffer=ibuf)
    port_map.set_buffer(port=node.get_port(key='output'), buffer=obuf)

    builder.run(port_map=port_map)

    obuf_bytes = obuf.read(num_data_bytes=len(odata_bytes))
    odata = np.frombuffer(obuf_bytes, dtype=np.int32)

    for i in range(4*4):
        assert odata[i] == 42
