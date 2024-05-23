# https://github.com/fixstars/ion-csharp/blob/master/test/Test.cs
from ionpy import Node, Builder, Buffer, Port, Param, Type, TypeCode
import numpy as np # TODO: rewrite with pure python


def test_all():
    # Test new Buffer API

    input_port = Port(name='input', type=Type.from_dtype(np.dtype(np.int32)), dim=2)
    value41 = Param(key='v', val='41')

    builder = Builder()
    builder.set_target(target='host')
    builder.with_bb_module(path='ion-bb-test')

    node = builder.add('test_inc_i32x2').set_iport([input_port]).set_param(params=[ value41, ])

    idata = np.full((4, 4), fill_value=1, dtype=np.int32)
    ibuf = Buffer(array=idata)

    odata = np.full((4, 4), fill_value=0, dtype=np.int32)
    obuf = Buffer(array=odata)

    input_port.bind(ibuf)
    output_port = node.get_port(name='output')
    output_port.bind(obuf)

    # First run
    builder.run()

    for y in range(4):
        for x in range(4):
            assert odata[y][x] == 42

    # Second
    for y in range(4):
        for x in range(4):
            idata[y][x] = 2

    builder.run()

    for y in range(4):
        for x in range(4):
            assert odata[y][x] == 43
    print("passed")
