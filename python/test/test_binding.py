import ctypes

from ionpy import Node, Builder, Buffer, PortMap, Port, Param, Type, TypeCode
import numpy as np


def test_binding():
    input_port = Port(name='input', type=Type.from_dtype(np.dtype(np.int32)), dim=2)
    value_port = Port(name='v', type=Type.from_dtype(np.dtype(np.int32)), dim=0)

    builder = Builder()
    builder.set_target(target='host')
    builder.with_bb_module(path='ion-bb-test')

    node = builder.add('test_incx_i32x2').set_iport([input_port, value_port])

    idata = np.array([[42, 42]], dtype=np.int32)
    ibuf = Buffer(array=idata)

    odata = np.array([[0, 0]], dtype=np.int32)
    obuf = Buffer(array=odata)

    input_port.bind(ibuf)
    output_port = node.get_port(name='output')
    output_port.bind(obuf)

    # First run
    value_port.bind(0)
    builder.run()
    assert odata[0][0] == 42

    # Second run
    value_port.bind(1)
    builder.run()
    assert odata[0][0] == 43

    # Third run
    value_port.bind(2)
    builder.run()
    assert odata[0][0] == 44
