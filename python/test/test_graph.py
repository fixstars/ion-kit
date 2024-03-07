from ionpy import Builder, Graph, Port, Buffer, Type
import numpy as np

def test_graph():
    input_port0 = Port(name='input', type=Type.from_dtype(np.dtype(np.int32)), dim=2)
    value_port0 = Port(name='v', type=Type.from_dtype(np.dtype(np.int32)), dim=0)

    builder = Builder()
    builder.set_target(target='host')
    builder.with_bb_module(path='ion-bb-test')

    graph0 = builder.add_graph("graph0")

    node0 = graph0.add('test_incx_i32x2').set_iport([input_port0, value_port0])

    idata0 = np.array([[42, 42]], dtype=np.int32)
    ibuf0 = Buffer(array=idata0)

    odata0 = np.array([[0, 0]], dtype=np.int32)
    obuf0 = Buffer(array=odata0)

    input_port0.bind(ibuf0)
    output_port0 = node0.get_port(name='output')
    output_port0.bind(obuf0)
    value_port0.bind(0)

    input_port1 = Port(name='input', type=Type.from_dtype(np.dtype(np.int32)), dim=2)
    value_port1 = Port(name='v', type=Type.from_dtype(np.dtype(np.int32)), dim=0)

    graph1 = builder.add_graph("graph1")
    # graph1 = Graph(builder =builder, name="graph1") # alternative
    node1 = graph1.add('test_incx_i32x2').set_iport([input_port1, value_port1])

    idata1 = np.array([[42, 42]], dtype=np.int32)
    ibuf1 = Buffer(array=idata1)

    odata1 = np.array([[0, 0]], dtype=np.int32)
    obuf1 = Buffer(array=odata1)

    input_port1.bind(ibuf1)
    output_port = node1.get_port(name='output')
    output_port.bind(obuf1)
    value_port1.bind(1)

    g = graph0 + graph1
    g.run()
    # alternative
    # graph1 += graph0
    # g = g(sub_graphs=[graph1,graph0])

    assert odata0[0][0] == 42
    assert odata1[0][0] == 43
