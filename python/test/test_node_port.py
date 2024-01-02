from ionpy import Node, Builder, Port, Type, TypeCode


def test_node_port():
    t = Type(code_=TypeCode.Int, bits_=32, lanes_=1)

    port_to_set = Port(name='input', type=t, dim=2)

    builder = Builder()
    builder.set_target(target='host')
    # make sure path includes libion-bb-test.so
    builder.with_bb_module(path='ion-bb-test')
    # builder.with_bb_module(path='ion-bb-test.dll') # for Windows

    n = builder.add('test_inc_i32x2').set_iport(ports=[ port_to_set, ])

    port_to_get = n.get_port('input')
    print(f'from node.get_port: {port_to_get}')

