from ion import Node, Port, Type, TypeCode


def test_node_port():
    t = Type(code_=TypeCode.Int, bits_=32, lanes_=1)

    port_to_set = Port(key='iamkey', type=t, dim=3)

    ports = [ port_to_set, ]

    n = Node()
    n.set_port(ports)

    port_to_get = n.get_port('iamkey')
    print(f'from node.get_port: {port_to_get}')
