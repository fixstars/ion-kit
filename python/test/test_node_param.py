from ionpy import Node, Param


def test_node_param():
    param_to_set = Param(key='iamkey', val='iamval')

    params = [ param_to_set, ]

    n = Node()
    n.set_param(params)
