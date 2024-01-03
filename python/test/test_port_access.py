from ionpy import Port, Type, TypeCode

import ctypes

def test_port():
    t = Type(code_=TypeCode.Int, bits_=32, lanes_=1)

    p = Port(name='iamkey', type=t, dim=3)
    p = p[1]
    print(p)
