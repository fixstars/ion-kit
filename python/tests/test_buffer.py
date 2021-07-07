from ionpy import Buffer, Type, TypeCode


def test_buffer():
    t = Type(code_=TypeCode.Float, bits_=32, lanes_=1)
    sizes = [ 10, ]

    b = Buffer(type=t, sizes=sizes)
    print(b)
