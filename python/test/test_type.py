from ionpy import Type, TypeCode


def test_type():
    t = Type(code_=TypeCode.Int, bits_=32, lanes_=1)
    print(t)
