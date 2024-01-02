from ionpy import Buffer, Type, TypeCode
import numpy as np

def test_buffer():
    # From scratch
    t = Type(code_=TypeCode.Float, bits_=32, lanes_=1)
    sizes = [ 10, ]

    b = Buffer(type=t, sizes=sizes)
    print(b)

    array = np.array([100, 200, 300, 400, 500], dtype=np.int16)

    # Pass-through ndarray
    b = Buffer(array=array)
    print(b)

