import ctypes
from typing import List

from .native import (
    c_ion_buffer_t,

    ion_buffer_create,
    ion_buffer_create_with_data,
    ion_buffer_destroy,

    ion_buffer_write,
    ion_buffer_read,
)

from .Type import Type


class Buffer:
    def __init__(self, type: Type=None, sizes: List[int]=None, array=None):
        c_buffer = c_ion_buffer_t()

        if array is None:
            num_sizes = len(sizes)
            c_sizes = (ctypes.c_int * num_sizes)(*sizes)
            ret = ion_buffer_create(ctypes.byref(c_buffer), type.to_cobj(), c_sizes, num_sizes)
        else:
            type = Type.from_dtype(array.dtype)
            c_sizes = (ctypes.c_int * array.ndim)(*reversed(array.shape))
            ret = ion_buffer_create_with_data(ctypes.byref(c_buffer), type.to_cobj(), array.ctypes.data_as(ctypes.c_void_p), c_sizes, array.ndim)

        if ret != 0:
            raise Exception('Invalid operation')

        self.obj = c_buffer

    def __del__(self):
        if self.obj: # check not nullptr
            ion_buffer_destroy(self.obj)


    # https://stackoverflow.com/questions/40732706/how-do-i-pass-void-array-to-a-c-function-via-ctypes
    def write(self, data: bytes):
        num_data_bytes = len(data)
        c_data = (ctypes.c_char * num_data_bytes)(*map(ctypes.c_char, data))

        ret = ion_buffer_write(self.obj, c_data, num_data_bytes)
        if ret != 0:
            raise Exception('Invalid operation')

    # https://stackoverflow.com/questions/15377338/convert-ctype-byte-array-to-bytes
    def read(self, num_data_bytes: int) -> bytes:
        c_data = (ctypes.c_char * num_data_bytes)()

        ret = ion_buffer_read(self.obj, c_data, num_data_bytes)
        if ret != 0:
            raise Exception('Invalid operation')

        return bytearray(c_data)
