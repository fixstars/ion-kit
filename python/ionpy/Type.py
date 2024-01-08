from dataclasses import dataclass

from .TypeCode import TypeCode
from .native import (
    c_ion_type_t,
)

import numpy as np

@dataclass
class Type:
    code_: TypeCode
    bits_: int
    lanes_: int

    def to_cobj(self) -> c_ion_type_t:
        return c_ion_type_t(self.code_.value, self.bits_, self.lanes_)

    @staticmethod
    def from_dtype(dtype):
        table = {
                  np.dtype(np.bool_):   Type(code_=TypeCode.Uint,  bits_=1,  lanes_=1),
                  np.dtype(np.int8):    Type(code_=TypeCode.Int,   bits_=8,  lanes_=1),
                  np.dtype(np.int16):   Type(code_=TypeCode.Int,   bits_=16, lanes_=1),
                  np.dtype(np.int32):   Type(code_=TypeCode.Int,   bits_=32, lanes_=1),
                  np.dtype(np.int64):   Type(code_=TypeCode.Int,   bits_=64, lanes_=1),
                  np.dtype(np.uint8):   Type(code_=TypeCode.Uint,  bits_=8,  lanes_=1),
                  np.dtype(np.uint16):  Type(code_=TypeCode.Uint,  bits_=16, lanes_=1),
                  np.dtype(np.uint32):  Type(code_=TypeCode.Uint,  bits_=32, lanes_=1),
                  np.dtype(np.uint64):  Type(code_=TypeCode.Uint,  bits_=64, lanes_=1),
                  np.dtype(np.float32): Type(code_=TypeCode.Float, bits_=32, lanes_=1),
                  np.dtype(np.float64): Type(code_=TypeCode.Float, bits_=64, lanes_=1)
                }
        if dtype in table:
            return table[dtype]
        else:
            raise Exception("Unknown dtype: {}".format(dtype))

    def to_dtype(self):
        if self.code_ == TypeCode.Int:
            if self.bits_ == 8:
                return np.int8
            elif self.bits_ == 16:
                return np.int16
            elif self.bits_ == 32:
                return np.int32
            elif self.bits_ == 64:
                return np.int64
        elif self.code_ == TypeCode.Uint:
            if self.bits_ == 1:
                return np.bool_
            if self.bits_ == 8:
                return np.uint8
            if self.bits_ == 16:
                return np.uint16
            if self.bits_ == 32:
                return np.uint32
            if self.bits_ == 64:
                return np.uint64
        elif self.code_ == TypeCode.Float:
            if self.bits_ == 32:
                return np.float32
            elif self.bits_ == 64:
                return np.float64
        else:
            raise Exception("Unknown Type: {}".format(Type))
