import ctypes
from typing import Optional, Union, List
import numpy as np

from .native import (
    c_ion_port_t,
    c_ion_buffer_t,
    ion_port_create,
    ion_port_create_with_index,
    ion_port_destroy,

    ion_port_bind_i8,
    ion_port_bind_i16,
    ion_port_bind_i32,
    ion_port_bind_i64,

    ion_port_bind_u1,
    ion_port_bind_u8,
    ion_port_bind_u16,
    ion_port_bind_u32,
    ion_port_bind_u64,

    ion_port_bind_f32,
    ion_port_bind_f64,
    ion_port_bind_buffer,
    ion_port_bind_buffer_array
)

from .Type import Type
from .Buffer import Buffer
from .TypeCode import TypeCode

class Port:
    def __init__(self,
        name: Optional[str] = None,
        type: Optional[Type] = None,
        dim: Optional[int] = None,
        # -- or
        obj_: Optional[c_ion_port_t] = None,
    ):
        if obj_ is None:
            obj_ = c_ion_port_t()
            type_cobj = type.to_cobj()

            ret = ion_port_create(ctypes.byref(obj_), name.encode(), type_cobj, dim)
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_
        self.dim = dim
        self.type = type
        self.bind_value = None  # default

    def __getitem__(self, index):
        new_obj = c_ion_port_t()
        ret = ion_port_create_with_index(ctypes.byref(new_obj), self.obj, index)
        if ret != 0:
            raise Exception('Invalid operation')
        return Port(obj_=new_obj)

    def __del__(self):
        if self.obj: # check not nullptr
            ion_port_destroy(self.obj)

    def bind(self, v: Union[int, float, Buffer, List[Buffer]]):
        if self.dim == 0:
            if self.bind_value is None:
                self.bind_value = np.ctypeslib.as_ctypes_type(self.type.to_dtype())(v)
            else:
                self.bind_value.value = v
            # scalar
            if self.type.code_ == TypeCode.Int:
                if self.type.bits_ == 8 and ion_port_bind_i8(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                elif self.type.bits_ == 16 and ion_port_bind_i16(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                elif self.type.bits_ == 32 and ion_port_bind_i32(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                elif self.type.bits_ == 64 and ion_port_bind_i64(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
            elif self.type.code_ == TypeCode.Uint:
                if self.type.bits_ == 1 and ion_port_bind_u1(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                if self.type.bits_ == 8 and ion_port_bind_u8(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                if self.type.bits_ == 16 and ion_port_bind_u16(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                if self.type.bits_ == 32 and ion_port_bind_u32(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                if self.type.bits_ == 64 and ion_port_bind_u64(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
            elif self.type.code_ == TypeCode.Float:
                if self.type.bits_ == 32 and ion_port_bind_f32(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
                if self.type.bits_ == 64 and ion_port_bind_f64(self.obj, ctypes.byref(self.bind_value)) != 0:
                    raise Exception('Invalid operation')
        #  vector
        else:

            self.bind_value = v
            if type(v) is not list:
                if ion_port_bind_buffer(self.obj, v.obj) != 0:
                    raise Exception('Invalid operation')
            else:
                num_buffers = len(v)
                c_buffers = (c_ion_buffer_t * num_buffers)()
                for i in range(num_buffers):
                    c_buffers[i] = v[i].obj
                if ion_port_bind_buffer_array(self.obj, c_buffers, num_buffers) != 0:
                    raise Exception('Invalid operation')
