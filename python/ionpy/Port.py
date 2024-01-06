import ctypes
from typing import Optional

from .native import (
    c_ion_port_t,
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
    ion_port_bind_buffer
)

from .Type import Type
from .Buffer import Buffer

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

    def __getitem__(self, index):
        new_obj = c_ion_port_t()
        ret = ion_port_create_with_index(ctypes.byref(new_obj), self.obj, index)
        if ret != 0:
            raise Exception('Invalid operation')
        return Port(obj_=new_obj)

    def __del__(self):
        if self.obj: # check not nullptr
            ion_port_destroy(self.obj)

    
        # should use numpy type?
    def bind_i8(self, v: int):
        if ion_port_bind_i8(self.obj,  ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_i16(self, v: int):
        if ion_port_bind_i16(self.obj, ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_i32(self, v: int):
        if ion_port_bind_i32(self.obj, ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_i64(self, v: int):
        if ion_port_bind_i64(self.obj, ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_u1(self, v: bool):
        if ion_port_bind_u1(self.obj, ctypes.byref(v))!= 0:
            raise Exception('Invalid operation')

    def bind_u8(self, v: int):
        if ion_port_bind_u8(self.obj,  ctypes.byref(v))!= 0:
            raise Exception('Invalid operation')

    def bind_u16(self, v: int):
        if ion_port_bind_u16(self.obj, ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_u32(self, v: int):
        if ion_port_bind_u32(self.obj,  ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_u64(self, v: int):
        if ion_port_bind_u64(self.obj, ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_f32(self,  v: float):
        if ion_port_bind_f32(self.obj,ctypes.byref(v)) != 0:
            raise Exception('Invalid operation')

    def bind_f64(self, v: float):
        if ion_port_bind_f64(self.obj, ctypes.byref(v))!= 0:
            raise Exception('Invalid operation')
    
    def bind_buffer(self, buffer: Buffer):
        if ion_port_bind_buffer(self.obj, buffer.obj) != 0:
            raise Exception('Invalid operation')
