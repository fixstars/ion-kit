import ctypes
from typing import List

from .native import (
    c_ion_port_map_t,
    c_ion_buffer_t,

    ion_port_map_create,
    ion_port_map_destroy,

    ion_port_map_set_i8,
    ion_port_map_set_i16,
    ion_port_map_set_i32,
    ion_port_map_set_i64,

    ion_port_map_set_u1,
    ion_port_map_set_u8,
    ion_port_map_set_u16,
    ion_port_map_set_u32,
    ion_port_map_set_u64,

    ion_port_map_set_f32,
    ion_port_map_set_f64,

    ion_port_map_set_buffer,
)

from .Port import Port
from .Buffer import Buffer


class PortMap:
    def __init__(self):
        c_port_map = c_ion_port_map_t()

        ret = ion_port_map_create(ctypes.byref(c_port_map))
        if ret != 0:
            raise Exception('Invalid operation')

        self.obj = c_port_map

    def __del__(self):
        if self.obj: # check not nullptr
            ion_port_map_destroy(self.obj)

    # should use numpy type?
    def set_i8(self, port: Port, v: int):
        if ion_port_map_set_i8(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_i16(self, port: Port, v: int):
        if ion_port_map_set_i16(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_i32(self, port: Port, v: int):
        if ion_port_map_set_i32(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_i64(self, port: Port, v: int):
        if ion_port_map_set_i64(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')


    def set_u1(self, port: Port, v: bool):
        if ion_port_map_set_u1(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_u8(self, port: Port, v: int):
        if ion_port_map_set_u8(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_u16(self, port: Port, v: int):
        if ion_port_map_set_u16(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_u32(self, port: Port, v: int):
        if ion_port_map_set_u32(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_u64(self, port: Port, v: int):
        if ion_port_map_set_u64(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')


    def set_f32(self, port: Port, v: float):
        if ion_port_map_set_f32(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')

    def set_f64(self, port: Port, v: float):
        if ion_port_map_set_f64(self.obj, port.obj, v) != 0:
            raise Exception('Invalid operation')


    def set_buffer(self, port: Port, buffer: Buffer):
        if ion_port_map_set_buffer(self.obj, port.obj, buffer.obj) != 0:
            raise Exception('Invalid operation')
