import ctypes
from typing import Optional

from .native import (
    c_ion_port_t,
    ion_port_create,
    ion_port_destroy,
    ion_port_create_with_index,
)

from .Type import Type


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

            ret = ion_port_create(ctypes.byref(obj_), self.name.encode(), type_cobj, self.dim)
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_

    def __getitem__(self, i):
        new_obj = c_ion_port_t()
        ret = ion_port_create_with_index(ctypes.byref(new_obj), self.obj, index)
        if ret != 0:
            raise Exception('Invalid operation')
        return new_obj

    def __del__(self):
        if self.obj: # check not nullptr
            ion_port_destroy(self.obj)
