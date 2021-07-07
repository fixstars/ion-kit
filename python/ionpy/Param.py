import ctypes
from typing import Optional

from .native import (
    c_ion_param_t,
    ion_param_create,
    ion_param_destroy,
)


class Param:
    def __init__(self,
        key: Optional[str] = None,
        val: Optional[str] = None,
        # -- or
        obj_: Optional[c_ion_param_t] = None,
    ):
        if obj_ is None:
            obj_ = c_ion_param_t()

            ret = ion_param_create(ctypes.byref(obj_), key.encode(),val.encode())
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_

    def __del__(self):
        if self.obj: # check not nullptr
            ion_param_destroy(self.obj)
