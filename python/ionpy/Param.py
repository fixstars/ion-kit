import ctypes
from typing import Optional, Any

from .native import (
    c_ion_param_t,
    ion_param_create,
    ion_param_destroy,
)


class Param:
    def __init__(self,
        key: Optional[str] = None,
        val: Any = None,
        # -- or
        obj_: Optional[c_ion_param_t] = None,
    ):
        if obj_ is None:
            obj_ = c_ion_param_t()
            if isinstance(val, bool):
                if val:
                    val = "true"
                else:
                    val = "false"
            ret = ion_param_create(ctypes.byref(obj_), key.encode(), str(val).encode())
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_

    def __del__(self):
        if self.obj: # check not nullptr
            ion_param_destroy(self.obj)
