import ctypes
from typing import List, Optional

from .native import (
    c_ion_node_t,
    c_ion_port_t,
    c_ion_param_t,
    ion_node_create,
    ion_node_destroy,
    ion_node_get_port,
    ion_node_set_iport,
    ion_node_set_param,
)
from .Type import Type
from .Port import Port
from .Param import Param


class Node:
    def __init__(self,
        obj_: Optional[c_ion_param_t] = None,
    ):
        if obj_ is None:
            obj_ = c_ion_node_t()

            ret = ion_node_create(ctypes.byref(obj_))
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_

    def __del__(self):
        if self.obj: # check not nullptr
            ion_node_destroy(self.obj)

    # TODO: Make it work well
    # def __call__(self, *args):
    #     self.set_iport(list(*args))

    def get_port(self, name: str) -> Port:
        c_port = c_ion_port_t()

        ret = ion_node_get_port(self.obj, name.encode(), ctypes.byref(c_port))
        if ret != 0:
            raise Exception('Invalid operation')

        return Port(obj_=c_port)

    def set_iport(self, ports: List[Port]) -> 'Node':
        num_ports = len(ports)
        c_ion_port_sized_array_t = c_ion_port_t * num_ports # arraysize == num_ports
        c_ports = c_ion_port_sized_array_t() # instance

        for i in range(num_ports):
            c_ports[i] = ports[i].obj

        ret = ion_node_set_iport(self.obj, c_ports, num_ports)
        if ret != 0:
            raise Exception('Invalid operation')

        return self

    def set_param(self, params: List[Param]) -> 'Node':
        num_params = len(params)
        c_ion_param_sized_array_t = c_ion_param_t * num_params # arraysize == num_params
        c_params = c_ion_param_sized_array_t() # instance

        for i in range(num_params):
            c_params[i] = params[i].obj

        ret = ion_node_set_param(self.obj, c_params, num_params)
        if ret != 0:
            raise Exception('Invalid operation')

        return self
