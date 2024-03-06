import ctypes
from typing import Optional

from .native import (
    c_ion_node_t,
    c_ion_graph_t,

    ion_graph_create,
    ion_graph_destroy,
    ion_graph_create_with_multiple,
    ion_graph_run,
    ion_graph_add_node
)

from .Node import Node


class Graph:
    def __init__(self,
        builder = None,
        name: Optional[str] = None,
        obj_: Optional[c_ion_graph_t] = None,
        sub_graphs: [] = None,

    ):
        if obj_ is None and builder is not None and name is not None:
            obj_ = c_ion_graph_t()
            ret = ion_graph_create(ctypes.byref(obj_), builder.obj, name.encode())
            if ret != 0:
                raise Exception('Invalid operation')
        elif obj_ is None and sub_graphs is not None:
            num_graphs = len(sub_graphs)
            c_ion_graph_sized_array_t = c_ion_graph_t * num_graphs  # arraysize == num_graphs
            c_graphs = c_ion_graph_sized_array_t()  # instance
            for i in range(num_graphs):
                c_graphs[i] = sub_graphs[i].obj
            obj_ = c_ion_graph_t()
            ret = ion_graph_create_with_multiple(ctypes.byref(obj_), c_graphs, num_graphs)
            if ret != 0:
                raise Exception('Invalid operation')

        self.obj = obj_

    def __del__(self):
        if self.obj: # check not nullptr
            ion_graph_destroy(self.obj)

        # adding two objects

    def __add__(self, other):
        if isinstance(other, Graph):
            c_ion_graph_sized_array_t = c_ion_graph_t * 2  # arraysize == num_graphs
            c_graphs = c_ion_graph_sized_array_t()  # instance
            c_graphs[0] = self.obj
            c_graphs[1] = other.obj
            new_obj = c_ion_graph_t()
            ret = ion_graph_create_with_multiple(ctypes.byref(new_obj), c_graphs, 2)
            if ret != 0:
                raise Exception('Invalid operation')
            return Graph(obj_=new_obj)

    def __iadd__(self, other):
        if isinstance(other, Graph):
            c_ion_graph_sized_array_t = c_ion_graph_t * 2  # arraysize == num_graphs
            c_graphs = c_ion_graph_sized_array_t()  # instance
            c_graphs[0] = self.obj
            c_graphs[1] = other.obj
            ret = ion_graph_create_with_multiple(ctypes.byref(self.obj), c_graphs, 2)
            if ret != 0:
                raise Exception('Invalid operation')
            return self


    def run(self):
        ret = ion_graph_run(self.obj)
        if ret != 0:
            raise Exception('Invalid operation')

    def add(self, key: str) -> Node:
        c_node = c_ion_node_t()

        ret = ion_graph_add_node(self.obj, key.encode(), ctypes.byref(c_node))
        if ret != 0:
            raise Exception('Invalid operation')

        return Node(obj_=c_node)
