from dataclasses import dataclass

from .TypeCode import TypeCode
from .native import (
    c_ion_type_t,
)


@dataclass
class Type:
    code_: TypeCode
    bits_: int
    lanes_: int

    def to_cobj(self) -> c_ion_type_t:
        return c_ion_type_t(self.code_.value, self.bits_, self.lanes_)
