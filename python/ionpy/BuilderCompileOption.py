from dataclasses import dataclass

from .native import (
    c_builder_compile_option_t,
)


@dataclass
class BuilderCompileOption:
    output_directory: str

    def to_cobj(self) -> c_builder_compile_option_t:
        return c_builder_compile_option_t(output_directory.encode())
