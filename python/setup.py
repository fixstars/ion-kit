from setuptools import setup, find_namespace_packages
import sys
from typing import List
import platform
import sysconfig


def get_plat():
    if platform.system() == 'Linux':
        plat_form = "manylinux1_x86_64"
    else:
        plat_form = sysconfig.get_platform()
    return plat_form


# This creates a list which is empty but returns a length of 1.
# Should make the wheel a binary distribution and platlib compliant.
# class EmptyListWithLength(list):
#     def __len__(self):
#         return 1

package_data: List[str] = []

if platform.system() == 'Windows':
    package_data = ["module/windows/*"]
elif platform.system() == 'Darwin':
    package_data = ["module/macos/*"]
elif platform.system() == 'Linux':
    package_data = ["module/linux/*"]

setup(
    packages=["ionpy"],
    package_data={"ionpy": package_data},
    # ext_modules=EmptyListWithLength(),
    include_package_data=False,
    options={
        "bdist_wheel": {
            "plat_name": get_plat(),
            "python_tag": "py3",
        },
    },
)
