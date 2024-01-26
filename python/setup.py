from setuptools import setup, find_namespace_packages
import sys
from typing import List
import platform

install_requires: List[str] = []
import sysconfig

python_version = sys.version_info
if python_version < (3, 7):
    install_requires += ['dataclasses']


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
    install_requires=install_requires,
    package_data={"ionpy": package_data},
    # ext_modules=EmptyListWithLength(),
    include_package_data=False,
    options={
        "bdist_wheel": {
            "plat_name": sysconfig.get_platform(),
            "python_tag": "py3",
        },
    },
)
