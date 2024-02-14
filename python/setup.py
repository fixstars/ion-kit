import io

from setuptools import setup
import sys
from typing import List
import platform
import sysconfig
import os

from setuptools._distutils.util import convert_path


def get_plat():
    if platform.system() == 'Linux':
        plat_form = "manylinux1_x86_64"
    else:
        plat_form = sysconfig.get_platform()
    return plat_form


def get_version():
    if os.environ.get("ION_KIT_VERSION") is not None:
        tag = os.environ.get("ION_KIT_VERSION")
    else:
        main_ns = {}
        ver_path = convert_path('./ionpy/version.py')
        with open(ver_path) as ver_file:
            exec(ver_file.read(), main_ns)
        tag = main_ns["__version__"]
    return tag


def main():
    long_description = io.open("README.md", encoding="utf-8").read()
    package_data: List[str] = []

    if platform.system() == 'Windows':
        package_data = ["module/windows/*"]
    elif platform.system() == 'Darwin':
        package_data = ["module/macos/*"]
    elif platform.system() == 'Linux':
        package_data = ["module/linux/*"]

    setup(
        name="ion-python",
        author="Takuro Iizuka",
        author_email="t_iizuka@fixstars.com",
        packages=["ionpy"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/fixstars/ion-kit",
        version=get_version(),
        python_requires=">=3.8.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        description="Python Binding for ion-kit",
        package_data={"ionpy": package_data},
        install_requires=["numpy>=1.24"],
        # ext_modules=EmptyListWithLength(),
        include_package_data=False,
        options={
        "bdist_wheel": {
            "plat_name": get_plat(),
            "python_tag": "py3",
        },
    },
    )


# This creates a list which is empty but returns a length of 1.
# Should make the wheel a binary distribution and platlib compliant.
class EmptyListWithLength(list):
    def __len__(self):
        return 1


if __name__ == "__main__":
    main()
