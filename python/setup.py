import io

from setuptools import setup
import sys
from typing import List
import platform
import sysconfig
import os

from setuptools._distutils.util import convert_path
from setuptools.command.egg_info import egg_info


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)


def get_plat():
    if platform.system() == 'Linux' and platform.machine() == 'x86_64':
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


def get_name():
    if os.environ.get("USE_CONTRIB") is not None:
        return "ion-contrib-python"
    return "ion-python"


def get_liciense():
    if os.environ.get("USE_CONTRIB") is not None:
        return ['LICENSE', 'LICENSE-3RD-PARTY']
    return ['LICENSE']


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
        name=get_name(),
        author="Takuro Iizuka",
        author_email="t_iizuka@fixstars.com",
        packages=["ionpy"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/fixstars/ion-kit",
        version=get_version(),
        python_requires=">=3.8.0",
        license="MIT License",
        license_files=get_liciense(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
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
