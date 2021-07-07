from setuptools import setup
import sys
from typing import List

install_requires: List[str] = []

python_version = sys.version_info
if python_version < (3, 7):
    install_requires += [ 'dataclasses' ]

setup(
    name='ionpy',
    version='0.1.0-alpha',

    packages=[ 'ionpy', ],
    install_requires=install_requires,

    author='Takuro Iizuka',
    author_email='t_iizuka@fixstars.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
