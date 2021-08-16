'''
Author: Ligcox
Date: 2021-07-01 00:16:13
LastEditors: Ligcox
LastEditTime: 2021-08-13 00:31:25
Description: C++ and Python hybrid Programming example
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

ext_modules = [
    Extension(
    'wrap',
        ['funcs.cpp', 'wrap.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='wrap',
    version='1.0.0',
    author='Ligcox',
    author_email='zyhbum@foxmail.com',
    description='C++ and Python hybrid Programming example',
    ext_modules=ext_modules,
)