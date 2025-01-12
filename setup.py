from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile_args = []
if sys.platform.startswith('win'):
    extra_compile_args.append('/EHsc')

ext_modules = [
    Pybind11Extension(
        "mathext",
        ["mathext.cpp"],
        cxx_std=11,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="mathext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.6",
    install_requires=['pybind11>=2.6.0'],
    setup_requires=['pybind11>=2.6.0'],
)