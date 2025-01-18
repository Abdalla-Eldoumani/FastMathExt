from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import platform

extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    extra_compile_args.extend([
        '/openmp', 
        '/O2',
        '/fp:fast',
        '/GL',
    ])
    extra_link_args.extend([
        '/LTCG',
    ])
else:
    extra_compile_args.extend([
        '-fopenmp',
        '-O3',
        '-march=native',
        '-ffast-math',
        '-funroll-loops',
    ])
    extra_link_args.extend([
        '-fopenmp',
    ])

ext_modules = [
    Pybind11Extension(
        "MathExt",
        ["MathExt.cpp"],
        cxx_std=14,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="MathExt",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.6",
    install_requires=['pybind11>=2.6.0'],
    setup_requires=['pybind11>=2.6.0'],
)