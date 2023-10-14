from Cython.Build import cythonize
import numpy
from setuptools import Extension
from setuptools import setup

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mise (efficient mesh extraction)
mise_module = Extension(
    "utils.libmise.mise",
    sources=["utils/libmise/mise.pyx"],
)

# Gather all extension modules
ext_modules = [
    mise_module,
]

setup(
    name="mise", # zhanglb_change
    ext_modules=cythonize(ext_modules),
    )