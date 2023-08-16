import numpy
import sysconfig
from distutils.core import setup, Extension

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args.append("-std=c++17")  # may need to be fine-tuned depending on the compiler


module = Extension(
    "arrayutils",
    sources=["cpp/arrayutils_module.cpp"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name="arrayutils",
    version="0.1",
    description="arrayutils",
    ext_modules=[module],
)
