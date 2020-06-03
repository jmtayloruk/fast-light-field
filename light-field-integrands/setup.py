# Cribbed from https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension("light_field_integrands",
                                    sources=["light_field_integrands.pyx"],
                                    include_dirs=['./', '/Volumes/Development/scatter/gsl/'],
                                    extra_link_args=['-L../../scatter/gsl/.libs', '-lgsl']),
                          annotate=True)
)
