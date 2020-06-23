# Cribbed from https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# Note the hard-coded relative path to scatter/gsl.
# TODO: obviously that needs sorting out before this is releasable code
# (and may need some contrived setting up on other platforms to make it work!)
setup(ext_modules=cythonize(Extension("light_field_integrands",
                                      sources=["light_field_integrands.pyx"],
                                      include_dirs=['./', '../../scatter/gsl/'],
                                      extra_link_args=['../../scatter/gsl/.libs/libgsl.a']),
                            annotate=True))
