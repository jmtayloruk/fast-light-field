# Cribbed from https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(name="light_field_integrands",
      version="1.0.0",
      description="Utility module used by fast-light-field, assisting with integrals for PSF generation",
      author="Jonathan Taylor, University of Glasgow",
      url="https://github.com/jmtayloruk/fast-light-field",
      ext_modules=cythonize(Extension("light_field_integrands",
                                      sources=["light_field_integrands.pyx", "bessel.c"]),
                            annotate=True),
      install_requires=[])
