# cython: profile=False

# Old code that relied on GSL, left for reference. Now replaced with much more lightweight code.

import numpy as np
import scipy.special
import cython

@cython.cdivision(True)
cdef api double integrandPSF_r(int n, double[4] args):
    theta = args[0]
    alphaFactor = args[1]
    uOver2 = args[2]
    vFactor = args[3]
    cosTheta = cos(theta)
    sinTheta = sin(theta)
    sint2 = sin(theta/2)
    cdef double temp2 = gsl_sf_bessel_J0(sinTheta*vFactor)
    return ((sqrt(cosTheta)) * (1+cosTheta)  \
            *  (cos(-(uOver2)* (sint2*sint2) * alphaFactor)) \
            *  (temp2)
            *  (sinTheta))

@cython.cdivision(True)
cdef api double integrandPSF_i(int n, double[4] args):
    theta = args[0]
    alphaFactor = args[1]
    uOver2 = args[2]
    vFactor = args[3]
    cosTheta = cos(theta)
    sinTheta = sin(theta)
    sint2 = sin(theta/2)
    cdef double temp2 = gsl_sf_bessel_J0(sinTheta*vFactor)
    return ((sqrt(cosTheta)) * (1+cosTheta)  \
            *  (sin(-(uOver2)* (sint2*sint2) * alphaFactor)) \
            *  (temp2) \
            *  (sinTheta))
