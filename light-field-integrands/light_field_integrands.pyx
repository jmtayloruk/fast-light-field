# cython: profile=False

import numpy as np
import scipy.special
import cython

# Note: this paper flips the sign of the exponential term
#  https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-10-1-29&id=402869
#  That does completely change the structure of the PSF.
#  I presume it's correct for their scenario, but I very much doubt it is a
#  correction that applies in the Prevedel scenario, or they would have noticed the issue before!
@cython.cdivision(True)
cdef api double integrandPSF_r(int n, double[4] args):
    theta = args[0]
    alphaFactor = args[1]
    uOver2 = args[2]
    vFactor = args[3]
    cosTheta = cos(theta)
    sinTheta = sin(theta)
    sint2 = sin(theta/2)
    cdef double temp2 = bessj0(sinTheta*vFactor)
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
    cdef double temp2 = bessj0(sinTheta*vFactor)
    return ((sqrt(cosTheta)) * (1+cosTheta)  \
            *  (sin(-(uOver2)* (sint2*sint2) * alphaFactor)) \
            *  (temp2) \
            *  (sinTheta))
