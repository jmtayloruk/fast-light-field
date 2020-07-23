cdef api double integrandPSF_r(int, double[4])
cdef api double integrandPSF_i(int, double[4])

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double sqrt(double)

#cdef extern from "specfunc/gsl_sf_bessel.h":
#    double gsl_sf_bessel_J0(const double x)

cdef extern from "bessel.h":
    double bessj0(double x)
