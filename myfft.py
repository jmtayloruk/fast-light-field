# Snippet that is looking into the use of pyfftw.
# This may be somewhat redundant now that MKL seems to be being used as a backend
# (although it might also be useful to help me disable that usage while investigating performance!)

import scipy.fftpack
from numpy.fft import fft, fftn, rfft, rfftn, irfftn
import warnings

if True:
    # Old FFT code
    def myFFT2(mat, shape):
        # Perform a 'float' FFT on the matrix we are passed.
        # It would probably be faster if there was a way to perform the FFT natively on the 'float' type,
        # but scipy does not seem to support that option
        #
        # With my Mac Pro install, we hit a FutureWarning within scipy.
        # This wrapper just suppresses that warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return scipy.fftpack.fft2(mat, shape).astype('complex64')

    def myIFFT2(mat, shape):
        return irfftn(mat, shape)
else:
    # New FFT code based on FFTW.
    # It may be possible to speed this up slightly be using their "proper" interface
    # rather than the scipy-like interface (see pyfftw documentation)
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10.0)
    
    def myFFT2(mat, shape):
        return pyfftw.interfaces.scipy_fftpack.fft2(mat, shape)
    
    def myIFFT2(mat, shape):
        return pyfftw.interfaces.numpy_fft.irfft2(mat, shape)