# Snippet that is looking into the use of pyfftw.
# This may be somewhat redundant now that MKL seems to be being used as a backend
# (although it might also be useful to help me disable that usage while investigating performance!)

# Note that (as of Mar 2023) there are special instructions that must be followed if you are
# installing pyfftw on Apple M1 hardware: https://github.com/andrej5elin/howto_fftw_apple_silicon

import scipy.fftpack
import numpy as np
from numpy.fft import fft, fftn, rfft, rfftn, irfftn
import warnings

try:
    import cupy as cp
except ImportError:
    pass

if True:
    # Old FFT code
    # I have reactivated this old code branch because of problems I have encountered with pyfftw.
    # That should not be a problem, since I don't use this myfft.py code for anything performance-critical.
    #
    # Detailed explanation for posterity:
    # If I have the pyfftw module installed on OS X then I get dylib errors because of
    # some sort of incompatibility with how it is built, and how distutils extensions
    # seem to be build for OS X 10.9 (at least they are on my laptop and I can't see how to change that).
    # The result is that my light field C code picks up the "wrong" fftw dependency dylib and
    # fails on launch as a consequence of this.
    # (additional reminder to myself of these instructions for installing pyfftw on Apple M1 hardware: https://github.com/andrej5elin/howto_fftw_apple_silicon)
    def myFFT2(mat, shape):
	# Perform a 'float' FFT on the matrix we are passed.
        # It would probably be faster if there was a way to perform the FFT natively on the 'float' type,
        # but scipy does not seem to support that option
        #
        # With my Mac Pro install, we hit a FutureWarning within scipy.
        # This wrapper just suppresses that warning.
        assert(mat.dtype == np.float32)    # This is what we expect. If not, our cast on the next line may be wrong
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return scipy.fftpack.fft2(mat, shape).astype('complex64')

    # TODO: is there a reason that I use irfftn instead of the scipy version? It's inconsistent with the above call.
    # I wonder if it might possibly be something to do with being able to do a 2D FFT on a 3D array?
    # I use the non-real forward FFT because I want the full array so that I can mirror it (add more detail to comment...)
    # but since the data is actually real I can use the (faster) irfftn.
    def myIFFT2(mat, shape):
        assert(mat.dtype == np.complex64)    # Because otherwise our cast on the next line will be wrong
        return irfftn(mat, shape).astype('float32')
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
        # Note: supplying threads=8 does not seem to help - it actually seems to *increase* run time by about 6x.
        # I have not tried to understand why that is. Also specifying FFTW_ESTIMATE does not seem to change anything,
        # but I don't feel I have *definitely* ruled out it being some sort of planning issue...
        # TODO: I suspect I don't do anything clever with 'shape', and should probably not accept it as a parameter since my C code wouldn't know what to do with it.
        return pyfftw.interfaces.numpy_fft.irfft2(mat, shape)


def myFFT2_gpu(mat, shape):
    return cp.fft.fft2(mat, shape)

def myIFFT2_gpu(mat, shape):
    return cp.fft.ifft2(mat, shape)
