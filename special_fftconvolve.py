import numpy as np
import py_symmetry as jps
import warnings
import scipy.fftpack
import myfft

# We start with various utility functions which are taken from the original source code of fftconvolve

from scipy._lib._version import NumpyVersion
from numpy.fft import fft, fftn, rfft, rfftn, irfftn
_rfft_mt_safe = (NumpyVersion(np.__version__) >= '1.9.0.dev-e24486e')

def _next_regular(target):
    """
        Find the next regular number greater than or equal to target.
        Regular numbers are composites of the prime factors 2, 3, and 5.
        Also known as 5-smooth numbers or Hamming numbers, these are the optimal
        size for inputs to FFTPACK.
        
        Target must be a positive integer.
        """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    currsize = np.array(arr.shape)
    newsize = np.asarray(newsize)
    if (len(currsize) > len(newsize)):
        newsize = np.append([currsize[0]], newsize)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def expand2Multiplier(bb, fshape, resultShape):
    return np.exp(-1j * bb * 2*np.pi / fshape[-2] * np.arange(resultShape[-2],dtype='complex64'))

def tempMul(bb,fshape,result):
    result *= expand2Multiplier(bb, fshape, result.shape)[...,np.newaxis]
    return result

def expand2(result, bb, aa, Nnum, fshape):
    tileFactor = (1,) * (len(result.shape)-2) + (Nnum, 1)
    return np.tile(result, tileFactor)

def expand(reducedF, bb, aa, Nnum, fshape, partial=False):
    tileFactor = (1,) * (len(reducedF.shape)-1) + (int(Nnum/2+1),)
    result = np.tile(reducedF, tileFactor)
    result = result[...,:int(fshape[-1]/2+1)]
    result *= np.exp(-1j * aa * 2*np.pi / fshape[-1] * np.arange(result.shape[-1],dtype='complex64'))
    if partial:
        return result
    result = expand2(result, bb, aa, Nnum, fshape)
    return tempMul(bb,fshape,result)

def special_rfftn(in1, bb, aa, Nnum, fshape, partial=False):
    # Compute the fft of elements in1[bb::Nnum,aa::Nnum], after in1 has been zero-padded out to fshape
    # We exploit the fact that fft(masked-in1) is fft(arr[::Nnum,::Nnum]) replicated Nnum times.
    reducedShape = ()
    for d in fshape:
        assert((d % Nnum) == 0)
        reducedShape = reducedShape + (int(d/Nnum),)
    reduced = in1[...,bb::Nnum,aa::Nnum]
    # Compute an array giving rfft(mask(in1)), i.e. the FFT for a smaller array consisting only of the pixels selected by the mask
    reducedF = myfft.myFFT2(reduced, reducedShape)
    # Expand this up to obtain the equivalent fourier transform for the full masked array (with intervening zeroes).
    return expand(reducedF, bb, aa, Nnum, fshape, partial=partial)

def convolutionShape(in1, in2Shape, Nnum):
    # Logic copied from fftconvolve source code
    s1 = np.array(in1.shape)
    s2 = np.array(in2Shape)
    if (len(s1) == 3):   # Cope with case where we are processing multiple reconstructions in parallel
        s1 = s1[1:]
    shape = s1 + s2 - 1
    if False:
        # TODO: I haven't worked out if/how I can do this yet.
        # This is the original code in fftconvolve, which says:
        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [_next_regular(int(d)) for d in shape]
    else:
        fshape = [int(np.ceil(d/float(Nnum)))*Nnum for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    return (fshape, fslice, s1)
    
def special_fftconvolve_part1(in1, bb, aa, Nnum, in2Shape, partial=False):
    assert((len(in1.shape) == 2) or (len(in1.shape) == 3))
    assert(len(in2Shape) == 2)
    (fshape, _, _) = convolutionShape(in1, in2Shape, Nnum)
    # Pre-1.9 NumPy FFT routines are not threadsafe - this code requires numpy 1.9 or greater
    assert(_rfft_mt_safe)
    fa = special_rfftn(in1, bb, aa, Nnum, fshape, partial=partial)
    return (fa, fshape)

def special_fftconvolve_part3b(fab, fshape, fslice, s1):
    assert(len(fab.shape) == 2)
    ret = myfft.myIFFT2(fab, fshape)[fslice].copy()    # TODO: what was the purpose of the copy() here? Is it necessary? Perhaps if fslice does something nontrivial, it makes the result compact? But even if so, is that important?
    return _centered(ret, s1)

def special_fftconvolve_part3(fab, fshape, fslice, s1):
    # TODO: This gymnastics is probably unnecessary now I call ifft2 rather than fftn
    if (len(fab.shape) == 2):
        return special_fftconvolve_part3b(fab, fshape, fslice, s1)
    else:
        results = []
        for n in range(fab.shape[0]):
            results.append(special_fftconvolve_part3(fab[n], fshape, fslice, s1))
        return np.array(results)

def special_fftconvolve(in1, bb, aa, Nnum, in2Shape, accum, fb):
    '''
    in1 consists of subapertures of size Nnum x Nnum pixels.
    We are being asked to convolve only pixel (bb,aa) within each subaperture, i.e.
        tempSlice = np.zeros(in1.shape, dtype=in1.dtype)
        tempSlice[bb::Nnum, aa::Nnum] = in1[bb::Nnum, aa::Nnum]
    This allows us to take a significant shortcut in computing the FFT for in1.
    '''
    (fa, _) = special_fftconvolve_part1(in1, bb, aa, Nnum, in2Shape)
    assert(fa.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    assert(fb.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision

    if accum is None:
        accum = fa*fb
    else:
        accum += fa*fb
    assert(accum.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    return accum

################################
# This next function is an experimental work in progress, trying to cover more of the performance-sensitive bits with fast C code
def special_fftconvolve2(in1, bb, aa, Nnum, in2Shape, accum, fb_unmirrored, validWidth, mirrorXMultiplier):
    (fa_partial, fshape) = special_fftconvolve_part1(in1, bb, aa, Nnum, in2Shape, partial=True)
    assert(fa_partial.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    assert(fb_unmirrored.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    expandMultiplier = expand2Multiplier(bb, fshape, fshape)
    return jps.special_fftconvolve(accum, fa_partial, fb_unmirrored, expandMultiplier, validWidth, mirrorXMultiplier)

