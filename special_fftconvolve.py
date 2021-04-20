import numpy as np
import py_light_field as plf
import warnings
import scipy.fftpack
import myfft

from numpy.fft import fft, fftn, rfft, rfftn, irfftn

# We start with various utility functions which are taken from the original source code of fftconvolve

# The mt_safe flag is associated with a comment that "Pre-1.9 NumPy FFT routines are not threadsafe",
# and I have not included the pre-1.9 code to deal with that issue.
# However, it causes me problems (which I don't understand) whereby sometimes I see:
#    ModuleNotFoundError: No module named 'scipy._lib'
# The only solutions on the internet seem to involve trying to uninstall and reinstall scipy!
# Anyway, numpy 1.9 is pretty old and surely this is not something I should have to worry about any more.
# This code isn't even code that I use in my fast implementation.
#from scipy._lib._version import NumpyVersion
#_rfft_mt_safe = (NumpyVersion(np.__version__) >= '1.9.0.dev-e24486e')
_rfft_mt_safe = True

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
    tileFactor = (1,) * (len(reducedF.shape)-1) + (int(Nnum/2+1),)
    result = np.tile(reducedF, tileFactor)
    result = result[...,:int(fshape[-1]/2+1)]
    result *= np.exp(-1j * aa * 2*np.pi / fshape[-1] * np.arange(result.shape[-1],dtype='complex64'))
    if partial:
        return result
    tileFactor = (1,) * (len(result.shape)-2) + (Nnum, 1)
    result = np.tile(result, tileFactor)
    result *= expand2Multiplier(bb, fshape, result.shape)[...,np.newaxis]
    return result

def prime_factors(n):
    # Utility function from the internet
    # This is useful for examining the GPU blocking possibilities for a given array shape
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def betterPrimes(n, Nnum, maxAcceptable = 7):
    # Find a better number slightly larger than n, that is a multiple of Nnum but does not contain other awkward prime factors
    # Note that _next_regular does this up to prime factors of 5.
    # Instead of this (probably inefficient) code, I could re-code that to cope with up to 7.
    result = int(n/Nnum)
    while np.max(prime_factors(result)) > maxAcceptable:
        result += 1
    return result*Nnum

_betterPrimeTable = dict()

def BetterPrimeTable(Nnum):
    if not Nnum in _betterPrimeTable:
        # TODO: I am just assuming an upper limit on this lookup table, i.e. assuming we don't have vast matrices to convolve!
        better = np.arange(1000)*Nnum
        for i in range(2, len(better)):
            better[i] = betterPrimes(better[i], Nnum)
        _betterPrimeTable[Nnum] = better
    return _betterPrimeTable[Nnum]

def convolutionShape(in1Shape, in2Shape, Nnum, padToSmallPrimes):
    # Logic copied from fftconvolve source code
    s1 = np.array(in1Shape)
    s2 = np.array(in2Shape)
    if (len(s1) == 3):   # Cope with case where we are processing multiple reconstructions in parallel
        s1 = s1[1:]
    shape = s1 + s2 - 1
    if False:
        # For reference: this is the original code in fftconvolve, which says:
        # Speed up FFT by padding to optimal size for FFTPACK
        # This doesn't work because I need things to be a multiple of Nnum
        fshape = [_next_regular(int(d)) for d in shape]
    elif padToSmallPrimes == False:
        # Whatever happens, we need to expand up to a multiple of Nnum.
        # That is necessary because the tiling tricks I use in special_fftconvolve etc only work (I think) under that condition.
        # This minimal amount of padding is fastest on the GPU
        fshape = [int(np.ceil(d/float(Nnum)))*Nnum for d in shape]
    else:
        # Pad up so that prime factors are nice small numbers.
        # This improves performance on the CPU
        # We are forced to have a multiple of Nnum, for my tiling tricks,
        # and unfortunately that limits what we can do for Nnum=19.
        # However, we can ensure that all the other prime factors are nice and small.
        # That significantly speeds up the calculation of F(H).
        # In theory it could slow down the convolutions (due to making the arrays larger),
        # but in practice the gain due to the partial FFT we do in there means that the convolutions
        # are actually faster too.
        # There therefore seem to be no downsides to using this code here
        better = BetterPrimeTable(Nnum)
        fshape = [better[int(np.ceil(d/float(Nnum)))] for d in shape]

    fslice = tuple([slice(0, int(sz)) for sz in shape])
    return (fshape, fslice, s1)
    
def special_fftconvolve_part1(in1, bb, aa, Nnum, in2Shape, padToSmallPrimes, partial=False):
    assert((len(in1.shape) == 2) or (len(in1.shape) == 3))
    assert(len(in2Shape) == 2)
    (fshape, _, _) = convolutionShape(in1.shape, in2Shape, Nnum, padToSmallPrimes)
    assert(_rfft_mt_safe)
    fa = special_rfftn(in1, bb, aa, Nnum, fshape, partial=partial)
    return (fa, fshape)

def special_fftconvolve_part3b(fab, fshape, fslice, s1, useCCode=False):
    assert(len(fab.shape) == 2)
    if useCCode:
        ret = plf.InverseRFFT(fab, fshape[0], fshape[1])
    else:
        ret = myfft.myIFFT2(fab, fshape)
    # TODO: what was the purpose of the copy() here? I think I have just copied this from the fftconvolve source code. Perhaps if fslice does something nontrivial, it makes the result compact..? But fslice seems to be the same as fshape for me, here
    return _centered(ret[fslice].copy(), s1)

def special_fftconvolve_part3(fab, fshape, fslice, s1, useCCode=False):
    # TODO: This gymnastics is probably unnecessary - it should be possible to do it all in one go.
    # The complication is that fslice is a 2d slice, whereas fab[n] will probably be a 3D array.
    # That can lead to unpredictable problems (wrong output shapes) unless I take a lot more care than I am doing at the moment!
    if (len(fab.shape) == 2):
        return special_fftconvolve_part3b(fab, fshape, fslice, s1, useCCode)
    else:
        results = []
        for n in range(fab.shape[0]):
            results.append(special_fftconvolve_part3(fab[n], fshape, fslice, s1, useCCode))
        return np.array(results)

def special_fftconvolve(in1, fb, bb, aa, Nnum, in2Shape, accum, padToSmallPrimes=True):
    '''
    in1 consists of subapertures of size Nnum x Nnum pixels.
    We are being asked to convolve only pixel (bb,aa) within each subaperture, i.e.
        tempSlice = np.zeros(in1.shape, dtype=in1.dtype)
        tempSlice[bb::Nnum, aa::Nnum] = in1[bb::Nnum, aa::Nnum]
    This allows us to take a significant shortcut in computing the FFT for in1.
    '''
    (fa, _) = special_fftconvolve_part1(in1, bb, aa, Nnum, in2Shape, padToSmallPrimes)
    assert(fa.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    assert(fb.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision

    if accum is None:
        accum = fa*fb
    else:
        accum += fa*fb
    assert(accum.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    return accum
