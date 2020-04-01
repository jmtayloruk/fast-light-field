import numpy as np
from scipy.signal import fftconvolve
import scipy.fftpack
import time, warnings, os
from joblib import Parallel, delayed
from jutils import tqdm_alias as tqdm

import py_light_field as plf
import special_fftconvolve as special
import psfmatrix
import jutils as util
import myfft

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.fftpack
    gpuAvailable = True
except:
    print('Unable to import cupy - no GPU support will be available')
    gpuAvailable = False


# Ensure existence of the directory we will use to log performance diagnostics
try:
    os.mkdir('perf_diags')
except:
    pass  # Probably the directory already exists

# Global variables that affect GPU kernel block selection (see code below for usage)
gAutoSelectBlockSize = False
gExpandXBlocks = (1, 1, 1)
gCalculateRowsBlocks = (1, 1, 1)
gMirrorXBlocks = (1, 1)
gMirrorYBlocks = (1, 1)
gCalculateRowsTargetBlocks = (1, 8, 8)
gCalculateRowsMaxBlocks = (1, 12, 12)
gPadFactor = 8

gTempFFTOption = 1

# Note: H.shape in python is (<num z planes>, Nnum, Nnum, <psf size>, <psf size>),
#                       e.g. (56, 19, 19, 343, 343)

#########################################################################
# Z Projector class performs the convolution between a given image
# and a (provided) PSF. It operates for a specific ZYX, 
# but does the projection for all symmetries that exist for that PSF
# (i.e. in practice it is likely to project for more than just the one specified ZYX)
#########################################################################        

class ProjectorForZ_base(object):
    # Note: the variable names in this class mostly imply we are doing the back-projection
    # (e.g. Ht, 'projection', etc. However, the same code also does forward-projection!)
    def __init__(self, projection, hMatrix, cc, fftPlan=None):
        # Note: H and Hts are not stored as class variables.
        # I had a lot of trouble with them and multithreading,
        # and eventually settled on having them in shared memory.
        # As I encapsulate more stuff in this class, I could bring them back as class variables...

        self.cpuTime = np.zeros(2)
        self.fftPlan = fftPlan    # Only currently used by GPU version, but our code expects it to be defined (as None) in other cases
        
        # Nnum: number of pixels across a lenslet array (after rectification)
        self.Nnum = hMatrix.Nnum
        
        # This next chunk of logic is copied from the fftconvolve source code.
        # s1, s2: shapes of the input arrays
        # fshape: shape of the (full, possibly padded) result array in Fourier space
        # fslice: slicing tuple specifying the actual result size that should be returned
        self.s1 = np.array(projection.shape[-2:])
        self.s2 = np.array(hMatrix.PSFShape(cc))
        (self.fshape, self.fslice, _) = special.convolutionShape(self.s1, self.s2, hMatrix.Nnum)
        
        # rfslice: slicing tuple to crop down full fft array to the shape that would be output from rfftn
        self.rfshape = (self.fshape[0], int(self.fshape[1]/2)+1)
        self.rfslice = (slice(0,self.fshape[0]), slice(0,int(self.fshape[1]/2)+1))
        # reducedShape: shape of the initial FFT, which we will then tile up to full size using the special_fftconvolve tricks
        self.reducedShape = ()
        for d in self.fshape:
            assert((d % self.Nnum) == 0)
            self.reducedShape = self.reducedShape + (int(d/self.Nnum),)
        # rfshape_xPadded: real FFT array shape but with extra padding to improve row alignment in memory.
        # This padding is crucial for GPU performance, and might even help a little for CPU-based code
        self.rfshape_xPadded = (self.rfshape[0], ((self.rfshape[1] + gPadFactor-1) // gPadFactor) * gPadFactor)

        # Precalculate arrays that are needed as part of the process of converting from FFT(H) to FFT(mirrorImage(H))
        padLength = self.fshape[0] - self.s2[0]
        self.mirrorXMultiplier = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[0]) * np.arange(self.fshape[0])).astype('complex64')
        padLength = self.fshape[1] - self.s2[1]
        self.mirrorYMultiplier = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[1]) * np.arange(self.fshape[1])).astype('complex64')

        # Precalculate various arrays used as lookups by my c code
        expandXMultiplier = np.empty((self.Nnum, self.fshape[-1]), dtype='complex64')
        for a in range(self.Nnum):
            expandXMultiplier[a] = np.exp(-1j * a * 2*np.pi / self.fshape[-1] * np.arange(self.fshape[-1], dtype='complex64'))
        expandYMultiplier = np.empty((self.Nnum, self.fshape[-2]), dtype='complex64')
        for b in range(self.Nnum):
            expandYMultiplier[b] = np.exp(-1j * b * 2*np.pi / self.fshape[-2] * np.arange(self.fshape[-2],dtype='complex64'))
        self.xAxisMultipliers = np.append(self.mirrorYMultiplier[np.newaxis,:], expandXMultiplier, axis=0)
        self.yAxisMultipliers = np.array([expandYMultiplier, expandYMultiplier * self.mirrorXMultiplier])

        return

    def convolve(self, projection, hMatrix, cc, bb, aa, backwards, accum):
        # The main function to be called from external code.
        # Convolves projection with hMatrix, returning a result that is still in Fourier space
        # (caller will accumulate in Fourier space and do the inverse FFT just once at the end)
        cent = int(self.Nnum/2)
        
        mirrorX = (bb != cent)
        mirrorY = (aa != cent)
        transpose = ((aa != bb) and (aa != (self.Nnum-bb-1)))
        
        # TODO: it would speed things up if I could avoid computing the full fft for Hts.
        # However, it's not immediately clear to me how to fill out the full fftn array from rfftn
        # in the case of a 2D transform.
        # For 1D it's the reversed conjugate, but for 2D it's more complicated than that.
        # It's possible that it's actually nontrivial, in spite of the fact that
        # you can get away without it when only computing fft/ifft for real arrays)
        fHtsFull = hMatrix.fH(cc, bb, aa, backwards, False, self.fshape)
        accum = self.convolvePart2(projection,bb,aa,fHtsFull,mirrorY,mirrorX, accum)
        if transpose:
            if (self.fshape[0] == self.fshape[1]):
                # For a square array, the FFT of the transpose is just the transpose of the FFT.
                # The copy() is because my C and GPU code currently can't cope with
                # a transposed array (non-contiguous strides in x)
                fHtsFull = fHtsFull.transpose().copy()
            else:
                # For a non-square array, we have to compute the FFT for the transpose.
                fHtsFull = hMatrix.fH(cc, bb, aa, backwards, True, self.fshape).copy()
            # Note that mx,my have been swapped here, which is necessary following the transpose
            accum = self.convolvePart2(projection,aa,bb,fHtsFull,mirrorX,mirrorY, accum)
        assert(accum.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
        return accum

    def convolvePart2(self, projection, bb, aa, fHtsFull, mirrorY, mirrorX, accum):
        accum = self.convolvePart3(projection,bb,aa,fHtsFull,mirrorX,accum)
        if mirrorY:
            fHtsFull = self.MirrorYArray(fHtsFull)
            accum = self.convolvePart3(projection,bb,self.Nnum-aa-1,fHtsFull,mirrorX,accum)
        return accum

#########################################################################
# Pure python implementation of helper functions
#########################################################################
class ProjectorForZ_python(ProjectorForZ_base):
    def MirrorXArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the X mirror of that same PSF
        fHtsFull = fHtsFull.conj() * self.mirrorXMultiplier[:,np.newaxis]
        fHtsFull[:,1::] = fHtsFull[:,1::][:,::-1]
        return fHtsFull

    def MirrorYArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the Y mirror of that same PSF
        fHtsFull = fHtsFull.conj() * self.mirrorYMultiplier
        fHtsFull[1::] = fHtsFull[1::][::-1]
        return fHtsFull
    
    def convolvePart3(self, projection, bb, aa, fHtsFull, mirrorX, accum):
        cpu0 = util.cpuTime('both')
        accum = special.special_fftconvolve(projection,fHtsFull[self.rfslice],bb,aa,self.Nnum,self.s2,accum)
        if mirrorX:
            fHtsMirror = self.MirrorXArray(fHtsFull)
            accum = special.special_fftconvolve(projection,fHtsMirror[self.rfslice],self.Nnum-bb-1,aa,self.Nnum,self.s2,accum)
        self.cpuTime += util.cpuTime('both')-cpu0
        return accum


#########################################################################
# New, faster code with helper functions implemented in C
#########################################################################
class ProjectorForZ_cHelpers(ProjectorForZ_base):
    def MirrorXArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the X mirror of that same PSF
        return plf.mirrorX(fHtsFull, self.mirrorXMultiplier)

    def MirrorYArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the Y mirror of that same PSF
        return plf.mirrorY(fHtsFull, self.mirrorYMultiplier)
    
    def convolvePart3(self, projection, bb, aa, fHtsFull, mirrorX, accum):
        cpu0 = util.cpuTime('both')
        accum = plf.special_fftconvolve(projection, fHtsFull, bb, aa, self.Nnum, 0, self.xAxisMultipliers, self.yAxisMultipliers, accum)
        if mirrorX:
            accum = plf.special_fftconvolve(projection, fHtsFull, self.Nnum-bb-1, aa, self.Nnum, 1, self.xAxisMultipliers, self.yAxisMultipliers, accum)
        self.cpuTime += util.cpuTime('both')-cpu0
        return accum


#########################################################################
# Even faster variant where the entire convolution is written in C
#########################################################################
class ProjectorForZ_allC(ProjectorForZ_base):
    def convolvePart2(self, projection, bb, aa, fHtsFull, mirrorY, mirrorX, accum):
        return plf.Convolve(projection, fHtsFull, bb, aa, self.Nnum, self.xAxisMultipliers, self.yAxisMultipliers, accum)


#########################################################################
# Helper functions implemented on GPU
#########################################################################
def prime_factors(n):
    # From the internet
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

def CallKernel(kernel, workShape, blockShape, params):
    workShape = np.asarray(workShape)
    blockShape = np.asarray(blockShape)
    gridShape = workShape//blockShape
    # Check that the block shape we were given is an exact factor of the total work shape
    if not np.all(gridShape*blockShape == workShape):
        print('Block shape {0} incompatible with work shape {1}'.format(blockShape, workShape))
        print('Prime factors for work dimensions:')
        for d in workShape:
            print(' {0}: {1}', d, prime_factors(d))
        assert(False)
    # Call through to the kernel
    kernel(tuple(gridShape), tuple(blockShape), params)
    cp.cuda.runtime.deviceSynchronize()

def element_strides(a):
    return np.asarray(a.strides) // a.itemsize

def BestFactorUpTo(n, max):
    for i in range(max, 0, -1):
        if (((n//i)*i) == n):
            return i
    assert(0)   # Should always return from the loop - with i=1 if nothing else!

def BestBlockFactors(jobShape, target, max=None):
    # First try forming getting each axis as close to 'target' as possible
    result = [1] * len(jobShape)
    for i in range(len(jobShape)):
        result[i] = BestFactorUpTo(jobShape[i], target[i])
        if (result[i] == 1) and (target[i] > 1) and (max is not None):
            result[i] = BestFactorUpTo(jobShape[i], max[i])
    return tuple(result)

class ProjectorForZ_gpuHelpers(ProjectorForZ_base):
    def __init__(self, projection, hMatrix, cc, fftPlan=None):
        super().__init__(projection, hMatrix, cc, fftPlan=fftPlan)
        self.xAxisMultipliers = cp.asarray(self.xAxisMultipliers)
        self.yAxisMultipliers = cp.asarray(self.yAxisMultipliers)
        self.mirrorXMultiplier = cp.asarray(self.mirrorXMultiplier)
        self.mirrorYMultiplier = cp.asarray(self.mirrorYMultiplier)
        self.mirrorX_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__ void mirrorX(const complex<float>* x, const complex<float>* mirrorXMultiplier, complex<float>* result)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int i_width = blockDim.x * gridDim.x;
                int j_width = blockDim.y * gridDim.y; 
                // We need to reverse the order of all elements in the final dimension EXCEPT the first element.
                int jDest = j_width - j;  // Will give correct indexing for all except j=0
                if (j == 0)               // ... which we fix here
                  jDest = 0;
                  
                result[i*j_width + jDest] = conj(x[i*j_width + j]) * mirrorXMultiplier[i];
            }
            ''', 'mirrorX')
        self.mirrorY_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__ void mirrorY(const complex<float>* x, const complex<float>* mirrorYMultiplier, complex<float>* result)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int i_width = blockDim.x * gridDim.x;
                int j_width = blockDim.y * gridDim.y; 
                // We need to reverse the order of all elements in the first dimension EXCEPT the first element.
                int iDest = i_width - i;  // Will give correct indexing for all except i=0
                if (i == 0)               // ... which we fix here
                  iDest = 0;
                  
                result[iDest*j_width + j] = conj(x[i*j_width + j]) * mirrorYMultiplier[j];
            }
            ''', 'mirrorY')

        self.expandX_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__
            void expandX(const complex<float>* x, const complex<float>* expandXMultiplier, int smallDimXY, int smallDimX, complex<float>* result)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int k = threadIdx.z + blockIdx.z * blockDim.z;
                int i_width = blockDim.x * gridDim.x;
                int j_width = blockDim.y * gridDim.y; 
                int k_width = blockDim.z * gridDim.z; 

                // Tile the result up to the length that is implied by expandXMultiplier (using that length saves us figuring out the length for ourselves)
                int kReduced = k % smallDimX;
                result[i*j_width*k_width + j*k_width + k] = x[i*smallDimXY + j*smallDimX + kReduced] * expandXMultiplier[k];
            }
            ''', 'expandX')


        self.calculateRows_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__
            void calculateRows(const complex<float>* partialFourierOfProjection, const complex<float>* fHTsFull_unmirrored,
            const complex<float>* yAxisMultipliers, int smallDimY, int smallDimX, int fHtsDimX, complex<float>* accum)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int k = threadIdx.z + blockIdx.z * blockDim.z;
                int i_width = blockDim.x * gridDim.x;
                int j_width = blockDim.y * gridDim.y;
                int k_width = blockDim.z * gridDim.z;
                
                int pos3 = i*j_width*k_width + j*k_width + k;
                int pos3_partial = i*smallDimY*smallDimX + (j%smallDimY)*smallDimX + k;
                int pos2 = j*fHtsDimX + k;
                complex<float> emy = yAxisMultipliers[j];
                accum[pos3] += partialFourierOfProjection[pos3_partial] * fHTsFull_unmirrored[pos2] * emy;
            }
            ''', 'calculateRows')

        self.calculateRowsMirrored_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__
            void calculateRowsMirrored(const complex<float>* partialFourierOfProjection, const complex<float>* fHTsFull_unmirrored,
            const complex<float>* yAxisMultipliers, int smallDimY, int smallDimX, int fHtsDimX, complex<float>* accum)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int k = threadIdx.z + blockIdx.z * blockDim.z;
                int i_width = blockDim.x * gridDim.x;
                int j_width = blockDim.y * gridDim.y;
                int k_width = blockDim.z * gridDim.z;
                
                int pos3 = i*j_width*k_width + j*k_width + k;
                int pos3_partial = i*smallDimY*smallDimX + (j%smallDimY)*smallDimX + k;
                // For accessing fHTs, we need to reverse the order of all elements in the final dimension EXCEPT the first element.
                // ***** TODO: I need to think about whether fHtsDimX is correct here - I want to use width of fHTsFull_unmirrored.
                int k2 = fHtsDimX - k;  // Will give correct indexing for all except j=0
                if (k == 0)            // ... which we fix here
                  k2 = 0;
                int pos2 = j*fHtsDimX + k2;
                complex<float> emy = yAxisMultipliers[j];
                accum[pos3] += partialFourierOfProjection[pos3_partial] * conj(fHTsFull_unmirrored[pos2]) * emy;
            }
            ''', 'calculateRowsMirrored')
        # Set the block sizes we use when calling our custom GPU kernels.
        # To *ensure* we can make that a block size that performs well, we will deliberately over-allocate space in our accumlator array,
        # which then allows us to use a nice round block size even if it overruns the array in the x dimension
        if gAutoSelectBlockSize:
            # Automatically select suitable block sizes, based on the array shapes we will be working with.
            numTimepoints = 1
            if len(projection.shape) >= 3:
                numTimepoints = projection.shape[-3]
            # With these block factors, mirrorY seems to take negligible time (<1s I think)
            self.mirrorXBlocks = BestBlockFactors(self.fshape, target=(1, 15, 15))
            self.mirrorYBlocks = BestBlockFactors(self.fshape, target=(1, 15, 15))
            # With these block factors, expandX seems to take ~9.5s.
            # I have not investigated performance exhaustively, but 15,15 performs better than 1,15 despite the relatively small problem size
            # 15,8 seems to perform about the same as 15,15 (it may be that it often ends up just using 8 as its factor anyway...)
            self.expandXBlocks = BestBlockFactors((numTimepoints, self.reducedShape[-2], self.rfshape_xPadded[-1]), target=(1, 15, 8))
            # With these block factors, special_fftconvolve2_nomirror takes 26s.
            # A while back, I did a fair amount of investigating performance for smallish test scenarios,
            # but I'm not sure whether I have properly tested for the actual scenarios I see in real DeconvRL problem sizes.
            self.calculateRowsBlocks = BestBlockFactors((numTimepoints, self.rfshape[-2], self.rfshape_xPadded[-1]), target=gCalculateRowsTargetBlocks, max=gCalculateRowsMaxBlocks)
        else:
            # Use global variables to rigidly set the block sizes
            # (Note that this will not work well when doing a full projection, where each z plane will have different array shapes.
            #  This branch is only really useful
            self.expandXBlocks = gExpandXBlocks
            self.calculateRowsBlocks = gCalculateRowsBlocks
            self.mirrorXBlocks = gMirrorXBlocks
            self.mirrorYBlocks = gMirrorYBlocks
        
        # Make our FFT plans, unless we have been provided with one that is already suitable (from the previous z plane that was processed)
        if (fftPlan is None) or (fftPlan.shape != self.reducedShape):
            dummy = cp.empty(projection[...,0::self.Nnum,0::self.Nnum].shape, dtype='complex64')
            self.fftPlan = cupyx.scipy.fftpack.get_fft_plan(dummy, shape=self.reducedShape, axes=(1,2))

    def MirrorXArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the X mirror of that same PSF
        result = cp.empty_like(fHtsFull)
        CallKernel(self.mirrorX_kernel, fHtsFull.shape, self.mirrorXBlocks,
                   (fHtsFull, self.mirrorXMultiplier, result))
        return result
    
    def MirrorYArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the Y mirror of that same PSF
        result = cp.empty_like(fHtsFull)
        CallKernel(self.mirrorY_kernel, fHtsFull.shape, self.mirrorYBlocks,
                   (fHtsFull, self.mirrorYMultiplier, result))
        return result

    def special_fftconvolve_py(self, projection, fHtsFull, bb, aa, Nnum, mirrorX, xAxisMultipliers, yAxisMultipliers, accum, earlyExit=0):
        assert(0)   # Should not be calling this, but I am leaving it here for testing purposes
        # First compute the FFT of 'projection'
        subset = projection[...,bb::Nnum,aa::Nnum]
        fftArray = np.fft.fftn(subset, self.reducedShape, axes=(1,2)).astype(np.complex64)

        # Tile the result up to the length that is implied by expandXMultiplier 
        # (using that length saves us figuring out the length for ourselves)
        tileFactor = (1,1,int(Nnum/2+1))
        result = np.tile(fftArray, tileFactor)
        expandXMultiplier = xAxisMultipliers[1+aa]
        result = result[...,:int(self.fshape[-1]/2+1)]
        result = result * expandXMultiplier[:result.shape[-1]]
        if (earlyExit == 1):
            return result
        tileFactor = (1, Nnum, 1)
        result = np.tile(result, tileFactor)
        if (earlyExit == 2):        # Note: with this condition we need to compare against a hacked version of the CUDA kernel that returns just the tiled part, without multipliers
            return result
        assert(mirrorX == 0)    # I don't support this here - fHtsFull needs to be mirrored before passing it in to us
        result *= fHtsFull[self.rfslice] * yAxisMultipliers[mirrorX,bb][...,np.newaxis]
        return accum + result

    def fftn(self, subset):
        # Note that we need to take a copy of our input array because the FFT plans insist on having contiguous arrays
        # We also need to convert it to the complex64 type.
        # My guess would be that that conversion would happen implicitly, internally to fftn, anyway.
        # Hopefully none of this will actually be a performance bottleneck(!)
        fftArray = cupyx.scipy.fftpack.fftn(subset.astype('complex64'), self.reducedShape, axes=(1,2), plan=self.fftPlan)
        cp.cuda.runtime.deviceSynchronize()
        return fftArray
    
    def special_fftconvolve(self, projection, fHtsFull, bb, aa, Nnum, mirrorX, xAxisMultipliers, yAxisMultipliers, accum):
        # First compute the FFT of 'projection'
        fftArray = self.fftn(projection[...,bb::Nnum,aa::Nnum])
        assert(fftArray.dtype == cp.complex64)
        # Now expand it in the horizontal direction
        partialFourierOfProjection = self.special_fftconvolve2_expand(fftArray, aa, xAxisMultipliers)
        # Now expand it in the vertical direction and do the multiplication
        if mirrorX:
            return self.special_fftconvolve2_mirror(partialFourierOfProjection, fHtsFull, bb, aa, Nnum, xAxisMultipliers, yAxisMultipliers, accum)
        else:
            return self.special_fftconvolve2_nomirror(partialFourierOfProjection, fHtsFull, bb, aa, Nnum, xAxisMultipliers, yAxisMultipliers, accum)

    def special_fftconvolve2_expand(self, fftArray, aa, xAxisMultipliers):
        # Expand our fft array in the x direction, making use of a padded array to improve GPU performance
        expandXMultiplier = xAxisMultipliers[1+aa,0:int(self.fshape[-1]/2+1)]
        _partialFourierOfProjection = cp.empty((fftArray.shape[0], fftArray.shape[1], self.rfshape_xPadded[1]), dtype='complex64')
        partialFourierOfProjection = _partialFourierOfProjection[:,:,0:expandXMultiplier.shape[0]]
        CallKernel(self.expandX_kernel, _partialFourierOfProjection.shape, self.expandXBlocks,
                   (fftArray, expandXMultiplier, np.int32(element_strides(fftArray)[0]), np.int32(element_strides(fftArray)[-2]), partialFourierOfProjection))
        return partialFourierOfProjection
    
    def special_fftconvolve2_mirror(self, partialFourierOfProjection, fHtsFull, bb, aa, Nnum, xAxisMultipliers, yAxisMultipliers, accum):
        # Adapt to the padded length of accum
        workShape = (accum.shape[0], accum.shape[1], element_strides(accum)[-2])
        CallKernel(self.calculateRowsMirrored_kernel, workShape, self.calculateRowsBlocks,
                   (partialFourierOfProjection, fHtsFull, yAxisMultipliers[1,bb],
                    np.int32(partialFourierOfProjection.shape[1]), np.int32(element_strides(partialFourierOfProjection)[-2]), np.int32(fHtsFull.shape[1]), accum))
        return accum

    def special_fftconvolve2_nomirror(self, partialFourierOfProjection, fHtsFull, bb, aa, Nnum, xAxisMultipliers, yAxisMultipliers, accum):
        # Adapt to the padded length of accum
        workShape = (accum.shape[0], accum.shape[1], element_strides(accum)[-2])
        CallKernel(self.calculateRows_kernel, workShape, self.calculateRowsBlocks,
                       (partialFourierOfProjection, fHtsFull, yAxisMultipliers[0,bb],
                        np.int32(partialFourierOfProjection.shape[1]), np.int32(element_strides(partialFourierOfProjection)[-2]), np.int32(fHtsFull.shape[1]), accum))
        return accum

    def convolvePart3(self, projection, bb, aa, fHtsFull, mirrorX, accum):
        cpu0 = util.cpuTime('both')
        accum = self.special_fftconvolve(projection, fHtsFull, bb, aa, self.Nnum, 0, self.xAxisMultipliers, self.yAxisMultipliers, accum)
        if mirrorX:
            accum = self.special_fftconvolve(projection, fHtsFull, self.Nnum-bb-1, aa, self.Nnum, 1, self.xAxisMultipliers, self.yAxisMultipliers, accum)
        self.cpuTime += util.cpuTime('both')-cpu0
        return accum


def SanityCheckMatrix(m):
  s = m.itemsize
  for i in range(len(m.shape)-1,0,-1):
    if (m.strides[i] != s):
      print('Problem with dimension {0} (stride {1}, expected {2}). {3}'.format(i, m.strides[i], s, m.strides))
    s *= m.shape[i]

#########################################################################        
# Classes to project for a partial or entire volume
# These are the functions I expect external code to call.
#########################################################################        
class Projector_base(object):
    def __init__(self):
        super().__init__()
        self.fftPlan = None
    
    def asnative(self, m):
        return np.asarray(m)
    
    def asnumpy(self, m):
        return np.asarray(m)
    
    def nativeZeros(self, shape, dtype=float):
        return np.zeros(shape, dtype)


class Projector_allC(Projector_base):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_allC

    def BackwardProjectACC(self, hMatrix, projection, planes, progress, logPrint, numjobs):
        Backprojection = np.zeros((hMatrix.numZ, projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        pos = 0
        results = []
        for cc in progress(planes, 'Backward-project - z', leave=False):
            proj = self.zProjectorClass(projection[0], hMatrix, cc)
            Hcc = hMatrix.Hcc(cc, True)
            fourierZPlane = plf.ProjectForZ(projection, Hcc, hMatrix.Nnum, \
                                             proj.fshape[-2], proj.fshape[-1], \
                                             proj.rfshape[-2], proj.rfshape[-1], \
                                             proj.xAxisMultipliers, proj.yAxisMultipliers)
            # Compute the FFT for each z plane
            Backprojection[cc] = special.special_fftconvolve_part3(fourierZPlane, proj.fshape, proj.fslice, proj.s1, useCCode=True)
        return Backprojection

    def ForwardProjectACC(self, hMatrix, realspace, planes, progress, logPrint, numjobs):
        TOTALprojection = None
        for cc in progress(planes, 'Forward-project - z', leave=False):
            # Project each z plane forward to the camera image plane
            proj = self.zProjectorClass(realspace[0,0], hMatrix, cc)
            Htcc = hMatrix.Hcc(cc, False)
            fourierProjection = plf.ProjectForZ(realspace[cc], Htcc, hMatrix.Nnum, \
                                                 proj.fshape[-2], proj.fshape[-1], \
                                                 proj.rfshape[-2], proj.rfshape[-1], \
                                                 proj.xAxisMultipliers, proj.yAxisMultipliers)
            # Transform back from Fourier space into real space
            # Note that we really do need to do a separate FFT for each plane, because fshape/convolutionShape will be different in each case
            thisProjection = special.special_fftconvolve_part3(fourierProjection, proj.fshape, proj.fslice, proj.s1, useCCode=True)
            if TOTALprojection is None:
                TOTALprojection = thisProjection
            else:
                TOTALprojection += thisProjection

        assert(TOTALprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
        return TOTALprojection


class Projector_pythonSkeleton(Projector_base):
    def BackwardProjectACC(self, hMatrix, projection, planes, progress, logPrint, numjobs):
        # Set up the work to iterate over each z plane
        work = []
        projection = self.asnative(projection)
        for cc in planes:
            for bb in hMatrix.iterableBRange:
                work.append((cc, bb, projection, hMatrix, True))

        # Run the multithreaded work
        results = self.DoMainWork(work, progress, numjobs, desc='Backward-project - z')

        # Gather together and sum the results for each z plane
        fourierZPlanes = [None]*hMatrix.numZ     # This has to be a list because in Fourier space the shapes are different for each z plane
        elapsedTime = 0
        for (result, cc, bb, t) in results:
            elapsedTime += t
            if fourierZPlanes[cc] is None:
                fourierZPlanes[cc] = result
            else:
                fourierZPlanes[cc] += result
    
        return self.asnumpy(self.InverseTransformBackwardProjection(fourierZPlanes, planes, hMatrix, projection))
        
    def InverseTransformBackwardProjection(self, fourierZPlanes, planes, hMatrix, projection):
        # Compute the FFT for each z plane
        Backprojection = np.zeros((len(planes), projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        for cc in planes:
            (fshape, fslice, s1) = special.convolutionShape(projection.shape, hMatrix.PSFShape(cc), hMatrix.Nnum)
            Backprojection[cc] = special.special_fftconvolve_part3(fourierZPlanes[cc], fshape, fslice, s1)
        return Backprojection

    def ForwardProjectACC(self, hMatrix, realspace, planes, progress, logPrint, numjobs):
        # Set up the work to iterate over each z plane
        work = []
        realspace = self.asnative(realspace)
        for cc in planes:
            for bb in hMatrix.iterableBRange:
                work.append((cc, bb, realspace[cc], hMatrix, False))

        # Run the multithreaded work
        results = self.DoMainWork(work, progress, numjobs, desc='Forward-project - z')

        # Gather together and sum all the results
        fourierProjection = [None]*hMatrix.numZ
        elapsedTime = 0
        for (result, cc, bb, t) in results:
            elapsedTime += t
            if fourierProjection[cc] is None:
                fourierProjection[cc] = result
            else:
                fourierProjection[cc] += result

        return self.asnumpy(self.InverseTransformForwardProjection(fourierProjection, planes, hMatrix, realspace))
    
    def InverseTransformForwardProjection(self, fourierProjection, planes, hMatrix, realspace):
        # Compute and accumulate the FFT for each z plane
        TOTALprojection = self.nativeZeros((len(planes), realspace.shape[1], realspace.shape[2], realspace.shape[3]), dtype='float32')
        for cc in planes:
            (fshape, fslice, s1) = special.convolutionShape(realspace[cc].shape, hMatrix.PSFShape(cc), hMatrix.Nnum)
            thisProjection = special.special_fftconvolve_part3(fourierProjection[cc], fshape, fslice, s1)
            TOTALprojection += thisProjection
        return TOTALprojection

    def ProjectForZY(self, cc, bb, source, hMatrix, backwards):
        assert(source.dtype == np.float32)   # Keep an eye out for if we are provided with double-precision inputs
        # This is a perhaps a bit of a hack for now - it ensures FFT(PSF) is calculated on the GPU
        # TODO: I should update this with a lambda (if I can work out how...?) that passes in an FFT plan that we have precomputed
        if self.zProjectorClass is ProjectorForZ_gpuHelpers:
            hMatrix.UpdateFFTFunc(myfft.myFFT2_gpu)
        else:
            hMatrix.UpdateFFTFunc(myfft.myFFT2)
        
        #f = open('perf_diags/%d_%d.txt'%(cc,bb), "w")
        t1 = time.time()
        singleJob = (len(source.shape) == 2)
        if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
            source = source[np.newaxis,:,:]
        projector = self.zProjectorClass(source, hMatrix, cc, self.fftPlan)
        # Cache the fftPlan from this projector, to reuse for the next z plane if possible
        self.fftPlan = projector.fftPlan
        
        # For the result, we actually allocate a larger array with a nice round-number x stride, and then take a 'view' into it that has our desired dimensions
        # This is crucial for good performance on the GPU
        if singleJob:
            _result = self.nativeZeros((1, projector.rfshape[0], projector.rfshape_xPadded[1]), dtype='complex64')
        else:
            _result = self.nativeZeros((source.shape[0], projector.rfshape[0], projector.rfshape_xPadded[1]), dtype='complex64')
        result = _result[:,:,0:projector.rfshape[1]]

        for aa in range(bb,int((hMatrix.Nnum+1)/2)):
            result = projector.convolve(source, hMatrix, cc, bb, aa, backwards, result)
        t2 = time.time()
        assert(result.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
        #f.write('%d\t%f\t%f\t%f\t%f\t%f\n' % (os.getpid(), t1, t2, t2-t1, projector.cpuTime[0], projector.cpuTime[1]))
        #f.close()
        if singleJob:
            return (result[0], cc, bb, t2-t1)
        else:
            return (result, cc, bb, t2-t1)

    def DoMainWork_parallel(self, work, progress, numjobs, desc):
        return Parallel(n_jobs=numjobs) (delayed(self.ProjectForZY)(*args) for args in progress(work, desc=desc, leave=False))

    def DoMainWork_singleThreaded(self, work, progress, desc):
        results = []
        for args in progress(work, desc=desc, leave=False):
            results.append(self.ProjectForZY(*args))
        return results
    
    def DoMainWork(self, work, progress, numjobs, desc):
        # Defaults to parallel, but subclasses may want to override this
        return self.DoMainWork_parallel(work, progress, numjobs, desc)


class Projector_python(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_python


class Projector_cHelpers(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_cHelpers


class Projector_gpuHelpers(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_gpuHelpers

    def asnative(self, m):
        return cp.asarray(m)

    def asnumpy(self, m):
        return cp.asnumpy(m)

    def nativeZeros(self, shape, dtype=float):
        return cp.zeros(shape, dtype)
    
    def DoMainWork(self, work, progress, numjobs, desc):
        # We do not parallelise the python code - the only parallelism will be on the GPU
        return self.DoMainWork_singleThreaded(work, progress, desc)

    def InverseTransformBackwardProjection(self, fourierZPlanes, planes, hMatrix, projection):
        # Compute the FFT for each z plane
        Backprojection = self.nativeZeros((len(fourierZPlanes), projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        for cc in planes:
            (fshape, fslice, s1) = special.convolutionShape(projection.shape, hMatrix.PSFShape(cc), hMatrix.Nnum)
            # This next code is copied from special_fftconvolve_part3
            results = []
            for n in range(fourierZPlanes[cc].shape[0]):
                inv = cp.fft.irfftn(fourierZPlanes[cc][n], fshape)
                # TODO: what was the purpose of the copy() here? I think I have just copied this from the fftconvolve source code. Perhaps if fslice does something nontrivial, it makes the result compact..? But fslice seems to be the same as fshape for me, here
                inv = special._centered(inv[fslice].copy(), s1)
                results.append(inv)
            Backprojection[cc] = cp.array(results)
        return Backprojection

    def InverseTransformForwardProjection(self, fourierProjection, planes, hMatrix, realspace):
        # Compute and accumulate the FFT for each z plane
        TOTALprojection = self.nativeZeros((realspace.shape[1], realspace.shape[2], realspace.shape[3]), dtype='float32')
        for cc in planes:
            (fshape, fslice, s1) = special.convolutionShape(realspace[cc].shape, hMatrix.PSFShape(cc), hMatrix.Nnum)
            # This next code is copied from special_fftconvolve_part3
            for n in range(fourierProjection[cc].shape[0]):
                inv = cp.fft.irfftn(fourierProjection[cc][n], fshape)
                # TODO: what was the purpose of the copy() here? I think I have just copied this from the fftconvolve source code. Perhaps if fslice does something nontrivial, it makes the result compact..? But fslice seems to be the same as fshape for me, here
                inv = special._centered(inv[fslice].copy(), s1)
                TOTALprojection[n] += inv
        return TOTALprojection


#########################################################################
# Older, simple versions of backprojection code, for reference.
# I don't expect these to be used except for internal testing within this module.
# Note: it is a bit of an anomaly that some of these are here, but forwardProjectACC etc are in lfdeconv...
#########################################################################

def ProjectForZ(hMatrix, backwards, cc, source, projectorClass=Projector_allC, progress=tqdm):
    # This is somewhat obsolete now, I think, but I will leave it for now
    result = None
    projector = projectorClass()
    for bb in progress(hMatrix.iterableBRange, leave=False, desc='Project - y'):
        (thisResult, _, _, _) = projector.ProjectForZY(cc, bb, source, hMatrix, backwards)
        if (result is None):
            result = thisResult
        else:
            result += thisResult
    # Actually, for forward projection we don't need to do this separately for every z,
    # but it's easier to do it for symmetry (and this function is not used in performance-critical code anyway)
    (fshape, fslice, s1) = special.convolutionShape(source.shape, hMatrix.PSFShape(cc), hMatrix.Nnum)
    return special.special_fftconvolve_part3(result, fshape, fslice, s1)

        
def ForwardProjectForZ_old(HCC, realspaceCC):
    singleJob = (len(realspaceCC.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        realspaceCC = realspaceCC[np.newaxis,:,:]
    # Iterate over each lenslet pixel
    Nnum = HCC.shape[1]
    TOTALprojection = np.zeros(realspaceCC.shape, dtype='float32')
    for bb in tqdm(range(Nnum), leave=False, desc='Forward-project - y'):
        for aa in tqdm(range(Nnum), leave=False, desc='Forward-project - x'):
            # Extract the part of H that represents this lenslet pixel
            Hs = HCC[bb, aa]
            for n in range(realspaceCC.shape[0]):
                # Create a workspace representing just the voxels cc,bb,aa behind each lenslet (the rest is 0)
                tempspace = np.zeros((realspaceCC[n].shape[0], realspaceCC[n].shape[1]), dtype='float32');
                tempspace[bb::Nnum, aa::Nnum] = realspaceCC[n, bb::Nnum, aa::Nnum]
                # Compute how those voxels project onto the sensor, and accumulate
                TOTALprojection[n] += fftconvolve(tempspace, Hs, 'same')
    if singleJob:
        return TOTALprojection[0]
    else:
        return TOTALprojection
    
def BackwardProjectForZ_old(HtCC, projection, progress=tqdm):
    singleJob = (len(projection.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        projection = projection[np.newaxis,:,:]
    # Iterate over each lenslet pixel
    Nnum = HtCC.shape[1]
    tempSliceBack = np.zeros(projection.shape, dtype='float32')        
    for aa in progress(range(Nnum), leave=False, desc='y'):
        for bb in range(Nnum):
            # Extract the part of Ht that represents this lenslet pixel
            Hts = HtCC[bb, aa]
            for n in range(projection.shape[0]):
                # Create a workspace representing just the voxels cc,bb,aa behind each lenslet (the rest is 0)
                tempSlice = np.zeros(projection[n].shape, dtype='float32')
                tempSlice[bb::Nnum, aa::Nnum] = projection[n, bb::Nnum, aa::Nnum]
                # Compute how those voxels back-project from the sensor
                tempSliceBack[n] += fftconvolve(tempSlice, Hts, 'same')
    if singleJob:
        return tempSliceBack[0]
    else:
        return tempSliceBack

def BackwardProjectACC_old(Ht, projection, CAindex, progress=tqdm, planes=None):
    backprojection = np.zeros((Ht.shape[0], projection.shape[0], projection.shape[1]), dtype='float32')
    # Iterate over each z plane
    if planes is None:
        planes = range(Ht.shape[0])
    for cc in progress(planes, desc='Back-project - z'):
        HtCC =  Ht[cc, :, :, CAindex[0,cc]-1:CAindex[1,cc], CAindex[0,cc]-1:CAindex[1,cc]]
        backprojection[cc] = BackwardProjectForZ_old(HtCC, projection, progress=progress)

    return backprojection

#########################################################################
# Self-test code: test the backprojection code against a slower definitive version
#########################################################################
def selfTest():
    # The strictest self-test would be against the original (very simple) code I wrote,
    # but that is very slow and requires us to load in the H matrices in a slow manner.
    # As a result, normally I would be satisfied to test against the newer, but still
    # pure-python implementation I have written (and which is stable code).
    testAgainstOriginalCode = False
    if testAgainstOriginalCode:
        print("Testing backprojection code against original code")
    else:
        print("Testing backprojection code against pure-python code")
    
    # Load the H matrix
    # We need the raw _H and _Ht values, since we are using old projection code to validate the results of my new optimized code
    matPath = 'PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.0.mat'
    if testAgainstOriginalCode:
        (_H, _Ht, _CAIndex, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = psfmatrix.LoadRawMatrixData(matPath)
        testHCC = _H[13]
        testHtCC = _Ht[13]

    # Test forward and back projection
    classesToTest = [Projector_allC, Projector_python, Projector_cHelpers]
    if gpuAvailable:
        classesToTest = [Projector_gpuHelpers]   #  classesToTest +    # TODO: eventually will want to prepend classesToTest
    for projectorClass in classesToTest:
        print(' Testing class:', projectorClass.__name__)
        for bk in [True, False]:
            print(' === bk', bk)
            # Test both square and non-square, since they use different code
            for shape in [(150,150), (150,300), (300,150)]:
                print(' === shape', shape)
                testHMatrix = psfmatrix.LoadMatrix(matPath, numZ=1, zStart=13)   # Needs to be in the loop here, because caching is confused by changing the image shape
                testProjection = np.random.random(shape).astype(np.float32)
                # Start by running old, definitive code that we trust
                if testAgainstOriginalCode:
                    if bk:
                        testResultOld = BackwardProjectForZ_old(testHtCC, testProjection)
                    else:
                        testResultOld = ForwardProjectForZ_old(testHCC, testProjection)
                else:
                    t1 = time.time()
                    testResultOld = ProjectForZ(testHMatrix, bk, 0, testProjection, Projector_python, progress=util.noProgressBar)
                    t2 = time.time()
                    print('Old took %.2fms'%((t2-t1)*1e3))

                # Now run the code we are actually testing.
                # Note that we call xxxProjectACC rather than ProjectForZ,
                # because the pure-C implementation does not have a ProjectForZ function.
                t1 = time.time()
                if bk:
                    testResultNew = projectorClass().BackwardProjectACC(testHMatrix, testProjection[np.newaxis,:,:], [0], progress=util.noProgressBar, logPrint=False, numjobs=1)
                else:
                    testResultNew = projectorClass().ForwardProjectACC(testHMatrix, testProjection[np.newaxis,np.newaxis,:,:], [0], progress=util.noProgressBar, logPrint=False, numjobs=1)
                t2 = time.time()
                print('New took %.2fms'%((t2-t1)*1e3))
                # Compare the results that we got
                comparison = np.max(np.abs(testResultOld - testResultNew))
                print('  test result (should be <<1): %e' % comparison)
                if (comparison > 1e-4):
                    print("   -> WARNING: disagreement detected")
                else:
                    print("   -> OK")
            
    print('Tests complete')

if __name__ == "__main__":
    selfTest()
