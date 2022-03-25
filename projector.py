import numpy as np
from scipy.signal import fftconvolve
import scipy.fftpack
import time, warnings, os
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
    import pycuda.driver as cuda
    gpuAvailable = True
except ImportError:
    #print('Unable to import cupy - no GPU support will be available')
    gpuAvailable = False


# Ensure existence of the directory we will use to log performance diagnostics
try:
    os.mkdir('perf_diags')
except FileExistsError:
    pass

#########################################################################
# Global variables that affect GPU kernel block selection (see code below for usage)
#########################################################################
gBlockSelection = 'measure'
gBlockSizeCache = dict()
# For hard-coded values
gExpandXBlocks = (1, 1, 1)
gCalculateRowsBlocks = (1, 1, 1)
gMirrorXBlocks = (1, 1)
gMirrorYBlocks = (1, 1)
# For auto-selected values (these are probably reasonably adequate)
gExpandXTargetBlocks = (1, 15, 8)
gCalculateRowsTargetBlocks = (3, 8, 8)
gCalculateRowsMaxBlocks = (3, 13, 13)

gPadFactor = 8
# Flag controlling a few debug checks.
# This has a slight performance penalty, for making a few self-consistency checks
gDebugChecks = False
# Flag controlling whether we call cp.cuda.runtime.deviceSynchronize()
# I am not sure if it is safe to turn this off (I am not completely sure how smart cupy is at making sure we wait before starting a dependent custom kernel)
# I think it is ok, though, and it does speed things up noticably!
# Presumably this is because it is able to continue running my python glue code while the GPU is busy, and so I am able to hide that CPU execution time
# Note that turning this off will make python (CPU) code profiler reports much harder to interpret
gSynchronizeAfterKernelCalls = False

# Note that padding fShape to small prime factors *degrades* performance on the GPU.
# Presumably either the FFT algorithm handles larger factors well, or the penalty of larger arrays
# is too much to make the padding worthwhile
padToSmallPrimesOnGPU = False
# Debug flag to enable verbose printing of GPU memory usage at key points during the running of the code
logGPUMemoryUsage = False

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
    # (e.g. 'Hts', 'projection', etc. However, the same code also does forward-projection!)
    def __init__(self, projection, hMatrix, cc, fftPlan=None, fftPlan2=None, padToSmallPrimes=True):
        assert(len(projection.shape) == 3)
        self.cpuTime = np.zeros(2)
        self.fftPlan = fftPlan       # Only currently used by GPU version, but our code
        self.fftPlan2 = fftPlan2     # expects it to be defined (as None) in other cases
        self.initialisedForCC = cc
        self.padToSmallPrimes = padToSmallPrimes
        
        # Nnum: number of pixels across a lenslet array (after rectification)
        self.Nnum = hMatrix.Nnum
        
        # This next chunk of logic is copied from the fftconvolve source code.
        # s1, s2: shapes of the input arrays
        # fshape: shape of the (full, possibly padded) result array in Fourier space
        # fslice: slicing tuple specifying the actual result size that should be returned
        self.s1 = np.array(projection.shape[-2:])
        self.s2 = np.array(hMatrix.PSFShape(cc))
        (self.fshape, self.fslice, _) = special.convolutionShape(self.s1, self.s2, self.Nnum, self.padToSmallPrimes)
        
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

    def nativeZeros(self, shape, dtype=float):
        return np.zeros(shape, dtype)

    def GetFH(self, hMatrix, cc, bb, aa, backwards, transpose):
        return hMatrix.fH(cc, bb, aa, backwards, transpose, self.fshape)

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
        fHtsFull = self.GetFH(hMatrix, cc, bb, aa, backwards, False)
        accum = self.convolvePart2(projection,bb,aa,fHtsFull,mirrorY,mirrorX, accum)
        if transpose:
            fHtsFull = self.GetFH(hMatrix, cc, bb, aa, backwards, True)
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

    def ProjectForZY(self, cc, bb, source, hMatrix, backwards):
        assert(source.dtype == np.float32)   # Keep an eye out for if we are provided with double-precision inputs
        #f = open('perf_diags/%d_%d.txt'%(cc,bb), "w")
        t1 = time.time()
        singleJob = (len(source.shape) == 2)
        if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
            source = source[np.newaxis,:,:]
        
        # For the result, we actually allocate a larger array with a nice round-number x stride, and then take a 'view' into it that has our desired dimensions
        # This is crucial for good performance on the GPU
        if singleJob:
            _result = self.nativeZeros((1, self.rfshape[0], self.rfshape_xPadded[1]), dtype='complex64')
        else:
            _result = self.nativeZeros((source.shape[0], self.rfshape[0], self.rfshape_xPadded[1]), dtype='complex64')
        result = _result[:,:,0:self.rfshape[1]]

        for aa in range(bb,int((hMatrix.Nnum+1)/2)):
            result = self.convolve(source, hMatrix, cc, bb, aa, backwards, result)
        t2 = time.time()
        assert(result.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
        #f.write('%d\t%f\t%f\t%f\t%f\t%f\n' % (os.getpid(), t1, t2, t2-t1, self.cpuTime[0], self.cpuTime[1]))
        #f.close()
        if singleJob:
            return result[0]
        else:
            return result

    def ProjectForZ(self, cc, source, hMatrix, backwards):
        assert(self.initialisedForCC == cc)
        result = None
        for bb in hMatrix.iterableBRange:
            r = self.ProjectForZY(cc, bb, source, hMatrix, backwards)
            if result is None:    # TODO: may want to tidy up to eliminate the need for this logic
                result = r
            else:
                result += r
        return result

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
    if gDebugChecks:
        if not np.all(gridShape*blockShape == workShape):
            print('Block shape {0} incompatible with work shape {1}'.format(blockShape, workShape))
            print('Prime factors for work dimensions:')
            for d in workShape:
                print(' {0}: {1}', d, prime_factors(d))
            assert(False)
    # Call through to the kernel
    kernel(tuple(gridShape), tuple(blockShape), params)
    if gSynchronizeAfterKernelCalls:
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
    def __init__(self, projection, hMatrix, cc, fftPlan=None, fftPlan2=None):
        super().__init__(projection, hMatrix, cc, fftPlan=fftPlan, fftPlan2=fftPlan2, padToSmallPrimes=padToSmallPrimesOnGPU)
        assert(len(projection.shape) == 3)
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
        if gBlockSelection == 'fixed':
            # Use global variables to rigidly set the block sizes
            # (Note that this will not work well when doing a full projection, where each z plane will have different array shapes.
            #  This mode is only really useful when running test cases where we have exact control over the inputs)
            self.mirrorXBlocks = gMirrorXBlocks
            self.mirrorYBlocks = gMirrorYBlocks
            self.expandXBlocks = gExpandXBlocks
            self.calculateRowsBlocks = gCalculateRowsBlocks
        elif gBlockSelection == 'auto':
            # Automatically select suitable block sizes, based on the array shapes we will be working with.
            # These block factors seem to perform reasonably well, but ideally I would write some code that would auto-calibrate
            # and work out the best block factors in each scenario. It might well be that this can be improved on.
            # Each individual kernel call takes ~1ms, so I could try quite a lot of different block factors in a manageable amount of time.
            numTimepoints = projection.shape[0]
            self.mirrorXBlocks = BestBlockFactors(self.fshape, target=(1, 15, 15))
            self.mirrorYBlocks = BestBlockFactors(self.fshape, target=(1, 15, 15))
            self.expandXBlocks = BestBlockFactors((numTimepoints, self.reducedShape[-2], self.rfshape_xPadded[-1]), target=gExpandXTargetBlocks)
            self.calculateRowsBlocks = BestBlockFactors((numTimepoints, self.rfshape[-2], self.rfshape_xPadded[-1]), target=gCalculateRowsTargetBlocks, max=gCalculateRowsMaxBlocks)
        elif gBlockSelection == 'measure':
            mirrorYWorkShape = (1, self.fshape[0], self.fshape[1])
            expandWorkShape = (projection.shape[0],self.reducedShape[0],self.rfshape_xPadded[1])
            calculateRowsWorkShape = (projection.shape[0],self.rfshape[0],self.rfshape_xPadded[1])
            cacheKey = str(mirrorYWorkShape) + str(expandWorkShape) + str(calculateRowsWorkShape)
            if cacheKey in gBlockSizeCache:
                (self.mirrorYBlocks, self.expandXBlocks, self.calculateRowsBlocks) = gBlockSizeCache[cacheKey]
            else:
                #print('Finding best block factors for Expand, on shape {0}'.format(expandWorkShape))
                testExpand = lambda dummyLoad,blocks: self.special_fftconvolve2_expand(dummyLoad, 0, blocks=blocks)
                self.expandXBlocks = self.CalibrateBlockFactors(testExpand, expandWorkShape)
                #print('Finding best block factors for CalculateRows, on shape {0}'.format(calculateRowsWorkShape))
                fHtsFull = cp.zeros(self.fshape)
                accum = cp.zeros((projection.shape[0],self.rfshape_xPadded[0],self.rfshape_xPadded[1]))
                testRows = lambda dummyLoad,blocks: self.special_fftconvolve2_nomirror(dummyLoad, fHtsFull, 0, 0, self.Nnum, accum, blocks=blocks)
                self.calculateRowsBlocks = self.CalibrateBlockFactors(testRows, calculateRowsWorkShape)
                #print('Finding best block factors for MirrorY, on shape {0}'.format(mirrorYWorkShape))
                testMirror = lambda dummyLoad,blocks: self.MirrorYArray(dummyLoad, blocks=blocks)
                self.mirrorYBlocks = self.CalibrateBlockFactors(testMirror, mirrorYWorkShape)
                self.mirrorYBlocks = (self.mirrorYBlocks[1], self.mirrorYBlocks[2])   # Actually it needs to be just a 2D block tuple
                gBlockSizeCache[cacheKey] = (self.mirrorYBlocks, self.expandXBlocks, self.calculateRowsBlocks)
            # TODO: this is not currently measured (but then I don't actually use mirrorX live, anyway!)
            self.mirrorXBlocks = BestBlockFactors(self.fshape, target=(1, 15, 15))
        else:
            assert(0)  # Invalid value for gBlockSelection
        
        # Make our FFT plans, unless we have been provided with one that is already suitable (from the previous z plane that was processed)
        self.batchFFTShape = (projection.shape[0], self.Nnum, self.Nnum, projection.shape[-2]//self.Nnum, projection.shape[-1]//self.Nnum)
        if (fftPlan is None) or (fftPlan.shape != self.reducedShape):
            dummy = cp.empty(self.batchFFTShape, dtype='complex64')
            self.fftPlan = cupyx.scipy.fftpack.get_fft_plan(dummy, shape=self.reducedShape, axes=(3,4))
        halfWidth = (self.Nnum+1)//2
        self.batchFHShape = (halfWidth,halfWidth,self.s2[0],self.s2[1])
        if (fftPlan2 is None) or (fftPlan2.shape != self.fshape):
            dummy = cp.empty(self.batchFHShape, dtype='complex64')
            # Note: I looked into doing the FFTs in larger batches (e.g. for a whole row of aa),
            # but it didn't improve overall performance much, if at all, and it requires more memory to
            # store the results, as well as making other memory-use optimisations harder to code.
            self.fftPlan2 = cupyx.scipy.fftpack.get_fft_plan(dummy[0,0], shape=self.fshape, axes=(0,1))

    def nativeZeros(self, shape, dtype=float):
        return cp.zeros(shape, dtype)
        
    def MirrorXArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the X mirror of that same PSF
        result = cp.empty_like(fHtsFull)
        CallKernel(self.mirrorX_kernel, fHtsFull.shape, self.mirrorXBlocks,
                   (fHtsFull, self.mirrorXMultiplier, result))
        return result
    
    def MirrorYArray(self, fHtsFull, blocks=None):
        # Utility function to convert the FFT of a PSF to the FFT of the Y mirror of that same PSF
        if blocks is None:
            blocks = self.mirrorYBlocks
        result = cp.empty_like(fHtsFull)
        CallKernel(self.mirrorY_kernel, fHtsFull.shape, blocks, (fHtsFull, self.mirrorYMultiplier, result))
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

    def special_fftconvolve(self, projection, fHtsFull, bb, aa, Nnum, mirrorX, accum):
        # First compute the FFT of 'projection'
        fftArray = self.precalculatedFFTArray[:,bb,aa,:,:]
        assert(fftArray.dtype == cp.complex64)
        # Now expand it in the horizontal direction
        partialFourierOfProjection = self.special_fftconvolve2_expand(fftArray, aa)
        # Now expand it in the vertical direction and do the multiplication
        if mirrorX:
            return self.special_fftconvolve2_mirror(partialFourierOfProjection, fHtsFull, bb, aa, Nnum, accum)
        else:
            return self.special_fftconvolve2_nomirror(partialFourierOfProjection, fHtsFull, bb, aa, Nnum, accum)

    def special_fftconvolve2_expand(self, fftArray, aa, blocks=None):
        # Expand our fft array in the x direction, making use of a padded array to improve GPU performance
        if blocks is None:
            blocks = self.expandXBlocks
        expandXMultiplier = self.xAxisMultipliers[1+aa,0:int(self.fshape[-1]/2+1)]
        _partialFourierOfProjection = cp.empty((fftArray.shape[0], fftArray.shape[1], self.rfshape_xPadded[1]), dtype='complex64')
        partialFourierOfProjection = _partialFourierOfProjection[:,:,0:expandXMultiplier.shape[0]]
        CallKernel(self.expandX_kernel, _partialFourierOfProjection.shape, blocks,
                   (fftArray, expandXMultiplier, np.int32(element_strides(fftArray)[0]), np.int32(element_strides(fftArray)[-2]), partialFourierOfProjection))
        return partialFourierOfProjection
    
    def CalibrateBlockFactors(self, kernelCall, workShape):
        maxThreadsPerBlock = cuda.Device(0).get_attributes()[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        numRepeats = 8  # How many times do we call the kernel (taking the average run time)
        def ExactDivisors(a, b):
          a = np.array(a)
          b = np.array(b)
          return np.all((b//a)*a == b)
        if ((workShape[2]%8) == 0):
            xSearchRange = range(8, np.minimum(workShape[2], 64)+1, 8)   # Non-multiples of 8 are almost definitely going to perform less well
        else:
            xSearchRange = range(1, np.minimum(workShape[2], 32)+1, 1)
        bestTime = 1
        bestShape = None
        numTries = 0
        tStart = time.time()
        for z in range(1, np.minimum(workShape[0], 9)+1):     # Larger z batch sizes do not seem to get picked, but we could be patient and still try z>8
            for y in range(1, workShape[1]+1):
                for x in xSearchRange:
                    thisBlockShape = (z,y,x)
                    if (ExactDivisors(thisBlockShape, workShape) and (z*y*x <= maxThreadsPerBlock)):
                      numTries += 1
                      minTime = 1
                      maxTime = 0
                      totalTime = 0
                      for n in range(numRepeats):
                        dummyWork = cp.empty(workShape)
                        t1 = time.time()
                        kernelCall(dummyWork, thisBlockShape)
                        if not gSynchronizeAfterKernelCalls:
                            # We would not normally synchronize as part of the kernel call, but we need to do that here
                            # so that we can have a meaningful measure of how long the kernel call took to run to completion.
                            # I still don't fully understand if/when deviceSynchronize is needed, and it's possible that calling this
                            # does not give the most "realistic" measure of run time. But it's the best option I know of, for now.
                            cp.cuda.runtime.deviceSynchronize()
                        t = time.time()-t1
                        minTime = np.minimum(minTime, t)
                        maxTime = np.maximum(maxTime, t)
                        totalTime += t
                        if (n == 0) and (t > bestTime*1.5):
                          totalTime *= numRepeats
                          break
                      #print('Try {0}, took {1:.2f} ({2:.2f}-{3:.2f})'.format(thisBlockShape, totalTime/numRepeats*1e3, minTime*1e3, maxTime*1e3))
                      if (totalTime/numRepeats < bestTime):
                        bestTime = totalTime/numRepeats
                        bestShape = thisBlockShape
        tEnd = time.time()
        #print('Best shape {0}, took {1:.2f}ms. Tried {2} in {3:.2f}s'.format(bestShape, bestTime*1e3, numTries, tEnd-tStart))
        return bestShape
    
    def special_fftconvolve2_mirror(self, partialFourierOfProjection, fHtsFull, bb, aa, Nnum, accum, blocks=None):
        if blocks is None:
            blocks = self.calculateRowsBlocks
        workShape = (accum.shape[0], accum.shape[1], element_strides(accum)[-2])        # Adapt to the padded length of accum
        CallKernel(self.calculateRowsMirrored_kernel, workShape, blocks,
                   (partialFourierOfProjection, fHtsFull, self.yAxisMultipliers[1,bb],
                    np.int32(partialFourierOfProjection.shape[1]), np.int32(element_strides(partialFourierOfProjection)[-2]), np.int32(fHtsFull.shape[1]), accum))
        return accum

    def special_fftconvolve2_nomirror(self, partialFourierOfProjection, fHtsFull, bb, aa, Nnum, accum, blocks=None):
        if blocks is None:
            blocks = self.calculateRowsBlocks
        workShape = (accum.shape[0], accum.shape[1], element_strides(accum)[-2])        # Adapt to the padded length of accum
        CallKernel(self.calculateRows_kernel, workShape, blocks,
                       (partialFourierOfProjection, fHtsFull, self.yAxisMultipliers[0,bb],
                        np.int32(partialFourierOfProjection.shape[1]), np.int32(element_strides(partialFourierOfProjection)[-2]), np.int32(fHtsFull.shape[1]), accum))
        return accum

    def convolvePart3(self, projection, bb, aa, fHtsFull, mirrorX, accum):
        cpu0 = util.cpuTime('both')
        accum = self.special_fftconvolve(projection, fHtsFull, bb, aa, self.Nnum, 0, accum)
        if mirrorX:
            accum = self.special_fftconvolve(projection, fHtsFull, self.Nnum-bb-1, aa, self.Nnum, 1, accum)
        self.cpuTime += util.cpuTime('both')-cpu0
        return accum

    def GetFH(self, hMatrix, cc, bb, aa, backwards, transpose):
        if transpose:
            # TODO: I haven't thought through this scenario properly yet.
            # I would have expected to need to actually perform a transpose,
            # but in the 'else' branch below I seem to get away without doing that...?
            # TODO: note also that we are not caching the FFT result here, but the caller should be able to do that.
            # I should take a big picture view on whether I need to do the actual transposing,
            # and then probably update the calling code throughout this source file
            aa,bb = bb,aa
        H = cp.asarray(hMatrix.Hcc(cc, backwards)[bb, aa])
        fHtsFull = cupyx.scipy.fftpack.fftn(H, self.fshape, axes=(0,1), plan=self.fftPlan2)
        return fHtsFull

    def PrecalculateFFTArray(self, cc, source):
        # Rejig 'projection' into a shape we can work with more easily
        # source.strides[2] should be contiguous - the x spacing in the original iamge
        # source.strides[1] is the y spacing in the original image
        # source.strides[0] is the spacing between timepoints in the original image
        # newStrides[4] should be an interval of self.Nnum elements
        # newStrides[3] should be an interval of self.Nnum rows
        newStrides = (source.strides[0], source.strides[1], source.strides[2], source.strides[1]*self.Nnum, source.strides[2]*self.Nnum)
        # Read out the strides, and make a copy to force creation of a new array with contiguous storage
        tempRejig = cp.lib.stride_tricks.as_strided(source, self.batchFFTShape, newStrides).astype('complex64', order='C')
        # Calculate all the FFTs in one big batch
        self.precalculatedFFTArray = cupyx.scipy.fftpack.fftn(tempRejig, self.reducedShape, axes=(3,4), plan=self.fftPlan)
        if gSynchronizeAfterKernelCalls:
            cp.cuda.runtime.deviceSynchronize()

    def ProjectForZ(self, cc, source, hMatrix, backwards):
        self.PrecalculateFFTArray(cc, source)
        # Now do the z projection
        return super().ProjectForZ(cc, source, hMatrix, backwards)


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
        self.fftPlan2 = None
        self.cacheFH = False  # Default - can be updated subsequently by the caller
    
    def asnative(self, m):
        return np.asarray(m)
    
    def asnumpy(self, m):
        return np.asarray(m)
    
    def nativeZeros(self, shape, dtype=float):
        return np.zeros(shape, dtype)
    
    def nativeDivide(self, a, b, out):
        return np.divide(a, b, out=out)

def CropAfterFFT(realZPlane, proj):
    result = []
    for n in range(realZPlane.shape[0]):
        result.append(special._centered(realZPlane[n][proj.fslice], proj.s1))
    return np.array(result)

class Projector_allC(Projector_base):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_allC
        self.name = 'Pure C'
        # Patient planning does not actually lead to a noticable overall improvement, and takes >5 minutes to plan!
        #plf.SetPlanningMode(32)   #patient

    def BackwardProjectACC(self, hMatrix, projection, planes, progress, logPrint, numjobs, keepNative=False, rlUpdateFunc=None):
        Backprojection = np.zeros((hMatrix.numZ, projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        pos = 0
        planeWork = []
        plf.SetNumThreadsToUse(numjobs)
        for cc in planes:
            proj = self.zProjectorClass(projection, hMatrix, cc)
            planeWork.append((projection, hMatrix.Hcc(cc, True), hMatrix.Nnum, 1, proj.fshape[-2], proj.fshape[-1], proj.rfshape[-2], proj.rfshape[-1], proj.xAxisMultipliers, proj.yAxisMultipliers))
        if (self.cacheFH):
            cacheIdentifierToUse = hMatrix.HPathFormat
            plf.EnableFHCachingWithIdentifier(cacheIdentifierToUse)
        else:
            cacheIdentifierToUse = None
            plf.DisableFHCaching()
        fourierZPlanes = plf.ProjectForZList(planeWork, cacheIdentifierToUse)
        if False:
            for cc in planes:
                # Compute the iFFT for each z plane
                proj = self.zProjectorClass(projection, hMatrix, cc)
                Backprojection[cc] = special.special_fftconvolve_part3(fourierZPlanes[cc], proj.fshape, proj.fslice, proj.s1, useCCode=True)
        else:
            inverseWork = []
            for cc in planes:
                proj = self.zProjectorClass(projection, hMatrix, cc)
                inverseWork.append((fourierZPlanes[cc], proj.fshape[-2], proj.fshape[-1]))
            realZPlanes = plf.InverseRFFTList(inverseWork)
            for cc in planes:
                Backprojection[cc] = CropAfterFFT(realZPlanes[cc], self.zProjectorClass(projection, hMatrix, cc))
        if rlUpdateFunc is not None:
            rlUpdateFunc(Backprojection, slice(None))
            return None
        else:
            return Backprojection
    
    def ForwardProjectACC(self, hMatrix, realspace, planes, progress, logPrint, numjobs, keepNative=False):
        TOTALprojection = None
        planeWork = []
        plf.SetNumThreadsToUse(numjobs)
        for cc in planes:
            # Project each z plane forward to the camera image plane
            proj = self.zProjectorClass(realspace[0], hMatrix, cc)
            planeWork.append((realspace[cc], hMatrix.Hcc(cc, False), hMatrix.Nnum, 0, proj.fshape[-2], proj.fshape[-1], proj.rfshape[-2], proj.rfshape[-1], proj.xAxisMultipliers, proj.yAxisMultipliers))
        if (self.cacheFH):
            cacheIdentifierToUse = hMatrix.HPathFormat
            plf.EnableFHCachingWithIdentifier(cacheIdentifierToUse)
        else:
            cacheIdentifierToUse = None
            plf.DisableFHCaching()
        fourierProjections = plf.ProjectForZList(planeWork, cacheIdentifierToUse)
        if False:
            for cc in planes:
                # Transform back from Fourier space into real space
                # Note that we really do need to do a separate FFT for each plane, because fshape/convolutionShape will be different in each case
                proj = self.zProjectorClass(realspace[0], hMatrix, cc)
                thisProjection = special.special_fftconvolve_part3(fourierProjections[cc], proj.fshape, proj.fslice, proj.s1, useCCode=True)
                if TOTALprojection is None:
                    TOTALprojection = thisProjection
                else:
                    TOTALprojection += thisProjection
        else:
            inverseWork = []
            for cc in planes:
                proj = self.zProjectorClass(realspace[0], hMatrix, cc)
                inverseWork.append((fourierProjections[cc], proj.fshape[-2], proj.fshape[-1]))
            thisProjection = plf.InverseRFFTList(inverseWork)
            for cc in planes:
                crop = CropAfterFFT(thisProjection[cc], self.zProjectorClass(realspace[0], hMatrix, cc))
                if TOTALprojection is None:
                    TOTALprojection = crop
                else:
                    TOTALprojection += crop
        assert(TOTALprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
        return TOTALprojection
    
    def BackwardProjectACC_old(self, hMatrix, projection, planes, progress, logPrint, numjobs, keepNative=False):
        # Plane-by-plane code left for reference
        Backprojection = np.zeros((hMatrix.numZ, projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        plf.SetNumThreadsToUse(numjobs)
        for cc in progress(planes, 'Backward-project - z', leave=False):
            proj = self.zProjectorClass(projection, hMatrix, cc)
            Hcc = hMatrix.Hcc(cc, True)
            fourierZPlane = plf.ProjectForZ(projection, Hcc, hMatrix.Nnum, \
                                             proj.fshape[-2], proj.fshape[-1], \
                                             proj.rfshape[-2], proj.rfshape[-1], \
                                             proj.xAxisMultipliers, proj.yAxisMultipliers)
            # Compute the iFFT for each z plane
            Backprojection[cc] = special.special_fftconvolve_part3(fourierZPlane, proj.fshape, proj.fslice, proj.s1, useCCode=True)
        return Backprojection

    def ForwardProjectACC_old(self, hMatrix, realspace, planes, progress, logPrint, numjobs, keepNative=False):
        TOTALprojection = None
        plf.SetNumThreadsToUse(numjobs)
        for cc in progress(planes, 'Forward-project - z', leave=False):
            # Project each z plane forward to the camera image plane
            proj = self.zProjectorClass(realspace[0], hMatrix, cc)
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


def LogMemory(description, garbageCollect=False):
    if gpuAvailable:
        _ = cp.zeros((1))  # Dummy to trigger cuda lazy initialization if we are called before any real work has been done
        before = cuda.mem_get_info()
        before = before[1]-before[0]
        if logGPUMemoryUsage:
            print("{0:.3f}GB {1}".format(before/1e9, description))
        if garbageCollect:
            cp.get_default_memory_pool().free_all_blocks()
            after = cuda.mem_get_info()
            after = after[1]-after[0]
            if logGPUMemoryUsage:
                if (before == after):
                    print("-> no garbage")
                else:
                    print("-> {0:.3f}GB (collected {1:.3f}GB)".format(after/1e9, (before-after)/1e9))

class Projector_pythonSkeleton(Projector_base):
    def BackwardProjectACC(self, hMatrix, projection, planes, progress, logPrint, numjobs, keepNative=False, rlUpdateFunc=None): # Ignores numjobs
        projection = self.asnative(projection)
        LogMemory("BackwardProjectACC", True)
        if rlUpdateFunc is None:
            result = self.nativeZeros((len(planes), projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        else:
            planeResult = self.nativeZeros((projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')

        # Project each z plane in turn
        fourierZPlanes = []     # This has to be a list because in Fourier space the shapes are different for each z plane
        for cc in progress(planes, desc='Backward-project - z', leave=False):
            thisFourierBackprojection = self.ProjectForZ(cc, projection, hMatrix, True)
            if rlUpdateFunc is not None:
                self.InverseTransformBackwardProjection(planeResult, thisFourierBackprojection, hMatrix, cc, projection)
                rlUpdateFunc(planeResult, cc)
            else:
                self.InverseTransformBackwardProjection(result[cc], thisFourierBackprojection, hMatrix, cc, projection)
            # Doing garbage collection at this point does slow things down a bit, but keeps the high water mark down a bit.
            # I don't know if it actually affects what we can cope with
            #LogMemory("ProjectForZ[{0}]".format(cc), True)
            
        if rlUpdateFunc is not None:
            del planeResult
        LogMemory("ProjectForZ finished", True)

        if rlUpdateFunc is not None:
            return None
        else:
            if keepNative:
                return result
            else:
                return self.asnumpy(result)

    def InverseTransformBackwardProjection(self, result, thisFourierBackprojection, hMatrix, cc, projection):
        # Compute the FFT for this z plane
        projector = self.zProjectorClass(projection, hMatrix, cc, self.fftPlan, self.fftPlan2)
        (fshape, fslice, s1) = special.convolutionShape(projection.shape, hMatrix.PSFShape(cc), hMatrix.Nnum, projector.padToSmallPrimes)
        result = special.special_fftconvolve_part3(thisFourierBackprojection, fshape, fslice, s1)

    def ForwardProjectACC(self, hMatrix, realspace, planes, progress, logPrint, numjobs, keepNative=False): # Ignores numjobs
        realspace = self.asnative(realspace)
        LogMemory("ForwardProjectACC", True)
        result = self.nativeZeros((realspace.shape[1], realspace.shape[2], realspace.shape[3]), dtype='float32')
        
        # Project each z plane in turn
        fourierProjections = []     # This has to be a list because in Fourier space the shapes are different for each z plane
        for cc in progress(planes, desc='Forward-project - z', leave=False):
            thisFourierForwardProjection = self.ProjectForZ(cc, realspace[cc], hMatrix, False)
            self.InverseTransformForwardProjection(result, thisFourierForwardProjection, hMatrix, cc, realspace)
            #LogMemory("ProjectForZ[{0}]".format(cc), True)
        LogMemory("ProjectForZ finished", True)

        if keepNative:
            return result
        else:
            return self.asnumpy(result)
    
    def InverseTransformForwardProjection(self, result, thisFourierForwardProjection, hMatrix, cc, realspace):
        # Compute and accumulate the FFT for this z plane
        projector = self.zProjectorClass(realspace[cc], hMatrix, cc, self.fftPlan, self.fftPlan2)
        (fshape, fslice, s1) = special.convolutionShape(realspace[cc].shape, hMatrix.PSFShape(cc), hMatrix.Nnum, projector.padToSmallPrimes)
        result += special.special_fftconvolve_part3(thisFourierForwardProjection, fshape, fslice, s1)

    def ProjectForZ(self, cc, source, hMatrix, backwards):
        # This is a perhaps a bit of a hack for now - it ensures FFT(PSF) is calculated on the GPU
        # TODO: I should update this with a lambda (if I can work out how...?) that passes in an FFT plan that we have precomputed
        if self.zProjectorClass is ProjectorForZ_gpuHelpers:
            hMatrix.UpdateFFTFunc(myfft.myFFT2_gpu)
        else:
            hMatrix.UpdateFFTFunc(myfft.myFFT2)
        projector = self.zProjectorClass(source, hMatrix, cc, self.fftPlan, self.fftPlan2)
        # Cache the fftPlan from this projector, to reuse for the next z plane if possible
        self.fftPlan = projector.fftPlan
        self.fftPlan2 = projector.fftPlan2
        return projector.ProjectForZ(cc, source, hMatrix, backwards)

class Projector_python(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_python
        self.name = 'Pure python'


class Projector_cHelpers(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_cHelpers
        self.name = 'C helpers'


class Projector_gpuHelpers(Projector_pythonSkeleton):
    def __init__(self):
        super().__init__()
        self.zProjectorClass = ProjectorForZ_gpuHelpers
        self.name = 'GPU helpers'

    def asnative(self, m):
        return cp.asarray(m)

    def asnumpy(self, m):
        return cp.asnumpy(m)

    def nativeZeros(self, shape, dtype=float):
        return cp.zeros(shape, dtype)
    
    def nativeDivide(self, a, b, out):
        return cp.divide(a, b, out=out)

    def InverseTransformBackwardProjection(self, result, thisFourierBackprojection, hMatrix, cc, projection):
        projector = self.zProjectorClass(projection, hMatrix, cc, self.fftPlan, self.fftPlan2)
        (fshape, fslice, s1) = special.convolutionShape(projection.shape, hMatrix.PSFShape(cc), hMatrix.Nnum, projector.padToSmallPrimes)
        # This next code is copied from special_fftconvolve_part3
        for n in range(thisFourierBackprojection.shape[0]):
            inv = cp.fft.irfftn(thisFourierBackprojection[n], fshape)
            result[n] = special._centered(inv[fslice], s1)

    def InverseTransformForwardProjection(self, result, thisFourierForwardProjection, hMatrix, cc, realspace):
        # Compute and accumulate the FFT for each z plane
        (fshape, fslice, s1) = special.convolutionShape(realspace[cc].shape, hMatrix.PSFShape(cc), hMatrix.Nnum, padToSmallPrimesOnGPU)
        # This next code is copied from special_fftconvolve_part3
        for n in range(thisFourierForwardProjection.shape[0]):
            inv = cp.fft.irfftn(thisFourierForwardProjection[n], fshape)
            result[n] += special._centered(inv[fslice], s1)


#########################################################################
# Older, simple versions of backprojection code, for reference.
# I don't expect these to be used except for internal testing within this module.
# Note: it is a bit of an anomaly that some of these are here, but forwardProjectACC etc are in lfdeconv...
#########################################################################

def ProjectForZ(hMatrix, backwards, cc, source, projectorClass=Projector_allC, progress=tqdm):
    # This is somewhat obsolete now, I think, but I will leave it for now
    projector = projectorClass()
    result = projector.ProjectForZ(cc, source, hMatrix, backwards)
    # Actually, for forward projection we don't need to do this separately for every z,
    # but it's easier to do it for consistency (and this function is not used in performance-critical code anyway)
    (fshape, fslice, s1) = special.convolutionShape(source.shape, hMatrix.PSFShape(cc), hMatrix.Nnum, False)
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
    backprojection = np.zeros((Ht.shape[0], projection.shape[-2], projection.shape[-1]), dtype='float32')
    # Iterate over each z plane
    if planes is None:
        planes = range(Ht.shape[0])
    for cc in progress(planes, desc='Back-project - z'):
        HtCC =  Ht[cc, :, :, CAindex[0,cc]-1:CAindex[1,cc], CAindex[0,cc]-1:CAindex[1,cc]]
        bp = BackwardProjectForZ_old(HtCC, projection, progress=progress)
        if len(bp.shape) == 3:  # Multi-timepoint result
            assert(bp.shape[0] == 1)
        backprojection[cc] = bp[...,:,:]

    return backprojection

#########################################################################
# Self-test code: test the backprojection code against a slower definitive version
#########################################################################
def selfTest(verbose=True):
    # The strictest self-test would be against the original (very simple) code I wrote,
    # but that is very slow and requires us to load in the H matrices in a slow manner.
    # As a result, normally I would be satisfied to test against the newer, but still
    # pure-python implementation I have written (and which is stable code).
    testAgainstOriginalCode = False
    if testAgainstOriginalCode:
        print('Testing backprojection code against original code')
    else:
        print('Testing backprojection code against pure-python code')
    
    # Load the H matrix
    # We need the raw _H and _Ht values, since we are using old projection code to validate the results of my new optimized code
    matPath = 'PSFmatrix/fdnormPSFmatrix_M40NA0.95MLPitch150fml3000from-6to-5zspacing0.5Nnum15lambda520n1.mat'
    if testAgainstOriginalCode:
        (_H, _Ht, _CAIndex, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = psfmatrix.LoadRawMatrixData(matPath, createPSF=True)
        testHCC = _H[0]
        testHtCC = _Ht[0]
    
    # I have removed Projector_cHelpers from this list of classes to test, because my changes to py_light_field
    # have removed several of the c helper functions. I might be able to reinstate them now the C code is stable again,
    # but I don't think it's a mode I actually care about keeping going. Its only benefit is to have a C counterpart
    # to the GPU helpers for testing purposes (but I could probably equally well test against a python counterpart if I wanted...)
    classesToTest = [Projector_allC, Projector_python]
    if gpuAvailable:
        classesToTest = classesToTest + [Projector_gpuHelpers]
    np.random.seed(0)
    testOutcomes = np.zeros((2))
    for projectorClass in classesToTest:
        print(' Testing class:', projectorClass.__name__)
        for bk in [True, False]:
            print('  === bk {0} ==='.format(bk))
            # Test both square and non-square, since they use different code
            for numTimepoints in [2]:
              for shape in [(numTimepoints,150,150), (numTimepoints,150,300), (numTimepoints,300,150)]:
                print('  === shape {0} ==='.format(shape))
                testHMatrix = psfmatrix.LoadMatrix(matPath, numZ=1, zStart=0, createPSF=True)   # Needs to be in the loop here, because caching is confused by changing the image shape
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
                    if verbose:
                        print('   Old took %.2fms'%((t2-t1)*1e3))

                # Now run the code we are actually testing.
                # Note that we call xxxProjectACC rather than ProjectForZ,
                # because the pure-C implementation does not have a ProjectForZ function.
                projector = projectorClass()
                projector.cacheFH = True
                plf.ClearFHCache()
                for n in range(2):  # Run twice to avoid counting the time spent making FFT plans and possibly caching FFT
                    t1 = time.time()
                    if bk:
                        testResultNew = projector.BackwardProjectACC(testHMatrix, testProjection, [0], progress=util.noProgressBar, logPrint=False, numjobs=1)
                    else:
                        testResultNew = projector.ForwardProjectACC(testHMatrix, testProjection[np.newaxis,:,:,:], [0], progress=util.noProgressBar, logPrint=False, numjobs=1)
                    t2 = time.time()
                if verbose:
                    print('   New took %.2fms'%((t2-t1)*1e3))
                # Compare the results that we got
                testOutcomes += util.CheckComparison(testResultOld, testResultNew, 1e-4, description='   Test result', shouldBe='<<1', verbose=verbose)

    print('Projector tests against reference implementation complete (passed %d/%d)' % (testOutcomes[0], testOutcomes[1]))
    return testOutcomes

if __name__ == "__main__":
    selfTest()
