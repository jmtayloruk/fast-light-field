import numpy as np
from scipy.signal import fftconvolve
import scipy.fftpack
import time, warnings, os
from jutils import tqdm_alias as tqdm

import py_symmetry as jps
import special_fftconvolve as special
import psfmatrix
import jutils as util

# Ensure existence of the directory we will use to log performance diagnostics
try:
    os.mkdir('perf_diags')
except:
    pass  # Probably the directory already exists


# Note: H.shape in python is (<num z planes>, Nnum, Nnum, <psf size>, <psf size>),
#                       e.g. (56, 19, 19, 343, 343)

#########################################################################        
# Projector class performs the convolution between a given image
# and a (provided) PSF. It operates for a specific ZYX, 
# but does the projection for all symmetries that exist for that PSF
# (i.e. in practice it is likely to project for more than just the one specified ZYX)
#########################################################################        

class Projector(object):
    # Note: the variable names in this class mostly imply we are doing the back-projection
    # (e.g. Ht, 'projection', etc. However, the same code also does forward-projection!)
    def __init__(self, projection, hMatrix, cc):
        # Note: H and Hts are not stored as class variables.
        # I had a lot of trouble with them and multithreading,
        # and eventually settled on having them in shared memory.
        # As I encapsulate more stuff in this class, I could bring them back as class variables...

        self.cpuTime = np.zeros(2)
        
        # Nnum: number of pixels across a lenslet array (after rectification)
        self.Nnum = hMatrix.Nnum(cc)
        
        # This next chunk of logic is copied from the fftconvolve source code.
        # s1, s2: shapes of the input arrays
        # fshape: shape of the (full, possibly padded) result array in Fourier space
        # fslice: slicing tuple specifying the actual result size that should be returned
        self.s1 = np.array(projection.shape)
        self.s2 = np.array(hMatrix.PSFShape(cc))
        shape = self.s1 + self.s2 - 1
        if False:
            # TODO: I haven't worked out if/how I can do this yet.
            # This is the original code in fftconvolve, which says:
            # Speed up FFT by padding to optimal size for FFTPACK
            self.fshape = [_next_regular(int(d)) for d in shape]
        else:
            self.fshape = [int(np.ceil(d/float(self.Nnum)))*self.Nnum for d in shape]
        self.fslice = tuple([slice(0, int(sz)) for sz in shape])
        
        # rfslice: slicing tuple to crop down full fft array to the shape that would be output from rfftn
        self.rfslice = (slice(0,self.fshape[0]), slice(0,int(self.fshape[1]/2)+1))
        
        # Precalculate arrays that are needed as part of the process of converting from FFT(H) to FFT(mirrorImage(H))
        padLength = self.fshape[0] - self.s2[0]
        self.mirrorXMultiplier = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[0]) * np.arange(self.fshape[0])).astype('complex64')
        padLength = self.fshape[1] - self.s2[1]
        self.mirrorYMultiplier = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[1]) * np.arange(self.fshape[1])).astype('complex64')

        return
    
    
    #########################################################################
    # Convolve a projection with a H matrix (PSF), taking advantage of all
    # the mirror symmetries that are present in the H matrix
    #########################################################################
    def MirrorXArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the X mirror of that same PSF
        if False:
            # Old code that does this in pure python
            fHtsFull = fHtsFull.conj() * self.mirrorXMultiplier[:,np.newaxis]
            fHtsFull[:,1::] = fHtsFull[:,1::][:,::-1]
            return fHtsFull
        else:
            # New, faster code written in C
            return jps.mirrorX(fHtsFull, self.mirrorXMultiplier)

    def MirrorYArray(self, fHtsFull):
        # Utility function to convert the FFT of a PSF to the FFT of the Y mirror of that same PSF
        if False:
            # Old code that does this in pure python
            fHtsFull = fHtsFull.conj() * self.mirrorYMultiplier
            fHtsFull[1::] = fHtsFull[1::][::-1]
            return fHtsFull
        else:
            # New, faster code written in C
            return jps.mirrorY(fHtsFull, self.mirrorYMultiplier)
        
    def convolvePart3(self, projection, bb, aa, fHtsFull, mirrorX, accum):
        cpu0 = util.cpuTime('both')
        if True:
            accum = special.special_fftconvolve2(projection,bb,aa,self.Nnum,self.s2,accum,fHtsFull,int(self.fshape[1]/2)+1,None)
        else:
            # Old pure-python code still here for reference, for now
            accum = special.special_fftconvolve(projection,bb,aa,self.Nnum,self.s2,accum,fb=fHtsFull[self.rfslice])
        self.cpuTime += util.cpuTime('both')-cpu0
        if mirrorX:
            cpu0 = util.cpuTime('both')
            if True:
                accum = special.special_fftconvolve2(projection,self.Nnum-bb-1,aa,self.Nnum,self.s2,accum,fHtsFull,int(self.fshape[1]/2)+1,self.mirrorXMultiplier)
            else:
                # Old pure-python code still here for reference, for now
                fHtsMirror = self.MirrorXArray(fHtsFull)[self.rfslice]
                accum = special.special_fftconvolve(projection,self.Nnum-bb-1,aa,self.Nnum,self.s2,accum,fb=fHtsMirror)
            self.cpuTime += util.cpuTime('both')-cpu0
        return accum

    def convolvePart2(self, projection, bb, aa, fHtsFull, mirrorY, mirrorX, accum):
        accum = self.convolvePart3(projection,bb,aa,fHtsFull,mirrorX,accum)
        if mirrorY:
            fHtsFull = self.MirrorYArray(fHtsFull)
            accum = self.convolvePart3(projection,bb,self.Nnum-aa-1,fHtsFull,mirrorX,accum)
        return accum

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
                # The copy() is because my C code currently can't cope with
                # a transposed array (non-contiguous strides in x)
                fHtsFull = fHtsFull.transpose().copy()
            else:
                # For a non-square array, we have to compute the FFT for the transpose.
                fHtsFull = hMatrix.fH(cc, bb, aa, backwards, True, self.fshape)
            # Note that mx,my have been swapped here, which is necessary following the transpose
            accum = self.convolvePart2(projection,aa,bb,fHtsFull,mirrorX,mirrorY, accum) 
        assert(accum.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
        return accum

    
#########################################################################        
# Functions to project for a partial or entire z plane.
# These are the functions I expect external code to call.
#########################################################################        
    
def ProjectForZY(cc, bb, source, hMatrix, backwards):
    f = open('perf_diags/%d_%d.txt'%(cc,bb), "w")
    t1 = time.time()
    singleJob = (len(source.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        source = source[np.newaxis,:,:]
    result = None
    projector = Projector(source[0], hMatrix, cc)
    projector.cpuTime = np.zeros(2)
    for aa in range(bb,int((hMatrix.Nnum(cc)+1)/2)):
        result = projector.convolve(source, hMatrix, cc, bb, aa, backwards, result)
    t2 = time.time()
    assert(result.dtype == np.complex64)   # Keep an eye out for any reversion to double-precision
    f.write('%d\t%f\t%f\t%f\t%f\t%f\n' % (os.getpid(), t1, t2, t2-t1, projector.cpuTime[0], projector.cpuTime[1]))
    f.close()
    if singleJob:
        return (result[0], cc, bb, t2-t1)
    else:
        return (np.array(result), cc, bb, t2-t1)

def ProjectForZ(hMatrix, backwards, cc, source):
    result = None
    for bb in tqdm(hMatrix.IterableBRange(cc), leave=False, desc='Project - y'):
        (thisResult, _, _, _) = ProjectForZY(cc, bb, source, hMatrix, backwards)
        if (result is None):
            result = thisResult
        else:
            result += thisResult
    # Actually, for forward projection we don't need to do this separately for every z,
    # but it's easier to do it for symmetry (and this function is not used in performance-critical code anyway)
    (fshape, fslice, s1) = special.convolutionShape(source, hMatrix.PSFShape(cc), hMatrix.Nnum(cc))
    return special.special_fftconvolve_part3(result, fshape, fslice, s1)

        
#########################################################################        
# Older, simple versions of backprojection code, for reference.
# I don't expect these to be used except for internal testing within this module.
# Note: it is a bit of an anomaly that these are here, but forwardProjectACC etc are in lfdeconv...
#########################################################################        

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

if __name__ == "__main__":
    #########################################################################
    # Self-test code: test the backprojection code against a slower definitive version
    #########################################################################
    print("Testing the backprojection code")
    
    # Load the H matrix
    # We need the raw _H and _Ht values, since we are using old projection code to validate the results of my new optimized code
    matPath = 'PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.0.mat'
    (_H, _Ht, _CAIndex, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = psfmatrix.LoadRawMatrixData(matPath)
    testHCC = _H[13]
    testHtCC = _Ht[13]

    # Test forward and back projection
    for bk in [True, False]:
        # Test both square and non-square, since they use different code
        for shape in [(200,200), (200,300), (300,200)]:
            testHMatrix = psfmatrix.LoadMatrix(matPath, numZ=1, zStart=13)   # Needs to be in the loop here, because caching is confused by changing the image shape
            testProjection = np.random.random(shape).astype(np.float32)
            if bk:
                testResultOld = BackwardProjectForZ_old(testHtCC, testProjection)
            else:
                testResultOld = ForwardProjectForZ_old(testHCC, testProjection)
            
            testResultNew = ProjectForZ(testHMatrix, bk, 0, testProjection)
            comparison = np.max(np.abs(testResultOld - testResultNew))
            print('test result (should be <<1): %e' % comparison)
            if (comparison > 1e-4):
                print(" -> WARNING: disagreement detected")
            else:
                print(" -> OK")
            
    print('Tests complete')
