import numpy as np
from scipy.signal import fftconvolve
import scipy.fftpack
import time, warnings, os
# This next import is a slightly strange construction, but allows the option of substituting
# the second 'tqdm' with 'tqdm_notebook' if running in notebook environment.
# TODO: does that mean it won't work as desired if my module here is imported from a notebook?
# I will need to investigate that...
from tqdm import tqdm as tqdm   

import py_symmetry as jps
import special_fftconvolve as special
import hmatrix
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
    def __init__(self, projection, HtCCBB, Nnum):
        # Note: H and Hts are not stored as class variables.
        # I had a lot of trouble with them and multithreading,
        # and eventually settled on having them in shared memory.
        # As I encapsulate more stuff in this class, I could bring them back as class variables...
        
        self.cpuTime = np.zeros(2)
        
        # Nnum: number of pixels across a lenslet array (after rectification)
        self.Nnum = Nnum
        
        # This next chunk of logic is copied from the fftconvolve source code.
        # s1, s2: shapes of the input arrays
        # fshape: shape of the (full, possibly padded) result array in Fourier space
        # fslice: slicing tuple specifying the actual result size that should be returned
        self.s1 = np.array(projection.shape)
        self.s2 = np.array(HtCCBB[0].shape)
        shape = self.s1 + self.s2 - 1
        if False:
            # TODO: I haven't worked out if/how I can do this yet.
            # This is the original code in fftconvolve, which says:
            # Speed up FFT by padding to optimal size for FFTPACK
            self.fshape = [_next_regular(int(d)) for d in shape]
        else:
            self.fshape = [int(np.ceil(d/float(Nnum)))*Nnum for d in shape]
        self.fslice = tuple([slice(0, int(sz)) for sz in shape])
        
        # rfslice: slicing tuple to crop down full fft array to the shape that would be output from rfftn
        self.rfslice = (slice(0,self.fshape[0]), slice(0,int(self.fshape[1]/2)+1))
        return
    
    def MirrorXArray(self, Hts, fHtsFull):
        padLength = self.fshape[0] - Hts.shape[0]
        if False:
            fHtsFull = fHtsFull.conj() * np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[0]) * np.arange(self.fshape[0],dtype='complex64')[:,np.newaxis])
            fHtsFull[:,1::] = fHtsFull[:,1::][:,::-1]
            return fHtsFull
        else:
            temp = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[0]) * np.arange(self.fshape[0])).astype('complex64')
            if False:
                # TODO: this jps feature is a new one that I need to check in!
                result = jps.mirrorX(fHtsFull, temp)
            else:
                result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
                result[:,0] = fHtsFull[:,0].conj()*temp
                for i in range(1,fHtsFull.shape[1]):
                    result[:,i] = (fHtsFull[:,fHtsFull.shape[1]-i].conj()*temp)
            return result

    def MirrorYArray(self, Hts, fHtsFull):
        padLength = self.fshape[1] - Hts.shape[1]
        if False:
            fHtsFull = fHtsFull.conj() * np.exp(1j * (1+padLength) * 2*np.pi / self.fshape[1] * np.arange(self.fshape[1],dtype='complex64'))
            fHtsFull[1::] = fHtsFull[1::][::-1]
            return fHtsFull
        else:
            temp = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[1]) * np.arange(self.fshape[1])).astype('complex64')
            if False:
                result = jps.mirrorY(fHtsFull, temp)
            else:
                result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
                result[0] = fHtsFull[0].conj()*temp
                for i in range(1,fHtsFull.shape[0]):
                    result[i] = (fHtsFull[fHtsFull.shape[0]-i].conj()*temp)
            return result
    
    def convolvePart3(self, projection, bb, aa, Hts, fHtsFull, mirrorX, accum):
        # TODO: to make this work, I need the full matrix for fHts and then I need to slice it
        # to the correct shape when I call through to special_fftconvolve here. Is fshape what I need?
        cpu0 = util.cpuTime('both')
        (accum,_,_,_) = special.special_fftconvolve(projection,bb,aa,self.Nnum,Hts,accum,fb=fHtsFull[self.rfslice])
        self.cpuTime += util.cpuTime('both')-cpu0
        if mirrorX:
            fHtsFull = self.MirrorXArray(Hts, fHtsFull)
            cpu0 = util.cpuTime('both')
            (accum,_,_,_) = special.special_fftconvolve(projection,self.Nnum-bb-1,aa,self.Nnum,Hts[::-1,:],accum,fb=fHtsFull[self.rfslice])
            self.cpuTime += util.cpuTime('both')-cpu0
        return accum
    
    def convolvePart2(self, projection, bb, aa, Hts, fHtsFull, mirrorY, mirrorX, accum):
        accum = self.convolvePart3(projection,bb,aa,Hts,fHtsFull,mirrorX,accum)
        if mirrorY:
            fHtsFull = self.MirrorYArray(Hts, fHtsFull)
            accum = self.convolvePart3(projection,bb,self.Nnum-aa-1,Hts[:,::-1],fHtsFull,mirrorX,accum)
        return accum
    
    def fft2(self, mat, shape):
        # Perform a 'float' FFT on the matrix we are passed.
        # It would probably be faster if there was a way to perform the FFT natively on the 'float' type,
        # but scipy does not seem to support that option
        #
        # With my Mac Pro install, we hit a FutureWarning within scipy.
        # This wrapper just suppresses that warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return scipy.fftpack.fft2(mat, shape).astype('complex64')

    def convolve(self, projection, bb, aa, Hts, accum):
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
        fHtsFull = self.fft2(Hts, self.fshape)
        accum = self.convolvePart2(projection,bb,aa,Hts,fHtsFull,mirrorY,mirrorX, accum)
        if transpose:
            if (self.fshape[0] == self.fshape[1]):
                # For a square array, the FFT of the transpose is just the transpose of the FFT.
                # The copy() is because my C code currently can't cope with
                # a transposed array (non-contiguous strides in x)
                fHtsFull = fHtsFull.transpose().copy()
            else:
                # For a non-square array, we have to compute the FFT for the transpose.
                fHtsFull = self.fft2(Hts.transpose(), self.fshape)

            # Note that mx,my need to be swapped following the transpose
            accum = self.convolvePart2(projection,aa,bb,Hts.transpose(),fHtsFull,mirrorX,mirrorY, accum)
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
    result = [None] * source.shape[0]
    # TODO: tidy these next two lines, which are a bit of a hack to figure out Nnum.
    # I think the best solution is to move the functionality into hMatrix by implementing HMatrix.IterableARange()
    # (but note that I still use Hccbb)
    Hccbb = hMatrix.Hcc(cc, transpose=backwards)[bb]
    Nnum = hMatrix.Nnum(cc)
    projector = Projector(source[0], Hccbb, Nnum)
    projector.cpuTime = np.zeros(2)
    for aa in range(bb,int((Nnum+1)/2)):
        for n in range(source.shape[0]):
            result[n] = projector.convolve(source[n], bb, aa, Hccbb[aa], result[n])
    t2 = time.time()
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
    (fshape, fslice, s1) = special.convolutionShape(source, np.empty(hMatrix.PSFShape(cc)), hMatrix.Nnum(cc))
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

def BackwardProjectForZ_old(HtCC, projection):
    singleJob = (len(projection.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        projection = projection[np.newaxis,:,:]
    # Iterate over each lenslet pixel
    Nnum = HtCC.shape[1]
    tempSliceBack = np.zeros(projection.shape, dtype='float32')
    for aa in tqdm(range(Nnum), leave=False, desc='y'):
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

def BackwardProjectACC_old(Ht, projection, CAindex, planes=None):
    backprojection = np.zeros((Ht.shape[0], projection.shape[0], projection.shape[1]), dtype='float32')
    # Iterate over each z plane
    if planes is None:
        planes = range(Ht.shape[0])
    for cc in tqdm(planes, desc='Back-project - z'):
        HtCC =  Ht[cc, :, :, CAindex[0,cc]-1:CAindex[1,cc], CAindex[0,cc]-1:CAindex[1,cc]]
        backprojection[cc] = BackwardProjectForZ_old(HtCC, projection)
    return backprojection


if __name__ == "__main__":
    #########################################################################
    # Self-test code: test the backprojection code against a slower definitive version
    #########################################################################
    print("Testing the backprojection code")
    
    # Load the H matrix
    # We need the raw _H and _Ht values, since we are using old projection code to validate the results of my new optimized code
    matPath = 'PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.0.mat'
    (_H, _Ht, _CAIndex, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = hmatrix.LoadRawMatrixData(matPath)
    testHCC = _H[13]
    testHtCC = _Ht[13]

    # Test forward and back projection
    for bk in [True, False]:
        # Test both square and non-square, since they use different code
        for shape in [(200,200), (200,300), (300,200)]:
            testHMatrix = hmatrix.LoadMatrix(matPath, numZ=1, zStart=13)   # Needs to be in the loop here, because caching is confused by changing the image shape
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
