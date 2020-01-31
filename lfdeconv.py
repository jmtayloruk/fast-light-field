import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import tifffile
import sys, time, os
from jutils import tqdm_alias as tqdm

import psfmatrix, lfimage
import projector
import special_fftconvolve as special
import jutils as util

# It has been suggested that low-level threading does not interact well with the joblib Parallel feature.
# Certainly that matches my experience.
# I have seen hangs, and also my code seems to take way longer to execute overall than it should.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

#########################################################################        
# Functions to implement deconvolution.
# This code uses Parallel() to run multithreaded for speed
#########################################################################        

def BackwardProjectACC(hMatrix, projection, planes=None, numjobs=multiprocessing.cpu_count(), progress=tqdm, logPrint=True):
    singleJob = (len(projection.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        projection = projection[np.newaxis,:,:]
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    ru1 = util.cpuTime('both')

    Backprojection = np.zeros((hMatrix.numZ, projection.shape[0], projection.shape[1], projection.shape[2]), dtype='float32')
        
    # Set up the work to iterate over each z plane
    work = []
    for cc in planes:
        for bb in hMatrix.IterableBRange(cc):
            work.append((cc, bb, projection, hMatrix, True))

    # Run the multithreaded work
    t0 = time.time()
    results = Parallel(n_jobs=numjobs)\
            (delayed(projector.ProjectForZY)(*args) for args in progress(work, desc='Back-project - z', leave=False))
    ru2 = util.cpuTime('both')

    # Gather together and sum the results for each z plane
    t1 = time.time()
    fourierZPlanes = [None]*hMatrix.numZ
    elapsedTime = 0
    for (result, cc, bb, t) in results:
        elapsedTime += t
        if fourierZPlanes[cc] is None:
            fourierZPlanes[cc] = result
        else:
            fourierZPlanes[cc] += result
    
    # Compute the FFT for each z plane
    for cc in planes:
        (fshape, fslice, s1) = special.convolutionShape(projection, hMatrix.PSFShape(cc), hMatrix.Nnum(cc))
        Backprojection[cc] = special.special_fftconvolve_part3(fourierZPlanes[cc], fshape, fslice, s1)        
    t2 = time.time()
    assert(Backprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
   
    # Save some diagnostics
    if logPrint:
        print('work elapsed wallclock time %f'%(t1-t0))
        print('work elapsed thread time %f'%elapsedTime)
        print('work delta rusage:', ru2-ru1)
        print('FFTs took %f'%(t2-t1))
    
    f = open('overall.txt', 'w')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\n' % (t0, t1, t1-t0, t2-t1, (ru2-ru1)[0], (ru2-ru1)[1]))
    f.close()

    if singleJob:
        return Backprojection[:,0]
    else:
        return Backprojection

def ForwardProjectACC(hMatrix, realspace, planes=None, numjobs=multiprocessing.cpu_count(), progress=tqdm, logPrint=True):
    singleJob = (len(realspace.shape) == 3)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        realspace = realspace[:,np.newaxis,:,:]
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    # Set up the work to iterate over each z plane
    work = []
    for cc in planes:
        for bb in hMatrix.IterableBRange(cc):
            work.append((cc, bb, realspace[cc], hMatrix, False))

    # Run the multithreaded work
    t0 = time.time()
    results = Parallel(n_jobs=numjobs)\
                (delayed(projector.ProjectForZY)(*args) for args in progress(work, desc='Forward-project - z', leave=False))

    # Gather together and sum all the results
    t1 = time.time()
    fourierProjection = [None]*hMatrix.numZ
    elapsedTime = 0
    for (result, cc, bb, t) in results:
        elapsedTime += t
        if fourierProjection[cc] is None:
            fourierProjection[cc] = result
        else:
            fourierProjection[cc] += result

    # Compute and accumulate the FFT for each z plane
    TOTALprojection = None
    for cc in planes:
        # A bit complicated here to set up the correct inputs for convolutionShape...
        (fshape, fslice, s1) = special.convolutionShape(realspace[cc], hMatrix.PSFShape(cc), hMatrix.Nnum(cc))
        thisProjection = special.special_fftconvolve_part3(fourierProjection[cc], fshape, fslice, s1)        
        if TOTALprojection is None:
            TOTALprojection = thisProjection
        else:
            TOTALprojection += thisProjection
    t2 = time.time()
    assert(TOTALprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
            
    # Print out some diagnostics
    if (logPrint):
        print('work elapsed wallclock time %f'%(t1-t0))
        print('work elapsed thread time %f'%elapsedTime)
        print('FFTs took %f'%(t2-t1))
        
    if singleJob:
        return TOTALprojection[0]
    else:
        return TOTALprojection

def DeconvRL(hMatrix, Htf, maxIter, Xguess, logPrint=True, numjobs=multiprocessing.cpu_count()):
    # Note:
    #  Htf is the *initial* backprojection of the camera image
    #  Xguess is the initial guess for the object
    ru1 = util.cpuTime('both')
    t1 = time.time()
    for i in tqdm(range(maxIter), desc='RL deconv'):
        t0 = time.time()
        HXguess = ForwardProjectACC(hMatrix, Xguess, numjobs=numjobs, logPrint=logPrint)
        HXguessBack = BackwardProjectACC(hMatrix, HXguess, numjobs=numjobs, logPrint=logPrint)
        errorBack = Htf / HXguessBack
        Xguess = Xguess * errorBack
        Xguess[np.where(np.isnan(Xguess))] = 0
        ttime = time.time() - t0
        print('iter %d | %d, took %.1f secs. Max val %f' % (i+1, maxIter, ttime, np.max(Xguess)))
    ru2 = util.cpuTime('both')
    t2 = time.time()
    print('Deconvolution elapsed wallclock time %f, rusage %f' % (t2-t1, np.sum(ru2-ru1)))

    return Xguess


if __name__ == "__main__":
    #########################################################################
    # Test code for deconvolution
    #########################################################################

    # Load the input image and matrix
    inputImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')
    hMatrix = psfmatrix.LoadMatrix('PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-26to0zspacing2Nnum15lambda520n1.0.mat')
    
    if ('basic' in sys.argv) or ('full' in sys.argv):
	    # Run my back-projection code (single-threaded) on a cropped version of Prevedel's data
        Htf = BackwardProjectACC(hMatrix, inputImage, planes=None, logPrint=False)
        if True:
            definitive = tifffile.imread('Data/03_Reconstructed/exampleData/definitive_worm_crop_X15_backproject.tif')
            definitive = np.transpose(definitive, axes=(0,2,1))
            util.CheckComparison(definitive[4], Htf[4]*10, 1.0, 'Compare against matlab result')
        else:
            # I am not totally sure what the purpose of this next code is,
            # My guess is that this is an output I generated from my code, after first checking consistency with Matlab.
            # I am not sure exactly what the purpose of it is, though - why is this better than using the actual matlab tiff?
            definitive = np.load('semi-definitive.npy')
            util.CheckComparison(definitive, Htf, 1.0, 'Compare against matlab result')

    if 'full' in sys.argv:
        # Run my full Richardson-Lucy code on a cropped version of Prevedel's data
        # Note that the back-projection must run first, since we use that as our initial guess
        deconvolvedResult = DeconvRL(hMatrix, Htf, maxIter=8, Xguess=Htf.copy(), logPrint=False)
        definitive = tifffile.imread('Data/03_Reconstructed/exampleData/definitive_worm_crop_X15_iter8.tif')
        definitive = np.transpose(definitive, axes=(0,2,1))
        util.CheckComparison(definitive, deconvolvedResult*1e3, 1.0, 'Compare against matlab result')

    if 'parallel' in sys.argv:
        # Code to test simultaneous deconvolution of an image pair
        # This does not test overall correctness, but it runs with two different (albeit proportional)
        # images and checks that the result matches the result for two totally independent calls on a single array.
        print("Testing image pair deconvolution:")
        candidate = np.tile(inputImage[np.newaxis,0,0], (2,1,1))
        candidate[1] *= 1.4
        planesToUse = None   # Use all planes
        if planesToUse is None:
            numPlanesToUse = hMatrix.numZ
        else:
            numPlanesToUse = len(planesToUse)

        print('Running (%d planes x2)'%numPlanesToUse)
        t1 = time.time()
        temp = BackwardProjectACC(hMatrix, candidate, planes=None, numjobs=1, logPrint=False)
        dualRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, logPrint=False)
        print('New method took', time.time()-t1)

        # Run for the individual images, and check we get the same result as with the dual round-trip
        temp = BackwardProjectACC(hMatrix, candidate[0], planes=None, numjobs=1, logPrint=False)
        firstRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, numjobs=1, logPrint=False)
        comparison = np.max(np.abs(firstRoundtrip - dualRoundtrip[0]))
        print(' Test result (should be <<1): %e' % comparison)
        if (comparison > 1e-6):
            print("  -> WARNING: disagreement detected")
        else:
            print("  -> OK")
        
        temp = BackwardProjectACC(hMatrix, candidate[1], planes=None, numjobs=1, logPrint=False)
        secondRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, numjobs=1, logPrint=False)
        comparison = np.max(np.abs(secondRoundtrip - dualRoundtrip[1]))
        print(' Test result (should be <<1): %e' % comparison)
        if (comparison > 1e-6):
            print("  -> WARNING: disagreement detected")
        else:
            print("  -> OK")

        # Run for 10 images in parallel.
        # Note that 'candidate' is already 2 images, so we tile by a factor of 5 to get a total of 10 images
        print('Running (%d planes x10)'%numPlanesToUse)
        t1 = time.time()
        temp = BackwardProjectACC(hMatrix, np.tile(candidate, (5,1,1)), planes=None, numjobs=1, logPrint=False)
        tenRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, logPrint=False)
        print('New method took', time.time()-t1)
    
       
    if 'parallel' in sys.argv:
        # Run to test parallelization
        # TODO: this just parasitises a previously-defined hMatrix and inputImage...
        # Note that I think the reason I do not currently check for correctness is that
        # there are small numerical variations depending on how the parallelism occurs in each run.
        print("Testing multithreaded execution:")
        result1 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=1, logPrint=False)
        result3 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=3, logPrint=False)
        comparison = np.max(np.abs(result1 - result3))
        print(' Test result (should be <<1): %e' % comparison)
        if (comparison > 1e-6):
            print("  -> WARNING: disagreement detected")
        else:
            print("  -> OK")
        result6 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=6, logPrint=False)
        comparison = np.max(np.abs(result1 - result6))
        print(' Test result (should be <<1): %e' % comparison)
        if (comparison > 1e-6):
            print("  -> WARNING: disagreement detected")
        else:
            print("  -> OK")

