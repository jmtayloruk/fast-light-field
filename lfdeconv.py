import numpy as np
import multiprocessing
import tifffile
import sys, time, os
import cProfile, pstats
from jutils import tqdm_alias as tqdm

import psfmatrix, lfimage
import projector
import special_fftconvolve as special
import jutils as util
import py_light_field as plf

# It has been suggested that low-level threading does not interact well with the joblib Parallel feature.
# I have seen hangs, and also my code seems to take way longer to execute overall than it should.
# However, in fact I am not using joblib now, so I could consider allowing MKL to parallelise things.
# I don't use that in my gold-standard fastest code, though, so it's probably less confusing if I leave it
# clear what is running single-threaded and what is multithreaded through deliberate choices I make
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

#########################################################################        
# Functions to implement deconvolution.
# Most of the core code has now been encapsulated in classes in projector.py
#########################################################################

def BackwardProjectACC(hMatrix, projection, planes=None, progress=tqdm, logPrint=True, numjobs=multiprocessing.cpu_count(), projector=projector.Projector_allC(), keepNative=False):
    singleJob = (len(projection.shape) == 2)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        projection = projection[np.newaxis,:,:]
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    ru1 = util.cpuTime('both')
    t1 = time.time()
    Backprojection = projector.BackwardProjectACC(hMatrix, projection, planes, progress, logPrint, numjobs, keepNative)
    assert(Backprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
    ru2 = util.cpuTime('both')
    t2 = time.time()
    elapsedTime = t2 - t1

    # Save some diagnostics
    if logPrint:
        print('Total work elapsed thread time %f'%elapsedTime)
        print('Total work delta rusage:', ru2-ru1)
    
    f = open('overall.txt', 'w')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\n' % (0, 0, 0, 0, (ru2-ru1)[0], (ru2-ru1)[1]))
    f.close()

    if singleJob:
        return Backprojection[:,0]
    else:
        return Backprojection

def ForwardProjectACC(hMatrix, realspace, planes=None, progress=tqdm, logPrint=True, numjobs=multiprocessing.cpu_count(), projector=projector.Projector_allC(), keepNative=False):
    singleJob = (len(realspace.shape) == 3)
    if singleJob:   # Cope with both a single 2D plane and an array of multiple 2D planes to process independently
        realspace = realspace[:,np.newaxis,:,:]
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    ru1 = util.cpuTime('both')
    t1 = time.time()
    TOTALprojection = projector.ForwardProjectACC(hMatrix, realspace, planes, progress, logPrint, numjobs, keepNative)
    assert(TOTALprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
    ru2 = util.cpuTime('both')
    t2 = time.time()
    elapsedTime = t2 - t1

    # Print out some diagnostics
    if (logPrint):
        print('work elapsed wallclock time %f'%(t1-t0))
        print('work elapsed thread time %f'%elapsedTime)
        print('FFTs took %f'%(t2-t1))
        
    if singleJob:
        return TOTALprojection[0]
    else:
        return TOTALprojection

def DeconvRL(hMatrix, Htf, maxIter, Xguess, logPrint=True, numjobs=multiprocessing.cpu_count(), projectorClass=projector.Projector_allC):
    # Note:
    #  Htf is the *initial* backprojection of the camera image
    #  Xguess is the initial guess for the object
    ru1 = util.cpuTime('both')
    t1 = time.time()
    projector = projectorClass()
    Htf = projector.asnative(Htf)
    Xguess = projector.asnative(Xguess)
    for i in tqdm(range(maxIter), desc='RL deconv'):
        t0 = time.time()
        HXguess = ForwardProjectACC(hMatrix, Xguess, numjobs=numjobs, progress=None, logPrint=logPrint, projector=projector, keepNative=True)
        HXguessBack = BackwardProjectACC(hMatrix, HXguess, numjobs=numjobs, progress=None, logPrint=logPrint, projector=projector, keepNative=True)
        errorBack = Htf / HXguessBack
        Xguess = Xguess * errorBack
        Xguess[np.where(np.isnan(Xguess))] = 0
        ttime = time.time() - t0
        print('iter %d | %d, took %.1f secs' % (i+1, maxIter, ttime))
    ru2 = util.cpuTime('both')
    t2 = time.time()
    print('Deconvolution elapsed wallclock time %f, rusage %f' % (t2-t1, np.sum(ru2-ru1)))
    return projector.asnumpy(Xguess)


def main(argv, projectorClass=projector.Projector_allC, maxiter=8, numParallel=32):
    #########################################################################
    # Test code for deconvolution
    #########################################################################

    # Load the input image and matrix
    inputImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')
    hMatrix = psfmatrix.LoadMatrix('PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-26to0zspacing2Nnum15lambda520n1.0.mat')
    deconvolvedResult = None
    
    if (len(argv) == 1):
        # No arguments passed
        print('NO ARGUMENTS PASSED. You should pass parameters to this script if you want to execute it directly to run self-tests (see script for what options are available)')
    
    if ('basic' in argv) or ('full' in argv) or ('full32' in argv):
        print('== Running basic (single-threaded backprojection) ==')
	    # Run my back-projection code (single-threaded) on a cropped version of Prevedel's data
        Htf = BackwardProjectACC(hMatrix, inputImage, planes=None, numjobs=1, progress=None, logPrint=False, projector=projectorClass())
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

    if 'prime' in argv:
        # Run a single RL iteration to prime things (my caches, GPU FFT plans, etc) for subsequent runs (since we may be interested in timing/profiling)
        print('== Priming caches with a single RL iteration ==')
        deconvolvedResult = DeconvRL(hMatrix, Htf, maxIter=1, Xguess=Htf.copy(), logPrint=False, projectorClass=projectorClass)

    if 'full' in argv:
        # Run my full Richardson-Lucy code on a cropped version of Prevedel's data
        # Note that the back-projection (i.e. the 'basic' branch, above) must run first, since we use that as our initial guess
        print('== Running full RL deconvolution ==')
        print('Problem size {0}={1}MB'.format(Htf.shape, Htf.size*Htf.itemsize/1e6))
        deconvolvedResult = DeconvRL(hMatrix, Htf, maxIter=maxiter, Xguess=Htf.copy(), logPrint=False, projectorClass=projectorClass)
        definitive = tifffile.imread('Data/03_Reconstructed/exampleData/definitive_worm_crop_X15_iter8.tif')
        definitive = np.transpose(definitive, axes=(0,2,1))
        util.CheckComparison(definitive, deconvolvedResult*1e3, 1.0, 'Compare against matlab result')

    if 'full32' in argv:
        print('== Running full32 (full RL deconvolution on {0} parallel timepoints) =='.format(numParallel))
        Htf_x32 = np.tile(Htf[:,np.newaxis,:,:], (1,numParallel,1,1))
        print('Problem size {0}={1}MB'.format(Htf_x32.shape, Htf_x32.size*Htf_x32.itemsize/1e6))
        pr = cProfile.Profile()
        pr.enable()
        deconvolvedResult = DeconvRL(hMatrix, Htf_x32, maxIter=maxiter, Xguess=Htf_x32.copy(), logPrint=False, projectorClass=projectorClass)
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)
    
    if 'parallel' in argv:
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
        deconvolvedResult = ForwardProjectACC(hMatrix, temp, planes=None, logPrint=False)
        print('New method took', time.time()-t1)
    
       
    if 'parallel-threading' in argv:
        # Run to test parallelization
        # Note that I think the reason I do not currently have code here to check for correctness is that
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

    # Return the last result we obtained
    return deconvolvedResult

if __name__ == "__main__":
    main(sys.argv, projectorClass=projector.Projector_python)
