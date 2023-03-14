import numpy as np
import tifffile
import sys, time, os
import cProfile, pstats
from jutils import tqdm_alias as tqdm

import psfmatrix, lfimage
import projector as proj
import special_fftconvolve as special
import jutils as util
import py_light_field as plf

# I originally included these restrictions because low-level threading does not interact well with the joblib Parallel feature.
# However, in fact I am not using joblib now, so I could consider allowing MKL to parallelise things.
# I don't use that in my gold-standard fastest code, though - I use FFTW threading.
# It is therefore probably less confusing if I leave it clear what is running single-threaded and what is multithreaded through deliberate choices I make,
# rather than allowing MKL to parallelise behind the scenes if it wants.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

#########################################################################        
# Functions to implement deconvolution.
# Most of the core code has now been encapsulated in classes in projector.py
#########################################################################

def BackwardProjectACC(hMatrix, projection, planes=None, progress=tqdm, logPrint=True, numjobs=util.PhysicalCoreCount(), projector=proj.Projector_allC(), rlUpdateFunc=None):
    assert(len(projection.shape) == 3)  # We only now support batch jobs (although batch size [dimension 0] can of course be 1)
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    ru1 = util.cpuTime('both')
    t1 = time.time()
    Backprojection = projector.BackwardProjectACC(hMatrix, projection, planes, progress, logPrint, numjobs, rlUpdateFunc)
    ru2 = util.cpuTime('both')
    t2 = time.time()

    # Save some diagnostics
    if logPrint:
        print('Backward project: elapsed wallclock time %f'%(t2-t1))
        print('Total work delta rusage:', ru2-ru1)
    
    f = open('overall.txt', 'w')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\n' % (t1, t2, 0, 0, (ru2-ru1)[0], (ru2-ru1)[1]))
    f.close()

    return Backprojection

def ForwardProjectACC(hMatrix, realspace, planes=None, progress=tqdm, logPrint=True, numjobs=util.PhysicalCoreCount(), projector=proj.Projector_allC()):
    assert(len(realspace.shape) == 4)  # We only now support batch jobs (although batch size [dimension 1] can of course be 1)
    if planes is None:
        planes = range(hMatrix.numZ)
    if progress is None:
        progress = util.noProgressBar

    ru1 = util.cpuTime('both')
    t1 = time.time()
    TOTALprojection = projector.ForwardProjectACC(hMatrix, realspace, planes, progress, logPrint, numjobs)
    assert(TOTALprojection.dtype == np.float32)   # Keep an eye out for any reversion to double-precision
    ru2 = util.cpuTime('both')
    t2 = time.time()

    # Print out some diagnostics
    if logPrint:
        print('Forward project: elapsed wallclock time %f'%(t2-t1))
        print('Total work delta rusage:', ru2-ru1)
    
    return TOTALprojection

def DeconvRL(hMatrix, Htf, maxIter, Xguess, logPrint=True, numjobs=util.PhysicalCoreCount(), projector=proj.Projector_allC(), im=None, progress=tqdm, prevedel=True, trustMask=None):
    '''
        Htf       ndarray  The *initial* backprojection of the camera image
        Xguess    ndarray  The initial guess for the object
        prevedel  bool     Use the variation of the Richardson-Lucy algorithm used in the code associated with Prevedel et al (2014).
                            This is subtly different from the standard Richardson-Lucy algorithm (and I'm not sure why they have done it differently).
                            The flag defaults to True for pixel-by-pixel compatibility with the output from Prevedel et al's code.
                            Note that if set to False it is important to also use trustMask to avoid background light issues leading to major native focal plane artefacts
        trustMask ndarray  Mask of pixels in the camera image that should be used in the reconstruction (only applicable when prevedel=False).
                            See note "reconstruction methods.md"
    '''
    ru1 = util.cpuTime('both')
    t1 = time.time()
    proj.LogMemory("Starting DeconvRL")
    if Htf is None:
        # Caller has not provided the initial backprojection, but has provided the camera image itself
        assert(im is not None)
        if (prevedel) or (Xguess is None):
            # We will need to know the backprojection of the image
            Htf = BackwardProjectACC(hMatrix, im, progress=None, projector=projector)
    else:
        # Caller has provided the initial backprojection (and so we don't expect them to provide the image)
        if projector.storeVolumesOnGPU:
            Htf = projector.asnative(Htf)
    proj.LogMemoryDetailed("after Htf")
    if Xguess is None:
        # Caller has not provided the initial guess - we will use the backprojection as the initial guess
        Xguess = Htf.copy()
    else:
        # Caller has provided initial guess
        if projector.storeVolumesOnGPU:
            Xguess = projector.asnative(Xguess)
    proj.LogMemoryDetailed("after Xguess")

    if trustMask is not None:
        wh = np.where(trustMask == 0)
        for _im in im:
            assert(np.sum(_im[wh]) == 0)   # Caller must ensure image=0 in untrusted regions
        trustBack = BackwardProjectACC(hMatrix, trustMask[np.newaxis], progress=None, projector=projector)
        
    def RLUpdateFuncPrevedel(Xguess, Htf, HXguessBack, cc=slice(None)):
        assert(trustMask is None)
        projector.assertNative(HXguessBack)
        proj.LogMemoryDetailed("RLUpdateFunc starting for {0}".format(cc))
        if isinstance(HXguessBack, np.ndarray):
            # We are working on the CPU
            errorBack = np.divide(Htf[cc], np.asarray(HXguessBack), out=HXguessBack)
            proj.LogMemoryDetailed("RLUpdateFunc after errorback")
            
            del HXguessBack   # Effectively this has been destroyed - storage is now used for errorBack
            Xguess[cc] *= errorBack
            proj.LogMemoryDetailed("RLUpdateFunc after Xguess multiplication")
            del errorBack
            Xguess[cc][np.where(np.isnan(Xguess[cc]))] = 0
            proj.LogMemoryDetailed("RLUpdateFunc after nan-masking")
        else:
            # We are presumably working with a GPU.
            #
            # Performance notes re nan masking:
            # 1. My cp.where construction is marginally faster than cp.nan_to_num(copy=False), at least when we don't have NaNs
            # 2. If I run a profiler, this appears to take almost 25% of the total run time.
            #    However, eliminating it doesn't speed things up, so I think this is just acting as a checkpoint for asynchronous GPU work.
            if projector.storeVolumesOnGPU:
                # All calculations and results stay on the GPU
                errorBack = projector.nativeDivide(Htf[cc], HXguessBack, out=HXguessBack)
                del HXguessBack   # Effectively this has been destroyed - storage is now used for errorBack
                Xguess[cc] *= errorBack
                del errorBack
                Xguess[cc][cp.where(cp.isnan(Xguess[cc]))] = 0
            else:
                # Intermediate calculations are on the GPU, but then we bring the final result back to the CPU
                errorBack = cp.divide(cp.asarray(Htf[cc]), HXguessBack, out=HXguessBack)
                del HXguessBack   # Effectively this has been destroyed - storage is now used for errorBack
                mul = cp.asarray(Xguess[cc]) * errorBack
                del errorBack
                mul[cp.where(cp.isnan(mul))] = 0
                Xguess[cc] = mul.get()
        proj.LogMemoryDetailed("RLUpdateFunc complete for {0}".format(cc))

    def RLUpdateFuncStandard(Xguess, errorBack, cc=slice(None)):
        projector.assertNative(errorBack)
        proj.LogMemoryDetailed("RLUpdateFunc starting for {0}".format(cc))
        if isinstance(errorBack, np.ndarray):
            # We are working on the CPU
            Xguess[cc] *= errorBack
            if (trustMask is not None):
                # Note that trustBack[cc] is 1xYxX whereas Xguess[cc] is  NxYxX.
                # Nevertheless broadcasting rules should ensure the operation acts on every volume in the batch
                Xguess[cc] /= trustBack[cc]
            proj.LogMemoryDetailed("RLUpdateFunc after Xguess multiplication")
            del errorBack
            Xguess[cc][np.where(np.isnan(Xguess[cc]))] = 0
            proj.LogMemoryDetailed("RLUpdateFunc after nan-masking")
        else:
            assert(trustMask is None) # Not yet implemented for GPU (shouldn't be hard...)
            # We are presumably working with a GPU.
            #
            # Performance notes re nan masking:
            # 1. My cp.where construction is marginally faster than cp.nan_to_num(copy=False), at least when we don't have NaNs
            # 2. If I run a profiler, this appears to take almost 25% of the total run time.
            #    However, eliminating it doesn't speed things up, so I think this is just acting as a checkpoint for asynchronous GPU work.
            if projector.storeVolumesOnGPU:
                # All calculations and results stay on the GPU
                Xguess[cc] *= errorBack
                del errorBack
                Xguess[cc][cp.where(cp.isnan(Xguess[cc]))] = 0
            else:
                # Intermediate calculations are on the GPU, but Xguess lives on the CPU
                # It therefore makes most sense to do the multiplication there too
                errorBack[cp.where(cp.isnan(errorBack))] = 0
                Xguess[cc] *= errorBack.get()
        proj.LogMemoryDetailed("RLUpdateFunc complete for {0}".format(cc))

    for i in progress(range(maxIter), desc='RL deconv'):
        proj.LogMemory("Start RL iter {0}/{1}".format(i+1, maxIter))
        t0 = time.time()
        HXguess = ForwardProjectACC(hMatrix, Xguess, numjobs=numjobs, progress=None, logPrint=logPrint, projector=projector)
        if prevedel:
            BackwardProjectACC(hMatrix, HXguess, numjobs=numjobs, progress=None, logPrint=logPrint, projector=projector, rlUpdateFunc=lambda x,cc:RLUpdateFuncPrevedel(Xguess, Htf, x, cc))
        else:
            error = im / HXguess
            BackwardProjectACC(hMatrix, error, numjobs=numjobs, progress=None, logPrint=logPrint, projector=projector, rlUpdateFunc=lambda x,cc:RLUpdateFuncStandard(Xguess, x, cc))
        proj.LogMemory("after Forward+BackwardProjectACC", True)
        ttime = time.time() - t0
        #print('iter %d/%d took %.1f secs' % (i+1, maxIter, ttime))
    proj.LogMemory("RL iterations complete", True)
    ru2 = util.cpuTime('both')
    t2 = time.time()
    if logPrint:
        print('Deconvolution elapsed wallclock time %f, rusage %f' % (t2-t1, np.sum(ru2-ru1)))
    return projector.asnumpy(Xguess)


def main(argv, projectorClass=proj.Projector_allC, maxiter=8, numParallel=32):
    #########################################################################
    # Test code for deconvolution
    #########################################################################

    if (len(argv) == 1):
        # No arguments passed
        print('NO ARGUMENTS PASSED. You should pass parameters to this script if you want to execute it directly to run self-tests (see script for what options are available)')
        
    print('Reminder: this test code will use a matrix that is not fully normalised, to reproduce results from original Matlab code')
    # Load the input image and PSF matrix
    inputImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')[np.newaxis].copy()
    # Note that the PSF matrix we are using here is not normalised in the way I believe it should be.
    # This is purely intended to replicate the previous results obtained by Prevedel's code.
    hMatrix = psfmatrix.LoadMatrix('PSFmatrix/reducedPSFmatrix_M40NA0.95MLPitch150fml3000from-24to0zspacing3Nnum15lambda520n1.mat', createPSF=True)
    print('** Tests will run with projector type: {0} **'.format(projectorClass().name))
    deconvolvedResult = None
    testOutcomes = np.zeros((2))
    
    projector = projectorClass()
    if ('cacheFH' in argv):
        # Enable caching of F(H).
        # This is fine on the problem size in this test script, but will require large amounts of memory for some large problems.
        print('** Caching of F(H) has been enabled **')
        projector.cacheFH = True
    
    if ('basic' in argv) or ('full' in argv) or ('full32' in argv):
        print('== Running basic (single-threaded backprojection) ==')
	    # Run my back-projection code (single-threaded) on a cropped version of Prevedel's data
        Htf = BackwardProjectACC(hMatrix, inputImage, planes=None, numjobs=1, progress=None, logPrint=False, projector=projector)
        definitive = tifffile.imread('Data/03_Reconstructed/exampleData/definitive_worm_crop_X15_backproject.tif')[:,np.newaxis]
        testOutcomes += util.CheckComparison(definitive, Htf*10, 1.0, 'Compare against matlab result')

    if 'prime' in argv:
        # Run a single RL iteration to prime things (my caches, GPU FFT plans, etc) for subsequent runs (since we may be interested in timing/profiling)
        print('== Priming caches with a single RL iteration ==')
        deconvolvedResult = DeconvRL(hMatrix, Htf, maxIter=1, Xguess=Htf.copy(), logPrint=False, projector=projector)

    if 'full' in argv:
        # Run my full Richardson-Lucy code on a cropped version of Prevedel's data
        # Note that the back-projection (i.e. the 'basic' branch, above) must run first, since we use that as our initial guess
        print('== Running full RL deconvolution ==')
        print('Problem size {0}={1}MB'.format(Htf.shape, Htf.size*Htf.itemsize/1e6))
        deconvolvedResult = DeconvRL(hMatrix, Htf, maxIter=maxiter, Xguess=Htf.copy(), logPrint=False, projector=projector)
        # Note that this is the file generated by the Matlab code, but after I have fixed the 'max' bug at z=0
        definitive = tifffile.imread('Data/03_Reconstructed/exampleData/definitive_worm_crop_X15_iter8.tif')[:,np.newaxis]
        testOutcomes += util.CheckComparison(definitive, deconvolvedResult*1e3, 1.0, 'Compare against matlab result')

    if 'profile32' in argv:
        # This test does not currently check for correctness...
        print('== Running full32 (full RL deconvolution on {0} parallel timepoints) =='.format(numParallel))
        Htf_x32 = np.tile(Htf[:,np.newaxis,:,:], (1,numParallel,1,1))
        print('Problem size {0}={1}MB'.format(Htf_x32.shape, Htf_x32.size*Htf_x32.itemsize/1e6))
        pr = cProfile.Profile()
        pr.enable()
        deconvolvedResult = DeconvRL(hMatrix, Htf_x32, maxIter=maxiter, Xguess=Htf_x32.copy(), logPrint=False, projector=projector)
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)
    
    if 'parallel' in argv:
        # Code to test simultaneous deconvolution of an image pair
        # This does not test overall correctness, but it runs with two different (albeit proportional)
        # images and checks that the result matches the result for two totally independent calls on a single array.
        print('Testing image pair deconvolution:')
        candidate = np.tile(inputImage, (2,1,1))
        candidate[1] *= 1.4
        planesToUse = None   # Use all planes
        if planesToUse is None:
            numPlanesToUse = hMatrix.numZ
        else:
            numPlanesToUse = len(planesToUse)

        print('Running (%d planes x2)'%numPlanesToUse)
        t1 = time.time()
        temp = BackwardProjectACC(hMatrix, candidate, planes=None, numjobs=1, logPrint=False, projector=projector)
        dualRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, logPrint=False, projector=projector)
        print('New method took', time.time()-t1)

        # Run for the individual images, and check we get the same result as with the dual round-trip
        # Note that these tests yield very large values in the result array,
        # so a comparison with a max error tolerance of 1.0 is reasonable.
        temp = BackwardProjectACC(hMatrix, candidate[0][np.newaxis], planes=None, numjobs=1, logPrint=False, projector=projector)
        firstRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, numjobs=1, logPrint=False, projector=projector)
        testOutcomes += util.CheckComparison(firstRoundtrip, dualRoundtrip[0], 1.0, 'Compare single and dual deconvolution #1', '<<1')

        temp = BackwardProjectACC(hMatrix, candidate[1][np.newaxis], planes=None, numjobs=1, logPrint=False, projector=projector)
        secondRoundtrip = ForwardProjectACC(hMatrix, temp, planes=None, numjobs=1, logPrint=False, projector=projector)
        testOutcomes += util.CheckComparison(secondRoundtrip, dualRoundtrip[1], 1.0, 'Compare single and dual deconvolution #2', '<<1')

    if 'parallel-threading' in argv:
        # Run to test parallelization
        # Note that I think the reason I do not currently have code here to check for correctness is that
        # there are small numerical variations depending on how the parallelism occurs in each run.
        # I have had to relax the condition for the 1 vs 3 or 6 thread comparison, but I think that is still not a cause for concern.
        print('Testing multithreaded execution:')
        result1 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=1, logPrint=False, projector=projector)
        result3 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=3, logPrint=False, projector=projector)
        testOutcomes += util.CheckComparison(result1, result3, 1e-3, 'Compare 1 and 3 threads', '<1')
        result6 = BackwardProjectACC(hMatrix, inputImage, planes=[0], numjobs=6, logPrint=False, projector=projector)
        testOutcomes += util.CheckComparison(result1, result6, 1e-3, 'Compare 1 and 6 threads', '<1')

    print('Regression tests complete (passed %d/%d)' % (testOutcomes[0], testOutcomes[1]))
    return testOutcomes

try:
    import pycuda.driver as cuda
    import cupy as cp
    def PrintKeyGPUAttributes():
      # For some reason if this runs before everything else then I get an error (pycuda initialization error)
      # As a workaround, I just do a dummy cupy operation first, and that seems to solve the issue
      _ = cp.zeros(2)
      # Now print out the GPU details
      for devicenum in range(cuda.Device.count()):
        device=cuda.Device(devicenum)
        attrs=device.get_attributes()
        print(' {0} threads x {1} processors'.format(attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR], attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]))
        print(' Clock speed {0}GHz, mem speed {1}GHz x {2}B = {3:.2f}GB/s, L2 {4:.2f}MB'.format(attrs[cuda.device_attribute.CLOCK_RATE]*1e3/1e9, attrs[cuda.device_attribute.MEMORY_CLOCK_RATE]*1e3/1e9, attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]//8, attrs[cuda.device_attribute.MEMORY_CLOCK_RATE]*attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]/8e6, attrs[cuda.device_attribute.L2_CACHE_SIZE]/1e6))
        total = cuda.mem_get_info()[1]
        print(' Total GPU RAM {0:.2f}GB'.format(total/1e9))
    hasGPU = True
except ImportError:
    hasGPU = False

if __name__ == "__main__":
    # TODO: I should make this like lf_performance_analysis.py, where it accepts a sequence of commands.
    # That way I can fold in the gpu testing here too.
    main(sys.argv)
    if hasGPU:
        main(sys.argv, projectorClass=proj.Projector_gpuHelpers)
        PrintKeyGPUAttributes()
