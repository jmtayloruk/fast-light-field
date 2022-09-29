# This file contains some snippets I use for investigating the performance of my light-field deconvolution code
import os, sys, time, warnings
import csv, glob
import cProfile, pstats
import numpy as np
import matplotlib.pyplot as plt

import jutils as util
import lfdeconv, psfmatrix, lfimage
import projector as proj
import py_light_field as plf

def AnalyzeTestResults(numJobsUsed):
    # Long function which analyses the data from the run that just happened
    # (data stored in 'overall.txt'), accumulates some summary statistics on it,
    # and appends the results to the 'stats.txt' file.
    # Clearly this function could do with more commenting to explain what's going on!!
    #
    # Note also that deconvolution-performance-analysis.ipynb does more detailed analysis
    # of thread performance of my C code.
    
    with open('overall.txt') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            pass
    startTime = float(row[0])
    endTime = float(row[1])
    userTime = float(row[4])
    sysTime = float(row[5])
    print('Elapsed time', endTime-startTime)
    print('User cpu time', userTime)
    print('System cpu time', sysTime)

    with open('stats.txt', 'a') as f:
        f.write('%f\t%f\t%f\t%f\n' % (numJobsUsed, endTime-startTime, userTime, sysTime))


def decomment(csvfile):
    # Strip comments from a CSV file
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def AnalyzeTestResults2(statsFilePath):
    # Plots graphs from a file that contains summary stats from multiple runs,
    # to understand how performance scales with number of threads
    rows = []
    with open(statsFilePath) as f:
        csv_reader = csv.reader(decomment(f), delimiter='\t')
        for row in csv_reader:
            rows.append(row)
    rows = np.array(rows).astype(np.float).transpose()

    plt.plot(rows[0], rows[2]/rows[2,0], label='work time')
    plt.plot(rows[0], np.sum(rows[5:8], axis=0)/(rows[0]*rows[1]), label='dead time')
    plt.plot(rows[0], rows[5]/(rows[0]*rows[1]), label='dead start')
    plt.plot(rows[0], rows[1]/(rows[1,0]/rows[0]), label='runtime excess')
    plt.ylim(0,2.5)
    plt.legend(loc=2)
    plt.show()

def SetNumJobs(nj):
    plf.SetNumThreadsToUse(nj)
    nj = plf.GetNumThreadsToUse()  # Read the value back, so we know the true number even if "0" was specified
    print('Will use {0} parallel threads'.format(nj))
    plf.SetThreadFileName('threads{0}.txt'.format(nj))
    return nj

def IsNumericArg(arg, stem):
    return arg.startswith(stem) and arg[len(stem):].isnumeric()

def SetImage(img, batchSize):
    return img[np.newaxis], np.tile(img[np.newaxis,:,:], (batchSize,1,1))

def main(argv, defaultImage=None, batchSize=30, matPath=None, planesToProcess=None, numJobs=plf.GetNumThreadsToUse(), projectorClass=proj.Projector_allC):
    #########################################################################
    # Test code for performance measurement
    #########################################################################
    if matPath is None:
        matPath = 'PSFmatrix/fdnormPSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.mat'

    if defaultImage is None:
        defaultImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')

    if not 'prime-cache' in argv:
        print('NOTE: cache is not being primed - timings for early runs will include FFT planning time')

    numJobs = SetNumJobs(numJobs)
    iterations = 4
    
    # Default to cpu mode unless/until explicitly specified otherwise
    args = ['no-cache-FH', 'volumes-on-cpu', 'cpu', 'back-project', 'no-profile', 'default-matrix', 'default-image'] + argv
    projector = None
    results = []

    for arg in args:
        RunThis = None
        if arg == 'gpu':
            projectorClass = proj.Projector_gpuHelpers
            projector = projectorClass()
            projector.cacheFH = cacheFH
            projector.storeVolumesOnGPU = storeVolumesOnGPU
        elif arg == 'cpu':
            projectorClass = proj.Projector_allC
            projector = projectorClass()
            projector.cacheFH = cacheFH
        elif arg == 'back-project':
            print('Benchmarking backprojection operator')
            backProjectOnly = True
        elif arg == 'deconv':
            print('Benchmarking full deconvolution')
            backProjectOnly = False
        elif arg == 'profile':
            profile = True
        elif arg == 'no-profile':
            profile = False
        elif arg == 'volumes-on-gpu':
            storeVolumesOnGPU = True
            if projector is not None:
                projector.storeVolumesOnGPU = storeVolumesOnGPU
        elif arg == 'volumes-on-cpu':
            storeVolumesOnGPU = False
            if projector is not None:
                projector.storeVolumesOnGPU = storeVolumesOnGPU
        elif IsNumericArg(arg, 'j'):
            numJobs = SetNumJobs(int(arg[1:]))
        elif IsNumericArg(arg, 'x'):
            batchSize = int(arg[1:])
            inputImage,inputImageBatch = SetImage(inputImage[0], batchSize)
        elif IsNumericArg(arg, 'i'):
            iterations = int(arg[1:])

        elif arg == 'default-matrix':
            hMatrix = psfmatrix.LoadMatrix(matPath)
        elif arg == 'piv-matrix':
            hMatrix = psfmatrix.LoadMatrix('PSFmatrix/fdnormPSFmatrix_M22.2NA0.5MLPitch125fml3125from-56to56zspacing4Nnum19lambda520n1.33.mat')
        elif arg == 'olaf-matrix':
            hMatrix = psfmatrix.LoadMatrix('PSFmatrix/fdnormPSFmatrix_M22.222NA0.5MLPitch125fml3125from-60to60zspacing5Nnum19lambda520n1.33.mat')
        elif arg == 'cache-FH':
            cacheFH = True
            if projector is not None:
                projector.cacheFH = cacheFH
        elif arg == 'no-cache-FH':
            cacheFH = False
            if projector is not None:
                projector.cacheFH = cacheFH

        elif arg == 'default-image':
            inputImage,inputImageBatch = SetImage(defaultImage, batchSize)
        elif arg == 'piv-image':
            inputImage,inputImageBatch = SetImage(np.zeros((19*19,19*19), dtype='float32'), batchSize)
        elif arg == 'smaller-image':
            inputImage,inputImageBatch = SetImage(inputImage[0,0:20*15,0:15*15], batchSize)
        elif arg == 'olaf-image':
            # This is just a dummy image with the same dimensions as Nils's test image for the Olaf code
            # Note that the image dimensions are deliberately the wrong way round, since that seems to be what we are given from Matlab for this dataset
            inputImage,inputImageBatch = SetImage(np.zeros((1463, 1273), dtype='float32'), batchSize)

        elif arg == 'parallel-scaling':
            # Investigate performance for different numbers of parallel CPU threads
            if (projectorClass is proj.Projector_gpuHelpers):
                print('NOTE: will not investigate parallel scaling on GPU')
            else:
                for _numJobs in range(1,numJobs+1):
                    print('Profiling with {0} parallel threads:'.format(_numJobs))
                    hMatrix.ClearCache()
                    ru1 = util.cpuTime('both')
                    if backProjectOnly:
                        temp = lfdeconv.BackwardProjectACC(hMatrix, inputImageBatch, planes=planesToProcess, progress=util.noProgressBar, numjobs=_numJobs, projector=projector, logPrint=False)
                    else:
                        temp = lfdeconv.DeconvRL(hMatrix, Htf=None, maxIter=iterations, Xguess=None, im=inputImageBatch, logPrint=False, numjobs=_numJobs, projector=projector)
                    ru2 = util.cpuTime('both')
                    print('overall delta rusage:', ru2-ru1)
                    AnalyzeTestResults(_numJobs)
        elif arg == 'analyze-saved-data':
            # Plot some analysis based on previously-acquired performance statistics
            plt.title('Dummy work on empty arrays')
            AnalyzeTestResults2('stats-dummy.txt')
            plt.title('Real work')
            AnalyzeTestResults2('stats-realwork.txt')
            plt.title('Smaller memory footprint - no improvement')
            AnalyzeTestResults2('stats-no-H.txt')
            plt.title('New code')
            AnalyzeTestResults2('stats-new-code.txt')

        elif arg == 'old':
            # Run old code (single-threaded)
            print('Benchmarking old slow code (single image)')
            matBase,matFile = os.path.split(matPath)
            oldMatPath = matPath
            try:
                i = matFile.index('PSFmatrix')
                if i > 1:
                    print('This looks like a matrix file with a prefix. The old code needs a full .mat file, not just one quadrant')
                    print('We will try loading counterpart file {0}'.format(matFile[i:]))
                    oldMatPath = os.path.join(matBase, matFile[i:])
                else:
                    assert(i == 0)
            except:
                print('Filename doesnt follow the expected format. We will try and load it anyway...')
            (_H, _Ht, _CAindex, _, _, _, _) = psfmatrix.LoadRawMatrixData(oldMatPath)
            def RunThis():
                return proj.BackwardProjectACC_old(_Ht, inputImage, _CAindex, planes=planesToProcess)
        elif arg == 'prime-cache':
            # Do a single-image run to take care of one-off work such as FFT planning,
            # so that is not included in the timings of subsequent tests
            # TODO: I think this will not fully prime everything when running on the GPU,
            # because the self-calibrating block sizes will vary depending on the number of images
            print('Priming cache')
            def RunThis():
                result = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, progress=util.noProgressBar, projector=projector, logPrint=False)
                return result
        elif arg == 'new':
            # Run my new fast code
            print('Benchmarking new fast code (single image)')
            def RunThis():
                if backProjectOnly:
                    return lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
                else:
                    return lfdeconv.DeconvRL(hMatrix, Htf=None, maxIter=iterations, Xguess=None, im=inputImage, logPrint=False, numjobs=numJobs, projector=projector)
        elif arg == 'new-piv':
            # Run my code in the sort of scenario I would expect to run it in for my PIV experiments.
            print('Benchmarking new fast code (PIV scenario)')
            def RunThis():
                return lfdeconv.BackwardProjectACC(hMatrix, inputImageBatch[0:2], planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
        elif arg == 'new-batch':
            # Run my code in the sort of scenario I would expect to run it in when batch-processing video
            print('Benchmarking new fast code (batch scenario)')
            def RunThis():
                if backProjectOnly:
                    return lfdeconv.BackwardProjectACC(hMatrix, inputImageBatch, planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
                else:
                    return lfdeconv.DeconvRL(hMatrix, Htf=None, maxIter=iterations, Xguess=None, im=inputImageBatch, progress=util.noProgressBar, logPrint=False, numjobs=numJobs, projector=projector)
        else:
            print('UNRECOGNISED ARGUMENT:', arg)
            
        if RunThis is not None:
            print('Running with batch image shape {0}, batch x{1}'.format(inputImageBatch.shape, batchSize))
            if profile:
                pr = cProfile.Profile()
                pr.enable()
            ru1 = util.cpuTime('both')
            t1 = time.time()
            temp = RunThis()
            t2 = time.time()
            ru2 = util.cpuTime('both')
            if profile:
                pr.disable()
                pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)
            print(' time: %.2f. overall delta rusage:' % (t2-t1), (ru2-ru1))
            results.append(t2-t1)
    return results

if __name__ == "__main__":
    main(sys.argv[1:])
