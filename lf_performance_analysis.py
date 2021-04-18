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
    with open('overall.txt') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            pass
    startTime = float(row[0])
    endTime = float(row[1])
    userTime = float(row[4])
    sysTime = float(row[5])

    rows = []
    for fn in glob.glob('perf_diags/*_*.txt'):
        with open(fn) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                pass
            rows.append(row)
    rows = np.array(rows).astype('float').transpose()
    firstPid = np.min(rows[0])
    rows[0] -= firstPid
    rows[1:3] -= startTime
    rows = rows[:,np.argsort(rows[1],kind='mergesort')]
    rows = rows[:,rows[0].argsort(kind='mergesort')]

    deadTimeStart = 0
    deadTimeMid = 0
    deadTimeEnd = 0
    threadWorkTime = 0
    thisThreadStartTime = 0
    longestThreadRunTime = 0
    longestThreadRunPid = -1
    latestStartTime = 0
    userTimeBreakdown = 0
    sysTimeBreakdown = 0
    for i in range(rows.shape[1]):
        pid = rows[0,i]
        t0 = rows[1,i]
        t1 = rows[2,i]
        userTimeBreakdown += rows[4,i]
        sysTimeBreakdown += rows[5,i]
        if (i == 0):
            deadTimeStart += t0
            thisThreadStartTime = t0
            latestStartTime = t0
        else:
            if (pid == rows[0,i-1]):
                deadTimeMid += t0 - rows[2,i-1]
            else:
                latestStartTime = max(latestStartTime, t0)
                thisThreadRunTime = rows[2,i-1]-thisThreadStartTime  # For previous pid
                if (thisThreadRunTime > longestThreadRunTime):
                    longestThreadRunPid = rows[0,i-1]
                    longestThreadRunTime = thisThreadRunTime
                thisThreadStartTime = t0
                deadTimeStart += t0
                deadTimeEnd += (endTime-startTime) - rows[2,i-1]
        threadWorkTime += t1-t0
        plt.plot([t0, t1], [pid, pid])
        plt.plot(t0, pid, 'x')
    thisThreadRunTime = t1-thisThreadStartTime
    if (thisThreadRunTime > longestThreadRunTime):
        longestThreadRunPid = pid
        longestThreadRunTime = thisThreadRunTime
    deadTimeEnd += (endTime-startTime) - rows[2,-1]
    print('Elapsed time', endTime-startTime)
    print('Longest thread run time', longestThreadRunTime, 'pid', int(longestThreadRunPid))
    print('Latest start time', latestStartTime)
    print('Thread work time', threadWorkTime)
    print('Dead time', deadTimeStart, deadTimeMid, deadTimeEnd)
    print(' Total', deadTimeStart + deadTimeMid + deadTimeEnd)
    print('User cpu time', userTime)
    print('System cpu time', sysTime)
    print('User cpu time for subset', userTimeBreakdown)
    print('System cpu time for subset', sysTimeBreakdown)

    with open('stats.txt', 'a') as f:
        f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (numJobsUsed, endTime-startTime, threadWorkTime, \
                        longestThreadRunTime, latestStartTime, deadTimeStart, deadTimeMid, deadTimeEnd, userTime, sysTime))

    plt.xlim(0, endTime-startTime)
    plt.ylim(-0.5,np.max(rows[0])+0.5)
    plt.show()
    
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
    nj = plf.GetNumThreadsToUse()  # Reread so we know the true number when "0" was specified
    print('Will use {0} parallel threads'.format(nj))
    plf.SetThreadFileName("threads{0}.txt".format(nj))
    return nj

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
    
    # Default to cpu mode unless/until explicitly specified otherwise
    args = ['no-cache-FH', 'cpu', 'no-profile', 'default-matrix', 'default-image'] + argv
    projector = None
    results = []

    for arg in args:
        RunThis = None
        if arg == 'gpu':
            projectorClass = proj.Projector_gpuHelpers
            projector = projectorClass()
            projector.cacheFH = cacheFH
        elif arg == 'cpu':
            projectorClass = proj.Projector_allC
            projector = projectorClass()
            projector.cacheFH = cacheFH
        elif arg == 'profile':
            profile = True
        elif arg == 'no-profile':
            profile = False
        elif arg == 'j0':
            numJobs = SetNumJobs(0)
        elif arg == 'j1':
            numJobs = SetNumJobs(1)
        elif arg == 'cache-FH':
            cacheFH = True
            if projector is not None:
                projector.cacheFH = cacheFH
        elif arg == 'no-cache-FH':
            cacheFH = False
            if projector is not None:
                projector.cacheFH = cacheFH
        elif arg == 'default-matrix':
            hMatrix = psfmatrix.LoadMatrix(matPath)
        elif arg == 'piv-matrix':
            hMatrix = psfmatrix.LoadMatrix('PSFmatrix/fdnormPSFmatrix_M22.2NA0.5MLPitch125fml3125from-56to56zspacing4Nnum19lambda520n1.33.mat')
        elif arg == 'default-image':
            inputImage = defaultImage
            inputImageBatch = np.tile(inputImage[np.newaxis,:,:], (batchSize,1,1))
        elif arg == 'piv-image':
            inputImage = np.zeros((19*19,19*19), dtype='float32')
            inputImageBatch = np.tile(inputImage[np.newaxis,:,:], (batchSize,1,1))
        elif arg == 'smaller-image':
            inputImage = inputImage[0:20*15,0:15*15]
            inputImageBatch = np.tile(inputImage[np.newaxis,:,:], (batchSize,1,1))
        elif arg == 'olaf-image':
            hMatrix = psfmatrix.LoadMatrix('/Users/jonny/Development/prevedel-matlab-light-field/PSFmatrix/PSFmatrix_M22.222NA0.5MLPitch125fml3125from-60to60zspacing5Nnum19lambda520n1.33.mat')
            # Note that the image dimensions are deliberately the wrong way round, since that seems to be what we are given from Matlab for this dataset
            inputImage = np.zeros((1, 1463, 1273), dtype='float32')
        elif arg == 'parallel-scaling':
            # Investigate performance for different numbers of parallel threads
            # Note that this is just for a single inputImage - I haven't used this code for a while.
            for _numJobs in range(1,numJobs):
                print('Profiling with {0} parallel threads:'.format(_numJobs))
                hMatrix.ClearCache()
                ru1 = util.cpuTime('both')
                temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=None, numjobs=_numJobs)
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
            (_H, _Ht, _CAindex, _, _, _, _) = psfmatrix.LoadRawMatrixData(matPath)
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
                return lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
        elif arg == 'new-piv':
            # Run my code in the sort of scenario I would expect to run it in for my PIV experiments.
            print('Benchmarking new fast code (PIV scenario)')
            def RunThis():
                return lfdeconv.BackwardProjectACC(hMatrix, inputImageBatch[0:2], planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
        elif arg == 'new-batch':
            # Run my code in the sort of scenario I would expect to run it in when batch-processing video
            print('Benchmarking new fast code (batch scenario)')
            def RunThis():
                return lfdeconv.BackwardProjectACC(hMatrix, inputImageBatch, planes=planesToProcess, progress=util.noProgressBar, numjobs=numJobs, projector=projector, logPrint=False)
        else:
            print('UNRECOGNISED:', arg)
            
        if RunThis is not None:
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
