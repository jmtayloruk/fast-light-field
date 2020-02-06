# This file contains some snippets I use for investigating the performance of my light-field deconvolution code
import os, sys, time
import csv, glob
import cProfile, pstats
import numpy as np
#import matplotlib.pyplot as plt   # For some reason this is crashing on my Mac Pro. This has persisted across a restart, not sure what's going on...!?
# The following is a workaround until I figure out what the root cause is:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import jutils as util
import lfdeconv, psfmatrix, lfimage, projector

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

if __name__ == "__main__":
    #########################################################################
    # Test code for performance measurement
    #########################################################################

    matPath = 'PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.0.mat'
    if ('no-raw' in sys.argv):
        print('Warning - we will not load the raw H matrices. Not all command line options will be usable - some (old code) will hit errors')
    else:
        (_H, _Ht, _CAindex, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = psfmatrix.LoadRawMatrixData(matPath)
    if ('smaller-matrix' in sys.argv):
        # This matrix is small enough to allow matrix caching (for backprojection only) with 8GB of RAM available
        hMatrix = psfmatrix.LoadMatrix(matPath, numZ=16)
    else:
        hMatrix = psfmatrix.LoadMatrix(matPath)
    inputImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')
    inputImage_x30 = np.tile(inputImage[np.newaxis,:,:], (30,1,1))
    inputImage_x10 = inputImage_x30[0:10]

    if ('parallel-scaling' in sys.argv):
        # Investigate performance for different numbers of parallel threads
        for numJobsForTesting in range(1,13):
            print('Running with {0} parallel threads:'.format(numJobsForTesting))
            hMatrix.ClearCache()
            ru1 = util.cpuTime('both')
            temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=None, numjobs=numJobsForTesting)
            ru2 = util.cpuTime('both')
            print('overall delta rusage:', ru2-ru1)
            AnalyzeTestResults(numJobsForTesting)

    if ('analyze-saved-data' in sys.argv):
        # Plot some analysis based on previously-acquired performance statistics
        plt.title('Dummy work on empty arrays')
        AnalyzeTestResults2('stats-dummy.txt')
        plt.title('Real work')
        AnalyzeTestResults2('stats-realwork.txt')
        plt.title('Smaller memory footprint - no improvement')
        AnalyzeTestResults2('stats-no-H.txt')
        plt.title('New code')
        AnalyzeTestResults2('stats-new-code.txt')

    planesToProcess = None    # Can be set differently to speed things up for shorter investigations

    if ('time-old' in sys.argv):
        # For fun, see how long the original code and my new code takes
        t0 = time.time()
        result1 = projector.BackwardProjectACC_old(_Ht, inputImage, _CAindex, planes=planesToProcess)
        print('Original code took %f'%(time.time()-t0))
        
    if ('time-new' in sys.argv):
        # Run my code (single-threaded)
        t0 = time.time()
        result2 = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, numjobs=1)
        print('New code (single threaded) took %f'%(time.time()-t0))

        try:
            util.CheckComparison(result1, result2, 1.0, 'Compare results from new and old code')
        except:
            print('Old code was probably not run, so we cannot compare results')

        # Run my code multi-threaded
        t0 = time.time()
        result3 = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess)
        print('New code (multithreaded) took %f'%(time.time()-t0))

        util.CheckComparison(result2, result3, 1.0, 'Compare single- and multi-threaded')

    if ('profile-old' in sys.argv):
        # Profile old code (single-threaded)
        ru1 = util.cpuTime('both')
        myStats = cProfile.run('temp = projector.BackwardProjectACC_old(_Ht, inputImage, _CAindex, planes=planesToProcess)', 'mystats')
        ru2 = util.cpuTime('both')
        print('overall delta rusage:', ru2-ru1)
        p = pstats.Stats('mystats')
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new' in sys.argv):
        # Profile my code (single-threaded)
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        print('hMatrix hits {0} misses {1}. Cache size {2}GB'.format(hMatrix.cacheHits, hMatrix.cacheMisses, hMatrix.cacheSize/1e9))
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new2' in sys.argv):
        # Profile the same code code for a second run, now that the matrices ought to be cached. (single-threaded)
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        print('hMatrix hits {0} misses {1}. Cache size {2}GB'.format(hMatrix.cacheHits, hMatrix.cacheMisses, hMatrix.cacheSize/1e9))
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-piv' in sys.argv):
        # Profile my code (single-threaded) in the sort of scenario I would expect to run it in for my PIV experiments
        tempInputImage = np.zeros((2,hMatrix.Nnum(0)*20,hMatrix.Nnum(0)*20)).astype('float32')
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, tempInputImage, planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-batch-prime-cache' in sys.argv):
        # Do a single-image run to fill the cache, to observe the speedup when we run subsequent code
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage_x10[0], planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        print('hMatrix hits {0} misses {1}. Cache size {2}GB'.format(hMatrix.cacheHits, hMatrix.cacheMisses, hMatrix.cacheSize/1e9))
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-batch' in sys.argv):
        # Profile my code (single-threaded) in the sort of scenario I would expect to run it in when batch-processing video
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage_x30, planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        print('hMatrix hits {0} misses {1}. Cache size {2}GB'.format(hMatrix.cacheHits, hMatrix.cacheMisses, hMatrix.cacheSize/1e9))
        p.strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-batch2' in sys.argv):
        # Repeat again (with cache already primed)
        myStats = cProfile.run('temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage_x30, planes=planesToProcess, numjobs=1)', 'mystats')
        p = pstats.Stats('mystats')
        print('hMatrix hits {0} misses {1}. Cache size {2}GB'.format(hMatrix.cacheHits, hMatrix.cacheMisses, hMatrix.cacheSize/1e9))
        p.strip_dirs().sort_stats('cumulative').print_stats(40)
