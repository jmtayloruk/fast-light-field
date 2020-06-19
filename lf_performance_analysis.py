# This file contains some snippets I use for investigating the performance of my light-field deconvolution code
import os, sys, time
import csv, glob
import cProfile, pstats
import numpy as np
import matplotlib.pyplot as plt

import jutils as util
import lfdeconv, psfmatrix, lfimage, projector
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

def main(argv, planesToProcess=None, projectorClass=projector.Projector_allC):
    #########################################################################
    # Test code for performance measurement
    #########################################################################
    matPath = 'PSFmatrix/PSFmatrix_M40NA0.95MLPitch150fml3000from-13to0zspacing0.5Nnum15lambda520n1.0.mat'
    hMatrix = psfmatrix.LoadMatrix(matPath)
    inputImage = lfimage.LoadLightFieldTiff('Data/02_Rectified/exampleData/20131219WORM2_small_full_neg_X1_N15_cropped_uncompressed.tif')
    if ('smaller-image' in argv):
        inputImage = inputImage[0:20*15,0:15*15]
    if not 'profile-prime-cache' in argv:
        warnings.warn('Cache not primed - timings for new code will include FFT planning time')

    inputImage_x30 = np.tile(inputImage[np.newaxis,:,:], (30,1,1))
    inputImage_x10 = inputImage_x30[0:10]

    numJobsForTesting = 8    # But may want to edit this!
    print('== Running with {0} parallel threads =='.format(numJobsForTesting))

    if ('parallel-scaling' in argv):
        # Investigate performance for different numbers of parallel threads
        # Note that this is just for a single inputImage - I haven't used this code for a while.
        for numJobsForTesting in range(1,13):
            print('Running with {0} parallel threads:'.format(numJobsForTesting))
            hMatrix.ClearCache()
            ru1 = util.cpuTime('both')
            temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=None, numjobs=numJobsForTesting)
            ru2 = util.cpuTime('both')
            print('overall delta rusage:', ru2-ru1)
            AnalyzeTestResults(numJobsForTesting)

    if ('analyze-saved-data' in argv):
        # Plot some analysis based on previously-acquired performance statistics
        plt.title('Dummy work on empty arrays')
        AnalyzeTestResults2('stats-dummy.txt')
        plt.title('Real work')
        AnalyzeTestResults2('stats-realwork.txt')
        plt.title('Smaller memory footprint - no improvement')
        AnalyzeTestResults2('stats-no-H.txt')
        plt.title('New code')
        AnalyzeTestResults2('stats-new-code.txt')

    if ('profile-old' in argv):
        # Profile old code (single-threaded)
        (_H, _Ht, _CAindex, _, _, _, _) = psfmatrix.LoadRawMatrixData(matPath)
        pr = cProfile.Profile()
        pr.enable()
        ru1 = util.cpuTime('both')
        temp = projector.BackwardProjectACC_old(_Ht, inputImage, _CAindex, planes=planesToProcess)
        ru2 = util.cpuTime('both')
        print('overall delta rusage:', ru2-ru1)
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-prime-cache' in argv):
        # Do a single-image run to take care of one-off work such as FFT planning,
        # so that is not included in the timings of subsequent tests
        pr = cProfile.Profile()
        pr.enable()
        temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, projector=projectorClass())
        print('Cache has been primed')
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)
        
    if ('profile-new' in argv):
        # Profile my code (single-threaded)
        pr = cProfile.Profile()
        pr.enable()
        ru1 = util.cpuTime('both')
        temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage, planes=planesToProcess, numjobs=numJobsForTesting, projector=projectorClass())
        ru2 = util.cpuTime('both')
        print('overall delta rusage:', ru2-ru1)
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-piv' in argv):
        # Profile my code (single-threaded) in the sort of scenario I would expect to run it in for my PIV experiments.
        tempInputImage = np.zeros((2,hMatrix.Nnum(0)*20,hMatrix.Nnum(0)*20)).astype('float32')
        pr = cProfile.Profile()
        pr.enable()
        temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage_x10[0:2], planes=planesToProcess, numjobs=numJobsForTesting, projector=projectorClass())
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)

    if ('profile-new-batch' in argv):
        # Profile my code (single-threaded) in the sort of scenario I would expect to run it in when batch-processing video
        pr = cProfile.Profile()
        pr.enable()
        temp = lfdeconv.BackwardProjectACC(hMatrix, inputImage_x30, planes=planesToProcess, numjobs=numJobsForTesting, projector=projectorClass())
        pr.disable()
        pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(40)

if __name__ == "__main__":
    main(sys.argv)
