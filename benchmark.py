# A little script to generate some timing benchmarks for scenarios I am interested in

import numpy as np
import multiprocessing
import tifffile
import sys, time, os, csv, warnings, glob
import psutil
import cProfile, pstats
from jutils import tqdm_alias as tqdm

import jutils as util
import py_light_field as plf
import lf_performance_analysis as perf
import projector as proj
import lfdeconv

def main(testThreadScaling=False, prefix=[], prefix2=[]):
    # Timing measurements for a 'typical scenario and for my PIV scenario
    results = []
    
    if testThreadScaling:
        # Enable this for more exhaustive benchmarking
        jobsToTest = [1]
        while (jobsToTest[-1]*2 <= util.PhysicalCoreCount()):
            jobsToTest.append(jobsToTest[-1]*2)
    else:
        jobsToTest = [util.PhysicalCoreCount()]

    platforms = ['cpu']
    if proj.gpuAvailable:
        platforms.append('gpu')

    for plat in platforms:
        print('Platform: {0}'.format(plat))
        if plat == 'cpu':
            prime = ['prime-cache']
        else:
            # On the GPU there is adaptive selection of block sizes,
            # so the only way to prime is to run it for real
            prime = ['i1', 'new-batch', 'i4']

        if True:
            # Benchmark scenario for a large-ish image
            for numJobs in jobsToTest:
                # Run the test, saving thread performance information.
                # Note that the problem is too large to consider caching FH
                plf.SetThreadFileName("thread-benchmarks/threads_new_%d.txt" % numJobs)
                results.append(perf.main(prefix+[plat]+prime+prefix2+['new-batch'], numJobs=numJobs)[1:])
            plf.SetThreadFileName("")

        if True:
            # Square image, PIV scenario, with and without caching FH
            for numJobs in jobsToTest:
                # Run the tests, saving thread performance information.
                # We run each test several times to monitor for variability
                plf.SetThreadFileName("thread-benchmarks/threads_square_%d.txt" % numJobs)
                results.append(perf.main(prefix+[plat, 'piv-image', 'piv-matrix']+prime+prefix2+['new-batch'],
                                         batchSize=2, numJobs=numJobs)[1:])
                plf.SetThreadFileName("thread-benchmarks/threads_square_cached2_%d.txt" % numJobs) # The 2 recognises that we now cache FFT and transpose
                results.append(perf.main(prefix+[plat, 'piv-image', 'piv-matrix', 'cache-FH']+prime+prefix2+['new-batch'],
                                         batchSize=2, numJobs=numJobs)[1:])
            plf.SetThreadFileName("")

    print("\033[1;32mBenchmarking results:\033[0m {0}".format(results))
    print("\033[1;32mMachine specs:\033[0m physical_cpus:{0} logical_cpus:{1} {2}".format(psutil.cpu_count(logical=False), psutil.cpu_count(logical=True), psutil.cpu_freq()))
    if hasattr(lfdeconv, 'PrintKeyGPUAttributes'):
        print("\033[1;32mGPU specs:\033[0m")
        # Note that this function call will fail if run before I actually execute any GPU code
        # (see comment inside this function). But if the GPU is there, we will have used it by now.
        lfdeconv.PrintKeyGPUAttributes()

    return results

if __name__ == "__main__":
    main(testThreadScaling=False)
