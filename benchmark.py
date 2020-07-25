# A little script to generate some timing benchmarks for scenarios I am interested in

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import tifffile
import sys, time, os, csv, warnings, glob
import cProfile, pstats
from jutils import tqdm_alias as tqdm

import psfmatrix, lfimage
import projector, lfdeconv
import special_fftconvolve as special
import jutils as util
import py_light_field as plf
import lf_performance_analysis as perf


def main(testThreadScaling=False):
    # Timing measurements for a 'typical scenario and for my PIV scenario
    results = []
    
    if testThreadScaling:
        # Enable this for more exhaustive benchmarking
        jobsToTest = [1]
        while (jobsToTest[-1]*2 <= util.PhysicalCoreCount()):
            jobsToTest.append(jobsToTest[-1]*2)
    else:
        jobsToTest = [util.PhysicalCoreCount()]

    if False:
        # Benchmark scenario for a large-ish image
        for numJobs in jobsToTest:
            # Run the test, saving thread performance information.
            # Note that the problem is too large to consider caching FH
            plf.SetThreadFileName("thread-benchmarks/threads_new_%d.txt" % numJobs)
            results.append(perf.main(['cpu', 'prime-cache', 'new-batch'], numJobs=numJobs)[1:])
        plf.SetThreadFileName("")

    if True:
        # Square image, PIV scenario, with and without caching FH
        for numJobs in jobsToTest:
            # Run the test, saving thread performance information
            plf.SetThreadFileName("thread-benchmarks/threads_square_%d.txt" % numJobs)
            results.append(perf.main(['cpu', 'piv-image', 'piv-matrix', 'prime-cache', 'new-batch', 'new-batch'],
                                     batchSize=2, numJobs=numJobs)[1:])
            plf.SetThreadFileName("thread-benchmarks/threads_square_cached2_%d.txt" % numJobs) # The 2 recognises that we now cache FFT and transpose
                #results.append(perf.main(['cpu', 'piv-image', 'piv-matrix', 'cache-FH', 'prime-cache', 'new-batch', 'new-batch'],
                #                    batchSize=2, numJobs=numJobs)[1:])
        plf.SetThreadFileName("")

    print("\033[1;32mBenchmarking results:\033[0m {0}".format(results))
    return results

if __name__ == "__main__":
    main(testThreadScaling=False)
