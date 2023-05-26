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

def RunBenchmark(plat, prefix=[], prefix2=[], suffix=[], outputFilename='performance_analysis_output.tif'):
    '''
        Run benchmarking sequence to measure speed of reconstruction.
        Parameters:
          plat     str         'cpu' or 'gpu'
          prefix   list[str]   General specifiers that are applied at the start of the command sequence
          prefix2  list[str]   Specifiers that are only applied after the initial priming run(s), for the actual measured run
          suffix   list[str]   Any specifiers to be run after the main benchmarking run
        Returns:
          the timing result from the benchmark run
    '''
    # Timing measurements for a 'typical' scenario and for my PIV scenario
    results = []
    if plat == 'cpu':
        prime = ['prime-cache']
    else:
        # On the GPU there is adaptive selection of block sizes,
        # so the only way to prime is to run it for real
        prime = ['i1', 'new-batch', 'i4']

    # Note that the [1:] skips the time measurement for the intial cache-priming run (which will be slower)
    return perf.main(prefix+[plat]+prime+prefix2+['new-batch']+suffix, outputFilename=outputFilename)[1:]

def PrintBenchmarkSummary(results):
    print("\033[1;32mBenchmarking results:\033[0m {0}".format(results))
    print("\033[1;32mMachine specs:\033[0m physical_cpus:{0} logical_cpus:{1} {2}".format(psutil.cpu_count(logical=False), psutil.cpu_count(logical=True), psutil.cpu_freq()))
    if hasattr(lfdeconv, 'PrintKeyGPUAttributes'):
        print("\033[1;32mGPU specs:\033[0m")
        # Note that this function call will fail if run before I actually execute any GPU code
        # (see comment inside this function). But if the GPU is there, we will have used it by now.
        lfdeconv.PrintKeyGPUAttributes()

def RunSimpleBenchmarks():
    platforms = ['cpu']
    if proj.gpuAvailable:
        platforms.append('gpu')
    else:
        print('NOTE: will not benchmark GPU - no available GPU detected')

    results = []
    for plat in platforms:
        print(f'Platform: {plat}')
        
        if True:
            # Benchmark scenario for a large-ish image
            # Note that the problem is too large to consider caching FH
            results.append(RunBenchmark(plat))

        if True:
            # Square image, PIV scenario, with and without caching FH
            # We run each test several times to monitor for variability
            results.append(RunBenchmark(plat, prefix=['x2', 'piv-image', 'piv-matrix']))
            results.append(RunBenchmark(plat, prefix=['x2', 'piv-image', 'piv-matrix', 'cache-FH']))
    PrintBenchmarkSummary(results)

if __name__ == "__main__":
    RunSimpleBenchmarks()
