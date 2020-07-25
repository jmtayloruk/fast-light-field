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


# Timing measurements for a 'typical scenario and for my PIV scenario

if cacheFH:
    cachedString = 'cached2_' # The 2 recognises that we now cache FFT and transpose
    print('Benchmarking with cached F(H)')
else:
    cachedString = ''
    print('Benchmarking without caching F(H)')

if True:
    for numJobs in [8, 4, 2, 1]:
        # Run the test, saving thread performance information.
        # Note that we ignore cacheFH, because the problem is simply too large to cache!
        plf.SetThreadFileName("threads_new_%d.txt" % numJobs)
        perf.main(['profile-prime-cache', 'profile-new-batch'],
                  inputImage=None, numJobs=numJobs, printProfileOutput=False)
    plf.SetThreadFileName("")

if True:
    inputImage = np.zeros((19*19,19*19), dtype='float32')
    for numJobs in [8, 4, 2, 1]:
        # Run the test, saving thread performance information
        plf.SetThreadFileName("threads_square_%s%d.txt" % (cachedString, numJobs))
        perf.main(['profile-prime-cache', 'profile-new-batch'],
                  matPath='PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-56to56zspacing4Nnum19lambda520n1.33.mat',
                  inputImage=inputImage, batchSize=2, numJobs=numJobs, printProfileOutput=False, cacheFH=cacheFH)
    plf.SetThreadFileName("")