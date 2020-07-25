import numpy as np
import tifffile
import sys, time, os
import argparse
from jutils import tqdm_alias as tqdm
import lfdeconv, lfimage, psfmatrix
import projector as proj

def ProcessBatch(projector, hMatrix, timepoints, destDir, maxIter):
    # Ensure that the destination directory exists
    try:
        os.mkdir(destDir)
    except FileExistsError:
        pass

    # Load the input images
    inputImages = []
    for t in timepoints:
        im = lfimage.LoadLightFieldTiff(t)
        inputImages.append(im)
        if (len(inputImages) > 1) and (inputImages[0].shape != inputImages[-1].shape):
            print("ERROR: not all images have the same dimensions - cannot deconvolve these all together")
            exit(1)
    inputImages = np.array(inputImages)

    # Deconvolve this batch of camera images
    Htf = lfdeconv.BackwardProjectACC(hMatrix, inputImages, progress=None, logPrint=False, projector=projector)
    deconvolvedResult = lfdeconv.DeconvRL(hMatrix, Htf, maxIter=maxIter, Xguess=Htf.copy(), logPrint=False, projector=projector)

    # Save the images
    for i in range(len(timepoints)):
        destFilePath = '%s/%s_deconvolved.tif' % (destDir, os.path.splitext(os.path.basename(timepoints[i]))[0])
        tifffile.imsave(destFilePath, deconvolvedResult[:,i])


def main(argv, maxIter=8, numParallel=32):
    parser = argparse.ArgumentParser(description='Deconvolves a time series of (pre-rectified) light field microscopy images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if proj.gpuAvailable:
        gpuHelp = ' or gpu'
    else:
        gpuHelp = ' [gpu unavailable]'
    parser.add_argument('-m', '--mode', dest='platform', metavar='PLATFORM', choices=['cpu', 'gpu'], default='cpu', help='deconvolve on cpu%s'%gpuHelp)
    parser.add_argument('-b', '--batch-size', dest='numParallel', metavar='BATCH-SIZE', type=int, default=32, help='number of simultaneous deconvolutions')
    parser.add_argument('-i', '--iters', dest='numIter', metavar='RL-ITERS', type=int, default=8, help='number of iterations of Richardson-Lucy to run')
    parser.add_argument('-c', '--cacheFH', dest='cacheFH', action='store_true', default=False, help='enable caching (see documentation - use with care)')
    parser.add_argument('-d', '--dest', dest='destDir', metavar='DEST-DIR', default='.', help='directory in which deconvolved output will be saved')
    parser.add_argument('-p', '--psf', dest='psfFile', metavar='PSF-FILE', required=True, default=argparse.SUPPRESS, help='path to PSF file (will be created if needed)')
    parser.add_argument('-t', '--timepoints', dest='timepoints', metavar='IMG', required=True, default=argparse.SUPPRESS, nargs='+', help='files to deconvolve')
    args = parser.parse_args()

    projectorClass = proj.Projector_allC
    if args.platform == 'gpu':
        if proj.gpuAvailable:
            projectorClass = proj.Projector_gpuHelpers
        else:
            print('\033[0;33mNOTE: GPU unavailable - reverting to CPU\033[0m')
    projector = projectorClass()
    projector.cacheFH = args.cacheFH
    hMatrix = psfmatrix.LoadMatrix(args.psfFile, createPSF=True)

    if (args.numParallel < 1):
        print('\033[0;31mERROR: Invalid batch size: %d\033[0m' % args.numParallel)
        exit(1)
    if (args.numIter < 1):
        print('\033[0;31mERROR: Invalid number of RL iterations: %d\033[0m' % args.numIter)
        exit(1)
    if args.cacheFH:
        print('\033[0;33mNOTE: Caching F(H). Check documentation and use with care - may exhaust available memory!\033[0m')

    if (len(args.timepoints) > args.numParallel):
        print('Deconvolving timepoints in batches of %d' % args.numParallel)
    else:
        print('Deconvolving all %d timepoints simultaneously' % len(args.timepoints))
    print('Note that the progress bar will only advance occasionally - be patient!')

    for t in tqdm(range(0, len(args.timepoints), args.numParallel), desc='Deconvolving total {0} files'.format(len(args.timepoints))):
        tEnd = np.minimum(t+args.numParallel, len(args.timepoints))
        ProcessBatch(projector, hMatrix, args.timepoints[t:tEnd], args.destDir, args.numIter)

if __name__ == "__main__":
    main(sys.argv)

# Quick test:
# python deconvolve_time_series.py --psf "/Volumes/Development/light-field-flow/PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-4to0zspacing4Nnum19lambda520n1.33.mat" --dest tempOutput --timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/Cam_Right_40_X1.tif"

# Should replicate Nils deconvolved tiffs
# python deconvolve_time_series.py --psf "/Volumes/Development/light-field-flow/PSFmatrix/buggyPSFMatrix_M22.2NA0.5MLPitch125fml3125from-156to156zspacing4Nnum19lambda520n1.33.mat" --dest tempOutput --timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/Cam_Right_40_X1.tif"


# TODO: test with a large number of images, and a small batch size (to confirm that my logic there works)
#python deconvolve_time_series.py -i=1 --psf "/Volumes/Development/light-field-flow/PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-4to0zspacing4Nnum19lambda520n1.33.mat" --batch-size 2 --dest tempOutputBatch2 --timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/"*.tif
#python deconvolve_time_series.py -i=1 --psf "/Volumes/Development/light-field-flow/PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-4to0zspacing4Nnum19lambda520n1.33.mat" --dest tempOutputBatch11 --timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/"*.tif
