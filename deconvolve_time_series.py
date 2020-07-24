import numpy as np
import tifffile
import sys, time, os
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
    destDir = '.'
    projectorClass = proj.Projector_allC
    hMatrix = None
    i = 1
    while (i < len(argv)):
        keyword = argv[i]
        i = i+1
        if keyword == '--cpu':
            projectorClass = proj.Projector_allC
            projector = projectorClass()
        elif keyword == '--gpu':
            projectorClass = proj.Projector_gpuHelpers
            projector = projectorClass()
        elif keyword == '--batch-size':
            numParallel = int(argv[i])
            i = i+1
        elif keyword == '--iters':
            maxIter = int(argv[i])
            i = i+1
        elif keyword == '--psf':
            hMatrix = psfmatrix.LoadMatrix(argv[i], createPSF=True)
            projector = projectorClass()
            i = i+1
        elif keyword == '--dest':
            destDir = argv[i]
            i = i+1
        elif keyword == '--cacheFH':
            print('Caching of F(H) has been enabled')
            projector.cacheFH = True
        elif keyword == '--timepoints':
            if hMatrix is None:
                print('ERROR: No PSF matrix file has been specified. Terminating.')
                exit(1)
            print('Deconvolving timepoints in batches of %d' % numParallel)
            print('Note that the progress bar will only advance occasionally - be patient!')
            timepoints = argv[i:]
            i = len(argv)
            for t in tqdm(range(0, len(timepoints), numParallel), desc='Deconvolving total {0} files'.format(len(timepoints))):
                tEnd = np.minimum(t+numParallel, len(timepoints))
                ProcessBatch(projector, hMatrix, timepoints[t:tEnd], destDir, maxIter)
        else:
            print('Unrecognised keyword {0}'.format(keyword))

if __name__ == "__main__":
    main(sys.argv)
