import numpy as np
import tifffile
import sys, time, os
import lfdeconv, lfimage, psfmatrix
import projector as proj

def ProcessBatch(projector, hMatrix, timepoints, destDir, maxIter):
    # Ensure that the destination directory exists
    try:
        os.mkdir(destDir)
    except:
        pass  # Probably the directory already exists

    # Load the input images
    inputImages = []
    for t in timepoints:
        im = lfimage.LoadLightFieldTiff(t)
        inputImages.append(im)
    inputImages = np.array(inputImages)

    # Deconvolve this batch of camera images
    Htf = lfdeconv.BackwardProjectACC(hMatrix, inputImages, progress=None, logPrint=False, projector=projector)
    deconvolvedResult = lfdeconv.DeconvRL(hMatrix, Htf, maxIter=maxIter, Xguess=Htf.copy(), logPrint=False, projector=projector)

    # Save the images
    for i in range(len(timepoints)):
        destFilePath = '%s/%s_deconvolved.tif' % (destDir, os.path.splitext(timepoints[i])[0])
        print('Saving to %s' % destFilePath)
        tifffile.imsave(deconvolvedResult[:,i], destFilePath)


def main(argv, maxIter=8, numParallel=32):
    i = 0
    destDir = '.'
    projectorClass = proj.Projector_allC
    while (i < len(argv)):
        keyword = argv[i]
        i = i+1
        if keyword == '-cpu':
            projectorClass = proj.Projector_allC
            projector = projectorClass()
        elif keyword == '-gpu':
            projectorClass = proj.Projector_gpuHelpers
            projector = projectorClass()
        elif keyword == '-batch-size':
            numParallel = int(argv[i])
            i = i+1
        elif keyword == '-iters':
            maxIter = int(argv[i])
            i = i+1
        elif keyword == '-psf':
            hMatrix = psfmatrix.LoadMatrix(argv[i], createPSF=True)
            projector = projectorClass()
            i = i+1
        elif keyword == '-dest':
            destDir = argv[i]
            i = i+1
        elif keyword == '-cacheFH':
            print('** Caching of F(H) has been enabled **')
            projector.cacheFH = True
        elif keyword == '-timepoints':
            print('Deconvolving timepoints in batches of %d' % numParallel)
            timepoints = argv[i:]
            i = len(argv)
            batch = np.minimum(len(timepoints), numParallel)
            if hMatrix is None:
                raise ValueError('A PSF matrix file has not been specified')
            ProcessBatch(projector, hMatrix, timepoints[:batch], destDir, maxIter)
            timepoints = timepoints[batch:]

if __name__ == "__main__":
    main(sys.argv)

# python deconvolve_time_series.py -psf "/Volumes/Development/light-field-flow/PSFmatrix/buggyPSFMatrix_M22.2NA0.5MLPitch125fml3125from-156to156zspacing4Nnum19lambda520n1.33.mat" -dest tempOutput -timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/Cam_Right_40_X1.tif"
# python deconvolve_time_series.py -psf "/Volumes/Development/light-field-flow/PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-4to0zspacing4Nnum19lambda520n1.33.mat" -dest tempOutput -timepoints "/Users/jonny/Movies/Nils files/Raw-camera-images/Right/Cam_Right_40_X1.tif"

