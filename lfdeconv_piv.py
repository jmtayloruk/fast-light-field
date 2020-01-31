import numpy as np
import sys, time, os
from jutils import tqdm_alias as tqdm

import lfdeconv

def ForwardProjectACC_PIV(hMatrix, obj, shifter, shiftDescription):
    # Compute the AB images obtained from the single object we are provided with
    # (with the B image being of the object shifted by shiftYX).
    # We give each image half the intensity in order to conserve energy.
    dualObject = np.tile(obj[:,np.newaxis,:,:] / 2.0, (1,2,1,1))
    dualObject[:,1,:,:] = shifter.ShiftObject(dualObject[:,1,:,:], shiftDescription)
    return lfdeconv.ForwardProjectACC(hMatrix, dualObject, logPrint=False, progress=None)

def DualBackwardProjectACC_PIV(hMatrix, dualProjection, shifter, shiftDescription):
    # Compute the reverse transform given the AB images (B image shifted by shiftYX).
    # First we do the reverse transformation on both images
    dualObject = lfdeconv.BackwardProjectACC(hMatrix, dualProjection, logPrint=False, progress=None)
    # Now we reverse the shift on the B object
    dualObject[:,1,:,:] = shifter.ShiftObject(dualObject[:,1,:,:], -shiftDescription)
    # Now, ideally the objects would match, but of course in practice there will be discrepancies,
    # especially if we are not using the correct shiftDescription.
    # To make the operation match the transpose of the forward operation,
    # we add the two objects and divide by 2 here
    return dualObject

def FusedBackwardProjectACC_PIV(hMatrix, dualProjection, shifter, shiftDescription):
    dualObject = DualBackwardProjectACC_PIV(hMatrix, dualProjection, shifter, shiftDescription)
    result = np.sum(dualObject, axis=1) / 2.0     # Merge the two backprojection
    return result

def DeconvRL_PIV(hMatrix, imageAB, maxIter, shifter, shiftDescription):
    # Note:
    #  Htf is the *initial* backprojection of the camera image
    #  Xguess is the initial guess for the object
    Htf = FusedBackwardProjectACC_PIV(hMatrix, imageAB, shifter, shiftDescription)
    Xguess = Htf.copy()
    for i in tqdm(range(maxIter), desc='RL deconv'):
        t0 = time.time()
        HXguess = ForwardProjectACC_PIV(hMatrix, Xguess, shifter, shiftDescription)
        HXguessBack = FusedBackwardProjectACC_PIV(hMatrix, HXguess, shifter, shiftDescription)
        errorBack = Htf / HXguessBack
        Xguess = Xguess * errorBack
        Xguess[np.where(np.isnan(Xguess))] = 0
        t1 = time.time() - t0
    return Xguess
