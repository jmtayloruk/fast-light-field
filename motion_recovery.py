from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage, scipy.optimize, scipy.io
from scipy.optimize import Bounds
import flow, lfdeconv_piv

class ShiftHistory:
    # Class used to keep track of the different trial shifts
    # that are considered over the course of an optimization 
    def __init__(self, shifter):
        self.Reset()
        self.shifter = shifter

    def __copy__(self):
        result = ShiftHistory(self.shifter)
        result.shiftHistory = self.shiftHistory
        result.scoreHistory = self.scoreHistory
        result.counter = self.counter
        return result

    def Reset(self):
        self.scoreHistory = []
        self.shiftHistory = []
        self.counter = 0
    
    def Update(self, shift, score):
        # Called by the optimizer every time it calculates a score for a new trial shift
        self.shiftHistory.append(shift)
        self.scoreHistory.append(score)
        self.counter = self.counter + 1
        
    def BestScore(self):
        return np.min(self.scoreHistory)

    def BestShift(self):
        return self.shiftHistory[np.argmin(self.scoreHistory)]

    def PlotHistory(self, onlyPlotEvery=1):
        # Displays some plots as a visual indicator of current optimizer progress towards convergence
        if ((self.counter%onlyPlotEvery) == 0) and (len(self.shiftHistory) > 0):
            print('best score so far: %e' % np.min(self.scoreHistory))
            # Plot one of the shifts
            shiftShape = self.shiftHistory[0].shape
            selectedItem = np.minimum(int(np.sqrt(shiftShape[0])/2), shiftShape[0]-1)
            selectedShift = np.array(self.shiftHistory)[:, selectedItem, -1]
            plt.title('Value of item {0} in the set of vectors'.format(selectedItem))
            plt.xlabel('Iteration')
            plt.plot(selectedShift)
            plt.show()
            # Plot scores, with a suitable y axis scaling to see the interesting parts.
            # We limit the y axis to avoid stupid guesses distorting the plot.
            improvement = self.scoreHistory[0] - np.min(self.scoreHistory)
            plt.ylim(np.min(self.scoreHistory), self.scoreHistory[0]+2*improvement)
            plt.title('Score history')
            plt.xlabel('Iteration')
            plt.plot(self.scoreHistory)
            plt.show()
            # Plot an indication of which values are being updated on which iteration
            for n in range(1, len(self.scoreHistory)):
                changes = np.array(np.where((self.shiftHistory[n] == self.shiftHistory[n-1]).flatten() == False))
                if (changes.size > 0):
                    plt.plot(n, changes, 'x', color='red')
            plt.title('Which item in vector set is being updated?')
            plt.xlabel('Iteration')
            plt.show()

            with open('scores.txt', 'a') as f:
                f.write('%f\t' % self.scoreHistory[-1])
                for n in self.shiftHistory[-1]:
                    if self.shifter.xMotionPermitted:
                        f.write('%f\t%f\t' % (n[0], n[1]))
                    else:
                        f.write('%f\t' % (n[0]))
                f.write('\n')
            return True
        else:
            return False   

def ScoreShift(candidateShiftYX, shifter, method, imageAB, hMatrix=None, shiftHistory=None, scaling=1.0, log=True, logPrint=False, comparator=None, maxIter=8):
    # Just returns a single score evaluating the extent to which the candidate shift can "explain" the AB images that were observed
    return ScoreShiftDetailed(candidateShiftYX, shifter, method, imageAB, hMatrix, shiftHistory, scaling, log, logPrint, comparator, maxIter=maxIter)[0]

def ScoreShiftDetailed(candidateShiftYX, shifter, method, imageAB, hMatrix=None, shiftHistory=None, scaling=1.0, log=True, logPrint=False, comparator=None, maxIter=8):
    # Returns a score evaluating the extent to which the candidate shift can "explain" the AB images that were observed,
    # along with additional detailed information that may be useful for debugging and closer visual investigation of this candidate scenario

    # Our input parameters get flattened, so we need to reshape them to Nx2 like my code is expecting
    # 'scaling' is useful for optimizers that insist on initial very small step sizes
    if shifter.xMotionPermitted:
        candidateShiftYX = candidateShiftYX.reshape(int(candidateShiftYX.shape[0]/2),2) * scaling
    else:
        candidateShiftYX = candidateShiftYX.reshape(candidateShiftYX.shape[0],1) * scaling
    # Sanity check and reminder that we have a 2xMxN AB image pair
    assert(len(imageAB.shape) == 3)  
    assert(imageAB.shape[0] == 2)
        
    if log:
        print('======== Score shift ========', candidateShiftYX.T)
    #        print('======== Score shift ========')

    if method == 'joint':
        # Perform the joint deconvolution to recover a single object
        res = lfdeconv_piv.DeconvRL_PIV(hMatrix, imageAB, maxIter, shifter, shiftDescription=candidateShiftYX, logPrint=logPrint)
        # Evaluate how well the forward-projected result matches the actual camera images, using SSD
        candidateImageAB = lfdeconv_piv.ForwardProjectACC_PIV(hMatrix, res, shifter, candidateShiftYX, logPrint=logPrint)
    elif method == 'joint-test-trivial':
        # Debugging method in which I use trivial projectors that behave like a delta function PSF
        res = DeconvRLTrivial(hMatrix, imageAB, maxIter, shifter, shiftDescription=candidateShiftYX)
        candidateImageAB = lfdeconv_piv.ForwardProjectTrivial(hMatrix, res, shifter, candidateShiftYX)
    else:
        # Just warp the raw B image manually and look at how the two images compare
        assert(method == 'naive')
        candidateImageAB = imageAB.copy()
        # A bit of dimensional gymnastics here, because ShiftObject expects an *object*,
        # i.e. a 3D volume, whereas in this case we just have a 2D image
        candidateImageAB[1,:,:] = shifter.ShiftObject(candidateImageAB[np.newaxis,0,:,:], candidateShiftYX)[0]
        res = None  # So that we have something to return
    # Sanity check and reminder that we have a 2xMxN AB image pair
    assert(len(candidateImageAB.shape) == 3)  
    assert(candidateImageAB.shape[0] == 2)

    imageToScore = candidateImageAB[:, 1:-1-shifter.actualImageExtendSize, 1:-1-shifter.actualImageExtendSize]
    referenceImage = imageAB[:, 1:-1-shifter.actualImageExtendSize, 1:-1-shifter.actualImageExtendSize]
    # Score by comparing the A and B images to the ones we are optimizing on.
    # Note: in some simulated or naive cases, the A camera images will always be a perfect match,
    # but for the real case the joint solution will be a compromise for both the A and B camera images.
    #
    # I have tried to renormalize to aid comparison between the images - based on the relative intensity
    # of the candidate and observed A images. I chose the A images because they will be identical in the case
    # of the 'naive' method (direct warping). However, for the 'joint' method they won't be.
    # TODO: I need to think more about whether this normalization is necessary and appropriate.
    # (I think I introduced it in the hope of fixing a problem,
    # but lack of normalization wasn't the fundamental issue in the end)
    renormHack = np.average(candidateImageAB[0]) / np.average(imageAB[0])
    ssdScore = np.sum((imageToScore/renormHack - referenceImage)**2)

    if comparator is not None:
        maxLoc = np.argmax(np.abs(imageToScore - comparator)[1:-1,1:-1])
        maxVal =    np.max(np.abs(imageToScore - comparator)[1:-1,1:-1])
        print('showing B image diffs')
        plt.imshow((imageToScore[1] - comparator)[170:,150:])
        plt.colorbar()
        plt.title('BRel (max %e)'%maxVal)
        print('Max val %f at %d (image scale %d)' % (maxVal, maxLoc, np.max(comparator)))
        plt.show()

    if shiftHistory is not None:
        shiftHistory.Update(candidateShiftYX, ssdScore)
        if log:
            if shiftHistory.PlotHistory(onlyPlotEvery=50):
                if method == 'joint':
                    dualObject = np.tile(res[:,np.newaxis,:,:] / 2.0, (1,2,1,1))
                    dualObject[:,1,:,:] = shifter.ShiftObject(dualObject[:,1,:,:], candidateShiftYX)
                    flow.ShowDualObjectAndFlow(dualObject, shifter, candidateShiftYX)
                else:
                    flow.ShowDualObjectAndFlow(candidateImageAB, shifter, candidateShiftYX)
                print('Last trial shift: ', candidateShiftYX.T)

    if log:
        print('return %e' % ssdScore)
    return (ssdScore, renormHack, np.average(candidateImageAB[0]), np.average(imageAB), candidateImageAB, res) 

def OptimizeToRecoverFlowField(method, imageAB, hMatrix, shifter, trueShiftDescription, initialShiftGuess, searchRangeXY=(10,10), shiftHistory=None, logPrint=False):
    # Main function which runs an optimizer to estimate the shift that occurred between two camera images.
    imageAB = imageAB.copy()    # This is just paranoia - I don't think it should get manipulated
    print('True shift:', trueShiftDescription.T)

    if shiftHistory is not None:
        warnings.warn('Overriding initial shift guess with best shift from history')
        initialShiftGuess = shiftHistory.BestShift()
        
    if False:
        plt.imshow(imageAB[0,:,:])
        plt.show()
        plt.imshow(imageAB[1,:,:])
        plt.show()

    if False:
        print('Score for correct shift:', ScoreShift(trueShiftDescription.flatten(), shifter, method, imageAB, hMatrix))
        print('Score for initial guess:', ScoreShift(initialShiftGuess.flatten(), shifter, method, imageAB, hMatrix))

    if True:
        optimizationAlgorithm = 'Powell'
        options = {'xtol': 1e-2}
    elif True:
        optimizationAlgorithm = 'L-BFGS-B'
        options = {'eps': 5e-03, 'gtol': 1e-6}
    else:
        optimizationAlgorithm = 'Nelder-Mead'
        options = {'eps': 5e-03, 'xatol': 1e-2, 'adaptive': True}

    if shiftHistory is None:
        shiftHistory = ShiftHistory(shifter)

    # Define the search bounds for the optimizer
    lb = []
    ub = []
    if shifter.xMotionPermitted:
        for n in range(trueShiftDescription.shape[0]):
            lb.extend([trueShiftDescription[n,0]-searchRangeXY[0], trueShiftDescription[n,1]-searchRangeXY[1]])
            ub.extend([trueShiftDescription[n,0]+searchRangeXY[0], trueShiftDescription[n,1]+searchRangeXY[1]])
    else:
        # TODO: what happens if I give the optimizer a search range of 0 along certain axes?
        # Would that be a way to save this separate code branch (and elsewhere), and just always assumed 2D motion
        # but if !xMotionPermitted then we just don't optimize across those parameters...?
        for n in range(trueShiftDescription.shape[0]):
            lb.extend([trueShiftDescription[n,0]-searchRangeXY[1]])
            ub.extend([trueShiftDescription[n,0]+searchRangeXY[1]])
    shiftSearchBounds = scipy.optimize.Bounds(lb, ub, True)

    # Optimize to obtain the best-matching shift
    try:
        shift = scipy.optimize.minimize(ScoreShift, initialShiftGuess, bounds=shiftSearchBounds, args=(shifter, method, imageAB, hMatrix, shiftHistory, 1.0, True, logPrint), method=optimizationAlgorithm, options=options)
        print('Optimizer finished:', str(shift.message), 'Final shift:', shift.x.T)
    except KeyboardInterrupt:
        # Catch keyboard interrupts so that we still return whatever shiftHistory we have built up so far.
        print('KEYBOARD INTERRUPT DURING OPTIMIZATION')
    return shiftHistory

def CheckConvergence(funcToCall, convergedShift, args):
    # A crude check to make sure our solution is not CLEARLY unconverged.
    # We try a few select perturbations from the true solution and make sure
    # they do indeed result in higher scores.
    initialScore = funcToCall(convergedShift.flatten(), *args)
    print('Sanity check: perturbing certain components just to check this doesnt result is a better score')
    print(' Initial score %e' % initialScore)
    ok = True
    paramsToTweak = [7, 8, 12, 13]    # Try various tweaking a few different control points
    if convergedShift.shape[0] < paramsToTweak[-1]:
        # Actually we have a very short control point vector - just test the first few
        paramsToTweak = range(np.minimum(4, convergedShift.shape[0]))
    for du in paramsToTweak:
        for n in paramsToTweak:
            temp = convergedShift.copy()
            temp[n] += du
            score = funcToCall(temp, *args)
            print(' Offset score %e' % score)
            if (score < initialScore):
                print(n, du, ' BETTER! (by %f%%)' % ((initialScore-score)/score*100))
                ok = False
    if ok:
       print('Passed sanity check') 

def ReportOnOptimizerConvergence(shiftHistory, shifter, method, obj, hMatrix=None):
    # Print out information relating to whether the optimizer seems to have converged to find the correct motion, or not.
    if shiftHistory is None:
        print('ReportOnOptimizerConvergence returning - called with shiftHistory=None')
        return
    bestShift = shiftHistory.BestShift()
    print('Best score: %e' % shiftHistory.BestScore())
    print('Best shift: np.array([', end='')
    for n in bestShift.flatten():
        print('%f, '%n, end='')
    print('])')
    CheckConvergence(ScoreShift, bestShift.flatten(), (shifter, method, obj, hMatrix, None, 1.0, False))
    return bestShift
