# This file contains some snippets I was using for investigating the behaviour of various
# subtly different implementations of the RL algorithm.
# TODO: I need to come back to this and look into this some more


# Temporary check to compare RL implementations
#def deconvRL(hMatrix, Htf, maxIter, Xguess, logPrint=True):

tempH = HMatrix(_HPathFormat, _HtPathFormat, _HReducedShape, numZ=1, zStart=13)
Htf = backwardProjectACC(tempH, inputImage, numjobs=1)
Xguess = Htf.copy();
maxIter = 1
deconvolvedResult = deconvRL(tempH, Htf, maxIter, Xguess)

def deconvRL_copy(hMatrix, Htf, maxIter, Xguess, logPrint=True):
    # Note:
    #  Htf is the *initial* backprojection of the camera image
    #  Xguess is the initial guess for the object
    for i in tqdm(range(maxIter), desc='RL deconv'):
        t0 = time.time()
        HXguess = forwardProjectACC(hMatrix, Xguess, logPrint=logPrint)
        HXguessBack = backwardProjectACC(hMatrix, HXguess, logPrint=logPrint)
        errorBack = Htf / HXguessBack
        Xguess = Xguess * errorBack
        Xguess[np.where(np.isnan(Xguess))] = 0
        ttime = time.time() - t0
        print('iter %d | %d, took %.1f secs. Max val %f' % (i+1, maxIter, ttime, np.max(Xguess)))
    return Xguess

def myDeconvRL(hMatrix, image, maxIter, Xguess):
    # I believed this to be the RL algorithm in the way I have written it in the past.
    # TODO: I should look into this and see if I've just made a mistake or if they are actually different.
    
    # Xguess is our single combined guess of the object
    Xguess = Xguess.copy()    # Because we will be updating it, and caller may not always be expecting that
    print('Xguess shape', Xguess.shape)
    for i in tqdm(range(maxIter), desc='RL deconv'):
        print('iter', i)
        temp = forwardProjectACC(hMatrix, Xguess)
        plt.imshow(temp)
        plt.show()
        print('temp range', np.min(temp), np.max(temp), np.min(np.abs(temp)))
        
        relativeBlur = image / forwardProjectACC(hMatrix, Xguess)
        # At this point, the forward projection has negative values (and zero values?)
        # I think that is the source of the problem.
        relativeBlur[np.where(np.isnan(Xguess))] = 0
        
        Xguess *= backwardProjectACC(hMatrix, relativeBlur)
    #        Xguess[np.where(np.isnan(Xguess))] = 0
    return Xguess

def myDeconvRL2(hMatrix, image, maxIter, Xguess):
    # Modifying my algorithm slightly in order to make it more like the Matlab one
    # -> Yes, this now gives identical results to the Matlab one.
    Xguess = Xguess.copy()    # Because we will be updating it, and caller may not always be expecting that
    print('Xguess shape', Xguess.shape)
    for i in tqdm(range(maxIter), desc='RL deconv'):
        print('iter', i)
        
        relativeBlur = backwardProjectACC(hMatrix, image) / backwardProjectACC(hMatrix, forwardProjectACC(hMatrix, Xguess))
        Xguess *= relativeBlur
        Xguess[np.where(np.isnan(Xguess))] = 0
    return Xguess

plt.imshow(Xguess[0])
plt.show()
deconvolvedResult2 = myDeconvRL(tempH, inputImage, maxIter, Xguess)

print(deconvolvedResult.shape, np.min(deconvolvedResult), np.max(deconvolvedResult))
print(deconvolvedResult2.shape, np.min(deconvolvedResult2), np.max(deconvolvedResult2))
plt.imshow(deconvolvedResult[0])
plt.show()
plt.imshow(deconvolvedResult2[0])
plt.show()
