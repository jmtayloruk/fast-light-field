def CalcFlowUsingVanillaPIV(imagePair, iwPos):
    # Calculate the flow based on traditional window-based PIV analysis
    shiftDescriptionPIV = np.zeros(iwPos.shape)
    smallIWSize = controlPointSpacing
    largeIWSize = 2 * controlPointSpacing
    for n in range(iwPos.shape[0]):
        a = imagePair[0, iwPos[n,1]-int(smallIWSize/2):iwPos[n,1]+int(smallIWSize/2),\
                             iwPos[n,0]-int(smallIWSize/2):iwPos[n,0]+int(smallIWSize/2)]
        b = imagePair[1, iwPos[n,1]-int(largeIWSize/2):iwPos[n,1]+int(largeIWSize/2),\
                             iwPos[n,0]-int(largeIWSize/2):iwPos[n,0]+int(largeIWSize/2)]
        sad_using_c_code = jpsad.sad_correlation(a, b)
        zeroPoint = np.array([1,1])*int((largeIWSize-smallIWSize)/2)
        shiftDescriptionPIV[n] = -(np.array(np.unravel_index(sad_using_c_code.argmin(), sad_using_c_code.shape))[::-1]-zeroPoint)
    if xMotionPermitted:
        return shiftDescriptionPIV
    else:
        return shiftDescriptionPIV[:,1:]

if source == 'synthetic':
    # Synthetic images - draw all vectors
    thresh = 0
else:
    # Experimental PIV images - suppress vectors in dark regions
    thresh = 6e5
    
shiftDescriptionPIVRaw = CalcFlowUsingVanillaPIV(dualObject[0], iwPos)
ShowDualObjectAndFlow(dualObject, shiftDescriptionPIVRaw*3, suppressDark=thresh)
shiftDescriptionPIVReconstructed = CalcFlowUsingVanillaPIV(dualObjectRecovered[0], iwPos)
ShowDualObjectAndFlow(dualObjectRecovered, shiftDescriptionPIVReconstructed*3, suppressDark=thresh)