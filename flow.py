import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

class Shifter:
    # Base class for Shifters (which warp an image according to a shift description they are provided with)
    def __init__(self, controlPointSpacing, actualImageExtendSize, xMotionPermitted):
        self.controlPointSpacing = controlPointSpacing
        self.actualImageExtendSize = actualImageExtendSize
        self.xMotionPermitted = xMotionPermitted


class UniformRollShifter(Shifter):
    # Applies a uniform shift, implemented using np.roll
    def ShiftObject(self, obj, shiftYX):
        # Transform a 3D object according to the flow information provided in shiftDescription
        # For now I just consider a uniform translation in xy
        # 
        # TODO: We need to worry about conserving energy during the shift. 
        # For now I will do a circular shift in order to avoid having to worry about this!
        result = RollNoninteger(obj, shiftYX[0,0], axis=len(obj.shape)-2)
        return RollNoninteger(result, shiftYX[0,1], axis=len(obj.shape)-1)

    def ExampleShiftDescriptionForObject(self, obj):
        # Just provides an example of an appropriate flow description
        return np.array([[-10, 20]])
    
    def VelocityShapeForObject(self, obj):
        # Provides the required shape for the shift description that this class expects
        return (2,)

    def IWCentresForObject(self, obj):
        # Specifies the coordinates at which flow vectors should be drawn (i.e. where the control points are for the deformation that is applied)
        return np.array([[int(obj.shape[-2]/2), int(obj.shape[-1]/2)]])

class UniformSKShifter(UniformRollShifter):
    # Applies a uniform shift, implemented using skimage.warp
    # This behaves otherwise like UniformRollShifter, so I just override the ShiftObject function in that class.
    def ShiftObject(self, obj, shiftYX):
        # Note that this has a fair amount of code duplication from PIVShifter, but I will not attempt to consolidate that for now...
        # Generate control points in the corners of the image
        src_cols = np.arange(0, obj.shape[-1]+1, obj.shape[-1])
        src_rows = np.arange(0, obj.shape[-2]+1, obj.shape[-2])
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        dst = src + shiftYX[0]
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        # Annoyingly, skimage insists that a float input is scaled between 0 and 1, so I must rescale here
        maxVal = np.max(np.abs(obj))
        if len(obj.shape) == 3:
            result = np.zeros(obj.shape)
            for cc in range(obj.shape[0]):
                result[cc] = warp(obj[cc]/maxVal, tform, mode='edge') * maxVal
            return result
        else:
            return warp(obj/maxVal, tform, mode='edge') * maxVal

class PIVShifter(Shifter):
    def IWCentresForObject(self, obj):
        startPos = 0
        # Reusing the code from the skimage example, since that actually does what we need:
        src_cols, src_rows = self.GetRowsAndCols(obj, startPos)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        return np.dstack([src_cols.flat, src_rows.flat])[0]

    def GetRowsAndCols(self, obj, startPos):
        src_cols = np.arange(startPos, obj.shape[-1]+1, self.controlPointSpacing)
        src_rows = np.arange(startPos, obj.shape[-2]+1-self.actualImageExtendSize, self.controlPointSpacing)
        return src_cols, src_rows

    def VelocityShapeForObject(self, obj):
        return self.IWCentresForObject(obj).shape
    
    def ExampleShiftDescriptionForObject(self, obj):
        peakVelocity = 7
        iwPos = self.IWCentresForObject(obj)
        shiftDescription = np.zeros(self.VelocityShapeForObject(obj))
        width = obj.shape[-1]
        for n in range(iwPos.shape[0]):
            quadraticProfile = ((width/2.)**2 - (iwPos[n,0]-width/2.)**2)
            quadraticProfile = quadraticProfile / ((width/2.)**2) * peakVelocity
            shiftDescription[n,1] = quadraticProfile
        if self.xMotionPermitted:
            return shiftDescription
        else:
            return shiftDescription[:,1:2]

    def ExtraDuplicateRow(self, shifts, add=None):
        assert(len(shifts.shape) == 2)
        rowLength = int(np.sqrt(shifts.shape[0]))
        shifts = np.reshape(shifts, (rowLength, rowLength, shifts.shape[1]))
        toAppend = shifts[:,-1:,:].copy()
        if add is not None:
            toAppend += add
        result = np.append(shifts, toAppend, axis=1)
        return result.reshape(result.shape[0]*result.shape[1], result.shape[2])

    def AddZeroEdgePadding(self, obj, src, shiftYX):
        # No-op function to allow PIVZeroEdgeShifter to override this.
        return src, shiftYX

    def ShiftObject(self, obj, shiftYX):
        # Transform a 3D object according to the flow information provided in shiftDescription
        # I use a piecewise affine transformation that should approximately correspond to
        # what I use for PIV analysis
        src = self.IWCentresForObject(obj)
        if (src.shape[0] != shiftYX.shape[0]):
            print(src.shape, shiftYX.shape, obj.shape)
            assert(src.shape[0] == shiftYX.shape[0])
            
        (src, shiftYX) = self.AddZeroEdgePadding(obj, src, shiftYX)
        
        if (self.actualImageExtendSize > 0):
            src = self.ExtraDuplicateRow(src, add=np.array([0, self.actualImageExtendSize]))
            if self.xMotionPermitted:
                dst = src + self.ExtraDuplicateRow(shiftYX)
            else:
                dst = src.copy().astype(shiftYX.dtype)
                dst[:,1] = dst[:,1] + self.ExtraDuplicateRow(shiftYX)[:,0]
        else:
            dst = src.copy().astype(shiftYX.dtype) + shiftYX
            
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        # Annoyingly, skimage insists that a float input is scaled between 0 and 1, so I must rescale here
        maxVal = np.max(np.abs(obj))
        if len(obj.shape) == 3:
            result = np.zeros(obj.shape)
            for cc in range(obj.shape[0]):
                result[cc] = warp(obj[cc]/maxVal, tform, mode='edge') * maxVal
            return result
        else:
            assert(len(obj.shape) == 2)
            return warp(obj/maxVal, tform, mode='edge') * maxVal


class PIVZeroEdgeShifter(PIVShifter):
    def GetRowsAndCols(self, obj, startPos):
        src_cols = np.arange(self.controlPointSpacing, obj.shape[-1], self.controlPointSpacing)
        src_rows = np.arange(self.controlPointSpacing, obj.shape[-2]-self.actualImageExtendSize, self.controlPointSpacing)
        return src_cols, src_rows

    def AddZeroEdgePadding(self, obj, src, shiftYX):
        paddedSrc = self.IWCentresForObject(obj, st='piv')
        paddedShifts = np.zeros(paddedSrc.shape)
        for i in range(src.shape[0]):
            match = False
            for j in range(paddedSrc.shape[0]):
                if (src[i] == paddedSrc[j]).all():
                    match = True
                    paddedShifts[j] = shiftYX[i]
            assert(match)
        return paddedSrc, paddedShifts
        



def ShowDualObjectAndFlow(dualObject, shifter, shiftDescription, otherObject=None, otherObject2=None, destFilename=None, suppressDark=0, histogram=0):
    # Display the recovered object (A and B images) annotated with the flow profile supplied by the caller
    plt.subplot(1, 2, 1)
    if (len(dualObject.shape) == 4):
        assert(dualObject.shape[1] == 2)
        plt.imshow(dualObject[0,0])
        plt.subplot(1, 2, 2)
        plt.imshow(dualObject[0,1])
        windowSource = dualObject[0,0]
    else:
        assert(len(dualObject.shape) == 3)  # It's actually a dual image not an object
        assert(dualObject.shape[0] == 2)
        plt.imshow(dualObject[1])
        windowSource = dualObject[0]
    iwPos = shifter.IWCentresForObject(dualObject)
    velocities = []
    for n in range(iwPos.shape[0]):
        aWindow = windowSource[np.maximum(iwPos[n,1]-int(shifter.controlPointSpacing/2),0):np.minimum(iwPos[n,1]+int(shifter.controlPointSpacing/2),dualObject.shape[-2]),\
                               np.maximum(iwPos[n,0]-int(shifter.controlPointSpacing/2),0):np.minimum(iwPos[n,0]+int(shifter.controlPointSpacing/2),dualObject.shape[-1])]
        if (aWindow.sum() > suppressDark):
            if shifter.xMotionPermitted == False:
                velocities.append(shiftDescription[n,0])
                plt.plot([iwPos[n,0], iwPos[n,0]], \
                         [iwPos[n,1], iwPos[n,1] - shiftDescription[n,0]/2.], color='red')
            else:
                velocities.append(np.sqrt(shiftDescription[n,0]**2 + shiftDescription[n,1]**2))

                plt.plot([iwPos[n,0], iwPos[n,0] - shiftDescription[n,0]/2.], \
                         [iwPos[n,1], iwPos[n,1] - shiftDescription[n,1]/2.], color='red')
    plt.xlim(0, dualObject.shape[-1])
    plt.ylim(dualObject.shape[-2], 0)
    if destFilename is not None:
        plt.savefig(destFilename, dpi=200)
    plt.show()
    if (histogram > 0):
        plt.hist(velocities, range=(0,histogram), bins=20)
        plt.show()
    if otherObject is not None:
        plt.imshow(otherObject[0])
        plt.show()        
    if otherObject2 is not None:
        plt.imshow(otherObject2[0])
        plt.show()   
    return np.array(velocities)

def ShowDifferences(im1, im2, fullIm1, sh):
    # Utility function to help understand how two images differ, since I have been having
    # a lot of problems related to warp(), where tiny changes in the shift values make a visible difference to the result
    # (these are largely due to edge effects of one type or another)
    diff = im1-im2
    print(diff.shape)
    print('Largest difference', np.max(np.abs(diff)), 'loc', np.argmax(np.abs(diff)), \
          np.argmax(np.abs(diff))%diff.shape[1], int(np.argmax(np.abs(diff))/diff.shape[1]))
    plt.imshow(diff)
    iwPos = IWCentresForObject(dualObject, st='piv')
    if False:
        for n in range(iwPos.shape[0]):
            plt.plot(iwPos[n,0], iwPos[n,1], 'x', color='red')
    elif True:
        src = IWCentresForObject(fullIm1[np.newaxis])
        if (src.shape[0] != sh.shape[0]):
            assert(src.shape[0] == sh.shape[0])
        if (shiftType == 'piv-zeroedge'):
            (src, sh) = AddZeroEdgePadding(fullIm1[np.newaxis], src, sh)
            print('padded')
        for n in range(sh.shape[0]):
            plt.plot([iwPos[n,0], iwPos[n,0]+sh[n,0]*1e9], \
                     [iwPos[n,1], iwPos[n,1]+sh[n,1]*1e9], color='red')
            if not sh[n,0] == 0:
                print('x', [iwPos[n,0], iwPos[n,0]+sh[n,0]*1e9])
                print('y', [iwPos[n,1], iwPos[n,1]+sh[n,1]*1e9])
    plt.xlim(-10,60)
    plt.ylim(80,-10)    
    plt.show()

def RollNoninteger(obj, amount, axis=0):
    # Utility function crudely approximating a noninteger shift.
    # We roll by integer amounts and interpolate to account for the noninteger part.
    # Note that we must use floor rather than casting to int, to ensure
    # correct behaviour for negative shifts
    intAmount = math.floor(amount)
    frac = amount - intAmount
    result1 = np.roll(obj, intAmount, axis=axis)
    result2 = np.roll(obj, intAmount+1, axis=axis)
    return result1 * (1-frac) + result2 * frac

