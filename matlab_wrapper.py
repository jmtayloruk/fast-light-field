import py_light_field as plf
import projector as proj
import numpy as np
import jutils
import time

# Note that in principle I think it should be possible to call EnableDiagnostics directly
# from matlab code, as long as the right import command is used (see MatlabCalling.m possibly?).
# But I haven't figured that out that yet
def EnableDiagnostics():
    plf.SetProgressReportingInterval(10.0)

def ErrorIfGPUUnavailable():
    # This should only be run when we are expecting to use GPU support,
    # as it will raise an exception if cupy is not available.
    # The function's purpose is simply to raise a more user-friendly error than the one we would otherwise get!
    try:
        _ = cp.zeros((1,))
    except:
        raise RuntimeError("GPU acceleration requested, but no Python/GPU support installed on this computer. Consult installation instructions for fast-light-field")

# Simple accessors to work around the fact that I'm not sure if I can access
# class variables directly from Matlab
def NumZForMatrix(H):
    return int(H.numZ)
def NnumForMatrix(H):
    return int(H.Nnum)

def BackwardProjectPLF(projector, H, projection, projectionShape, planesMatlabIndexing, logPrint, numJobs):
    projectionShape = np.array(projectionShape, dtype='int')
    # We receive an object from matlab that is not a true numpy array.
    # The object contains the array data but does not have any shape
    # (casting the input to np.array gives us a 1D array).
    # We therefore pass in the shape manually and reshape the array so it
    # looks like a Python array (with C-order indexing like we expect in Python code).
    #print(f"BackwardProjectPLF python wrapper received shape {projectionShape}. Strides unknown (not a ndarray)")
    #print(f"reshaping according to {projectionShape[::-1]}")
    t1 = time.time()
    _projection = np.array(projection)
    projection = _projection.reshape(projectionShape[::-1])
    t2 = time.time()
    #print(f"Reshape took {t2-t1:.3f}")
    #print(f"Intermediate array was shape {_projection.shape} strides {_projection.strides})")
    #print(f"Final array is shape {projection.shape} strides {projection.strides})")
    if len(projection.shape) == 2:
        # Matlab has a habit of flattening dimensions of size 1,
        # so we need to be able to accept a 2D array here
        projection = projection[np.newaxis,:,:]
    if planesMatlabIndexing is None:
        planes = range(H.numZ)
    else:
        planes = np.array(planesMatlabIndexing, dtype='int') - 1
        if (-1 in planes):
            raise IndexError("Plane index 0 passed in. This Matlab interface expects 1-based Matlab indexing of planes")

    result = projector.BackwardProjectACC(H, projection, planes, jutils.noProgressBar, logPrint, np.int64(numJobs))

    # We now have a result array (with C-order indexing) that we need to pass back to Matlab.
    # To do this we need to create an array in which the axis order is reversed but the data
    # is F-ordered. This should not require anything to be copied at all, we should be able
    # to do this using array views only.
    # Tests suggest that this next line should be instantaneous (we are just changing how we access the data,
    # but the data itself is not changed or copied). But I should monitor this a bit
    t1 = time.time()
    resultMatlab = (result.ravel()).reshape(result.shape[::-1], order='F')
    t2 = time.time()
    #print(f"Matlab-izing took {t2-t1:.3f}")
    #print(f"Final array is shape {resultMatlab.shape} strides {resultMatlab.strides})")
    return resultMatlab

def ForwardProjectPLF(projector, H, realspace, realspaceShape, planesMatlabIndexing, logPrint, numJobs):
    realspaceShape = np.array(realspaceShape, dtype='int')
    #print(f"ForwardProjectPLF python wrapper received shape {realspaceShape}. Strides unknown (not a ndarray)")
    #print(f"reshaping according to {realspaceShape[::-1]}")
    t1 = time.time()
    _realspace = np.array(realspace)
    realspace = _realspace.reshape(realspaceShape[::-1])
    t2 = time.time()
    #print(f"Reshape took {t2-t1:.3f}")
    #print(f"Intermediate array was shape {_realspace.shape} strides {_realspace.strides})")
    #print(f"Final array is shape {realspace.shape} strides {realspace.strides})")
    if planesMatlabIndexing is None:
        planes = range(H.numZ)
    else:
        planes = np.array(planesMatlabIndexing, dtype='int') - 1
        if (-1 in planes):
            raise IndexError("Plane index 0 passed in. This Matlab interface expects 1-based Matlab indexing of planes")
    result = projector.ForwardProjectACC(H, realspace, planes, jutils.noProgressBar, logPrint, np.int64(numJobs))

    t1 = time.time()
    resultMatlab = (result.ravel()).reshape(result.shape[::-1], order='F')
    t2 = time.time()
    #print(f"Matlab-izing took {t2-t1:.3f}")
    #print(f"Final array is shape {resultMatlab.shape} strides {resultMatlab.strides})")
    return resultMatlab

