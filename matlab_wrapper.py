import py_light_field as plf
import projector as proj
import numpy as np
import jutils

def BackwardProjectPLF(projector, H, projection, projectionShape, planesMatlabIndexing, logPrint, numJobs):
    projectionShape = np.array(projectionShape, dtype='int')
    projection = np.array(projection).reshape(projectionShape[::-1])
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

    return projector.BackwardProjectACC(H, projection, planes, jutils.noProgressBar, logPrint, np.int64(numJobs))

def ForwardProjectPLF(projector, H, realspace, realspaceShape, planesMatlabIndexing, logPrint, numJobs):
    realspaceShape = np.array(realspaceShape, dtype='int')
    realspace = np.array(realspace).reshape(realspaceShape[::-1])
    if planesMatlabIndexing is None:
        planes = range(H.numZ)
    else:
        planes = np.array(planesMatlabIndexing, dtype='int') - 1
        if (-1 in planes):
            raise IndexError("Plane index 0 passed in. This Matlab interface expects 1-based Matlab indexing of planes")
    return projector.ForwardProjectACC(H, realspace, planes, jutils.noProgressBar, logPrint, np.int64(numJobs))
