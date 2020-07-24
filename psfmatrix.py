import numpy as np
import os, sys, h5py
from jutils import tqdm_alias as tqdm
import myfft
import generate_psf as psf

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    # If cupy is *not* present then we should not find ourselves in the position of calling any cupy functions.
    # The caller would have to do something *really* weird for that to happen.
    pass

class HMatrix(object):
    def __init__(self, HPathFormat, HtPathFormat, HReducedShape, numZ=None, zStart=0, cacheMMap=True, cacheH=False):
        self.HPathFormat = HPathFormat
        self.HtPathFormat = HtPathFormat
        self.HReducedShape = HReducedShape   # Same for Ht
        if numZ is not None:
            self.numZ = numZ
        else:
            self.numZ = len(HReducedShape)
        self.zStart = zStart
        # We can deduce Nnum indirectly from looking at the dimensions of the H matrix we are loading.
        # We only store one quadrant of the H matrix, since we know the other quadrants are just mirror images
        self.Nnum = self.HReducedShape[0][0]*2-1
        self.iterableBRange = range(self.HReducedShape[0][0])

        self.cacheH = cacheH   # Obsolete feature that was only useful for small problem sizes (and ignored by my C code anyway)
        self.Hcache = dict()
        self.cacheHits = 0
        self.cacheMisses = 0
        self.cacheSize = 0
        self.cacheMMap = cacheMMap
        self.mappedH = dict()
        self.mappedHt = dict()
        self._fftFunc = myfft.myFFT2
  
    def UpdateFFTFunc(self, fftFunc):
        if self._fftFunc is not fftFunc:
            self.HCache = dict()
            self.mappedH = dict()
            self.mappedHt = dict()
            self._fftFunc = fftFunc
    
    def Hcc(self, cc, useHt):
        if useHt:
            if (self.cacheMMap and (str(cc) in self.mappedHt)):
                return self.mappedHt[str(cc)]
            pathFormat = self.HtPathFormat
        else:
            if (self.cacheMMap and (str(cc) in self.mappedH)):
                return self.mappedH[str(cc)]
            pathFormat = self.HPathFormat
        # Note we need to cast to tuple for safety - strange errors seem to occur if we pass a numpy array instead
        result = np.memmap(pathFormat.format(z=cc+self.zStart), dtype='float32', mode='r', shape=tuple(self.HReducedShape[cc+self.zStart]))

        if self._fftFunc is myfft.myFFT2_gpu:
            result = cp.asarray(np.array(result))
        
        if self.cacheMMap:
            if useHt:
                self.mappedHt[str(cc)] = result
            else:
                self.mappedH[str(cc)] = result
        return result

    def fH_uncached(self, cc, bb, aa, useHt, transposePSF, fshape):
        if transposePSF:
            Hcc = self.Hcc(cc, useHt)[bb, aa].transpose()
        else:
            Hcc = self.Hcc(cc, useHt)[bb, aa]
        return self._fftFunc(Hcc, fshape)

    def fH(self, cc, bb, aa, useHt, transposePSF, fshape):
        key = '%d,%d,%d,%d,%d'%(cc, bb, aa, int(useHt), int(transposePSF))
        if (self.cacheH and key in self.Hcache):
            self.cacheHits += 1
            return self.Hcache[key]
        result = self.fH_uncached(cc, bb, aa, useHt, transposePSF, fshape)
        if self.cacheH:
            self.Hcache[key] = result
            self.cacheSize += result.nbytes
            self.cacheMisses += 1
        return result
    
    def ClearCache(self):
        self.Hcache.clear()

    def PSFShape(self, cc):
        return (self.HReducedShape[cc+self.zStart][2], self.HReducedShape[cc+self.zStart][3])

def GetPathFormats(matPath):
    # Use the .mat filename, but without that extension, as a folder name to store our memory-mapped data
    mmapPath = os.path.splitext(matPath)[0]
    hPathFormat = mmapPath+'/H{z:02d}.array'
    htPathFormat = mmapPath+'/Ht{z:02d}.array'
    return (mmapPath, hPathFormat, htPathFormat)

def LoadRawMatrixData(matPath, expectedNnum=None, createPSF=True):
    # Load the contents of a .mat file (and generate memmap backing files that my optimized code actually uses)
    # Normally this function should not be called directly, but it is needed if we want to run old code that
    # makes direct use of the matrices _H and _Ht.
    mmapPath, hPathFormat, htPathFormat = GetPathFormats(matPath)
    try:
        # Ensure the folder we will use to store our mmapped matrices actually exists
        os.mkdir(mmapPath)
    except FileExistsError:
        pass

    if createPSF:
        # Generate the PSF file if it does not exist (slow!)
        psf.EnsureMatrixFileExists(matPath)
    
    # Load the matrices from the .mat file.
    # This is slow since they must be decompressed and are rather large! (9.5GB each, in single-precision FP)
    hReducedShape = []
    htReducedShape = []
    with h5py.File(matPath, 'r') as f:
        # Note that in the following code, h5py requires a weird syntax such as f['CAindex'][()] to access a keyed value from the HDF5 file
        print('Load CAindex')
        sys.stdout.flush()
        _CAindex = f['CAindex'][()].astype('int')
        # I have got into a mess whereby I have sometimes been saving CAIndex in transposed form.
        # This test is a hack to fix that:
        if _CAindex.shape[1] == 2:
            _CAindex = _CAindex.T
        Nnum = f['Nnum'][()].astype('int')
        if (expectedNnum is not None) and (Nnum != expectedNnum):
            warnings.warn('Nnum={0} from file does not match expected value {1}'.format(Nnum, expectedNnum))
        try:
            reducedMatrix = f['ReducedMatrix'][()].astype('int')
        except KeyError:
            print('No reduced flag - will assume this is a full matrix')
            reducedMatrix = False
        
        print('Load H')
        sys.stdout.flush()
        _H = f['H'][()].astype('float32')
        aabbRange = int((Nnum+1)/2)
        if reducedMatrix:
            assert(_H.shape[2] == aabbRange)
        else:
            assert(_H.shape[2] == Nnum)
        for cc in tqdm(range(_H.shape[0]), desc='memmap H'):
            HCC =  _H[cc, :aabbRange, :aabbRange, _CAindex[0,cc]-1:_CAindex[1,cc], _CAindex[0,cc]-1:_CAindex[1,cc]]
            hReducedShape.append(HCC.shape)
            a = np.memmap(hPathFormat.format(z=cc), dtype='float32', mode='w+', shape=HCC.shape)
            a[:,:,:,:] = HCC[:,:,:,:]
            a.flush()
            del a
        
        print('Load Ht')
        sys.stdout.flush()
        _Ht = f['Ht'][()].astype('float32')
        for cc in tqdm(range(_Ht.shape[0]), desc='memmap Ht'):
            HtCC =  _Ht[cc, :aabbRange, :aabbRange, _CAindex[0,cc]-1:_CAindex[1,cc], _CAindex[0,cc]-1:_CAindex[1,cc]]
            htReducedShape.append(HtCC.shape)
            a = np.memmap(htPathFormat.format(z=cc), dtype='float32', mode='w+', shape=HtCC.shape)
            a[:,:,:,:] = HtCC[:,:,:,:]
            a.flush()
            del a

    np.save(mmapPath+'/HReducedShape.npy', hReducedShape)
    np.save(mmapPath+'/HtReducedShape.npy', htReducedShape)

    return (_H, _Ht, _CAindex, hPathFormat, htPathFormat, hReducedShape, htReducedShape)

def LoadMatrix(matPath, numZ=None, zStart=0, forceRegeneration=False, cacheH=False, createPSF=False):
    # Obtain a HMatrix object based on a .mat file
    # (although we will jump straight to previously-generated memmap backing files if they exist)
    # This is the function that user code should normally be calling
    mmapPath, hPathFormat, htPathFormat = GetPathFormats(matPath)

    if createPSF:
        # Generate the PSF file if it does not exist (slow!)
        psf.EnsureMatrixFileExists(matPath)

    try:
        if forceRegeneration:
            raise FileNotFoundError('transferring control to except branch to force regeneration')
        hReducedShape = np.load(mmapPath+'/HReducedShape.npy')
        htReducedShape = np.load(mmapPath+'/HtReducedShape.npy')
    except FileNotFoundError:
        print("Failed to load array shapes. There are probably no memmap files either.")
        print("We will now try to generate them from scratch from the .mat file, which will take a little while")
        (_, _, _, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = LoadRawMatrixData(matPath)
    return HMatrix(hPathFormat, htPathFormat, hReducedShape, numZ=numZ, zStart=zStart, cacheH=cacheH, cacheMMap=True)
