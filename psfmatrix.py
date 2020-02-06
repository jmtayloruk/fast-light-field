import numpy as np
import os, sys, h5py
from jutils import tqdm_alias as tqdm
import myfft

class HMatrix:
    def __init__(self, HPathFormat, HtPathFormat, HReducedShape, numZ=None, zStart=0):
        self.HPathFormat = HPathFormat
        self.HtPathFormat = HtPathFormat
        self.HReducedShape = HReducedShape   # Same for Ht
        if numZ is not None:
            self.numZ = numZ
        else:
            self.numZ = len(HReducedShape)
        self.zStart = zStart
        self.Hcache = dict()
        self.cacheHits = 0
        self.cacheMisses = 0
        self.cacheSize = 0
  
    def Hcc(self, cc, useHt):
        if useHt:
            pathFormat = self.HtPathFormat
        else:
            pathFormat = self.HPathFormat
        # Note we need to cast to tuple for safety - strange errors seem to occur if we pass a numpy array instead
        result = np.memmap(pathFormat.format(z=cc+self.zStart), dtype='float32', mode='r', shape=tuple(self.HReducedShape[cc+self.zStart]))
        return result

    def fH_uncached(self, cc, bb, aa, useHt, transposePSF, fshape):
        if transposePSF:
            return myfft.myFFT2(self.Hcc(cc, useHt)[bb, aa].transpose(), fshape)
        else:
            return myfft.myFFT2(self.Hcc(cc, useHt)[bb, aa], fshape)
    
    def fH(self, cc, bb, aa, useHt, transposePSF, fshape):
        key = '%d,%d,%d,%d,%d'%(cc, bb, aa, int(useHt), int(transposePSF))
        if not key in self.Hcache:
            result = self.fH_uncached(cc, bb, aa, useHt, transposePSF, fshape)
            self.Hcache[key] = result
            self.cacheSize += result.nbytes
            self.cacheMisses += 1
        else:
            self.cacheHits += 1
        return self.Hcache[key]
    
    def ClearCache(self):
        self.Hcache.clear()
    
    def IterableBRange(self, cc):
        return range(self.HReducedShape[cc+self.zStart][0])
    
    def PSFShape(self, cc):
        return (self.HReducedShape[cc+self.zStart][2], self.HReducedShape[cc+self.zStart][3])
    
    def Nnum(self, cc):
        return self.HReducedShape[cc+self.zStart][0]*2-1

def GetPathFormats(matPath):
    # Use the .mat filename, but without that extension, as a folder name to store our memory-mapped data
    mmapPath = os.path.splitext(matPath)[0]
    hPathFormat = mmapPath+'/H{z:02d}.array'
    htPathFormat = mmapPath+'/Ht{z:02d}.array'
    return (mmapPath, hPathFormat, htPathFormat)

def LoadRawMatrixData(matPath, forceRegeneration = False):
    # Load the contents of a .mat file (and generate memmap backing files that my optimized code actually uses)
    # Normally this function should not be called directly, but it is needed if we want to run old code that
    # makes direct use of the matrices _H and _Ht.
    mmapPath, hPathFormat, htPathFormat = GetPathFormats(matPath)
    try:
        # Ensure the folder exists: we will use to store our mmapped matrices actually exists
        os.mkdir(mmapPath)
    except:
        pass  # Probably the directory already exists

    # Load the matrices from the .mat file.
    # This is slow since they must be decompressed and are rather large! (9.5GB each, in single-precision FP)
    hReducedShape = []
    htReducedShape = []
    with h5py.File(matPath, 'r') as f:
        print('Load CAindex')
        sys.stdout.flush()
        _CAindex = f['CAindex'].value.astype('int')
        
        print('Load H')
        sys.stdout.flush()
        _H = f['H'].value.astype('float32')
        Nnum = _H.shape[2]
        aabbRange = int((Nnum+1)/2)
        for cc in tqdm(range(_H.shape[0]), desc='memmap H'):
            HCC =  _H[cc, :aabbRange, :aabbRange, _CAindex[0,cc]-1:_CAindex[1,cc], _CAindex[0,cc]-1:_CAindex[1,cc]]
            hReducedShape.append(HCC.shape)
            a = np.memmap(hPathFormat.format(z=cc), dtype='float32', mode='w+', shape=HCC.shape)
            a[:,:,:,:] = HCC[:,:,:,:]
            del a
        
        print('Load Ht')
        sys.stdout.flush()
        _Ht = f['Ht'].value.astype('float32')
        for cc in tqdm(range(_Ht.shape[0]), desc='memmap Ht'):
            HtCC =  _Ht[cc, :aabbRange, :aabbRange, _CAindex[0,cc]-1:_CAindex[1,cc], _CAindex[0,cc]-1:_CAindex[1,cc]]
            htReducedShape.append(HtCC.shape)
            a = np.memmap(htPathFormat.format(z=cc), dtype='float32', mode='w+', shape=HtCC.shape)
            a[:,:,:,:] = HtCC[:,:,:,:]
            del a

    np.save(mmapPath+'/HReducedShape.npy', hReducedShape)
    np.save(mmapPath+'/HtReducedShape.npy', htReducedShape)

    return (_H, _Ht, _CAindex, hPathFormat, htPathFormat, hReducedShape, htReducedShape)

def LoadMatrix(matPath, numZ=None, zStart=0, forceRegeneration = False):
    # Obtain a HMatrix object based on a .mat file
    # (although we will jump straight to previously-generated memmap backing files if they exist)
    mmapPath, hPathFormat, htPathFormat = GetPathFormats(matPath)
    try:
        if forceRegeneration:
            raise('transferring control to except branch to force regeneration')
        hReducedShape = np.load(mmapPath+'/HReducedShape.npy')
        htReducedShape = np.load(mmapPath+'/HtReducedShape.npy')
    except:
        print("Failed to load array shapes. There are probably no memmap files either.")
        print("We will now try to generate them from scratch from the .mat file, which will take a little while")
        (_, _, _, hPathFormat, htPathFormat, hReducedShape, htReducedShape) = LoadRawMatrixData(matPath, forceRegeneration)
    return HMatrix(hPathFormat, htPathFormat, hReducedShape, numZ=numZ, zStart=zStart)
