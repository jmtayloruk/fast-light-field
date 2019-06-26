# Run the following command:
#	python setup.py build; python setup.py install
# in order to compile and install the j_py_sad_correlation module
import py_symmetry as ps
import numpy as np
import time
import sys
sys.path.insert(0, 'build/lib.macosx-10.6-x86_64-2.7')

fHtsFull = np.random.random((400,500)).astype(np.complex64)
temp = np.random.random((400)).astype(np.complex64)
#temp = np.exp((1j * (1+padLength) * 2*np.pi / self.fshape[0]) * np.arange(self.fshape[0]))
numRepeats = 1000

t1 = time.time()
result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
result[:,0] = fHtsFull[:,0].conj()*temp
for i in range(1,fHtsFull.shape[1]):
    result[:,i] = (fHtsFull[:,fHtsFull.shape[1]-i].conj()*temp)
print('python took', time.time()-t1)

t1 = time.time()
for n in range(numRepeats):
    result2 = ps.mirrorX(fHtsFull, temp)
print('c took', (time.time()-t1)/numRepeats)

print(np.max(np.abs(result - result2)));


temp = np.random.random((500)).astype(np.complex64)
t1 = time.time()
result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
result[0] = fHtsFull[0].conj()*temp
for i in range(1,fHtsFull.shape[0]):
    result[i] = (fHtsFull[fHtsFull.shape[0]-i].conj()*temp)
print('python took', time.time()-t1)

t1 = time.time()
for n in range(numRepeats):
    result2 = ps.mirrorY(fHtsFull, temp)
print('c took', (time.time()-t1)/numRepeats)

print(np.max(np.abs(result - result2)));
