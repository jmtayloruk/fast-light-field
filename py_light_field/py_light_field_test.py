# Run the following command:
#	python setup.py build; python setup.py install
# in order to compile and install the py_light_field module
import py_light_field as plf
import numpy as np
import sys, time, warnings
#sys.path.insert(0, 'build/lib.macosx-10.6-x86_64-2.7')

warnings.warn("For now this script only tests mirrorX and mirrorY - not the more recent functions in this module")
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
    result2 = plf.mirrorX(fHtsFull, temp)
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
    result2 = plf.mirrorY(fHtsFull, temp)
print('c took', (time.time()-t1)/numRepeats)

print(np.max(np.abs(result - result2)));
