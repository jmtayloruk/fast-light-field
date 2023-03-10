// new (fails)

#ifndef __JPYTHONARRAY_H__
#define __JPYTHONARRAY_H__ 1

#define NEW_CODE 1
#ifndef JPA_BOUNDS_CHECK
    // This should normally be set to 0, but (at the cost of a performance hit) it can be enabled
    // to perform bounds checking when accessing python arrays through my JPythonArray wrapper.
    #if 1
        #define JPA_BOUNDS_CHECK 0
    #else
        #warning "JPA bounds checking enabled (slow)"
        #define JPA_BOUNDS_CHECK 1
    #endif
#endif

#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include "jAssert.h"

// This should be specialized, in JPythonArray.cpp, for all required types
template<class Type> int ArrayType(void);

template<class Type> struct BackingData
{
    Type *data;
    size_t refcount, allocatedSize;
    
    BackingData(size_t inSize)
    {
        allocatedSize = inSize*sizeof(Type);
        data = (Type *)malloc(allocatedSize);        // Note: I use malloc, and this means values are not initialized to zero on creation. Caller must zero explicitly if required.
        ALWAYS_ASSERT(data != NULL);
        refcount = 1;
    }
    
    void Release(void)
    {
        size_t oldVal = __sync_fetch_and_sub(&refcount, 1);
        ALWAYS_ASSERT(oldVal > 0);
        if (oldVal == 1)
            delete this;
    }

  private:        // Callers should always call Release()
    ~BackingData()
    {
        ALWAYS_ASSERT(refcount == 0);
        free(data);
    }
};

const char *StringForPythonType(int objType);

template<class Type> class JPythonArray
{
  protected:
#if NEW_CODE
    static const int kMaxDims = 4;
    npy_intp		dims[kMaxDims];
    npy_intp		strides[kMaxDims];
#else
    npy_intp		*dims;
    npy_intp		*strides;
#endif
    int				numDims;
    Type			*data;
    BackingData<Type>     *backingData;    // Only non-NULL if we allocated the data ourselves
    //	PyArrayObject	*obj;		// I prefer not to store the object, because I think that's easier when dealing with sub-arrays.
    // It may be helpful to refcount it, though
    
    // Making this protected to force subclasses to have their own constructors (to make sure 2D cannot be constructed using 1D, for example)
    JPythonArray(int _numDims)
    {
        backingData = NULL;
        data = NULL;
        numDims = _numDims;
    }
    
    void AllocDims(int inNum, npy_intp *inDims, npy_intp *inStrides, int divideFactor = 1)
    {
#if NEW_CODE
        ALWAYS_ASSERT(inNum <= kMaxDims);
        numDims = inNum;
        memcpy(dims, inDims, sizeof(npy_intp) * numDims);
#else
        numDims = inNum;
        dims = new npy_intp[numDims];
        memcpy(dims, inDims, sizeof(npy_intp[numDims]));
        strides = new npy_intp[numDims];
#endif
        ALWAYS_ASSERT(inStrides != NULL);
        for (int i = 0; i < numDims; i++)
            strides[i] = inStrides[i] / divideFactor;
    }
    
    static bool CheckArrayType(PyArrayObject *obj, int expectedDims = 0)
    {
        if (PyArray_TYPE(obj) != ArrayType())
        {
            // If this error is hit then the wrong array type was passed to the JPythonArray class
            // Note that array enums can be checked at anaconda/lib/python3.5/site-packages/numpy/core/include/numpy/ndarraytypes.h
            PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Array type %d (%s) didn't match expected type %d (%s)", PyArray_TYPE(obj), StringForPythonType(PyArray_TYPE(obj)), ArrayType(), StringForPythonType(ArrayType()));
            printf("Array type %d (%s) didn't match expected type %d (%s)\n", PyArray_TYPE(obj), StringForPythonType(PyArray_TYPE(obj)), ArrayType(), StringForPythonType(ArrayType()));
            throw std::invalid_argument("array type incorrect");
        }
        int dimCount = PyArray_NDIM(obj);
        if ((expectedDims != 0) && (dimCount != expectedDims))
        {
            // If this error is hit then an array with the wrong dimensions was passed to the JPythonArray class
            PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Array type check failed: array had the wrong number of dimensions (got %d, expected %d)\n", dimCount, expectedDims);
            printf("Array type check failed: array had the wrong number of dimensions (got %d, expected %d)\n", dimCount, expectedDims);
            throw std::invalid_argument("array dimensions incorrect");
        }
        return true;
    }	

  public:
	
	void Construct(PyArrayObject *obj, int expectedDims = 0)
	{
		CheckArrayType(obj, expectedDims);
		AllocDims(PyArray_NDIM(obj), PyArray_DIMS(obj), PyArray_STRIDES(obj), sizeof(Type));
        backingData = NULL;
        data = (Type *)PyArray_DATA(obj);
    }
    
    JPythonArray(PyArrayObject *obj, int expectedDims)
    {
        Construct(obj, expectedDims);
    }
    
    JPythonArray(PyObject *obj, int expectedDims)
    {
        if (PyArray_Check(obj))
            Construct((PyArrayObject *)obj, expectedDims);
        else
        {
            // If this error is hit then the wrong array type was passed to the JPythonArray class
            PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Object is not an array object");
        }
    }
    
    JPythonArray(Type *inData, int inNum, npy_intp *inDims, npy_intp *inStrides)
    {
        if (inData != NULL)
        {
            // We have been provided with a data buffer
            AllocDims(inNum, inDims, inStrides);
            backingData = NULL;
            data = inData;
        }
        else
        {
            // Caller expects us to allocate and manage a data buffer
            npy_intp tempStrides[inNum];
            if (inStrides == NULL)
            {
                // Caller expects us to choose suitable strides. We will just make it contiguous.
                int s = 1;
                for (int n = inNum - 1; n >= 0; n--)
                {
                    tempStrides[n] = s;
                    s *= inDims[n];
                }
                inStrides = tempStrides;
            }
            
            npy_intp largestStride = 0;
            size_t s = 0;
            for (int n = 0; n < inNum; n++)
            {
                if (inStrides[n] > largestStride)
                {
                    largestStride = inStrides[n];
                    s = inStrides[n] * inDims[n];
                }
            }
            AllocDims(inNum, inDims, inStrides);
            backingData = new BackingData<Type>(s);
            data = backingData->data;
        }
    }
    
    JPythonArray &operator =(const JPythonArray &copy)
    {
        memcpy(dims, copy.dims, sizeof(dims));
        memcpy(strides, copy.strides, sizeof(strides));
        ALWAYS_ASSERT(numDims == copy.numDims);     // We should already have been initialized as e.g. a 2D array, and it would be mad to copy between different types
        numDims = copy.numDims;
        data = copy.data;
        if (backingData != NULL)
            backingData->Release();
        backingData = copy.backingData;
        if (backingData != NULL)
            __sync_fetch_and_add(&backingData->refcount, 1);
        return *this;
    }
    
    virtual ~JPythonArray()
    {
        if (backingData != NULL)
            backingData->Release();
#if NEW_CODE
#else
        delete[] dims;
        delete[] strides;
#endif
    }
    
    void SetZero(void)
    {
        // Set every element to zero (taking correct account of strides)
        int i;
        
        // If things are contiguous (or if we allocated ourselves and so we know any gaps are just padding)
        // then we can do this much faster.
        bool isContiguous = true;
        if (backingData == NULL)
        {
            for (i = 1; i < numDims; i++)
                if (strides[i-1] != dims[i])
                    isContiguous = false;
        }
        
        if (isContiguous)
        {
            memset(data, 0, dims[0] * strides[0] * sizeof(Type));
        }
        else
        {
            npy_intp indices[numDims];
            memset(indices, 0, sizeof(indices));
            do
            {
                npy_intp offset = 0;
                for (i = 0; i < numDims; i++)
                    offset += strides[i] * indices[i];
                data[offset] = 0;
                for (i = numDims-1; i >= 0; i--)
                {
                    indices[i]++;
                    if (indices[i] == dims[i])
                        indices[i] = 0;
                    else
                        break;
                }
            }
            while (i != -1);
        }
    }
    
    npy_intp NumElements(void) const
    {
        npy_intp count = 1;
        for (int i = 0; i < numDims; i++)
            count *= dims[i];
        return count;
    }
    
    bool FinalDimensionUnitStride(void) const
    {
        return (strides[numDims-1] == 1);
    }
    
    bool Contiguous(void) const
    {
        npy_intp expected = 1;
        for (int i = numDims - 1; i >= 0; i--)
        {
            if (strides[i] != expected)
                return false;
            expected *= dims[i];
        }
        return true;
    }
    
    void SetData(Type *inData, npy_intp len)
    {
        ALWAYS_ASSERT(len == NumElements());
        if (Contiguous())
        {
            memcpy(data, inData, len * sizeof(Type));
        }
        else
        {
            int i;
            npy_intp inPos = 0;
            npy_intp indices[numDims];
            memset(indices, 0, sizeof(indices));
            do
            {
                npy_intp offset = 0;
                for (i = 0; i < numDims; i++)
                    offset += strides[i] * indices[i];
                data[offset] = inData[inPos++];
                for (i = numDims; i >= 0; i--)
                {
                    indices[i]++;
                    if (indices[i] == dims[i])
                        indices[i] = 0;
                    else
                        break;
                }
            }
            while (i != -1);
        }
    }
    
    int NDims(void) const { return numDims; }
    npy_intp *Dims(void) { return dims; }		// This should be const, and return a const array, but PyArray_SimpleNew takes a non-const parameter for some reason
    npy_intp *Strides(void) { return strides; }	// This should be const, and return a const array, but PyArray_SimpleNew takes a non-const parameter for some reason
    int Dims(int n) const { return (int)dims[n]; }
    int Strides(int n) const { return (int)strides[n]; }
    Type *Data(void) const { return data; }
    static int ArrayType(void) { return ::ArrayType<Type>(); }
};

template<class Type> class JPythonArray1D : public JPythonArray<Type>
{
public:
    JPythonArray1D(PyArrayObject *init) : JPythonArray<Type>(init, 1) { }
    JPythonArray1D(PyObject *init) : JPythonArray<Type>(init, 1) { }
    JPythonArray1D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 1, inDims, inStrides) { }
    
    JPythonArray1D(const JPythonArray1D<Type> &copy) : JPythonArray<Type>(1)
    {
        JPythonArray<Type>::operator=(copy);
    }
    
    Type &operator[](int i)		// Note we return a reference here, so that this can be used as an lvalue, e.g. my1DArray[0] = 1.0, or my2DArray[0][0] = 1.0;
    {
        //		printf("Access element %d of %d\n", i, JPythonArray<Type>::dims[0]);
#if JPA_BOUNDS_CHECK
        ALWAYS_ASSERT(i < JPythonArray<Type>::dims[0]);
        return JPythonArray<Type>::data[i * JPythonArray<Type>::strides[0]];
#else
        // TODO: I am also assuming a stride of 1 here. I need to think about how to enforce that. Possibly a second subclass that does the checking and can be constructed from JPythonArray1D
        return JPythonArray<Type>::data[i];
#endif
    }
    Type *ElementPtr(int x) { return (Type *)(((const char*)JPythonArray<Type>::data) + JPythonArray<Type>::strides[0] * x); }
    
    Type &GetIndex_CanPromote(int i)
    {
        // Behaves like operator[], but if we have a single value in the array then returns that value regardless of i
        // This isn't ideal - it's a way of working around the fact that the object used to initialize this array may be a scalar value
        if (JPythonArray<Type>::dims[0] == 1)
            return JPythonArray<Type>::data[0];
        else
            return operator[](i);
    }
};

template<class Type> class JPythonArray2D : public JPythonArray<Type>
{
public:
    JPythonArray2D(PyArrayObject *init) : JPythonArray<Type>(init, 2) { }
    JPythonArray2D(PyObject *init) : JPythonArray<Type>(init, 2) { }
    JPythonArray2D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 2, inDims, inStrides) { }
    JPythonArray2D(npy_intp *inDims) : JPythonArray<Type>(NULL, 2, inDims, NULL) { }
    JPythonArray2D(const JPythonArray2D<Type> &copy) : JPythonArray<Type>(2)
    {
        JPythonArray<Type>::operator=(copy);
    }
    
    JPythonArray1D<Type> operator[](int i)
    {
#if JPA_BOUNDS_CHECK
        ALWAYS_ASSERT(i < JPythonArray<Type>::dims[0]);
#endif
        return JPythonArray1D<Type>(JPythonArray<Type>::data + JPythonArray<Type>::strides[0] * i, JPythonArray<Type>::dims + 1, JPythonArray<Type>::strides + 1);
    }
    Type *ElementPtr(int y, int x) { return (Type *)(((const char*)JPythonArray<Type>::data) + JPythonArray<Type>::strides[1] * y + JPythonArray<Type>::strides[0] * x); }
};

template<class Type> class JPythonArray3D : public JPythonArray<Type>
{
public:
    JPythonArray3D(PyArrayObject *init) : JPythonArray<Type>(init, 3) { }
    JPythonArray3D(PyObject *init) : JPythonArray<Type>(init, 3) { }
    JPythonArray3D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 3, inDims, inStrides) { }
    JPythonArray3D(npy_intp *inDims) : JPythonArray<Type>(NULL, 3, inDims, NULL) { }
    JPythonArray3D(const JPythonArray3D<Type> &copy) : JPythonArray<Type>(3)
    {
        JPythonArray<Type>::operator=(copy);
    }
    
    JPythonArray2D<Type> operator[](int i)
    {
#if JPA_BOUNDS_CHECK
        ALWAYS_ASSERT(i < JPythonArray<Type>::dims[0]);
#endif
        return JPythonArray2D<Type>(JPythonArray<Type>::data + JPythonArray<Type>::strides[0] * i, JPythonArray<Type>::dims + 1, JPythonArray<Type>::strides + 1);
    }
};

template<class Type> class JPythonArray4D : public JPythonArray<Type>
{
public:
    JPythonArray4D(PyArrayObject *init) : JPythonArray<Type>(init, 4) { }
    JPythonArray4D(PyObject *init) : JPythonArray<Type>(init, 4) { }
    JPythonArray4D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 4, inDims, inStrides) { }
    JPythonArray4D(const JPythonArray4D<Type> &copy) : JPythonArray<Type>(4)
    {
        operator=(copy);
    }
    
    JPythonArray3D<Type> operator[](int i)
    {
#if JPA_BOUNDS_CHECK
        ALWAYS_ASSERT(i < JPythonArray<Type>::dims[0]);
#endif
        return JPythonArray3D<Type>(JPythonArray<Type>::data + JPythonArray<Type>::strides[0] * i, JPythonArray<Type>::dims + 1, JPythonArray<Type>::strides + 1);
    }
};

template<class Type> JPythonArray2D<Type> PromoteTo2D(PyArrayObject *init)
{
    if (PyArray_NDIM(init) == 1)
    {
        npy_intp dims[2] = { 1, PyArray_DIMS(init)[0] };
        npy_intp strides[2] = { 0, PyArray_STRIDES(init)[0] / sizeof(Type) };
        return JPythonArray2D<Type>((Type *)PyArray_DATA(init), dims, strides);
    }
    else
    {
        // This could fail (if for example we are given a 3D array), but if that happens then a suitable error should be reported
        return JPythonArray2D<Type>(init);
    }
}

#endif
