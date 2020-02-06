#define PY_ARRAY_UNIQUE_SYMBOL j_symmetry_pyarray
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "common/jAssert.h"
#include "common/VectorFunctions.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jPythonCommon.h"
#include "common/PIVImageWindow.h"

template<class TYPE> void SetImageWindowForPythonWindow(ImageWindow<TYPE> &imageWindow, JPythonArray2D<TYPE> &pythonWindow)
{
    imageWindow.baseAddr = pythonWindow.Data();
    imageWindow.width = pythonWindow.Dims(1);
    imageWindow.height = pythonWindow.Dims(0);
    imageWindow.elementsPerRow = pythonWindow.Strides(0);
}

typedef complex<float> TYPE;

void CalculateRow(TYPE *ac, const TYPE *fap, const TYPE *fbu, const TYPE emy, int lim)
{
    for (int x = 0; x < lim; x++)
    {
        ac[x] += fap[x] * fbu[x] * emy;
    }
}

void CalculateRowMirror(TYPE *ac, const TYPE *fap, const TYPE *fbu, const TYPE emy, int lim, int trueLength)
{
    ac[0] += fap[0] * conj(fbu[0]) * emy;
    for (int x = 1; x < lim; x++)
    {
        ac[x] += fap[x] * conj(fbu[trueLength-x]) * emy;
    }
}

extern "C" PyObject *special_fftconvolve(PyObject *self, PyObject *args)
{
    PyArrayObject *_accum, *_fa_partial, *_fb_unmirrored, *_expandMultiplier, *_mirrorXMultiplier;
    int validWidth;
    
    // parse the input arrays from *args
    if (!PyArg_ParseTuple(args, "OO!O!O!iO",
                          &_accum,
                          &PyArray_Type, &_fa_partial,
                          &PyArray_Type, &_fb_unmirrored,
                          &PyArray_Type, &_expandMultiplier,
                          &validWidth,
                          &_mirrorXMultiplier))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
    }

    if ((PyArray_TYPE(_fa_partial) != NPY_CFLOAT) ||
        (PyArray_TYPE(_fb_unmirrored) != NPY_CFLOAT))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unsuitable array types %d %d passed in", (int)PyArray_TYPE(_fa_partial), (int)PyArray_TYPE(_fb_unmirrored));
        return NULL;
    }

    JPythonArray3D<TYPE> fa_partial(_fa_partial);
    JPythonArray2D<TYPE> fb_unmirrored(_fb_unmirrored);
    JPythonArray1D<TYPE> expandMultiplier(_expandMultiplier);
    JPythonArray1D<TYPE> *mirrorXMultiplier = NULL;
    if (_mirrorXMultiplier != (PyArrayObject *)Py_None)
    {
        mirrorXMultiplier = new JPythonArray1D<TYPE>(_mirrorXMultiplier);
    }
    if (!gPythonArraysOK)
    {
        printf("Returning NULL due to python error\n");
        return NULL;        // When flag was cleared, a PyErr should have been set up with the details.
    }

    if ((fa_partial.Strides(2) != 1) ||
        (fb_unmirrored.Strides(1) != 1) ||
        (expandMultiplier.Strides(0) != 1) ||
        ((mirrorXMultiplier != NULL) && (mirrorXMultiplier->Strides(0) != 1)))
    {
        delete mirrorXMultiplier;
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "One of the input arrays does not have unit stride (%zd %zd %zd)", fa_partial.Strides(2), fb_unmirrored.Strides(1), expandMultiplier.Strides(0));
        return NULL;
    }

    // Create a new array for accum, if we were not passed in an existing one
    npy_intp output_dims[3] = { fa_partial.Dims(0), fb_unmirrored.Dims(0), validWidth };
    PyArrayObject *__accum = _accum;
    if (__accum == (PyArrayObject *)Py_None)
    {
        __accum = (PyArrayObject *)PyArray_ZEROS(3, output_dims, NPY_CFLOAT, 0);
    }
    else
        Py_INCREF(__accum);      // Although I haven't found a definite statement to this effect, I suspect I need to incref on an input variable because pyarray_return presumably decrefs it?
    JPythonArray3D<TYPE> accum(__accum);
    
    // Generate each strip in the output array
    for (int z = 0; z < output_dims[0]; z++)
    {
        if (mirrorXMultiplier == NULL)
        {
            for (int y = 0; y < output_dims[1]; y++)
            {
                int ya = y % fa_partial.Dims(1);
                // Note that the array accessors take time, so must be lifted out of CalculateRow.
                // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
                // What I'd really like to do is to profile this code and see how well it does, because there may be other things I can improve.
                CalculateRow(&accum[z][y][0], &fa_partial[z][ya][0], &fb_unmirrored[y][0], expandMultiplier[y], output_dims[2]);
            }
        }
        else
        {
            for (int y = 0; y < output_dims[1]; y++)
            {
                int ya = y % fa_partial.Dims(1);
                CalculateRowMirror(&accum[z][y][0], &fa_partial[z][ya][0], &fb_unmirrored[y][0], expandMultiplier[y] * (*mirrorXMultiplier)[y], validWidth, fb_unmirrored.Dims(1));
            }
        }
    }
    
    if (mirrorXMultiplier != NULL)
        delete mirrorXMultiplier;
    return PyArray_Return(__accum);
}


extern "C" PyObject *mirrorXY(PyObject *self, PyObject *args, bool x)
{
    // This function takes two parameters, a and b, which should be numpy arrays.
    // It is expected that b will be larger than a, and an array will be returned
    // giving the SAD values between b and every possible shifted position of a within b

    // inputs
	PyArrayObject *a, *b;

	// parse the input arrays from *args
	if (!PyArg_ParseTuple(args, "O!O!",
			&PyArray_Type, &a,
			&PyArray_Type, &b))
	{
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse array!");
		return NULL;
	}
    if ((PyArray_TYPE(a) != NPY_CFLOAT) ||
        (PyArray_TYPE(b) != NPY_CFLOAT))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unsuitable array types %d %d passed in", (int)PyArray_TYPE(a), (int)PyArray_TYPE(b));
    }
    
    JPythonArray2D<TYPE> aa(a);
    ImageWindow<TYPE> fHtsFull;
    SetImageWindowForPythonWindow(fHtsFull, aa);

    JPythonArray1D<TYPE> temp(b);

    // result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
    npy_intp output_dims[2] = { aa.Dims(0), aa.Dims(1) };
    PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_CFLOAT);
    JPythonArray2D<TYPE> rr(r);
    ImageWindow<TYPE> result;
    SetImageWindowForPythonWindow(result, rr);
    
    int height = aa.Dims(0);
    int width = aa.Dims(1);
    if (x)
    {
        // result[:,0] = fHtsFull[:,0].conj()*temp
        for (int y = 0; y < height; y++)
            result.SetXY(0, y, conj(fHtsFull.PixelXY(0,y)) * temp[y]);
        // for i in range(1,fHtsFull.shape[1]):
        //     result[:,i] = (fHtsFull[:,fHtsFull.shape[1]-i].conj()*temp)
        for (int y = 0; y < height; y++)
        {
            int x = 1;
#if 0
            // This would be where I would write my own vectorized code.
            // However, the experience with the y case (below) is that actually the auto-vectorized code
            // is better than anything I can write by hand myself! (Which is no bad thing - saves me the effort!)
            for (; x+2 <= width; x++) {}
#endif
            
            for (; x < width; x++)
                result.SetXY(x, y, conj(fHtsFull.PixelXY(width-x,y)) * temp[y]);
        }
    }
    else
    {
        // result[0] = fHtsFull[0].conj()*temp
        for (int x = 0; x < width; x++)
            result.SetXY(x, 0, conj(fHtsFull.PixelXY(x,0)) * temp[x]);
        // for i in range(1,fHtsFull.shape[0]):
        //    result[i] = (fHtsFull[fHtsFull.shape[0]-i].conj()*temp)
        for (int y = 1; y < height; y++)
        {
            vFloat *tempPos = (vFloat *)temp.ElementPtr(0);
            vFloat *fPos = (vFloat *)(fHtsFull.baseAddr + (height-y)*fHtsFull.elementsPerRow);
            vFloat *rPos = (vFloat *)(result.baseAddr + y*result.elementsPerRow);
            int x = 0;
            if (false)//for (int x = 0; x < width/2; x++)
            {
                // This is slightly slower than the longhand C code. Why is that...? Not sure. It looks like the C code is already being vectorized?
                rPos[x] = vCMul(vNegateImag(fPos[x]), tempPos[x]);
                
//                result.SetXY(x, y, conj(fHtsFull.PixelXY(x,height-y)) * temp[x]);
            }
            for (x *= 2; x < width; x++)
                result.SetXY(x, y, conj(fHtsFull.PixelXY(x,height-y)) * temp[x]);
        }
    }
    return PyArray_Return(r);
}

extern "C" PyObject *mirrorX(PyObject *self, PyObject *args)
{
    return mirrorXY(self, args, true);
}

extern "C" PyObject *mirrorY(PyObject *self, PyObject *args)
{
    return mirrorXY(self, args, false);
}

/* Define a methods table for the module */

static PyMethodDef symm_methods[] = {
	{"mirrorX", mirrorX, METH_VARARGS},
    {"mirrorY", mirrorY, METH_VARARGS},
    {"special_fftconvolve", special_fftconvolve, METH_VARARGS},
	{NULL,NULL} };



/* initialisation - register the methods with the Python interpreter */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef py_symmetry =
{
    PyModuleDef_HEAD_INIT,
    "py_symmetry", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symm_methods
};

PyMODINIT_FUNC PyInit_py_symmetry(void)
{
    import_array();
    return PyModule_Create(&py_symmetry);
}

#else

extern "C" void initpy_symmetry(void)
{
    (void) Py_InitModule("py_symmetry", symm_methods);
    import_array();
}

#endif
