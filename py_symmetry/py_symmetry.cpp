#define PY_ARRAY_UNIQUE_SYMBOL j_symmetry_pyarray
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "common/jAssert.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jPythonCommon.h"
#include "common/PIVImageWindow.h"

template<class TYPE> void SetImageWindowForPythonWindow(ImageWindow<TYPE> &imageWindow, JPythonArray2D<TYPE> &pythonWindow)
{
    imageWindow.baseAddr = pythonWindow.Data();
    imageWindow.width = pythonWindow.Dims()[1];
    imageWindow.height = pythonWindow.Dims()[0];
    imageWindow.elementsPerRow = pythonWindow.Strides()[0];
}

typedef complex<float> TYPE;

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
    npy_intp output_dims[2] = { aa.Dims()[0], aa.Dims()[1] };
    PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_CFLOAT);
    JPythonArray2D<TYPE> rr(r);
    ImageWindow<TYPE> result;
    SetImageWindowForPythonWindow(result, rr);
    
    int height = aa.Dims()[0];
    int width = aa.Dims()[1];
    if (x)
    {
        // result[:,0] = fHtsFull[:,0].conj()*temp
        for (int y = 0; y < height; y++)
            result.SetXY(0, y, conj(fHtsFull.PixelXY(0,y)) * temp[y]);
        // for i in range(1,fHtsFull.shape[1]):
        //     result[:,i] = (fHtsFull[:,fHtsFull.shape[1]-i].conj()*temp)
        for (int y = 0; y < height; y++)
            for (int x = 1; x < width; x++)
                result.SetXY(x, y, conj(fHtsFull.PixelXY(width-x,y)) * temp[y]);
    }
    else
    {
        // result[0] = fHtsFull[0].conj()*temp
        for (int x = 0; x < width; x++)
            result.SetXY(x, 0, conj(fHtsFull.PixelXY(x,0)) * temp[x]);
        // for i in range(1,fHtsFull.shape[0]):
        //    result[i] = (fHtsFull[fHtsFull.shape[0]-i].conj()*temp)
        for (int y = 1; y < height; y++)
            for (int x = 0; x < width; x++)
                result.SetXY(x, y, conj(fHtsFull.PixelXY(x,height-y)) * temp[x]);
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
