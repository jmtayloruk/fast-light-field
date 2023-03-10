#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "jPythonArray.h"
#include <complex>

template<> int ArrayType<double>(void) { return NPY_DOUBLE; }
template<> int ArrayType<float>(void) { return NPY_FLOAT; }
template<> int ArrayType<int>(void) { return NPY_INT32; }
template<> int ArrayType<unsigned char>(void) { return NPY_UBYTE; }
template<> int ArrayType<unsigned short>(void) { return NPY_USHORT; }
template<> int ArrayType< std::complex<float> >(void) { return NPY_CFLOAT; }
template<> int ArrayType< std::complex<double> >(void) { return NPY_CDOUBLE; }

const char *pythonArrayTypeStrings[] =
{ "np.bool",
    "np.byte/np.int8",
    "np.ubyte/np.uint8",
    "np.short/np.int16",
    "np.ushort/np.uint16",
    "np.intc [probably int]",
    "np.uintc [probably uint]",
    "np.int_ [probably long]",
    "np.uint [probably ulong]",
    "np.longlong",
    "np.ulonglong",
    "np.single [probably float32]",
    "np.double [probably float64]",
    "np.longdouble",
    "np.csingle [probably complex64]",
    "np.cdouble [probably complex128]",
    "np.clongdouble",
    "object",
    "string",
    "np.unicode",
    "void" };

const char *StringForPythonType(int objType)
{
    if ((size_t)objType < sizeof(pythonArrayTypeStrings)/sizeof(pythonArrayTypeStrings[0]))
        return pythonArrayTypeStrings[objType];
    return "<unknown>";
}
