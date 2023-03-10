#ifndef __JPYTHONCOMMON_H__
#define __JPYTHONCOMMON_H__ 1

#include <Python.h>
#include <numpy/arrayobject.h>

void RequireObjectType(PyObject *obj, PyTypeObject &type);
bool ObjectIsNone(PyObject *obj);

#if PY_MAJOR_VERSION >= 3
    #define PyInt_Type PyLong_Type
    #define PyInt_AsLong PyLong_AsLong
#endif

#include "jPythonList.h"
#include "jPythonArray.h"

#endif
