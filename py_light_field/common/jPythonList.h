#ifndef __JPYTHONLIST_H__
#define __JPYTHONLIST_H__ 1

#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <stdlib.h>
#include "jAssert.h"
#include "jPythonCommon.h"

class JPythonList
{
  protected:
	PyObject *list;
  public:
	JPythonList(PyObject *in) : list(in)
	{
		RequireObjectType(in, PyList_Type);
	}
	JPythonList(void)
	{
		// Create an empty list
		list = PyList_New(0);
	}
	
	size_t Length(void) const { return PyList_Size(list); }
	bool ItemIsNone(size_t i) { return ObjectIsNone(PyList_GetItem(list, i)); }
	JPythonList ItemAsList(size_t i) { return JPythonList(PyList_GetItem(list, i)); }
	template<class Type> Type ItemAsNumArray(size_t i)
	{
		PyObject *result = PyList_GetItem(list, i);
		RequireObjectType(result, PyArray_Type);
		return Type(result);
	}		
	double ItemAsDouble(size_t i)
	{
		PyObject *result = PyList_GetItem(list, i);
		RequireObjectType(result, PyFloat_Type);
		return PyFloat_AsDouble(result);
	}
	long ItemAsLong(size_t i)
	{
		PyObject *result = PyList_GetItem(list, i);
		RequireObjectType(result, PyInt_Type);
		return PyInt_AsLong(result);
	}
	void AppendItem(PyObject *item)
	{
		PyList_Append(list, item);
	}
	PyObject *ListObject(void) { return list; }
};

#endif
