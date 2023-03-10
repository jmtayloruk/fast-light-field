#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "jPythonCommon.h"

void RequireObjectType(PyObject *obj, PyTypeObject &type)
{
	bool ok = PyObject_TypeCheck(obj, &type);
	if (!ok)
	{
		printf("ERROR - got object type %s, expected %s\n", obj->ob_type->tp_name, type.tp_name);
		printf("%p %p %p %p %s\n", obj, Py_None, obj->ob_type, Py_None->ob_type, obj->ob_type->tp_name);
		ALWAYS_ASSERT(ok);
	}
}

bool ObjectIsNone(PyObject *obj)
{
	return (obj == Py_None);
}
