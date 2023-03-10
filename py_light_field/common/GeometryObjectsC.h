/*
 *  GeometryObjectsC.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *
 */

#ifndef __GEOMETRY_OBJECTS_C_H__
#define __GEOMETRY_OBJECTS_C_H__ 1

struct IntegerPoint
{
	int x, y;
	IntegerPoint() : x(0), y(0) { }
	IntegerPoint(int a, int b) : x(a), y(b) { }
	IntegerPoint& operator += (IntegerPoint n) { x += n.x; y += n.y; return *this; }
	IntegerPoint& operator -= (IntegerPoint n) { x -= n.x; y -= n.y; return *this; }
};

// Sadly I can't seem to get pass-by-reference to work with this - some of my ObjC property-based
// code doesn't compile if I pass a and b by reference. Not sure if this would be fix-able, but
// I'm just going to leave it as-is for now.
inline bool operator!=(IntegerPoint a, IntegerPoint b) { return ((a.x != b.x) || (a.y != b.y)); }
inline IntegerPoint operator+(IntegerPoint a, IntegerPoint b) { return IntegerPoint(a.x+b.x, a.y+b.y); }
inline IntegerPoint operator-(IntegerPoint a, IntegerPoint b) { return IntegerPoint(a.x-b.x, a.y-b.y); }

struct IntegerPoint3D
{
    int x, y, z;
    IntegerPoint3D() : x(0), y(0), z(0) { }
    IntegerPoint3D(int a, int b, int c) : x(a), y(b), z(c) { }
    IntegerPoint3D& operator += (IntegerPoint3D n) { x += n.x; y += n.y; z += n.z; return *this; }
    IntegerPoint3D& operator -= (IntegerPoint3D n) { x -= n.x; y -= n.y; z -= n.z; return *this; }
};

// Sadly I can't seem to get pass-by-reference to work with this - some of my ObjC property-based
// code doesn't compile if I pass a and b by reference. Not sure if this would be fix-able, but
// I'm just going to leave it as-is for now.
inline bool operator!=(IntegerPoint3D a, IntegerPoint3D b) { return ((a.x != b.x) || (a.y != b.y) || (a.z != b.z)); }
inline IntegerPoint3D operator+(IntegerPoint3D a, IntegerPoint3D b) { return IntegerPoint3D(a.x+b.x, a.y+b.y, a.z+b.z); }
inline IntegerPoint3D operator-(IntegerPoint3D a, IntegerPoint3D b) { return IntegerPoint3D(a.x-b.x, a.y-b.y, a.z-b.z); }

#endif
