//
//  jCoord.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	C++ objects that wrap 2D and 3D vector coordinates
//	The idea is that these can be acted on like primitive types, using standard C++ operators for arithmetic.
//	Various vector-related utility functions are also defined.
//
//	Note that there is currently no type distinction between cartesian
//	and spherical coordinates, for example, and it is down to the caller
//	to keep track (e.g. through variable naming) which they are dealing with.
//

#ifndef __JCOORD_H__
#define __JCOORD_H__

#include <math.h>
#include <complex>
#include "jCommon.h"
#include "jComplex.h"

// This is defined in jUtils.h, but we get some circular header problems and it's easier just to define it here as well
extern const double PI;

/************************ 2D COORDINATE CLASS *******************************/

struct coord2
{
	// Note that not all operators are implemented - I have only implemented the ones I need.
  public:
	double	x, y;

	coord2() { }
	coord2(double inX, double inY) { x = inX; y = inY; }
	coord2& operator += (coord2 n) { x += n.x; y += n.y; return *this; }
	coord2 operator + (coord2 n) const { return coord2(*this) += n; }
	coord2& operator -= (coord2 n) { x -= n.x; y -= n.y; return *this; }
	coord2 operator - (coord2 n) const { return coord2(*this) -= n; }
	coord2& operator *= (double n) { x *= n; y *= n; return *this; }
	coord2 operator * (double n) const { return coord2(*this) *= n; }
	coord2& operator /= (double n) { x /= n; y /= n; return *this; }
	coord2 operator / (double n) const { return coord2(*this) /= n; }
	
	// Dot product
	double Dot(coord2 n) const { return x * n.x + y * n.y; }

	coord2& operator = (coord2 n) { x = n.x; y = n.y; return *this; }
	void Set(double inX, double inY) { x = inX; y = inY; }
	
	coord2& Normalize(void) { double invLen = 1 / sqrt(x*x + y*y); x *= invLen; y *= invLen; return *this; }
	double DistanceTo(coord2 &b) const { return sqrt(SQUARE(x - b.x) + SQUARE(y - b.y)); }
	double AngleWith(coord2 &b) const
	{
		double angle = b.Angle() - Angle();
		if (angle > PI) angle -= 2.0 * PI;
		if (angle < -PI) angle += 2.0 * PI;
		return angle;
	}
    coord2 RotateByRadians(double radians);
    coord2 RotateByDegrees(double degrees) { return RotateByRadians(degrees / 180.0 * PI); }

	inline double LengthSquared(void) const { return SQUARE(x) + SQUARE(y); }
	inline double Length(void) const { return sqrt(LengthSquared()); }
	double Angle(void) const { return atan2(y, x); }
	void Print(const char *suffix = "") const { printf("(%.12lg, %.12lg)%s", x, y, suffix); }
};

inline void Print(coord2 c, const char *suffix = "") { c.Print(suffix); }

inline coord2 operator*(const double l, const coord2 r)
{
	return r * l;
}

inline coord2 operator-(const double l, const coord2 r)
{
	return coord2(l - r.x, l - r.y);
}

inline coord2 operator-(const coord2 &r)
{
	return coord2(-r.x, -r.y);
}

/************************ 3D COORDINATE CLASS *******************************/

template<class Type> struct coord3T
{
	// Note that not all operators are implemented - I have only implemented the ones I need.
  public:
	Type	x, y, z;

	/*	Constructors - to create a coord3 object, write something like:
			coord3 a(1.0, 1.4, 1.1);
			a += coord3(2.4, 1.2, 0.0);		*/
	coord3T<Type>() : x(), y(), z() { }
	coord3T<Type>(Type inX, Type inY, Type inZ) { x = inX; y = inY; z=inZ; }
	static coord3T<Type> zero(void) { return coord3T<Type>(Type(0), Type(0), Type(0)); }

  #ifdef __GSL_COMPLEX_H__
	// These are extra constructors for interfacing with the open source GSL library
	// They should only be compiled in if you include the GSL headers
	// (so this file will work fine if you do not have GSL installed or have never heard of it)
				coord3T<Type>(gsl_vector *inVector);
				coord3T<Type>(gsl_vector *inVector, int offset);
	gsl_vector	*AllocGSLVector(void) const;
  #endif

	/*	Overloaded operators which allow you to write code like:
			myCoord = myOtherCoord + coord3(2.4, 1.2, 0.0);			*/
	coord3T<Type>& operator += (const coord3T<Type> &n) { x += n.x; y += n.y; z += n.z; return *this; }
	coord3T<Type> operator + (const coord3T<Type> &n) const { return coord3T<Type>(*this) += n; }
	coord3T<Type>& operator -= (const coord3T<Type> &n) { x -= n.x; y -= n.y; z -= n.z; return *this; }
	coord3T<Type> operator - (const coord3T<Type> &n) const { return coord3T<Type>(*this) -= n; }
	coord3T<Type>& operator *= (Type n) { x *= n; y *= n; z *= n; return *this; }
	coord3T<Type> operator * (Type n) const { return coord3T<Type>(*this) *= n; }
	coord3T<Type>& operator /= (Type n) { return (*this) *= (1/n); }
	coord3T<Type> operator / (Type n) const { return coord3T<Type>(*this) /= n; }
	inline coord3T<Type>& operator = (const coord3T<Type> &n) { x = n.x; y = n.y; z = n.z; return *this; }
	// Comparison operators (BUT be aware of the general issues of doing comparisons between floating-point variables!)
	bool operator == (const coord3T<Type> &n) { return ((x == n.x) && (y == n.y) && (z == n.z)); }
	bool operator != (const coord3T<Type> &n) { return !(operator==(n)); }
	
	// Dot and cross products - e.g. myCoord.Dot(myOtherCoord);
	inline Type Dot(const coord3T<Type> &n) const { return x * n.x + y * n.y + z * n.z; }
	coord3T<Type> Cross(const coord3T<Type> &n) const
	{
		return coord3T<Type>(y * n.z - z * n.y,
					  z * n.x - x * n.z,
					  x * n.y - y * n.x);
	}	

	// Utility functions. Some of these are used for cartesian to spherical conversions etc.
	inline void Set(Type inX, Type inY, Type inZ) { x = inX; y = inY; z = inZ; }
	void	RotateFromSphericalSystem(Type theta, Type phi)
	{
		// Convert from a local cartesian basis defined for a point in a spherical coordinate system,
		// to a pure global cartesian basis.
		/*		_r_ =		[ cos(phi)sin(theta), sin(phi)sin(theta), cos(theta) ]
		 _theta_ =	[ cos(phi)cos(theta), sin(phi)cos(theta), -sin(theta) ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]			*/
		Type		newX = cos(phi) * sin(theta) * x
		+ cos(phi) * cos(theta) * y
		- sin(phi) * z;
		Type		newY = sin(phi) * sin(theta) * x
		+ sin(phi) * cos(theta) * y
		+ cos(phi) * z;
		Type		newZ = cos(theta) * x
		- sin(theta) * y;
		Set(newX, newY, newZ);
	}
	void	RotateToSphericalSystem(Type theta, Type phi)
	{
		// Convert to a local cartesian basis defined for a point in a spherical coordinate system,
		// from a pure global cartesian basis.
		/*		_r_ =		[ cos(phi)sin(theta), sin(phi)sin(theta), cos(theta) ]
		 _theta_ =	[ cos(phi)cos(theta), sin(phi)cos(theta), -sin(theta) ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]			*/
		double		newX = cos(phi) * sin(theta) * x
		+ sin(phi) * sin(theta) * y
		+ cos(theta) * z;
		double		newY = cos(phi) * cos(theta) * x
		+ sin(phi) * cos(theta) * y
		- sin(theta) * z;
		double		newZ = -sin(phi) * x
		+ cos(phi) * y;
		Set(newX, newY, newZ);
	}

	void	RotateFromCylindricalSystem(Type phi)
	{
		/*		_r_ =		[ cos(phi), sin(phi), 0 ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]
		 _z_ =		[ 0, 0, 1 ]			*/
		double	newX = cos(phi) * x - sin(phi) * y;
		double	newY = sin(phi) * x + cos(phi) * y;
		double	newZ = z;
		Set(newX, newY, newZ);
	}
	void	RotateToCylindricalSystem(Type phi)
	{
		// Rotate to a cylindrical system for a point at angle phi in the cylindrical system
		/*		_r_ =		[ cos(phi), sin(phi), 0 ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]
		 _z_ =		[ 0, 0, 1 ]			*/
		double	newX = cos(phi) * x + sin(phi) * y;
		double	newY = -sin(phi) * x + cos(phi) * y;
		double	newZ = z;
		Set(newX, newY, newZ);
	}
	void	MultiplyByComponents(const coord3T<Type> &n) { x *= n.x; y *= n.y; z *= n.z; }
	
	// More utility functions
	coord3T<Type>& Normalize(void) { Type invLen = 1 / sqrt(x*x + y*y + z*z); x *= invLen; y *= invLen; z *= invLen; return *this; }
	inline Type SquaredDistanceTo(const coord3T<Type> &b) const { return (*this - b).LengthSquared(); }
	inline Type DistanceTo(const coord3T<Type> &b) const { return (*this - b).Length(); }
	inline Type LengthSquared(void) const { return SQUARE(x) + SQUARE(y) + SQUARE(z); }
	inline Type Length(void) const { return sqrt(LengthSquared()); }

	void Print(const char *suffix = "") const;
	
	// Extract one indexed component of the vector
	// This is not very efficient. To improve the efficiency of this function, it would be nice to redefine x, y, z as a 3 element array, but that will alter rather a lot of the arithmetic code in this struct definition!
	Type component(int c) { return (c==0) ? x : ((c==1) ? y : z); }
};
typedef coord3T<double> coord3;
typedef coord3T<jreal> coord3R;

typedef std::vector<coord3> coord3Vector;

void Print(coord3 c, const char *suffix = "");

/*	More operator overloading to allow mixing with scalar values
	e.g. myCoord = 3.0 * myOtherCoord;	*/
template<class Type> inline coord3T<Type> operator*(const Type l, const coord3T<Type> r)
{
	return r * l;
}

template<class Type> inline coord3T<Type> operator-(const Type l, const coord3T<Type> r)
{
	// Subtract from a scalar. Think carefully about whether you really want to do this!
	return coord3(l - r.x, l - r.y, l - r.z);
}

template<class Type> inline coord3T<Type> operator-(const coord3T<Type> &r)
{
	// Negation operator
	return coord3(-r.x, -r.y, -r.z);
}

/************************ COMPLEX 3D COORDINATE CLASS *******************************/

/*	It would be nice to have a typed differentiation between polar and cartesian coordinates
	to prevent them being mixed by mistake. Can't do that very easily with subclasses, though
	because all the operators return coordC3, so an action like theCoord = theCoord * 2 doesn't compile.
	Might be possible to achieve what I want using a dummy template argument to a templated coordC3.	*/
using std::conj;
template<class Type, class DoubleType> struct coordC3T
{
	// Note that not all operators are implemented - I have only implemented the ones I need.
  public:
	Type	x, y, z;

	coordC3T<Type, DoubleType>() : x(), y(), z() { }
	coordC3T<Type, DoubleType>(const Type &inX, const Type &inY, const Type &inZ) { Set(inX, inY, inZ); }
	explicit coordC3T<Type, DoubleType>(coord3T<DoubleType> r) : x(r.x), y(r.y), z(r.z) { }
	static coordC3T<Type, DoubleType> zero(void) { return coordC3T<Type, DoubleType>(DoubleType(0), DoubleType(0), DoubleType(0)); }

	coordC3T<Type, DoubleType>& operator += (const coordC3T<Type, DoubleType> &n) { x += n.x; y += n.y; z += n.z; return *this; }
	coordC3T<Type, DoubleType> operator + (const coordC3T<Type, DoubleType> &n) const { return coordC3T<Type, DoubleType>(*this) += n; }
	coordC3T<Type, DoubleType>& operator -= (const coordC3T<Type, DoubleType> &n) { x -= n.x; y -= n.y; z -= n.z; return *this; }
	coordC3T<Type, DoubleType> operator - (const coordC3T<Type, DoubleType> &n) const { return coordC3T<Type, DoubleType>(*this) -= n; }
	coordC3T<Type, DoubleType>& operator *= (DoubleType n) { x *= n; y *= n; z *= n; return *this; }
	coordC3T<Type, DoubleType> operator * (DoubleType n) const { return coordC3T<Type, DoubleType>(*this) *= n; }
	coordC3T<Type, DoubleType>& operator /= (DoubleType n) { return (*this) *= (1/n); }
	coordC3T<Type, DoubleType> operator / (DoubleType n) const { return coordC3T<Type, DoubleType>(*this) /= n; }
	coordC3T<Type, DoubleType>& operator *= (const Type &n) { x *= n; y *= n; z *= n; return *this; }
	coordC3T<Type, DoubleType> operator * (const Type &n) const { return coordC3T<Type, DoubleType>(*this) *= n; }
	bool operator == (const coordC3T<Type, DoubleType> &n) { return ((x == n.x) && (y == n.y) && (z == n.z)); }
	
	// Dot and cross products
	inline Type Dot(const coordC3T<Type, DoubleType> &n) const { return x * n.x + y * n.y + z * n.z; }
	inline Type Dot(const coord3T<DoubleType> &n) const { return x * n.x + y * n.y + z * n.z; }
	coordC3T<Type, DoubleType> Cross(const coordC3T<Type, DoubleType> &n) const
	{
		return coordC3T<Type, DoubleType>(y * n.z - z * n.y,
					   z * n.x - x * n.z,
					   x * n.y - y * n.x);
	}

	inline coordC3T<Type, DoubleType>& operator = (const coordC3T<Type, DoubleType> &n) { x = n.x; y = n.y; z = n.z; return *this; }
	inline void Set(const Type &inX, const Type &inY, const Type &inZ) { x = inX; y = inY; z = inZ; }
	void	RotateFromSphericalSystem(DoubleType theta, DoubleType phi)
	{
		// Convert from a local cartesian basis defined for a point in a spherical coordinate system,
		// to a pure global cartesian basis.
		/*		_r_ =		[ cos(phi)sin(theta), sin(phi)sin(theta), cos(theta) ]
		 _theta_ =	[ cos(phi)cos(theta), sin(phi)cos(theta), -sin(theta) ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]			*/
		Type	newX = cos(phi) * sin(theta) * x
		+ cos(phi) * cos(theta) * y
		- sin(phi) * z;
		Type	newY = sin(phi) * sin(theta) * x
		+ sin(phi) * cos(theta) * y
		+ cos(phi) * z;
		Type	newZ = cos(theta) * x
		- sin(theta) * y;
		Set(newX, newY, newZ);
	}
	void	RotateToSphericalSystem(DoubleType theta, DoubleType phi)
	{
		// Convert to a local cartesian basis defined for a point in a spherical coordinate system,
		// from a pure global cartesian basis.
		/*		_r_ =		[ cos(phi)sin(theta), sin(phi)sin(theta), cos(theta) ]
		 _theta_ =	[ cos(phi)cos(theta), sin(phi)cos(theta), -sin(theta) ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]			*/
		Type	newX = cos(phi) * sin(theta) * x
		+ sin(phi) * sin(theta) * y
		+ cos(theta) * z;
		Type	newY = cos(phi) * cos(theta) * x
		+ sin(phi) * cos(theta) * y
		- sin(theta) * z;
		Type	newZ = -sin(phi) * x
		+ cos(phi) * y;
		Set(newX, newY, newZ);
	}
	void	RotateFromCylindricalSystem(DoubleType phi)
	{
		/*		_r_ =		[ cos(phi), sin(phi), 0 ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]
		 _z_ =		[ 0, 0, 1 ]			*/
		Type	newX = cos(phi) * x - sin(phi) * y;
		Type	newY = sin(phi) * x + cos(phi) * y;
		Type	newZ = z;
		Set(newX, newY, newZ);
	}
	void	RotateToCylindricalSystem(DoubleType phi)
	{
		/*		_r_ =		[ cos(phi), sin(phi), 0 ]
		 _phi_ =		[ -sin(phi), cos(phi), 0 ]
		 _z_ =		[ 0, 0, 1 ]			*/
		Type	newX = cos(phi) * x + sin(phi) * y;
		Type	newY = -sin(phi) * x + cos(phi) * y;
		Type	newZ = z;
		Set(newX, newY, newZ);
	}
	
	inline DoubleType LengthSquared(void) const { return SQUARE(x.real()) + SQUARE(x.imag()) + SQUARE(y.real()) + SQUARE(y.imag()) + SQUARE(z.real()) + SQUARE(z.imag()); }
	inline DoubleType Length(void) const { return sqrt(LengthSquared()); }
	coord3T<DoubleType> component_abs(void) const { return coord3T<DoubleType>(abs(x), abs(y), abs(z)); }
	coord3T<DoubleType> real(void) const { return coord3T<DoubleType>(x.real(), y.real(), z.real()); }
	coord3T<DoubleType> imag(void) const { return coord3T<DoubleType>(x.imag(), y.imag(), z.imag()); }
	coordC3T<Type, DoubleType> conj(void) const { return coordC3T<Type, DoubleType>(::conj(x), ::conj(y), ::conj(z)); }
	void Print(const char *suffix = "") const;
};
typedef coordC3T<jComplex, double> coordC3;
typedef coordC3T<jComplexR, jreal> coordC3R;

void Print(coordC3 c, const char *suffix = "");

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> operator*(const Type &l, const coordC3T<Type, DoubleType> &r)
{
	return r * l;
}

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> operator*(const DoubleType &l, const coordC3T<Type, DoubleType> &r)
{
	return r * l;
}

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> operator-(const Type &l, const coordC3T<Type, DoubleType> &r)
{
	return coordC3T<Type, DoubleType>(l - r.x, l - r.y, l - r.z);
}

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> operator-(const coordC3T<Type, DoubleType> &r)
{
	return coordC3T<Type, DoubleType>(-r.x, -r.y, -r.z);
}

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> operator * (const coord3T<DoubleType> &a, Type n) { return coordC3T<Type, DoubleType>(a.x, a.y, a.z) * n; }

template<class Type, class DoubleType> inline coordC3T<Type, DoubleType> conj(const coordC3T<Type, DoubleType> &r)
{
	return r.conj();
}


template<class Type> coord3T<Type> CartesianToSpherical(coord3T<Type> source);
template<class Type> coord3T<Type> SphericalToCartesian(coord3T<Type> source);
coord3 CartesianToCylindrical(coord3 source);
coord3 CylindricalToCartesian(coord3 source);
//coordC3 CartesianToSpherical(coordC3 source);
//coordC3 SphericalToCartesian(coordC3 source);
coordC3 RotateFromSphericalSystem(coordC3 c, double theta, double phi);
coordC3 RotateToSphericalSystem(coordC3 c, double theta, double phi);
coord3 RotateFromSphericalSystem(coord3 c, double theta, double phi);
coord3 RotateToSphericalSystem(coord3 c, double theta, double phi);
coordC3 RotateFromCylindricalSystem(coordC3 c, double phi);
coordC3 RotateToCylindricalSystem(coordC3 c, double phi);

/************************ UTILITY FUNCTIONS *******************************/

inline coordC3 ConvertToComplex(coord3 &a)
{
	// Type conversion from a real 3D coordinate to a complex 3D coordinate
	return coordC3(a.x, a.y, a.z);
}

/*	Some more rotation utility functions.
	These ones are templated so you can use either the coord3 or the coordC3 class with them
	e.g.
		coord3 myCoord(0, 0, 0);
		coordC3 myComplexCoord(0, 0, 0);
		myCoord = RotateInXYPlane(myCoord, 3.14159);					// both these
		myComplexCoord = RotateInXYPlane(myComplexCoord, 3.14159);		// are valid
*/

template<class COORD, class PhiType> COORD RotateInXYPlane(COORD v, PhiType phi)
{
	PhiType sinPhi = sin(phi), cosPhi = cos(phi);
	return COORD(v.x * cosPhi - v.y * sinPhi,
				 v.x * sinPhi + v.y * cosPhi,
				 v.z);
}

template<class COORD, class PhiType> COORD RotateInXZPlane(COORD v, PhiType phi)
{
	PhiType sinPhi = sin(phi), cosPhi = cos(phi);
	return COORD(v.x * cosPhi - v.z * sinPhi,
				 v.y,
				 v.x * sinPhi + v.z * cosPhi);
}

coord3 AllowPrecisionLossReadingValue(coord3R val);
coord3R AllowPrecisionLossOnParam(coord3 val);
coordC3 AllowPrecisionLossReadingValue(coordC3R val);
coordC3R AllowPrecisionLossOnParam(coordC3 val);

#endif
