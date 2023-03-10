/*
 *	jComplexAsVector.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class to handle complex numbers, based on the Altivec/SSE vector double type.
 *	See discussion in jComplex.h on the efficiency of this implementation.
 */

#ifndef __JCOMPLEX_AS_VECTOR_H__
#define __JCOMPLEX_AS_VECTOR_H__

#include "VectorTypes.h"
#include "VectorFunctions.h"

#ifdef __GSL_COMPLEX_H__
	#include <gsl/gsl_complex_math.h>
#endif

class jComplexAsVector
{
	// This class represents a complex number, with operator overloading allowing instance variables to be included
	// in standard arithmetic expressions without any special treatment.
	// Note that not all operators are implemented - I have only implemented the ones I need.
  protected:
	vDouble __z;
	
  public:

	double real(void) const { return vLower(__z); }
	double imag(void) const { return vLower(vSwapD(__z)); } 
	vDouble z(void) const { return __z; }
	void SetReIm(double inRe, double inIm) { __z = (vDouble) { inRe, inIm }; }
	
	jComplexAsVector() { SetReIm(0.0, 0.0); }
	jComplexAsVector(double n) { SetReIm(n, 0); }
	jComplexAsVector(double inRe, double inIm) { SetReIm(inRe, inIm); }
	jComplexAsVector(vDouble inZ) : __z(inZ) { }
	jComplexAsVector(const jComplexAsVector &inZ) { __z = inZ.z(); }
	jComplexAsVector(const complex<double> &inZ) { SetReIm(inZ.real(), inZ.imag()); }
  #ifdef __GSL_COMPLEX_H__
	explicit jComplexAsVector(const gsl_complex &z);
  #endif
	
	jComplexAsVector& operator += (const jComplexAsVector &n) { __z = _mm_add_pd(__z, n.z()); return *this; }
	jComplexAsVector operator + (const jComplexAsVector &n) const { return jComplexAsVector(*this) += n; }
	jComplexAsVector& operator -= (const jComplexAsVector &n) {__z = _mm_sub_pd(__z, n.z()); return *this; }
	jComplexAsVector operator - (const jComplexAsVector &n) const { return jComplexAsVector(*this) -= n; }
	jComplexAsVector& operator *= (const jComplexAsVector &n)
	{
		vDouble x = __z;
		vDouble y = n.z();
		vDouble ySwap = vSwapD(y);	// { Im y, Re y }
		vDouble rrii = _mm_mul_pd(x, y);					// { (Re x * Re x), (Im x * Im y) }
		vDouble riir = _mm_mul_pd(x, ySwap);				// { (Re x * Im y), (Im x * Re y) }
		// These next two instructions contain redundant work that could
		// be combined if multiplying two quantities simultaneously
		vDouble re = _mm_hsub_pd(rrii, rrii);			// (Re x * Re x) - (Im x * Im y)
		vDouble im = _mm_hadd_pd(riir, riir);			// (Re x * Im y) + (Im x * Re y)
		__z = _mm_shuffle_pd(re, im, _MM_SHUFFLE2(0, 0));		// { Re x*y, Im x*y }
		return *this;
	}

	jComplexAsVector operator * (const jComplexAsVector &n) const { return jComplexAsVector(*this) *= n; }
	jComplexAsVector& operator /= (const jComplexAsVector &n)
	{
		return operator*= (n.conj() / n.norm());
	}
	jComplexAsVector operator / (const jComplexAsVector &n) const { return jComplexAsVector(*this) /= n; }

	jComplexAsVector& operator += (const double &n) { __z = _mm_add_sd(__z, (vDouble){n, 0.0}); return *this; }
	jComplexAsVector operator + (const double &n) const { return jComplexAsVector(*this) += n; }
	jComplexAsVector& operator -= (const double &n) { __z = _mm_sub_sd(__z, (vDouble){n, 0.0}); return *this; }
	jComplexAsVector operator - (const double &n) const { return jComplexAsVector(*this) -= n; }
	jComplexAsVector& operator *= (double n) { __z = _mm_mul_pd(__z, (vDouble){n, n}); return *this; }
	jComplexAsVector operator * (double n) const { return jComplexAsVector(*this) *= n; }
	jComplexAsVector& operator /= (double n) {  __z = _mm_div_pd(__z, (vDouble){n, n}); return *this; }
	jComplexAsVector operator / (double n) const { return jComplexAsVector(*this) /= n; }
	
	jComplexAsVector& operator = (double n) { SetReIm(n, 0); return *this; }

	// I want to keep this as a static member function to avoid confusion in the calling code
	// A particular concern is distinguishing it from integer loop variables named i
	static jComplexAsVector i(void) { return jComplexAsVector(0, 1); }

	double norm(void) const { vDouble squares = _mm_mul_pd(__z, __z); return vLower(_mm_hadd_pd(squares, squares)); }
	double Intensity(void) const { return norm(); }
	double abs(void) const { return sqrt(Intensity()); }
	jComplexAsVector conj(void) const { return jComplexAsVector(_mm_xor_pd(__z, (vDouble) { 0.0, -0.0 })); }
	double Theta(void) const { return atan2(imag(), real()); }
	void Print(const char *suffix = "") const { printf("{%lg,%lg}%s", real(), imag(), suffix); }
	
  #ifdef __GSL_COMPLEX_H__
 	static void Set(jComplexAsVector &outZ, gsl_complex inZ)
	{
		outZ = jComplexAsVector(GSL_REAL(inZ), GSL_IMAG(inZ));
	}
	
	static gsl_complex ToGSL(const jComplexAsVector &z)
	{
		return gsl_complex_rect(z.real(), z.imag());
	}
  #endif
	complex<float> ToFloat(void) const
	{
		return complex<float>(float(real()), float(imag()));
	}
};

inline jComplexAsVector operator*(const double l, const jComplexAsVector &r)
{
	return r * l;
}

inline jComplexAsVector operator-(const double l, const jComplexAsVector &r)
{
	return jComplexAsVector(l - r.real(), -r.imag());
}

inline jComplexAsVector operator-(const jComplexAsVector &r)
{
	return 0.0 - r;
}

inline jComplexAsVector operator+(const jComplexAsVector &r)
{
	return r;
}

inline jComplexAsVector operator+(const double l, const jComplexAsVector &r)
{
	return jComplexAsVector(l + r.real(), r.imag());
}

inline jComplexAsVector operator/(const double l, const jComplexAsVector &r)
{
	return l * r.conj() / r.norm();
}

inline bool operator != (const jComplexAsVector& x, const jComplexAsVector& y)
{
	return x.real() != y.real() || x.imag() != y.imag();
}

inline bool operator != (const jComplexAsVector &x, const double y)
{
	return x.real() != y || x.imag() != 0.0;
}

inline bool operator == (const jComplexAsVector &x, const jComplexAsVector &y)
{
	return x.real() == y.real() && x.imag() == y.imag();
}
	
inline bool operator == (const jComplexAsVector &x, const double y)
{
	return x.real() == y && x.imag() == 0.0;
}

inline double norm(const jComplexAsVector &z)
{
	return z.norm();
}

inline double abs(const jComplexAsVector &z)
{
	return z.abs();
}

inline jComplexAsVector polar(const double __rho, const double __theta)
{
	return jComplexAsVector(__rho * cos(__theta), __rho * sin(__theta));
}

inline jComplexAsVector sin(const jComplexAsVector& __z)
{
	const double __x = __z.real();
	const double __y = __z.imag();
	return jComplexAsVector(sin(__x) * cosh(__y), cos(__x) * sinh(__y)); 
}

inline jComplexAsVector cos(const jComplexAsVector& __z)
{
	const double __x = __z.real();
	const double __y = __z.imag();
	return jComplexAsVector(cos(__x) * cosh(__y), -sin(__x) * sinh(__y));
}

inline double arg(const jComplexAsVector& __z)
{
	return atan2(__z.imag(), __z.real());
}

inline jComplexAsVector log(const jComplexAsVector& __z)
{
	return jComplexAsVector(log(abs(__z)), arg(__z));
}

inline jComplexAsVector exp(const jComplexAsVector& __z)
{
	return polar(exp(__z.real()), __z.imag());
}

inline jComplexAsVector pow(const jComplexAsVector& __x, const double __y)
{
	if (__x.imag() == 0.0 && __x.real() > 0.0)
		return pow(__x.real(), __y);

	jComplexAsVector __t = log(__x);
	return polar(exp(__y * __t.real()), __y * __t.imag());
}

inline jComplexAsVector sqrt(const jComplexAsVector& __z)
{
	double __x = __z.real();
	double __y = __z.imag();

	if (__x == 0.0)
	{
		double __t = sqrt(fabs(__y) / 2);
		return jComplexAsVector(__t, __y < 0.0 ? -__t : __t);
	}
	else
	{
		double __t = sqrt(2 * (abs(__z) + fabs(__x)));
		double __u = __t / 2;
		return __x > 0.0
				? jComplexAsVector(__u, __y / __t)
				: jComplexAsVector(fabs(__y) / __t, __y < 0.0 ? -__u : __u);
	}
}

inline jComplexAsVector conj(const jComplexAsVector &z)
{
	return z.conj();
}

inline double real(const jComplexAsVector &z)
{
	return z.real();
}

inline double imag(const jComplexAsVector &z)
{
	return z.imag();
}

#endif
