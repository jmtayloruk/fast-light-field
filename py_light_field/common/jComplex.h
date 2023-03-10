/*
 *	jComplex.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class to handle complex numbers.
 *	There are in fact two variants, one based on std::complex and one based on an altivec/SSE vector.
 *	The latter is in fact not very efficient on intel because the sort of instructions required for
 *	some operations are slow or unavailable. Fortunately this isn't too much of a problem because
 *	we are generally able to use jComplexPair for performance-critical code, and this works much
 *	better in a vector implementation.
 */
 
#ifndef __JCOMPLEX_H__
#define __JCOMPLEX_H__

#include <stdio.h>

#include <complex>
#include <vector>

#if COMPILE_JCOMPLEX_GSL_INTERFACE
	// We may want to link against GSL, but not include this in a prefix header
	// In that case the macro USES_GSL should be defined in order to ensure the
	// relevant parts of the jComplex class are defined
	#include <gsl/gsl_complex.h>
#endif


using std::complex;

#include "jCommon.h"
#include "jComplexAsStd.h"
#if __SSE3__
	#include "jComplexAsVector.h"
#endif

typedef jComplexAsStd jComplex;
typedef std::vector<jComplex> jComplexVector;

#ifdef USE_JREAL
	typedef jComplexAsStdBase<jreal> jComplexR;
	typedef std::vector<jComplexR> jComplexVectorR;
	void Print(jComplexR, const char *suffix = "");
#else
	typedef jComplex jComplexR;
	typedef jComplexVector jComplexVectorR;
#endif

jComplex PowerOfI(int n);

jComplex AllowPrecisionLossReadingValue(jComplexR val);
jComplexR AllowPrecisionLossOnParam(jComplex val);
#ifdef USE_JREAL
	inline jreal j_norm(const jComplexR &z) { return SQUARE(real(z)) + SQUARE(imag(z)); }
	jComplexR PowerOfI_r(int n);
	jComplexR exp_i(jreal radianAngle);
	jComplexR exp_i(jComplexR z);
#endif

using std::polar;
using std::norm;

inline jComplex exp_i(double radianAngle)
{
	// This is a bit circuitous to make it work with jComplexAsVector
	// The compiler should tidy it all up for us, though
	complex<double> result = std::polar(1.0, radianAngle);
	return jComplex(real(result), imag(result));
}

inline jComplex exp_i(jComplex z)
{
	// exp(i(a+ib)) = exp(ia) * exp(-b)
	return exp_i(z.real()) * exp(-z.imag());
}

inline jComplex PowerOfI(int n)
{
	const jComplex powers[4] = { jComplex(1, 0), jComplex(0, 1), jComplex(-1, 0), jComplex(0, -1) };
	return powers[n&3];		// Note can't write n%4 as this does the wrong thing for n<0 !
}

inline jComplexR PowerOfI_r(int n)
{
	const jComplexR powers[4] = { jComplexR(1, 0), jComplexR(0, 1), jComplexR(-1, 0), jComplexR(0, -1) };
	return powers[n&3];		// Note can't write n%4 as this does the wrong thing for n<0 !
}

inline jComplex powerOfMinusI(int n)
{
	n = (n & 3);
	if (n == 0)
		return 1.0;
	if (n == 1)
		return jComplex(0, -1);
	if (n == 2)
		return -1.0;
	return jComplex::i();
}

void Print(jComplex z, const char *suffix = "");
#ifdef __GSL_COMPLEX_H__
	void Print(gsl_complex z, const char *suffix = "");
#endif

inline double j_norm(const jComplex &z) { return SQUARE(real(z)) + SQUARE(imag(z)); }

inline jComplex cacos(const jComplex &z)
{
	return -jComplex::i() * log(z + jComplex::i() * sqrt(1.0 - z*z));
}

#endif
