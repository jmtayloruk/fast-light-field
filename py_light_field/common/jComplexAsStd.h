/*
 *	jComplexAsStd.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class to handle complex numbers, derived from std::complex.
 *	There is very little to this derived class - it just adds a few extra
 *	utility functions which may come in handy.
 */

#ifndef __JCOMPLEX_AS_STD_H__
#define __JCOMPLEX_AS_STD_H__

template<class Type> class jComplexAsStdBase : public complex<Type>
{
  public:
	jComplexAsStdBase<Type>() : complex<Type>() {}
	jComplexAsStdBase<Type>(Type x) : complex<Type>(x) { }
	jComplexAsStdBase<Type>(Type re, Type im) : complex<Type>(re, im) { }
	jComplexAsStdBase<Type>(const jComplexAsStdBase<Type> &z) : complex<Type>(z.real(), z.imag()) { }
	jComplexAsStdBase<Type>(const complex<Type> &z) : complex<Type>(z.real(), z.imag()) { }

  #ifdef __GSL_COMPLEX_H__
	explicit jComplexAsStdBase<Type>(const gsl_complex &z);
  #endif

	// I want to keep this as a static member function to avoid confusion in the calling code
	// A particular concern is distinguishing it from integer loop variables named i
	static jComplexAsStdBase<Type> i(void) { return jComplexAsStdBase<Type>(0, 1); }

  #ifdef __GSL_COMPLEX_H__
	static gsl_complex ToGSL(const jComplexAsStdBase<Type> &z)
	{
		return gsl_complex_rect(AllowPrecisionLossReadingValue_mayAlreadyBeDouble(z.real()),
								AllowPrecisionLossReadingValue_mayAlreadyBeDouble(z.imag()));
	}
	
	static void Set(jComplexAsStdBase<Type> &outZ, gsl_complex inZ)
	{
		outZ = jComplexAsStdBase<Type>(GSL_REAL(inZ), GSL_IMAG(inZ));
	}
  #endif

	complex<float> ToFloat(void) const
	{
		return complex<float>(complex<Type>::real(), complex<Type>::imag());
	}
};

typedef jComplexAsStdBase<double> jComplexAsStd;

#endif
