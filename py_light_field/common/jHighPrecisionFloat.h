/*
 *	jHighPrecisionFloat.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing a floating-point number, but potentially to higher (or lower) precision
 *	than supported by the ubiquitous 'double' type.
 */

#ifndef __JHIGH_PRECISION_FLOAT_H__
#define __JHIGH_PRECISION_FLOAT_H__

#ifndef USE_JREAL
    typedef double jreal;
	typedef class jreal_as_double_consts jreal_consts;

    #include <float.h>
    extern const double NaN;
	class jreal_as_double_consts
	{
	  public:
		static double dbl_max(void) { return DBL_MAX; }
		static double dbl_min(void) { return DBL_MIN; }
		static double nan(void) { return NaN; }
        static double lnpi(void) { return 1.14472988584940017414342735135; /* ln(pi) */ }
        static double ln2(void) { return 0.69314718055994530941723212146; /* ln(2) */ }
		static double epsilon(void) { return DBL_EPSILON; }
	};

	#include <vector>
	typedef std::vector<double> realVector;

#else
	#define JREAL_DEFINED 1

	#if USE_JREAL == 1
		typedef class jHighPrecisionFloat_mpfr jreal;
		typedef class jHighPrecisionFloat_mpfr jreal_consts;
		#include "jHighPrecisionFloat_mpfr.h"
	#else
		typedef class jHighPrecisionFloat jreal;
		typedef class jHighPrecisionFloat jreal_consts;
		#include "jHighPrecisionFloat_stubAsDouble.h"
	#endif

	// Generic prototypes for implementation-specific functions

	int floor_int(const jreal &val);		// Calculate floor(val) and convert to int
	bool is_nan(const jreal &val);
	jreal fabs(const jreal &val);
	// I would like to eliminate abs and just have fabs for clarity, but std::complex expects abs() to be implemented, so this next function has to remain
	jreal abs(const jreal &val);
	jreal exp(const jreal &val);
	jreal pow(const jreal &val, const jreal &power);
	jreal sqrt(const jreal &val);
	jreal log(const jreal &val);
	jreal sin(const jreal &x);
	jreal sinh(const jreal &x);
	jreal cos(const jreal &x);
	jreal cosh(const jreal &x);
	jreal tan(const jreal &x);
	jreal asin(const jreal &x);
	jreal acos(const jreal &x);
	jreal atan2(const jreal &y, const jreal &x);
	jreal sign(const jreal &val);

	jreal gsl_sf_lnpoch(const jreal &a, const jreal &x);
	int gsl_sf_legendre_sphPlm_array(const int lmax, int m, const jreal &x, jreal *result_array);
	jreal gsl_sf_log_1plusx(const jreal &x);
	int gsl_sf_bessel_Jn_array(int nmin, int nmax, jreal &x, jreal *result_array);// TODO: when I implement this I need to make sure it can handle underflow gracefully and silently. I should then check everywhere I call this, because it looks like there are hacks in several different places!
	int gsl_sf_bessel_jl_array(const int lmax, const jreal &x, jreal *result_array);

	#include <vector>
	typedef std::vector<jreal> realVector;

#endif


double AllowPrecisionLossReadingValue(jreal val);
double AllowPrecisionLossReadingValue_mayAlreadyBeDouble(jreal val);
double AllowPrecisionLossReadingValue_mayAlreadyBeDouble(double val);
jreal AllowPrecisionLossOnParam(double val);
#ifdef USE_JREAL
void Print(jreal x, const char *suffix = "");
#endif
void Print(double x, const char *suffix = "");

#endif
