/*
 *	jCommon.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  A selection of generic macros etc
 */

#ifndef __JCOMMON_H__
#define __JCOMMON_H__
 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <algorithm>

#include "jOSMacros.h"		// Defines the macro OS_X, among other things!
#if OS_X
	// This should be a proper conditional (from ./configure script...), but for now I'll just use it for parameter checking
	// on my OS X machines without worrying about availability on other machines.
	#define PRINTFLIKE(A,B) __printflike((A),(B))
#else
	#define PRINTFLIKE(A,B) 
#endif
#define RESTRICT __restrict

#include "jAssert.h"
#include "jHighPrecisionFloat.h"      // Will just define jreal as double, unless compile-time flag specifically set

/*	There are some asserts which cause the code to run
	ORDERS OF MAGNITUDE more slowly if they are compiled in.
	They are truly for debugging purposes only.
	They should be accompanied by higher-level tests
	which will spot that _something_ has gone wrong.		*/
#define EXTRA_ASSERTS 0
#if EXTRA_ASSERTS
	#define ASSERT2(CONDITION) ALWAYS_ASSERT(CONDITION)
#else
	#define ASSERT2(CONDITION) IGNORE_CONDITION(CONDITION)
#endif

template<class T> T SQUARE(const T &a) { return a*a; }
template<class T> T CUBE(const T &a) { return a*a*a; }

#ifndef MIN
	#define MIN(A, B) std::min((A), (B))
#endif
#ifndef MAX
	#define MAX(A, B) std::max((A), (B))
#endif
#ifndef LIMIT
    #define LIMIT(N, L, U) (std::max(std::min((N), (U)), (L)))
#endif

#define SOCKET_ERROR        -1
extern const int noSigPipe;

#define SWF NSString stringWithFormat

#define WANT_INLINE __attribute__((always_inline))

/*	The intention here was to offer a macro to assist with branch prediction,
	by indicating whether the condition is expected to be true or false.
	Intended use: 
		if (EXPECT(COMPARISON, RESULT))
			stuff;
	However I found (around 2011 I suspect?) that on an intel mac __builtin_expect
	seemed to generate poor code. As a result, for now this macro doesn't do anything
	beyond just evaluate the comparison	*/
#define EXPECT(COMPARISON, RESULT) (COMPARISON)

inline double random_01(void)
{
	// Return a random number between 0 and 1
	return ((double)random()) / 2147483647.0;		// (2^31 - 1)
}

inline double random_pm1(void)
{
	// Return a random number between -1 and 1
	return -1.0 + ((double)random()) / 1073741823.0;		// (2^30 - 1)
}

#endif
