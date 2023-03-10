/*
 *	VectorTypes.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *  Platform-independent definitions of basic CPU vector types
 *
 */

#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__

#include "jOSMacros.h"

#define HAS_VECTOR_SUPPORT 1        /* will be overridden, below, if we are just using a scalar substitute */

#if HAS_SSE
	#if __SSE3__
		#include <pmmintrin.h>
        #include <tmmintrin.h>		// SSSE3 (supplemental SSE3)      // TODO: there should probably be a separate #if switch for this.
	#else
		#include <xmmintrin.h>
	#endif

    /*  Note: care is ndeed on OS X because vecLibTypes.h (included e.g. by Accelerate.h)
        wants to redefine some of the same type names that I use, but it defines them in a special way on modern versions of GCC.
        The simplest solution seems to be just to echo here what is in vecLibTypes.h (with a few tweaks to type names to match what I have been using).
        It does look as if there should be much better type safety available from doing it the way they do in vecLibTypes.h. */
    #if defined(__GNUC__)
        typedef float vFloat __attribute__ ((__vector_size__ (16)));
    #else /* not __GNUC__ */
        typedef __m128 vFloat;
    #endif /* __GNUC__ */

    #if defined(__GNUC__)
        #if defined(__GNUC_MINOR__) && (((__GNUC__ == 3) && (__GNUC_MINOR__ <= 3)) || (__GNUC__ < 3))
            typedef __m128i                 vUChar;
            typedef __m128i                 vInt8;
            typedef __m128i                 vUInt16;
            typedef __m128i                 vInt16;
            typedef __m128i                 vUInt32;
            typedef __m128i                 vInt32;
            typedef __m128i                 vBool32;
            typedef __m128i                 vUInt64;
            typedef __m128i                 vInt64;
            typedef __m128d                 vDouble;
        #else /* gcc-3.5 or later */
            typedef unsigned char           vUChar          __attribute__ ((__vector_size__ (16)));
            typedef char                    vInt8          __attribute__ ((__vector_size__ (16)));
            typedef unsigned short          vUInt16         __attribute__ ((__vector_size__ (16)));
            typedef short                   vInt16         __attribute__ ((__vector_size__ (16)));
            typedef unsigned int            vUInt32         __attribute__ ((__vector_size__ (16)));
            typedef int                     vInt32         __attribute__ ((__vector_size__ (16)));
            typedef unsigned int            vBool32         __attribute__ ((__vector_size__ (16)));
            typedef unsigned long long      vUInt64         __attribute__ ((__vector_size__ (16)));
            typedef long long               vInt64         __attribute__ ((__vector_size__ (16)));
            typedef double                  vDouble         __attribute__ ((__vector_size__ (16)));
        #endif /* __GNUC__ <= 3.3 */
    #else /* not __GNUC__ */
            typedef __m128i                 vUChar;
            typedef __m128i                 vInt8;
            typedef __m128i                 vUInt16;
            typedef __m128i                 vInt16;
            typedef __m128i                 vUInt32;
            typedef __m128i                 vInt32;
            typedef __m128i                 vBool32;
            typedef __m128i                 vUInt64;
            typedef __m128i                 vInt64;
            typedef __m128d                 vDouble;
    #endif /* __GNUC__ */
#elif __arm__
    #include <arm_neon.h>
    typedef uint32x4_t vUInt32;
    typedef int32x4_t vInt32;
#elif HAS_ALTIVEC
	#if __SPU__
		#include <spu_intrinsics.h>
		#include <spu_mfcio.h> /* constant declarations for the MFC */
		#include <simdmath.h>
	#else
		#ifndef __APPLE_ALTIVEC__
			#include <altivec.h>

			// altivec.h defines bool for its own purposes, but my existing code uses it
			// in its normal form all over the place. Undefine it to prevent compile errors!
			#undef bool
		#endif
	#endif
	
	typedef vector float vFloat;
	typedef vector unsigned int vUInt32;
	typedef vector unsigned char vUChar;
#else
    // Providing minimal support in the non-vector case, because it may make life easier
    // to be able to write some simple bits of code so they compile and work even in the absence
    // of any vector support at all.
    #undef HAS_VECTOR_SUPPORT
    #define HAS_VECTOR_SUPPORT 0
    // Note that we define this as a struct, because C does not allow us to return an array from a function
    // (but we can return a struct).
    typedef struct
    {
        uint32_t i[4];
    } vUInt32;
#endif

#if HAS_ALTIVEC || HAS_SSE
	typedef union
	{
		vFloat	vf;
	//	vUInt32	v32;
		vUChar	vc;
		float	f[4];
		long	l[4];
		short	s[8];
	} VecUnion;
#endif

#endif
