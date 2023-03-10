//
//  jOSMacros.h
//
//  Copyright 2009-2015 Jonathan Taylor. All rights reserved.
//
//	Checks compiler-supplied platform-dependent macros to determine
//	whether specific features are available.
//	This could also be done through an autoconfig script
//

#ifndef __J_OS_MACROS_H__
#define __J_OS_MACROS_H__

// Define OS-dependent macros
// n.b. HAS_OS_X_GUI will be defined from within XCode.

#define OS_X __MACH__

#if __SSE__
	#define HAS_SSE 1
#else
	#define HAS_SSE 0
#endif
#if __AVX__
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif

#if __ppc__ || PS3 || __SPU__
	#define HAS_ALTIVEC 1
#else
	#define HAS_ALTIVEC 0
#endif

#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1060
	#define CAN_USE_GRAND_CENTRAL 1
#else
	#define CAN_USE_GRAND_CENTRAL 0
#endif

#define SYNC_FETCH_AVAILABLE ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 2)))
#if SYNC_FETCH_AVAILABLE
	#ifdef __cplusplus
		template<class Type> Type __sync_fetch(Type *addr) { return __sync_fetch_and_or(addr, 0); }
	#endif
#endif

#endif
