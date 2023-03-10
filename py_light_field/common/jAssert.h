/*
 *	jAssert.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Code to handle assertions
 */
#ifndef __JASSERT_H__
#define __JASSERT_H__

#include <stdio.h>
#include "DebugPrintf.h"

/*	There is the option of having asserts inline, but the preferred option is to call an
	object which will handle the assertion. This for example opens up the possibility of
	overriding the default assertion handler, as well as potentially keeping the code
	smaller	*/
#ifdef J_INLINE_ASSERTS
	#define ALWAYS_ASSERT(CONDITION) do { if (!(CONDITION)) { DebugPrintf("Assertion failed on line %d, function %s, file %s\n", (int)__LINE__, __PRETTY_FUNCTION__, __FILE__); *((int *)0L) = 0; } } while(0)
	#define CHECK(CONDITION) do { if (!(CONDITION)) { DebugPrintf("Check failed on line %d, function %s, file %s\n", (int)__LINE__, __PRETTY_FUNCTION__, __FILE__); } } while(0)
#else
	class BaseAssertionHandler
	{
	  protected:
		virtual void PullDownCode(void) __attribute__((__noreturn__));
	  public:
		virtual ~BaseAssertionHandler() { }
		virtual void AssertionFailed(int line, const char *function, const char *file) __attribute__((__noreturn__));
		virtual bool CheckCondition(bool condition, int line, const char *function, const char *file);
	};
	extern BaseAssertionHandler *assertionHandler;

	#define ALWAYS_ASSERT(CONDITION) do { if (__builtin_expect(!(CONDITION), false)) { assertionHandler->AssertionFailed(__LINE__, __PRETTY_FUNCTION__, __FILE__); } } while (0)
	#define CHECK(CONDITION) assertionHandler->CheckCondition((CONDITION), __LINE__, __PRETTY_FUNCTION__, __FILE__)
#endif

/* Note: a good place to go looking for a definition of old-style error codes is:
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/CarbonCore.framework/Versions/A/Headers/MacErrors.h
 */
#define ALWAYS_ASSERT_NOERR(RESULT) do { if (RESULT != 0) { DebugPrintf("Error code %d encountered\n", (int)(RESULT)); ALWAYS_ASSERT(0); } } while(0)
#define IGNORE_CONDITION(CONDITION) do { } while(0)

// Some assertions are only defined in the debug build
#if DEBUGGING
	#define ASSERT(CONDITION) ALWAYS_ASSERT(CONDITION)
	#define HARMLESS_ASSERT(CONDITION) ALWAYS_ASSERT(CONDITION)
	#define ASSERT_NOERR(RESULT) ALWAYS_ASSERT_NOERR((RESULT))
#else
	#define ASSERT(CONDITION) IGNORE_CONDITION(CONDITION)
	#define HARMLESS_ASSERT(CONDITION) IGNORE_CONDITION(CONDITION)
	#define ASSERT_NOERR(RESULT) IGNORE_CONDITION(CONDITION)
#endif

#endif
