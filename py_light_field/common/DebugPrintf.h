//
//  DebugPrintf.h
//
//  Copyright 2015 Jonathan Taylor. All rights reserved.
//
//	Header file defining a debug logging function
//	The function is intended for occasional but serious error logging, e.g. a failed assertion
//	and implementations may not give particularly good performance for routine log output.
//
//	Only one platform-specific file implementing DebugPrintf should be included in a project,
//	or else there will be linker errors due to multiple function definitions.
//

#ifndef __DebugPrintf__
#define __DebugPrintf__

#include <sys/cdefs.h>      // for __printflike
#include "jCommon.h"

void DebugPrintf(const char *format, ...) PRINTFLIKE(1, 2);
void DebugPrintfFatal(const char *errorIntro, const char *format, ...) PRINTFLIKE(2, 3);

#endif /* defined(__DebugPrintf__) */
