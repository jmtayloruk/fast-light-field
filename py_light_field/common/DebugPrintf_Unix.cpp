//
//  DebugPrintf_Unix.cpp
//
//  Copyright 2015 Jonathan Taylor. All rights reserved.
//
//	Implementation of DebugPrintf suitable for running on Unix
//	Print the message to stderr.
//
//	Only one platform-specific implementation file like this one should be included in a project,
//	or else there will be linker errors due to multiple function definitions.
//

#include "DebugPrintf.h"
#include <stdarg.h>
#include <stdio.h>

void DebugPrintf(const char *format, ...)
{
	// Just call through to stderr
	va_list args;
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);
}

void DebugPrintfFatal(const char *errorIntro, const char *format, ...)
{
    fprintf(stderr, "Fatal error - %s: ", errorIntro);
    
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
