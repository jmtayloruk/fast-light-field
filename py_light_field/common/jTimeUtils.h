/*
 *  jTimeUtils.h
 *  
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *	Utilities to determine elapsed time etc
 *
 */

#ifndef __J_TIME_UTILS_H__
#define __J_TIME_UTILS_H__ 1

#include "jOSMacros.h"

double GetTime(void);
double GetTimeAbsolute(void);
double GetTimebaseStart(void);
void ProcessorTime(double *userTime, double *sysTime, double *bothTime);

void ReportElapsedTime(double t1, double t2, const char *theText);
void ReportElapsedTime_us(double t1, double t2, const char *theText);
double CalcElapsedSecs(double t1, double t2);
void PauseFor(double secs);

#endif
