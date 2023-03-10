//
//	jMutex.cpp
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	This module defines a class implementing a mutex
//	The helper class LocalGetMutex can be used as a stack variable which will
//	acquire the specified mutex while the instance remains in scope, releasing it
//	when the instance goes out of scope
//

#include "jMutex.h"
#include "jAssert.h"
#include "jTimeUtils.h"
#include "DebugPrintf.h"
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/errno.h>

JMutex::JMutex()
{
	lockCount = 0;
	int result = pthread_mutex_init(&mutex, NULL);
	ALWAYS_ASSERT(result == 0);
#if MUTEX_TIMESTAMPS
	wrapped = false;
	historyPos = 0;
	creationTime = GetTime();
#endif
}

JMutex::~JMutex()
{
	ALWAYS_ASSERT(lockCount == 0);
	int result = pthread_mutex_destroy(&mutex);
	ALWAYS_ASSERT_NOERR(result);
}

void JMutex::Lock(int line)
{
//	printf("locking on line %ld\n", line);
	int result = -99;
#if MUTEX_TIMESTAMPS
	int tryResult = 0;
#endif

#if MUTEX_TIMESTAMPS
	double t1;
	t1 = GetTime();
#endif

#if MUTEX_TIMESTAMPS
	result = pthread_mutex_trylock(&mutex);
	if (!tryResult)
		tryResult = result;
	
	if (result)
		result = pthread_mutex_lock(&mutex);
#else
	result = pthread_mutex_lock(&mutex);
#endif
	ALWAYS_ASSERT(result == 0);

	if (lockCount != 0)
		printf("lock count %d\n", lockCount);
	ALWAYS_ASSERT(lockCount == 0);
	lockCount++;

#if MUTEX_TIMESTAMPS
	mutexTime[historyPos] = GetTime();
	mutexBlockTime[historyPos] = t1;
	historyTryResult[historyPos] = tryResult;
	got[historyPos] = true;
	historyLine[historyPos] = line;
	historyPos++;
	if (historyPos == kMutexHistorySize)
	{
		historyPos = 0;
		wrapped = true;
	}
#endif
//	printf("locked on line %ld\n", line);
}

bool JMutex::TryLock(int line)
{
    //	printf("trying lock on line %ld\n", line);
    int result = -99;
    
#if MUTEX_TIMESTAMPS
    double t1;
    t1 = GetTime();
#endif
    
    result = pthread_mutex_trylock(&mutex);
    
    if (result == 0)
    {
        if (lockCount != 0)
            printf("lock count %d\n", lockCount);
        ALWAYS_ASSERT(lockCount == 0);
        lockCount++;
        
#if MUTEX_TIMESTAMPS
        mutexTime[historyPos] = GetTime();
        mutexBlockTime[historyPos] = t1;
        historyTryResult[historyPos] = tryResult;
        got[historyPos] = true;
        historyLine[historyPos] = line;
        historyPos++;
        if (historyPos == kMutexHistorySize)
        {
            historyPos = 0;
            wrapped = true;
        }
#endif
        //	printf("locked on line %ld\n", line);
        return true;
    }
    else
        return false;
}

void JMutex::Unlock(int line)
{
//	printf("unlocking on line %ld\n", line);
	ALWAYS_ASSERT(lockCount == 1);
	lockCount--;

#if MUTEX_TIMESTAMPS
	mutexTime[historyPos] = GetTime();
	got[historyPos] = false;
	historyLine[historyPos] = line;
	historyPos++;
	if (historyPos == kMutexHistorySize)
	{
		historyPos = 0;
		wrapped = true;
	}
#endif

	int result = pthread_mutex_unlock(&mutex);
	ALWAYS_ASSERT(result == 0);
}

int JMutex::BlockWaitingForSignal(pthread_cond_t *cond, int line, bool mayBeAges)
{
#if MUTEX_TIMESTAMPS
	mutexTime[historyPos] = GetTime();
	got[historyPos] = false;
	historyLine[historyPos] = line;
	historyPos++;
	if (historyPos == kMutexHistorySize)
	{
		historyPos = 0;
		wrapped = true;
	}
#endif

	ALWAYS_ASSERT(lockCount == 1);
	lockCount--;
	
	int result;
#if !FEDORA_LINUX
	/*	I have a timedwait with a very long delay.
		If a bug means we never get signalled, we will eventually come out
		of the blocking call and assert. The delay is long enough that we should
		never be meant to wait that long.
		However, the work threads may block for a very long time depending on what
		we are doing and how many of them are active. Hence we offer an override
		for the timeout	*/
	if (!mayBeAges)
	{
		struct timespec t;
		struct timeval tv;
		struct timezone tz;
		gettimeofday(&tv, &tz);
		t.tv_sec = tv.tv_sec + 1000;
		t.tv_nsec = tv.tv_usec * 1000;
		result = pthread_cond_timedwait(cond, &mutex, &t);
	//	ALWAYS_ASSERT(result == 0);
		ALWAYS_ASSERT(result != EINVAL);
		ALWAYS_ASSERT(result != ETIMEDOUT);
	}
	else
#endif
	{
		// Unfortunately it seems that on the mac pro (Fedora), pthread_cond_timedwait
		// can return early with pthread_cond_timedwait. As a result we just block forever. 
		result = pthread_cond_wait(cond, &mutex);
		ALWAYS_ASSERT(result == 0);
	}

	ALWAYS_ASSERT(lockCount == 0);
	lockCount++;

#if MUTEX_TIMESTAMPS
	mutexTime[historyPos] = GetTime();
	historyTryResult[historyPos] = 0;
	got[historyPos] = true;
	historyLine[historyPos] = line;
	historyPos++;
	if (historyPos == kMutexHistorySize)
	{
		historyPos = 0;
		wrapped = true;
	}
#endif

	return result;
}

void JMutex::DumpHistory(int maxItems)
{
#if MUTEX_TIMESTAMPS
	maxItems = MIN(maxItems, kMutexHistorySize);
	int pos = historyPos - maxItems;
	if (pos < 0)
	{
		if (wrapped)
		{
			pos += kMutexHistorySize;
			printf("Last %ld entries:\n", maxItems);
		}
		else
		{
			pos = 0;
			printf("All %ld entries:\n", historyPos);
		}
	}
	
	while (pos < historyPos)
	{
		printf(" %s: time %le, line %ld.", got[pos] ? "GOT  " : "FREE", CalcElapsedSecs(creationTime, mutexTime[pos]), historyLine[pos]);
		if (got[pos])
			printf(" Blocked at time %le. Try result was %d (%s)", CalcElapsedSecs(creationTime, mutexBlockTime[pos]), historyTryResult[pos], strerror(historyTryResult[pos]));
		printf("\n");
		pos++;
	}
#endif
}

void JMutex::ResetHistory(bool resetZero)
{
#if MUTEX_TIMESTAMPS
	historyPos = 0;
	wrapped = false;
	if (resetZero)
		creationTime = GetTime();
#endif
}

#if RW_TIMESTAMPS
	RWLockEvent rwLogBuffer[kRWLogBufferSize];
	int rwLockBufferPos = 0;
	
	void DumpLogBuffer(void)
	{
		bool rwLockWrapped = (rwLockBufferPos >= kRWLogBufferSize);
		
		if (rwLockWrapped)
		{
			for (int i = (int)(rwLockBufferPos % kRWLogBufferSize); i < kRWLogBufferSize; i++)
				rwLogBuffer[i].Dump();
		}
		for (int i = 0; i < (int)(rwLockBufferPos % kRWLogBufferSize); i++)
			rwLogBuffer[i].Dump();
	}
#endif

JRWLock::JRWLock()
{
	lockCount = 0;
	int result = pthread_rwlock_init(&lock, NULL);
	ALWAYS_ASSERT(result == 0);
}

JRWLock::~JRWLock()
{
	ALWAYS_ASSERT(lockCount == 0);
	int result = pthread_rwlock_destroy(&lock);
	ALWAYS_ASSERT(result == 0);
}

void JRWLock::WriteLock(int line)
{
#if RW_TIMESTAMPS
	int pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGetWriteLock, this, line);
#endif

	int result = pthread_rwlock_wrlock(&lock);
	ALWAYS_ASSERT(result == 0);
	
#if RW_TIMESTAMPS
	pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGotWriteLock, this, line);
#endif
}

void JRWLock::ReadLock(int line, JRWLock *secondaryLock)
{
#if RW_TIMESTAMPS
	int pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGetReadLock, this, line);
#endif

	if (secondaryLock != NULL)
	{
#if RW_TIMESTAMPS
		pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
		rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGet2ndWriteLock, secondaryLock, line);
#endif
		secondaryLock->WriteLock(__LINE__);
#if RW_TIMESTAMPS
		pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
		rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGot2ndWriteLock, secondaryLock, line);
#endif
	}

	int result = pthread_rwlock_rdlock(&lock);
	ALWAYS_ASSERT(result == 0);

	if (secondaryLock != NULL)
	{
#if RW_TIMESTAMPS
		pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
		rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventDoUnlock, secondaryLock, line);
#endif
		secondaryLock->Unlock(__LINE__);
#if RW_TIMESTAMPS
		pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
		rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventDoneUnlock, secondaryLock, line);
#endif
	}

#if RW_TIMESTAMPS
	pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventGotReadLock, this, line);
#endif
}

void JRWLock::Unlock(int line)
{
#if RW_TIMESTAMPS
	int pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventDoUnlock, this, line);
#endif
	int result = pthread_rwlock_unlock(&lock);
	ALWAYS_ASSERT(result == 0);
#if RW_TIMESTAMPS
	pos = __sync_fetch_and_add(&rwLockBufferPos, 1);
	rwLogBuffer[pos % kRWLogBufferSize].Init(kLockEventDoneUnlock, this, line);
#endif
}

LocalGetMutex::LocalGetMutex(JMutex *inMutex)
{
	mutex = inMutex;
	// We support inMutex being NULL, since this makes it easier to use this class in
	// a case where we will sometimes want to acquire a mutex but sometimes not care.
	if (mutex != NULL)
	{
		mutex->Lock(-1);
		locked = true;
	}
}

LocalGetMutex::LocalGetMutex(JMutex *inMutex, long *outAcquireTime_us)
{
    // Alternative constructor that reports how much time it took to acquire the mutex
    mutex = inMutex;
    // We support inMutex being NULL, since this makes it easier to use this class in
    // a case where we will sometimes want to acquire a mutex but sometimes not care.
    if (mutex != NULL)
    {
        double t1 = GetTime();
        mutex->Lock(-1);
        double t2 = GetTime();
        long elapsedMicroseconds = long((t2-t1)*1e6);
        __sync_fetch_and_add(outAcquireTime_us, elapsedMicroseconds);
        locked = true;
    }
}

LocalGetMutex::~LocalGetMutex()
{
	if ((mutex != NULL) && (locked))
		mutex->Unlock(-1);
}

void LocalGetMutex::Unlock(int line)
{
	if (mutex != NULL)
	{
		ALWAYS_ASSERT(locked);
		mutex->Unlock(line);
		locked = false;
	}	
}

void LocalGetMutex::Lock(int line)
{
	if (mutex != NULL)
	{
		ALWAYS_ASSERT(!locked);
		mutex->Lock(line);
		locked = true;
	}	
}

LocalGetReadLock::LocalGetReadLock(JRWLock *inLock, int line, JRWLock *secondaryLock)
{
	lock = inLock;
	lineForGet = line;
	lock->ReadLock(line, secondaryLock);
}

LocalGetReadLock::~LocalGetReadLock()
{
	lock->Unlock(lineForGet);
}

LocalGetWriteLock::LocalGetWriteLock(JRWLock *inLock, int line)
{
	lock = inLock;
	lock->WriteLock(line);
}

LocalGetWriteLock::~LocalGetWriteLock()
{
	lock->Unlock(lineForGet);
}
