//
//	jMutex.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	This module defines a class implementing a mutex
//	The helper class LocalGetMutex can be used as a stack variable which will
//	acquire the specified mutex while the instance remains in scope, releasing it
//	when the instance goes out of scope
//
#ifndef __JMUTEX_H__
#define __JMUTEX_H__

#include <pthread.h>
#include "jAssert.h"

// This compile-time flag maintains a timestamped history of mutex-related activity,
// as an aid to debugging
#define MUTEX_TIMESTAMPS 0

class JMutex
{
  protected:
	pthread_mutex_t	mutex;
	pthread_mutex_t	mutex2;
	int			lockCount;
	
#if MUTEX_TIMESTAMPS
	enum { kMutexHistorySize = 128 };
	int			historyPos;
	UnsignedWide	creationTime;
	UnsignedWide	mutexTime[kMutexHistorySize];
	UnsignedWide	mutexBlockTime[kMutexHistorySize];
	bool			got[kMutexHistorySize];
	int			historyLine[kMutexHistorySize];
	int				historyTryResult[kMutexHistorySize];
	bool			wrapped;
#endif

	
  public:
			JMutex();
			~JMutex();

    // We are absolutely not set up to tolerate copying of JMutex objects!
    // These lines make that explicitly clear (using C++07 syntax)
    JMutex(JMutex &a) =delete;
    JMutex& operator=(JMutex &a) = delete;
    
	void	Lock(int line);
    bool    TryLock(int line);
	void	Unlock(int line);
	int		BlockWaitingForSignal(pthread_cond_t *cond, int line, bool mayBeAges = false);
	void	DumpHistory(int maxItems);
	void	ResetHistory(bool resetZero);
};

class JRWLock
{
  protected:
	pthread_rwlock_t lock;
	int			lockCount;
	
  public:
			JRWLock();
			~JRWLock();
	void	ReadLock(int line, JRWLock *secondaryLock = NULL);
	void	WriteLock(int line);
	void	Unlock(int line);
};

class LocalGetMutex
{
  protected:
	JMutex	*mutex;
	bool	locked;
  public:
			LocalGetMutex(JMutex *inMutex);
            LocalGetMutex(JMutex *inMutex, long *outAcquireTime_us);
			~LocalGetMutex();
	void	Unlock(int line);
	void	Lock(int line);

	// Prototypes to catch misuse of class
	LocalGetMutex(const LocalGetMutex &) : mutex(NULL), locked(false) { ALWAYS_ASSERT(0); }
	LocalGetMutex &operator=(const LocalGetMutex &) { ALWAYS_ASSERT(0); return *this; }
};

class LocalGetReadLock
{
  protected:
	JRWLock	*lock;
	int		lineForGet;
  public:
			LocalGetReadLock(JRWLock *inLock, int line, JRWLock *secondaryLock = NULL);
			~LocalGetReadLock();
	// Prototypes to catch misuse of class
	LocalGetReadLock(const LocalGetReadLock &) : lock(NULL), lineForGet(0) { ALWAYS_ASSERT(0); }
	LocalGetReadLock &operator=(const LocalGetReadLock &) { ALWAYS_ASSERT(0); return *this; }
};

class LocalGetWriteLock
{
  protected:
	JRWLock	*lock;
	int		lineForGet;
  public:
			LocalGetWriteLock(JRWLock *inLock, int line);
			~LocalGetWriteLock();
	// Prototypes to catch misuse of class
	LocalGetWriteLock(const LocalGetWriteLock &) : lock(NULL), lineForGet(0) { ALWAYS_ASSERT(0); }
	LocalGetWriteLock &operator=(const LocalGetWriteLock &) { ALWAYS_ASSERT(0); return *this; }
};

#define LocalGetMutex2(M) LocalGetMutex lgm##__LINE__((M));
#define LocalGetReadLock2(M) LocalGetReadLock lgl##__LINE__((M));
#define LocalGetWriteLock2(M) LocalGetWriteLock lgl##__LINE__((M));


#define RW_TIMESTAMPS 0
/*	Note that the recording of this information is NOT fully correct
	in a multithreaded environment. For example, when DumpLogBuffer()
	is called there may be recent reserved entries that have not yet been filled in.
	It should however be useful as a source of information for debugging,
	as long as it is understood it may not be 100% reliable.
	One attempt to make thie more obvious is that the time variable is the final one
	to be initialized when filling out a new entry...	*/
#if RW_TIMESTAMPS
	const int kRWLogBufferSize = 1<<12;
	enum
	{
		kLockEventGetWriteLock = 0,
		kLockEventGotWriteLock,
		kLockEventGet2ndWriteLock,
		kLockEventGot2ndWriteLock,
		kLockEventGetReadLock,
		kLockEventGotReadLock,
		kLockEventDoUnlock,
		kLockEventDoneUnlock,
		kLockEventLaunchThreads,
		kLockEventNumEvents
	};
	
	struct RWLockEvent
	{
		int type, line;
		double time;
		void *lock, *threadId;
		
		void Init(int t, JRWLock *l, int inLine)
		{
			time = -1.0;		// For informat, to help indicate that structure is being initialized
			type = t;
			lock = l;
			line = inLine;
			threadId = pthread_self();
			double GetTime(void);
			time = GetTime();
		}
		
		void Dump(void)
		{
			const char *eventNames[] = { "W?", "W√", "W2?", "W2√", "R?", "R√", "X", "X√", "LT" };
			if (type == kLockEventLaunchThreads)
				printf("\n\n%lf\tLAUNCH\t%d\n", time, line);
			else
				printf("%lf\t%s\t%d\t%p\t%p\n", time, (type >= 0 && type < kLockEventNumEvents) ? eventNames[type] : "???", line, lock, threadId);
		}
	};
	
	extern RWLockEvent rwLogBuffer[kRWLogBufferSize];
	extern int rwLockBufferPos;
#endif

#endif

