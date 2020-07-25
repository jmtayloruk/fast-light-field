// Attempted code to set thread/core affinity, to investigate whether that affects my low-level benchmarks.
// This code does not seem to actually have the desired effect, though I don’t know why.
// I have shelved this as not a high priority (major rabbit hole...)




#define SYSCTL_CORE_COUNT   "machdep.cpu.core_count"

typedef struct cpu_set {
    uint32_t    count;
} cpu_set_t;

static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int
CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

#include <mach/thread_policy.h>
#include <sys/sysctl.h>
#include <mach/thread_act.h>

int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set)
{
    int32_t core_count = 0;
    size_t  len = sizeof(core_count);
    int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, 0, 0);
    if (ret) {
        printf("error while get core count %d\n", ret);
        return -1;
    }
    cpu_set->count = 0;
    for (int i = 0; i < core_count; i++) {
        cpu_set->count |= (1 << i);
    }
    
    return 0;
}

int pthread_setaffinity_np(pthread_t thread, size_t cpu_size,
                           cpu_set_t *cpu_set)
{
    thread_port_t mach_thread;
    int core = 0;
    
    for (core = 0; core < 8 * cpu_size; core++) {
        if (CPU_ISSET(core, cpu_set)) break;
    }
    printf("binding to core %d\n", core);
    thread_affinity_policy_data_t policy = { core };
    mach_thread = pthread_mach_thread_np(thread);
    kern_return_t result = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, 1);
    printf("thread_policy_set returned %d\n", result);
    return 0;
}

struct ThreadInfo
{
    int                     threadIDCounter;
    size_t                  workCounter[kNumWorkTypes];
    JMutex                  *workQueueMutex;
    long                    *workQueueMutexBlock_us;
    double                  *pollingTime;
    std::vector<WorkItem *> *work[kNumWorkTypes];
    
    void *ThreadFunc(void)
    {
        int         thisThreadID;
        {
            LocalGetMutex lgm(workQueueMutex, workQueueMutexBlock_us);
            thisThreadID = threadIDCounter++;
        }

        
        pthread_t thread = pthread_self();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
#if 0
        /* Set affinity mask to include CPUs 0 to 7 */
        for (int j = 0; j < 8; j++)
            CPU_SET(j, &cpuset);
#else
        CPU_SET(thisThreadID, &cpuset);
#endif
        int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        ALWAYS_ASSERT(s == 0);
