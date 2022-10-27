#include <unistd.h>
#include <map>
#include <sys/resource.h>
#include "common/jAssert.h"
#include "common/VectorFunctions.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jMutex.h"
#include "common/jPythonCommon.h"
#include "common/PIVImageWindow.h"
#include "fftw-3.3.8/api/fftw3.h"

// Note: the TESTING macro is set when we build within Xcode, but not when we build using setup.py

int NumActualProcessorsAvailable(void)
{
    /*  Frustratingly, I haven't found a good way (in C or Python) to identify that we are running with hyperthreaded CPUs.
        Performance is best when *not* using extra hyperthreads, so I just divide the reported number of processors by 2.
        TODO: if this is run on a machine where no hyperthreading is available, it will not use the optimum number of processors.
        An ideal solution might be to actually have a function that runs timing tests on dummy data and decides how many processors to use,
        but that is probably overkill for now!  */
    return int(sysconf(_SC_NPROCESSORS_ONLN) / 2);
}

/* Define the FFTW planning strategy we will use. Run times:
    FFTW_ESTIMATE took 5.46
    FFTW_MEASURE took 5.01   (after 2s the first time, and <1ms on the later times)
    FFTW_PATIENT took 4.58   (after 29s the first time)
   In other words, there is a further gain available from FFTW_PATIENT, but it's too slow to be convenient until I really am running in anger with large datasets
 */
int gFFTPlanMethod = FFTW_MEASURE;
int gNumThreadsToUse = NumActualProcessorsAvailable();        // But can be modified using API call
// This is a special constant used when sorting work items (see Compare function).
// It is set by the calling code just before the sort command.
int gMutexContentionFactor;

#if 1
    /*  This whole subclass exists to work around a weird shortcoming of the std::complex implementation that comes with LLVM.
        Its implementation for operator* has some bizarre additional tests for NaN which appear to be intended to handle some
        sort of obscure edge case in a pedantically-correct manner. The gcc std::complex does not do this, so I'm not sure why
        the LLVM implementation feels the need to do that.
        Anyway, this causes a huge problem on OS X, because my code runs at half speed overall(!) when these extra tests are present.
        I haven't found a way of overloading or anything like that, to force the extra code to be eliminated.
        The only solution I have found is to subclass std::complex and provide my own implementation for operator*=.
        I then am forced to also implement various constructors etc, to make my subclass work, and I have also placed various
        "landmines" to protect against accidental reversion to std::complex for intermediate values during composite calculations.
        This is not perfect, but it's more than enough for my code as it stands right now. I will however need to keep an eye on performance,
        and maybe even examine disassembly of inner loops if I have any concerns in future. */
    template<class T> class complex_fast : public std::complex<T>
    {
    public:
        typedef T value_type;
        complex_fast(const value_type& __re = value_type(), const value_type& __im = value_type()) : std::complex<T>(__re, __im) {}
        template<class _Xp> complex_fast(const complex_fast<_Xp>& __c) : std::complex<T>(__c.real(), __c.imag()) {}
        complex_fast& operator= (const value_type& __re)
          {std::complex<T>::operator=(__re); return *this;}
        template<class _Xp> complex_fast& operator= (const complex_fast<_Xp>& __c)
          {std::complex<T>::operator=(__c); return *this;}

        template<class _Tp> static inline complex_fast<_Tp> conj(const complex<_Tp>& __c)
        {
            return complex_fast<_Tp>(__c.real(), -__c.imag());
        }

        /*  This exists as a landmine to prevent operator+= from allowing accidental use of vanilla 'complex' in intermediate values.
            That could happen for example by the use of a function call (e.g. exp) that I have not anticipated like I have with conj().
            It surprised me that operator+= 'worked' with a vanilla complex input, but I suppose it's because
            I don't actually *use* the result (with the return type of the base class).
            Anyway, this protects against that eventuality. Obviously other operators exist, but I can't rewrite the whole class...
         
         */
        template<class _Xp> complex_fast& operator+=(const complex<_Xp>& __c) =delete;
        template<class _Xp> complex_fast& operator+=(const complex_fast<_Xp>& __c)
        {
            complex<_Xp>::operator+=(__c); return *this;
        };
    };

#if 0
    // This is helpful to me, but does not compile on all platforms (some compilers complain about redefining an existing function).
    // As a result, this is disabled by default, but I may want to re-enable it temporarily if I have been working on this source file.
    template<class _Tp> static inline complex_fast<_Tp> conj(const complex<_Tp>& __c)
    {
        /*  This exists as a landmine, to ensure that code that calls vanilla conj() fails at compile-time.
         *Because* that code fails, I cannot just define this function, as far as I can see.
         Instead we must call TYPE::conj() - see definition above, within complex_fast.  */
        return complex_fast<_Tp>(__c.real(), -__c.imag());
    }
#endif

    template<class _Tp> complex_fast<_Tp> operator*(const complex_fast<_Tp>& __z, const complex_fast<_Tp>& __w)
    {
        _Tp __a = __z.real();
        _Tp __b = __z.imag();
        _Tp __c = __w.real();
        _Tp __d = __w.imag();
        _Tp __ac = __a * __c;
        _Tp __bd = __b * __d;
        _Tp __ad = __a * __d;
        _Tp __bc = __b * __c;
        _Tp __x = __ac - __bd;
        _Tp __y = __ad + __bc;
        return complex_fast<_Tp>(__x, __y);
    }
    template<class _Tp> complex_fast<_Tp> operator+(const complex_fast<_Tp>& __z, const complex_fast<_Tp>& __w)
    {
        _Tp __x = __z.real() + __w.real();
        _Tp __y = __z.imag() + __w.imag();
        return complex_fast<_Tp>(__x, __y);
    }

    // Note specialisation to float-only here, to make sure we don't accidentally accept vanilla 'complex' inputs.
    template<class _Tp> inline complex_fast<_Tp> operator*(const complex_fast<_Tp>& __x, const float& __y)
    {
        complex_fast<_Tp> __t(__x);
        __t *= __y;
        return __t;
    }

    template<class _Tp> inline complex_fast<_Tp> operator*(const float& __x, const complex_fast<_Tp>& __y)
    {
        complex_fast<_Tp> __t(__y);
        __t *= __x;
        return __t;
    }

    typedef complex_fast<float> TYPE;
    template<> int ArrayType<TYPE>(void) { return NPY_CFLOAT; }
#else
    // This ought to be all that is needed, were it not for the issue (described above) that I need to work around!
    typedef complex<float> TYPE;
#endif
typedef float RTYPE;

// Constants used to index into the x/yAxisMultipliers arrays we are passed from the python code
enum
{
    // x axis multipliers
    kXAxisMultiplierMirrorY = 0,
    kXAxisMultiplierExpandXStart,
    // y axis multipliers
    kYAxisMultiplierNoMirror = 0,   /* expandMultiplier[bb][y] */
    kYAxisMultiplierMirrorX         /* expandMultiplier[bb][y] * mirrorXMultiplier[y] */
};

double _GetTime(void)
{
    // Standard BSD function
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    return t.tv_sec + t.tv_usec * 1e-6;
}
double GetTime(void)
{
    static double t0 = _GetTime();
    return _GetTime() - t0;
}

void PauseFor(double secs)
{
    // Block for the specified number of seconds before returning
    struct timeval timeout;
    timeout.tv_sec = (int)secs;
    timeout.tv_usec = (int)((secs - (int)secs) * 1000000);
    select(0, NULL, NULL, NULL, &timeout);
}

char *gThreadFileName = NULL;
FILE *gStatsFile = NULL;

struct TimeStruct
{
    double wall;
    rusage _self;
    rusage children;
    
    TimeStruct()
    {
        wall = GetTime();
        getrusage(RUSAGE_SELF, &_self);
        getrusage(RUSAGE_CHILDREN, &children);
    }
    
    static double Secs(struct timeval t) { return(t.tv_sec + 1e-6*t.tv_usec); }
    
    void Dump(const char *desc, FILE *outFile=NULL)
    {
        if (outFile == NULL)
            outFile = gStatsFile;
        if (outFile != NULL)
        {
            fprintf(outFile, "%s\t%lf\t%lf\t%lf\t%lf\t%lf\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", desc, wall,
                                Secs(_self.ru_utime), Secs(_self.ru_stime),
                                Secs(children.ru_utime), Secs(children.ru_stime),
                                _self.ru_minflt, _self.ru_majflt, _self.ru_nswap, _self.ru_inblock, _self.ru_oublock,
                                children.ru_minflt, children.ru_majflt, children.ru_nswap, children.ru_inblock, children.ru_oublock);
        }
    }
};

extern "C" PyObject *GetNumThreadsToUse(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", gNumThreadsToUse);
}

extern "C" PyObject *SetNumThreadsToUse(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &gNumThreadsToUse))
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    if (gNumThreadsToUse <= 0)
        gNumThreadsToUse = NumActualProcessorsAvailable();
    Py_RETURN_NONE;
}

extern "C" PyObject *GetPlanningMode(PyObject *self, PyObject *args)
{
    
    return Py_BuildValue("i", gFFTPlanMethod);
}

extern "C" PyObject *SetPlanningMode(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &gFFTPlanMethod))
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    Py_RETURN_NONE;
}

extern "C" PyObject *SetStatsFile(PyObject *self, PyObject *args)
{
    // I will leave this here in case it is of future use, but this feature is much less interesting
    // now that I am doing full parallelism rather than block-parallelism across individual discrete tasks.
    const char *path;
    int append;
    if (!PyArg_ParseTuple(args, "si", &path, &append))
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    if (gStatsFile != NULL)
        fclose(gStatsFile);
    if (strlen(path) > 0)
    {
        gStatsFile = fopen(path, append ? "a" : "w");
        if (gStatsFile == NULL)
        {
            PyErr_Format(PyErr_NewException((char*)"exceptions.OSError", NULL, NULL), "Unable to open \"%s\"", path);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

extern "C" PyObject *SetThreadFileName(PyObject *self, PyObject *args)
{
    // If the input parameter is set to something non-empty, we will dump information about multithreaded performance
    // to a file at the end of each projection run.
    const char *filename;
    if (!PyArg_ParseTuple(args, "z", &filename))    // 'z' specifies a string but also accepts None (which maps to NULL)
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    if (gThreadFileName != NULL)
        delete[] gThreadFileName;
    if (filename == NULL)
        gThreadFileName = NULL;
    else
    {
        gThreadFileName = new char[strlen(filename)+1];
        strcpy(gThreadFileName, filename);
    }
    Py_RETURN_NONE;
}

void MirrorYArray(JPythonArray2D<TYPE> &fHtsFull, JPythonArray1D<TYPE> &mirrorYMultiplier, JPythonArray2D<TYPE> &result)
{
    // Given F(H), return F(mirrorY(H)), where mirrorY(H) is the vertical reflection of H.
    int height = fHtsFull.Dims(0);
    int width = fHtsFull.Dims(1);
    for (int x = 0; x < width; x++)
        result[0][x] = TYPE::conj(fHtsFull[0][x]) * mirrorYMultiplier[x];
    for (int y = 1; y < height; y++)
    {
        // Note that I tried writing hand-optimized SSE code and the longhand C code was faster.
        // That was surprising, but the C code seems to be being auto-vectorized rather well already.
        // Oh well, that saves me faffing around doing it myself!
        JPythonArray1D<TYPE> _result = result[y];
        JPythonArray1D<TYPE> _fHtsFull = fHtsFull[height-y];
        for (int x = 0; x < width; x++)
            _result[x] = TYPE::conj(_fHtsFull[x]) * mirrorYMultiplier[x];
    }
}

void CalculateRow(TYPE *ac, const TYPE *fap, const TYPE *fbu, const TYPE emy, int lim)
{
    for (int x = 0; x < lim; x++)
    {
        ac[x] += fap[x] * fbu[x] * emy;
    }
}

void CalculateRowMirror(TYPE *ac, const TYPE *fap, const TYPE *fbu, const TYPE emy, int lim, int trueLength)
{
    ac[0] += fap[0] * TYPE::conj(fbu[0]) * emy;
    for (int x = 1; x < lim; x++)
    {
        ac[x] += fap[x] * TYPE::conj(fbu[trueLength-x]) * emy;
    }
}

void CalculateRowBoth(TYPE *ac, const TYPE *fap1, const TYPE *fap2, const TYPE *fbu, const TYPE emy1, const TYPE emy2, int lim, int trueLength)
{
    /*  Increase memory efficiency by processing two different pixel locations (fap1, fap2) in one go, for the same z,t - i.e. accumulating into the same accumulator.
        This has the benefit of reusing the same row of fbu for a second operation, while it is resident in the cache.
        The gains are not massive, but are definitely measurable.
        I believe the gains are because we only have to read/write ac once from main memory. I don't expect a gain from fap,
        because that should be cache-resident (it is reused in the y loop - see modulo operation in calling code).
     
        The question then is, can I parallelise any more than this?
        I am saying that I think the bandwidth limitations are in accessing ac, and reading fbu.
        Therefore, there are two potential ways I could speed things up more:
        1. Process a second PSF (fbu2) into the same accumulator. That would further amortise the read and write of ac.
        2. Process a second z plane or timepoint, into a different accumulator. That would amortise the read of fbu.
     */
    ac[0] += fap1[0] * fbu[0] * emy1;
    ac[0] += fap2[0] * TYPE::conj(fbu[0]) * emy2;
    for (int x = 1; x < lim; x++)
    {
        ac[x] += fap1[x] * fbu[x] * emy1;
        ac[x] += fap2[x] * TYPE::conj(fbu[trueLength-x]) * emy2;
    }
}

class WorkItem
{
private:  // To ensure callers go through our accessor functions
    bool        complete, persistentMemory;
    WorkItem    *dependency;
    int         dependencyCount;
public:
    int         cc, order;
    double      runStartTime, runEndTime, mutexWaitStartTime[2], mutexWaitEndTime[2];
    size_t      memoryUsage;
    int         ranOnThread;
    
    WorkItem(int _cc, int _order) : complete(false), persistentMemory(false), dependency(NULL), dependencyCount(0), cc(_cc), order(_order)
    {
        mutexWaitStartTime[0] = mutexWaitEndTime[0] = 0;
        mutexWaitStartTime[1] = mutexWaitEndTime[1] = 0;
    }
    virtual ~WorkItem() { ALWAYS_ASSERT(dependencyCount == 0); }
    virtual void Reset(void) { complete = false; }      // For if we are recycling for another full run through all the work
    bool PersistentMemory(void) const { return persistentMemory; }
    void SetPersistentMemory(void) { persistentMemory = true; }
    virtual void Run(void) = 0;
    virtual void CleanUpAllocations(void) = 0;
    void RunComplete(void)
    {
        // Care is needed here about thread safety, but for now I think I am ok to do this without a mutex,
        // since I am just polling in the case where a thread is blocked waiting on another piece of work.
        // That case only happens at the end of the entire projection operation - until then, *something*
        // should always be available to run.
        complete = true;
        // And here again, AdjustDependencyCount is designed to be threadsafe
        if (dependency != NULL)
            dependency->AdjustDependencyCount(-1);
    }
    bool Complete(void)
    {
        return complete;
    }
    bool CanRun(void)   // Note that this is not fully threadsafe, and caller must be aware of window conditions (or can just poll!)
    {
        if (dependency != NULL)
            return dependency->Complete();
        else
            return true;
    }
    void AdjustDependencyCount(int delta)
    {
        // This function is designed to be threadsafe. We update dependencyCount in a threadsafe manner,
        // and only one thread will cause CleanUpAllocations to be called for a given object.
        int newVal = __sync_add_and_fetch(&dependencyCount, delta);
        ALWAYS_ASSERT(newVal >= 0);
        if ((newVal == 0) && (!persistentMemory))
        {
            // All dependencies have run. We should be safe to free any allocated memory now
            CleanUpAllocations();
        }
    }
    void AddDependency(WorkItem *dep)
    {
        dependency = dep;
        dep->AdjustDependencyCount(+1);
    }
#if 0
    // Reorder so that all items associated with the same offset within a lenslet are batched together (for all z).
    // The idea is that this reduces mutex contention when processing only a few timepoints at a time.
    static int Compare(WorkItem *a, WorkItem *b)        // Note that the pointers are because the std::vector itself is a vector of pointers
    {
        return a->order < b->order;
    }
#elif 0
    // Attempted reorder to reduce the amount of allocated memory floating around, i.e. try and have FFTs used
    // as soon as possible after they are computed. For small problem sizes this *might* improve cache usage a little bit.
    // The problem, though, is that it leads to mutex contention. I have not tried to resolve that limitation yet.
    static int Compare(const WorkItem *a, const WorkItem *b)        // Note that the pointers are because the std::vector itself is a vector of pointers
    {
        const WorkItem *aRoot = a, *bRoot = b;
        int aLen = 0, bLen = 0;
        while (aRoot->dependency != NULL)
        {
            aRoot = aRoot->dependency;
            aLen++;
        }
        while (bRoot->dependency != NULL)
        {
            bRoot = bRoot->dependency;
            bLen++;
        }
        // Primary sort order: ultimate root dependency order
        if (aRoot->order != bRoot->order)
            return aRoot->order < bRoot->order;
        if (aRoot->cc != bRoot->cc)
            return aRoot->cc < bRoot->cc;
        // Then shallowest dependency chain first, since they will become available first.
        if (aLen != bLen)
            return aLen < bLen;
        // Then z [for sorting of FFTs]
        // Then in-place sort order, by default
        return a->cc < b->cc;
    }
#else
    /*  This sort order tries to optimise two criteria simultaneously:
        1. Minimise memory usage, i.e. try and have FFTs used as soon as possible after they are computed.
            This avoids potentially using hundreds of times more RAM than the size of a volume(!),
            which is catastrophic for large problem sizes.
        2. Minimise mutex contention, i.e. make sure there are sufficient other work items between any two
            that refer to the same z and t value.
     
        To do this we aim to process each z in sequence, but we may batch up in z if there are insufficient
        timepoints to enable each thread to work independently on a different timepoint (if this is not the
        case then we risk mutex contention)
     
        The z coordinate does not feature in the "order" variable, in other words it has us do the equivalent
        operations for all z and t in sequence, before moving on to the next a,b lenslet position.
        What we need to do is to batch things up into small groups of z, and do *everything* for that group
        before moving on to other z coordinates.
        The z batch size needs to be just big enough to avoid mutex contention (see gMutexContentionFactor),
        but otherwise as small as possible to avoid memory bloat with intermediate FFT results etc hanging
        around for any longer than needed.
        TODO: after that first priority, it will be interesting to see if either one of my two above strategies is better than the other.
        for
     */
    static int Compare(const WorkItem *a, const WorkItem *b)        // Note that the pointers are because the std::vector itself is a vector of pointers
    {
        const WorkItem *aRoot = a, *bRoot = b;
        int aLen = 0, bLen = 0;
        while (aRoot->dependency != NULL)
        {
            aRoot = aRoot->dependency;
            aLen++;
        }
        while (bRoot->dependency != NULL)
        {
            bRoot = bRoot->dependency;
            bLen++;
        }
        // Primary sort order: ultimate root dependency order
        if (aRoot->order != bRoot->order)
            return aRoot->order < bRoot->order;
        // Sort kind-of on z coordinate, but adjusted such that that we may process a few
        // z coordinates together, to avoid mutex contention if we have more threads than timepoints
        int aRootCC = aRoot->cc / gMutexContentionFactor;
        int bRootCC = bRoot->cc / gMutexContentionFactor;
        if (aRootCC != bRootCC)
            return aRootCC < bRootCC;
        // Then shallowest dependency chain first, since they will become available first.
        if (aLen != bLen)
            return aLen < bLen;
        // Then z [for sorting of FFTs]
        // And finally in-place sort order, by default
        int aCC = a->cc / gMutexContentionFactor;
        int bCC = b->cc / gMutexContentionFactor;
        return aCC < bCC;
    }
#endif
    virtual void *DestArray() = 0;
    virtual void *SrcArray1() = 0;
    virtual void *SrcArray2() = 0;
};

enum
{
    kBenchmarkRead = 0,
    kBenchmarkDualRead,
    kBenchmarkWrite,
    kBenchmarkReadWrite,
    kBenchmarkIncrement,
    kBenchmarkCalculateRow,
    kBenchmarkCalculateRowAdd,  // Dummy to remove computational load of multiplication
    kBenchmarkCalculateRow2,
    kNumBenchmarkTypes
};

/*  We need to make sure our batch is large enough to saturate the caches, but (on OS X at least) the initial setup
    is actually really slow so I don't want to make this larger than I have to. This maps to 16MB per memory block.
    Even for the simplest benchmark, this should be enough to saturate the L3 cache, which is 8MB per processor on my mac pro.
 
 */
const int kNumElementsInBatch = 2*1000*1000;

class BenchmarkWorkItem : public WorkItem
{
public:
    int numElements, benchmarkType;
    TYPE *mem1, *mem2, *mem3, *mem4;

    BenchmarkWorkItem(int _numElements, int _benchmarkType) : WorkItem(0, 0), numElements(_numElements), benchmarkType(_benchmarkType)
    {
        //printf("%p initializing\n", this);
        mem1 = new TYPE[kNumElementsInBatch];
        mem2 = new TYPE[kNumElementsInBatch];
        mem3 = new TYPE[kNumElementsInBatch];
        mem4 = new TYPE[kNumElementsInBatch];
        for (int i = 0; i < kNumElementsInBatch; i++)
        {
            mem1[i] = i;
            mem2[i] = i;
            mem3[i] = i;
            mem4[i] = i;
        }
        //printf("%p initialized\n", this);
    }
    virtual ~BenchmarkWorkItem()
    {
        delete[] mem1;
        delete[] mem2;
        delete[] mem3;
        delete[] mem4;
    }
    static void CalculateRowDummy(TYPE *ac, const TYPE *fap, const TYPE *fbu, const TYPE emy, int lim)
    {
        // Multiplies replaced with addition to reduce computational load
        for (int x = 0; x < lim; x++)
            ac[x] += fap[x] + fbu[x] + emy;
    }

    virtual void Run(void)
    {
        TYPE sum = 0;
        int workRemaining = numElements;
        while (workRemaining)
        {
            int numInBatch = MIN(workRemaining, kNumElementsInBatch);
            TYPE *__restrict__ _mem1 = mem1;
            TYPE *__restrict__ _mem2 = mem2;
            TYPE *__restrict__ _mem3 = mem3;
            TYPE *__restrict__ _mem4 = mem4;
            if (benchmarkType == kBenchmarkRead)
            {
                for (int i = 0; i < numInBatch; i++)
                    sum += _mem1[i];
            }
            else if (benchmarkType == kBenchmarkDualRead)
            {
                for (int i = 0; i < numInBatch; i++)
                    sum += _mem1[i] + _mem2[i];
            }
            else if (benchmarkType == kBenchmarkWrite)
            {
                TYPE val = _mem1[0]; // Ensure the value we will write is not known to compiler
                for (int i = 0; i < numInBatch; i++)
                    _mem1[i] = val;
            }
            else if (benchmarkType == kBenchmarkReadWrite)
            {
                for (int i = 0; i < numInBatch; i++)
                    _mem1[i] = _mem2[i];
            }
            else if (benchmarkType == kBenchmarkIncrement)
            {
                for (int i = 0; i < numInBatch; i++)
                    _mem1[i] += TYPE(1.0f,0.1f);
            }
            /*  TODO: these next benchmarks may not be entirely representative of my real workload.
                fap will be read from an array that is about 50-150kB in size.
                My Mac pro has 256kB L2 cache per core, and the L3 cache is 8MB.
                fap is therefore expected to be resident in the L3 cache, though probably not L2.
                Access to L3 will be faster than uncached reads (although my measurements suggest possibly only 2x faster).
                I will probably need to construct a benchmark quite carefully, then, if I want to represent that fairly.
                I would need to think about how much faster it is - is it fast enough that I can just treat that as a constant for my benchmark?
             */
            else if (benchmarkType == kBenchmarkCalculateRow)
            {
#if 0
                CalculateRow(_mem1, _mem2, _mem3, _mem1[0], numInBatch);
#else
                // Attempt to reflect the fact that fab1,2 will be smaller arrays that should be resident in the L3 cache (reused ~10 times in my real use-case).
                const int kNumElementsInSubBatch = 10000;
                int subWorkRemaining = numInBatch;
                int numInSubBatch = MIN(subWorkRemaining, kNumElementsInSubBatch);
                size_t off = 0;
                while (subWorkRemaining)
                {
                    CalculateRow(_mem1+off, _mem2, _mem3+off, _mem1[0], numInSubBatch);
                    off += numInSubBatch;
                    subWorkRemaining -= numInSubBatch;
                }
#endif
            }
            else if (benchmarkType == kBenchmarkCalculateRowAdd)
                CalculateRowDummy(_mem1, _mem2, _mem3, _mem1[0], numInBatch);
            else if (benchmarkType == kBenchmarkCalculateRow2)
            {
#if 0
                CalculateRowBoth(_mem1, _mem2, _mem3, _mem4, _mem1[0], _mem2[0], numInBatch, numInBatch);
#else
                // Attempt to reflect the fact that fab1,2 will be smaller arrays that should be resident in the L3 cache (reused ~10 times in my real use-case).
                const int kNumElementsInSubBatch = 10000;
                int subWorkRemaining = numInBatch;
                int numInSubBatch = MIN(subWorkRemaining, kNumElementsInSubBatch);
                size_t off = 0;
                while (subWorkRemaining)
                {
                    CalculateRowBoth(_mem1+off, _mem2, _mem3, _mem4+off, _mem1[0], _mem2[0], numInSubBatch, numInSubBatch);
                    off += numInSubBatch;
                    subWorkRemaining -= numInSubBatch;
                }
#endif
            }
            else
                ALWAYS_ASSERT(0);
            workRemaining -= numInBatch;
        }
        mem1[0] = sum;   // Ensure side-effect from the result
        RunComplete();
    }
    virtual void CleanUpAllocations(void) { }
    virtual void *DestArray() { return mem1; }
    virtual void *SrcArray1() { return mem2; }
    virtual void *SrcArray2() { return mem3; }
};

size_t gMemoryUsage = 0;

void TrackAllocation(JPythonArray<TYPE> *array, bool dealloc=false)
{
    size_t size = sizeof(TYPE);
    for (int i = 0; i < array->NDims(); i++)
        size *= array->Dims()[i];
    if (dealloc)
        __sync_fetch_and_sub(&gMemoryUsage, size);
    else
        __sync_fetch_and_add(&gMemoryUsage, size);
}

void TrackDeallocation(JPythonArray<TYPE> *array)
{
    TrackAllocation(array, true);
}

class FHWorkItemBase : public WorkItem
{
public:
    int                     fshapeY, fshapeX, backwards;
    JPythonArray2D<TYPE>    *fftResult;
    void                    *destArrayForLogging;       // Needed because we will clean up our result array once we have finished with it
    
    FHWorkItemBase(int _fshapeY, int _fshapeX, int _backwards, int _cc, int _order) : WorkItem(_cc, _order), fshapeY(_fshapeY), fshapeX(_fshapeX), backwards(_backwards), fftResult(NULL)
    {
    }
    
    void AllocateResultArray(void)
    {
        // Allocate an array to hold the results
        npy_intp dims[2] = { fshapeY, fshapeX };
        npy_intp strides[2] = { int((fshapeX + 15)/16)*16, 1 };     // Ensure 16-byte alignment of each row, which seems to make things *slightly* faster
        fftResult = new JPythonArray2D<TYPE>(NULL, dims, strides);
        TrackAllocation(fftResult);
        ALWAYS_ASSERT(!(((size_t)fftResult->Data()) & 0xF));        // Check base alignment. In fact, just plain malloc seems to give sufficient alignment.
        destArrayForLogging = fftResult->Data();
    }
    
    virtual ~FHWorkItemBase()
    {
        // Memory should already have been freed by a call to CleanUpAllocations
        // (Note that that's only the case because we know all FHWorkItems have a dependency count)
        ALWAYS_ASSERT(fftResult == NULL);
    }

    virtual void CleanUpAllocations(void)
    {
        ALWAYS_ASSERT(fftResult != NULL);
        TrackDeallocation(fftResult);
        delete fftResult;
        fftResult = NULL;
    }
    virtual void *DestArray() { return destArrayForLogging; }
};

class FHWorkItem : public FHWorkItemBase
{
public:
    JPythonArray2D<RTYPE>   Hts;
    bool                    transpose;
    fftwf_plan              plan, plan2;
    
    
    FHWorkItem(JPythonArray2D<RTYPE> _Hts, int _fshapeY, int _fshapeX, int _backwards, bool _transpose, int _cc, int _order)
        : FHWorkItemBase(_fshapeY, _fshapeX, _backwards, _cc, _order), Hts(_Hts), transpose(_transpose)
    {
        /*  Set up the FFT plan.
            The complication here is that we cannot afford to allocate memory for every 
            FFT array simultaneously, as we would exhause all the available memory.
            Because of this we temporarily allocate an array for FFTW planning purposes.
            I am imagining this will never be a bottleneck, but I will want to keep an eye on how long the setup time takes.  */
        fftwf_plan_with_nthreads(1);
        AllocateResultArray();      // Just temporarily, for FFT planning! We will delete it at the end of this function
#if 0
        // Original - 35s
        int nx[1] = { fftResult->Dims(1) };
        // Compute the horizontal 1D FFTs, for only the nonzero rows
        // Note that we manually split up the 2D FFT into horizontal and vertical 1D FFTs, to enable us to skip the rows that we know are all zeroes.
        // Note also that this order (horizontal rows first, then vertical) seems to perform slightly faster under some circumstances
        // I don't know if that's because it uses SSE instructions to parallelise the independent column operations, or what.
        plan = fftwf_plan_many_dft(1, nx, Hts.Dims(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   FFTW_FORWARD, gFFTPlanMethod);
        // Compute the vertical 1D FFTs
        int ny[1] = { fftResult->Dims(0) };
        plan2 = fftwf_plan_many_dft(1, ny, fftResult->Dims(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    FFTW_FORWARD, gFFTPlanMethod);
#else
        // Swapped order - 32s
        // Actually, under my benchmarking circumstances, it's slightly faster to do the FFTs in this order
        int nx[1] = { fftResult->Dims(1) };
        int ny[1] = { fftResult->Dims(0) };
        // Compute the vertical 1D FFTs, for only the nonzero rows
        // Note that we manually split up the 2D FFT into vertical and horizontal 1D FFTs, to enable us to skip the rows that we know are all zeroes.
        plan = fftwf_plan_many_dft(1, ny, Hts.Dims(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    FFTW_FORWARD, gFFTPlanMethod);
        // Compute the horizontal 1D FFTs
        plan2 = fftwf_plan_many_dft(1, nx, fftResult->Dims(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   FFTW_FORWARD, gFFTPlanMethod);
#endif
        //PySys_WriteStdout("Plan %d %d %d %d %d\n", _cc, fftResult->Dims(0), fftResult->Dims(1), fftResult->Strides(0), fftResult->Strides(1));
        CleanUpAllocations();
    }
    
    virtual ~FHWorkItem()
    {
        fftwf_destroy_plan(plan);
        fftwf_destroy_plan(plan2);
    }
    
    void Run(void)
    {
        /*  Performance note: examination of threads.txt reveals that there is huge variation in how long it takes to do
            the initial zeroing and the setting of the array. It is often very fast, but occasionally takes almost as long
            as the FFT itself. I don't know why that is, but it could be something to do with whether or not we are reusing
            a previously-allocated block of memory. I'm not too worried about this, though, since my aim is to amortise away the FFTs anyway.  */
        AllocateResultArray();
        fftResult->SetZero();
        assert(Hts.Dims(0) == Hts.Dims(1));     // Sanity check - this should be true. If it is not, then among other things our whole transpose logic gets messed up.
        if (transpose)
            for (int y = 0; y < Hts.Dims(0); y++)
            {
                JPythonArray1D<TYPE> _result = (*fftResult)[y];
                for (int x = 0; x < Hts.Dims(1); x++)
                    _result[x] = Hts[x][y];
            }
        else
            for (int y = 0; y < Hts.Dims(0); y++)
            {
                JPythonArray1D<TYPE> _result = (*fftResult)[y];
                JPythonArray1D<RTYPE> _Hts = Hts[y];
                for (int x = 0; x < Hts.Dims(1); x++)
                    _result[x] = _Hts[x];
            }
        
        mutexWaitEndTime[0] = GetTime();
        // Compute the full 2D FFT (i.e. not just the RFFT)
        fftwf_execute_dft(plan, (fftwf_complex *)fftResult->Data(), (fftwf_complex *)fftResult->Data());
        mutexWaitEndTime[1] = GetTime();
        fftwf_execute_dft(plan2, (fftwf_complex *)fftResult->Data(), (fftwf_complex *)fftResult->Data());
        RunComplete();
    }
    
    virtual void *SrcArray1() { return Hts.Data(); }
    virtual void *SrcArray2() { return NULL; }
};

class TransposeWorkItem : public FHWorkItemBase
{
public:
    FHWorkItemBase        *sourceFFTWorkItem;
    
    TransposeWorkItem(FHWorkItemBase *_sourceFFTWorkItem, int _backwards, int _cc, int _order)
    : FHWorkItemBase(_sourceFFTWorkItem->fshapeY, _sourceFFTWorkItem->fshapeX, _backwards, _cc, _order), sourceFFTWorkItem(_sourceFFTWorkItem)
    {
        ALWAYS_ASSERT(fshapeY == fshapeX);  // Transpose only works when FFT(H) is a square array
        AddDependency(sourceFFTWorkItem);
    }
    void Run(void)
    {
        ALWAYS_ASSERT(fshapeY == sourceFFTWorkItem->fshapeY);
        ALWAYS_ASSERT(fshapeX == sourceFFTWorkItem->fshapeX);
        ALWAYS_ASSERT(sourceFFTWorkItem->fftResult->Strides(1) == 1);   // Assumption relied on in inner loop
        AllocateResultArray();
        for (int y = 0; y < fshapeY; y++)
        {
            JPythonArray1D<TYPE> _result = (*fftResult)[y];
            /*  Performance is measurably impacted if I write a naive inner loop that uses (*sourceFFTWorkItem->fftResult)[x][y].
                Instead, I manually compute the strides myself (to avoid creating temporary objects)    */
            TYPE *sourceColBase = sourceFFTWorkItem->fftResult->Data() + y;
            size_t stride = sourceFFTWorkItem->fftResult->Strides(0);
            for (int x = 0; x < fshapeX; x++)
                _result[x] = sourceColBase[x*stride];
        }
        RunComplete();
    }
    virtual void *SrcArray1() { return sourceFFTWorkItem->DestArray(); }
    virtual void *SrcArray2() { return NULL; }
};

class MirrorWorkItem : public FHWorkItemBase
{
public:
    FHWorkItemBase        *sourceFFTWorkItem;
    JPythonArray1D<TYPE>  mirrorYMultiplier;
    
    MirrorWorkItem(FHWorkItemBase *_sourceFFTWorkItem, JPythonArray1D<TYPE> _mirrorYMultiplier, int _backwards, int _cc, int _order)
        : FHWorkItemBase(_sourceFFTWorkItem->fshapeY, _sourceFFTWorkItem->fshapeX, _backwards, _cc, _order), sourceFFTWorkItem(_sourceFFTWorkItem), mirrorYMultiplier(_mirrorYMultiplier)
    {
        AddDependency(sourceFFTWorkItem);
    }
    void Run(void)
    {
        AllocateResultArray();
        MirrorYArray(*sourceFFTWorkItem->fftResult, mirrorYMultiplier, *fftResult);
        RunComplete();
    }
    virtual void *SrcArray1() { return sourceFFTWorkItem->DestArray(); }
    virtual void *SrcArray2() { return NULL; }
};

class IFFTWorkItem : public FHWorkItemBase
{
    /*  This work item is a bit different - we run all the IFFTs in a separate batch, rather than mixing them in
        with the other work items. Apart from anything else, this is because they have many, many dependencies!
        Note also that I do not use fftResult from the base class - the caller sets up 'result' and passes it in to us.
    */
public:
    JPythonArray2D<TYPE>    fab;
    JPythonArray2D<TYPE>    *paddedMatrix;
    JPythonArray2D<RTYPE>   result;
    fftwf_plan              plan;
    
    IFFTWorkItem(JPythonArray2D<TYPE> _fab, JPythonArray2D<RTYPE> _result, int _fshapeY, int _fshapeX, int _cc, int _order)
    : FHWorkItemBase(_fshapeY, _fshapeX, false, _cc, _order), fab(_fab), result(_result)
    {
        //  Set up the FFT plan.
        fftwf_plan_with_nthreads(1);
        AllocatePaddedMatrix();
        int dims[2] = { result.Dims(0), result.Dims(1) };
        int inFullShape[2] = { paddedMatrix->Dims(0), paddedMatrix->Strides(0) };
        int outFullShape[2] = { result.Dims(0), result.Strides(0) };
        ALWAYS_ASSERT(fab.Strides(1) == 1);     // We assume contiguous
        ALWAYS_ASSERT(paddedMatrix->Strides(1) == 1);     // We assume contiguous
        ALWAYS_ASSERT(result.Strides(1) == 1);  // We assume contiguous
        fftwf_plan_with_nthreads(1);
        plan = fftwf_plan_many_dft_c2r(2/*2D FFT*/, dims, 1/*howmany*/,
                                       (fftwf_complex *)paddedMatrix->Data(), inFullShape,
                                       1/*stride*/, 0/*unused*/,
                                       result.Data(), outFullShape,
                                       1/*stride*/, 0/*unused*/,
                                       gFFTPlanMethod);
        DeallocatePaddedMatrix();
    }
    
    virtual ~IFFTWorkItem()
    {
        fftwf_destroy_plan(plan);
    }
    
    void AllocatePaddedMatrix(void)
    {
        npy_intp paddedInputDims[2] = { fshapeY, int(fshapeX/2)+1 };
        paddedMatrix = new JPythonArray2D<TYPE>(NULL, paddedInputDims, NULL);
        TrackAllocation(paddedMatrix);
    }
    
    void DeallocatePaddedMatrix(void)
    {
        TrackDeallocation(paddedMatrix);
        delete paddedMatrix;
    }
    
    virtual void Run(void)
    {
        result.SetZero();
        
        // TODO: need to confirm whether I use result.Dims() or the dims of something else, if I consider a case with extra padding.
        // The answer seems to be result.Dims since my code works like this, but I should check for sure and write a definitive comment here.
        float inverseTotalSize = 1.0f / (float(result.Dims(0)) * float(result.Dims(1)));
        AllocatePaddedMatrix();
        ALWAYS_ASSERT(paddedMatrix->Dims(0) >= fab.Dims(0)); // This assertion is because we don't support cropping the input matrix, only padding
        ALWAYS_ASSERT(paddedMatrix->Dims(1) >= fab.Dims(1));
        paddedMatrix->SetZero();     // Inefficient, but I will do this for now. TODO: update this code to only zero out the (small number of) values we won't be overwriting in the loop
        mutexWaitEndTime[0] = GetTime();    // No real need for these, but it makes it easier to process the threads file if we record these similar to with a standard FFT
        for (int y = 0; y < fab.Dims(0); y++)
        {
            JPythonArray1D<TYPE> _fab = fab[y];
            JPythonArray1D<TYPE> _paddedMatrix = (*paddedMatrix)[y];
            for (int x = 0; x < fab.Dims(1); x++)
            _paddedMatrix[x] = _fab[x] * inverseTotalSize;
        }
        mutexWaitEndTime[1] = GetTime();
        
        // Compute the full 2D FFT (i.e. not just the RFFT)
        fftwf_execute_dft_c2r(plan, (fftwf_complex *)paddedMatrix->Data(), result.Data());
        
        DeallocatePaddedMatrix();
    }
    virtual void *DestArray() { return result.Data(); };
    virtual void *SrcArray1() { return fab.Data(); };
    virtual void *SrcArray2() { return paddedMatrix->Data(); };
};

class ConvolveWorkItem : public WorkItem
{
    JPythonArray2D<RTYPE> projection;
    int                   bbUnmirrored, aa, Nnum;
    FHWorkItemBase        *fhWorkItem_unXmirrored;
    bool                  mirrorX;
    JPythonArray2D<TYPE>  xAxisMultipliers;
    JPythonArray3D<TYPE>  yAxisMultipliers3;
    JPythonArray2D<TYPE>  accum;
    JMutex                *accumMutex;
    
    fftwf_plan            plan;     // Pointer to plan held by the caller

public:
    ConvolveWorkItem(JPythonArray2D<RTYPE> _projection, int _bb, int _aa, int _Nnum, FHWorkItemBase *_fhWorkItem_unXmirrored, bool _mirrorX, JPythonArray2D<TYPE> _xAxisMultipliers, JPythonArray3D<TYPE> _yAxisMultipliers3, JPythonArray2D<TYPE> _accum, JMutex *_accumMutex, fftwf_plan _plan, int _cc, int _order)
                       : WorkItem(_cc, _order), projection(_projection), bbUnmirrored(_bb), aa(_aa), Nnum(_Nnum), fhWorkItem_unXmirrored(_fhWorkItem_unXmirrored), mirrorX(_mirrorX),
                         xAxisMultipliers(_xAxisMultipliers), yAxisMultipliers3(_yAxisMultipliers3), accum(_accum), accumMutex(_accumMutex), plan(_plan)
    {
        AddDependency(fhWorkItem_unXmirrored);
    }
    
    static fftwf_plan GetFFTPlan(FHWorkItemBase *_fhWorkItem_unXmirrored, int Nnum)
    {
        // The convolution itself is actually fast enough that it takes a non-negligible amount of time to request the FFT plan if we do it separately for every convolution!
        // To avoid this, the main code should call this once and pass the result to many ConvolveWorkItems.
        // Define the FFTW plan that we will use as part of our convolution task
        npy_intp smallDims[2] = { _fhWorkItem_unXmirrored->fshapeY/Nnum, _fhWorkItem_unXmirrored->fshapeX/Nnum };
        JPythonArray2D<TYPE> fftArray(smallDims);
        // A reminder that this planning may modify the data in fftArray, especially if using FFTW_MEASURE.
        // That's fine, becasue we have not filled it in yet.
        fftwf_plan result = fftwf_plan_dft_2d((int)smallDims[0], (int)smallDims[1], (fftwf_complex *)fftArray.Data(), (fftwf_complex *)fftArray.Data(), FFTW_FORWARD, gFFTPlanMethod);
        ALWAYS_ASSERT(result != NULL);
        return result;
    }
    
    virtual ~ConvolveWorkItem()
    {
    }
    
    virtual void CleanUpAllocations(void)
    {
    }
    
    virtual const JMutex *AccumMutex(void) const { return accumMutex; }     // Just an accessor to be able to log which work items are dependent on the same mutex
    
    void special_fftconvolve_part1(JPythonArray2D<TYPE> &result, JPythonArray1D<TYPE> expandXMultiplier, int bb, int aa)
    {
        // Select the pixels indexed by bb,aa, and pad to the size we need based on what we will eventually tile up to the same shape as fHtsFull_unXmirrored
        npy_intp smallDims[2] = { fhWorkItem_unXmirrored->fshapeY/Nnum, fhWorkItem_unXmirrored->fshapeX/Nnum };
        JPythonArray2D<TYPE> fftArray(smallDims);
        fftArray.SetZero();    // Note that we may not be setting every element, I think, due to padding - so we do need to set to zero initially
        for (int y = bb, y2 = 0; y < projection.Dims(0); y += Nnum, y2++)
            for (int x = aa, x2 = 0; x < projection.Dims(1); x += Nnum, x2++)
                fftArray[y2][x2] = projection[y][x];
        
        // Compute the full 2D FFT (i.e. not just the RFFT)
        fftwf_execute_dft(plan, (fftwf_complex *)fftArray.Data(), (fftwf_complex *)fftArray.Data());
        
        // Tile the result up to the length that is implied by expandXMultiplier (using that length saves us figuring out the length for ourselves)
        int outputLength = expandXMultiplier.Dims(0);
        for (int y = 0; y < fftArray.Dims(0); y++)
        {
            JPythonArray1D<TYPE> _result = result[y];
            JPythonArray1D<TYPE> _fftArray = fftArray[y];
            for (int x = 0, inputX = 0; x < outputLength; x++, inputX++)
            {
                if (inputX == fftArray.Dims(1))     // Modulo operator is surprisingly time-consuming - it is faster to do this test instead
                    inputX = 0;
                _result[x] = _fftArray[inputX] * expandXMultiplier[x];
            }
        }
    }
    
    void ConvolvePart4Nomirror(int bb, int aa, JPythonArray2D<TYPE> yAxisMultipliers)
    {
        ALWAYS_ASSERT(!mirrorX);
        // We take advantage of the fact that we have been passed fHTsFull, to tell us what the padded array dimensions should be for the FFT.
        npy_intp output_dims[2] = { fhWorkItem_unXmirrored->fshapeY/Nnum, xAxisMultipliers.Dims(1) };
        JPythonArray2D<TYPE> partialFourierOfProjection(output_dims);
        special_fftconvolve_part1(partialFourierOfProjection, xAxisMultipliers[kXAxisMultiplierExpandXStart+aa], bb, aa);
        
        /*  Protect accum with a mutex, to avoid multiple threads potentially overwriting each other.
         It is up to the caller to provide scope for parallelism by providing fine-grained mutexes
         (e.g. different timepoints, and different z coordinates, are not using the same actual 2D accum array).
         In reality I think what we are protecting against is a tiny window condition in the += operator,
         but I cannot in good conscience ignore it, since it would lead to incorrect numerical results.   */
        mutexWaitStartTime[0] = GetTime();
        LocalGetMutex lgm(accumMutex);
        mutexWaitEndTime[0] = GetTime();
        // Do the actual updates to accum
        for (int y = 0; y < accum.Dims(0); y++)
        {
            int ya = y % partialFourierOfProjection.Dims(0);
            // Note that the array accessors take time, so need to be lifted out of CalculateRow.
            // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
            CalculateRow(&accum[y][0], &partialFourierOfProjection[ya][0], &(*fhWorkItem_unXmirrored->fftResult)[y][0], yAxisMultipliers[bb][y], accum.Dims(1));
        }
    }
    
    void ConvolvePart4Mirror(int bb1, int bb2, int aa, JPythonArray2D<TYPE> yAxisMultipliers1, JPythonArray2D<TYPE> yAxisMultipliers2)
    {
        // This code does both the standard and the x-mirrored convolution in parallel.
        // By reducing overall memory accesses, this gives a ~12% speedup to the entire projection operation
        
        // We take advantage of the fact that we have been passed fHTsFull, to tell us what the padded array dimensions should be for the FFT.
        npy_intp output_dims[2] = { fhWorkItem_unXmirrored->fshapeY/Nnum, xAxisMultipliers.Dims(1) };
        JPythonArray2D<TYPE> partialFourierOfProjection1(output_dims);
        special_fftconvolve_part1(partialFourierOfProjection1, xAxisMultipliers[kXAxisMultiplierExpandXStart+aa], bb1, aa);
        JPythonArray2D<TYPE> partialFourierOfProjection2(output_dims);
        special_fftconvolve_part1(partialFourierOfProjection2, xAxisMultipliers[kXAxisMultiplierExpandXStart+aa], bb2, aa);
        
        /*  Protect accum with a mutex, to avoid multiple threads potentially overwriting each other.
         It is up to the caller to provide scope for parallelism by providing fine-grained mutexes
         (e.g. different timepoints, and different z coordinates, are not using the same actual 2D accum array).
         In reality I think what we are protecting against is a tiny window condition in the += operator,
         but I cannot in good conscience ignore it, since it would lead to incorrect numerical results.   */
        mutexWaitStartTime[0] = GetTime();
        LocalGetMutex lgm(accumMutex);
        mutexWaitEndTime[0] = GetTime();
        // Do the actual updates to accum
        for (int y = 0; y < accum.Dims(0); y++)
        {
            int ya = y % partialFourierOfProjection1.Dims(0);
            // Note that the array accessors take time, so need to be lifted out of CalculateRow.
            // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
            CalculateRowBoth(&accum[y][0],
                             &partialFourierOfProjection1[ya][0], &partialFourierOfProjection2[ya][0],
                             &(*fhWorkItem_unXmirrored->fftResult)[y][0],
                             yAxisMultipliers1[bb1][y], yAxisMultipliers2[bb2][y],
                             accum.Dims(1), fhWorkItem_unXmirrored->fftResult->Dims(1));
        }
    }

    void Run(void)
    {
        // This operation plays the role that special_fftconvolve2 does in the python code.
        if (mirrorX)
            ConvolvePart4Mirror(bbUnmirrored, Nnum-bbUnmirrored-1, aa, yAxisMultipliers3[kYAxisMultiplierNoMirror], yAxisMultipliers3[kYAxisMultiplierMirrorX]);
        else
            ConvolvePart4Nomirror(bbUnmirrored, aa, yAxisMultipliers3[kYAxisMultiplierNoMirror]);
        
        RunComplete();
    }
    virtual void *DestArray() { return accum.Data(); };
    virtual void *SrcArray1() { return fhWorkItem_unXmirrored->DestArray(); };
    virtual void *SrcArray2() { return projection.Data(); };
};

enum
{
    // Reminder: if I add another work type then I need to manually update
    // the ThreadInfo initializers in RunWork(). I may not get a compiler warning about that.
    kWorkFFT = 0,
    kWorkTranspose,
    kWorkMirrorY,
    kWorkConvolve,
    kNumWorkTypes
};
const char *workNames[kNumWorkTypes] = { "fft", "trans", "mirror", "conv" };

double gLastReportingTime = GetTime();

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
        int         w, thisThreadID;
        {
            LocalGetMutex lgm(workQueueMutex, workQueueMutexBlock_us);
            thisThreadID = threadIDCounter++;
        }

//        PySys_WriteStdout("%.3lf === Running thread %d === \n", GetTime(), thisThreadID);
        while (1)
        {
//            printf("== %d Picking a work item ==\n", thisThreadID);
            WorkItem *workItem = NULL;
            // Pick a work item to run.
            double t1 = GetTime();
            bool polled = false;
            {
            repeat:
                LocalGetMutex lgm(workQueueMutex, workQueueMutexBlock_us);
                
                if (t1 > gLastReportingTime + 20.0)
                {
                    PySys_WriteStdout("Time %.1lf. Memory usage: %.3lfGB\n", t1, gMemoryUsage/1e9);
                    gLastReportingTime = t1;
                }
                
                // Run anything that is not blocked, prioritising the convolution work
                for (w = kNumWorkTypes - 1; w >= 0; w--)
                {
                    if (workCounter[w] < work[w]->size())
                    {
                        workItem = (*work[w])[workCounter[w]];
                        if (workItem->CanRun())
                        {
                            // First item of work[w] is not blocked - we should run it
//                            PySys_WriteStdout("%.3lf Work %s[%zd] object %p can run\n", GetTime(), workNames[w], workCounter[w], workItem);
                            workCounter[w]++;
                            break;
                        }
                        else
                        {
//                            printf("Work %s[%zd] object %p cannot run (waiting for %p)\n", workNames[w], workCounter[w], workItem, workItem->dependency);
                            workItem = NULL;
                        }
                    }
                }
                if (workItem == NULL)
                {
                    // All work types are either complete or are blocked.
                    // We should wait for something to unblock.
                    // We *MUST* prioritise the early work types (which in practice means the mirror),
                    // because otherwise we could end up deadlocked.
                    for (w = 0; w < kNumWorkTypes; w++)
                    {
                        if (workCounter[w] < work[w]->size())
                        {
                            /*  There is work remaining, but nothing was ready to run.
                                Rather than set up a complicated semaphore scheme here,
                                I really am just going to poll, by going back around the loop again until work is available.
                                This is inefficient, but in practice it will only happen for a short period at the very end
                                of the work list, so I am not concerned about this wasteful CPU load. 
                                (In practice, only one thread will be burning at any one time, since they will all be fighting
                                 for workQueueMutex!) */
                            //printf("%d polling!\n", thisThreadID);
                            polled = true;
                            workItem = NULL;
                            goto repeat;
                        }
                    }
                    // If we get here then all work is complete
                    break;
                }
                if (polled)
                    *pollingTime += GetTime()-t1;
            }
            // If we get here then we should have a work item to run
            ALWAYS_ASSERT(workItem != NULL);
            ALWAYS_ASSERT(workItem->CanRun());
            workItem->ranOnThread = thisThreadID;
            workItem->runStartTime = GetTime();
            workItem->Run();
            workItem->runEndTime = GetTime();
            workItem->memoryUsage = __sync_fetch_and_add(&gMemoryUsage, 0);
        }
        return NULL;
    }
};

void *ThreadFunc(void *params)
{
    ThreadInfo  *threadInfo = (ThreadInfo *)params;
    return threadInfo->ThreadFunc();
}

void RunWork(std::vector<WorkItem *> work[kNumWorkTypes])
{
    JMutex workQueueMutex;
    long workQueueMutexBlock_us = 0;
    double pollingTime = 0;
    if ((false))
    {
        // Initially doing this single-threaded (and in the correct order of work types), for test purposes
        printf("Running single-threaded\n");
        for (int w = 0; w < kNumWorkTypes; w++)
            for (size_t i = 0; i < work[w].size(); i++)
                work[w][i]->Run();
    }
    else if ((false))
    {
        // Intermediate code, for test purposes. This code is parallelised but does all the FFT work first, then all mirroring, then the actual convolutions
        printf("Running semi-parallelised\n");
        for (int w = 0; w < kNumWorkTypes; w++)
        {
            printf("Run work (%d)\n", w);
            ThreadInfo threadInfo { 0, {0, 0, 0, 0}, &workQueueMutex, &workQueueMutexBlock_us, &pollingTime, {&work[w], new std::vector<WorkItem *>(), new std::vector<WorkItem *>(), new std::vector<WorkItem *>()} };     // 'new' leaks, but this is only temporary code anyway
            std::vector<pthread_t> threads(gNumThreadsToUse);
            for (int i = 0; i < gNumThreadsToUse; i++)
                pthread_create(&threads[i], NULL, ThreadFunc, &threadInfo);
            for (int i = 0; i < gNumThreadsToUse; i++)
                pthread_join(threads[i], NULL);
        }
        printf("Returning from RunWork\n");
    }
    else
    {
        // Final parallelised code
        ThreadInfo threadInfo { 0, {0, 0, 0, 0}, &workQueueMutex, &workQueueMutexBlock_us, &pollingTime, {&work[0], &work[1], &work[2], &work[3]} };
        std::vector<pthread_t> threads(gNumThreadsToUse);
        for (int i = 0; i < gNumThreadsToUse; i++)
            pthread_create(&threads[i], NULL, ThreadFunc, &threadInfo);
        for (int i = 0; i < gNumThreadsToUse; i++)
            pthread_join(threads[i], NULL);
//        printf("%.1lfms spent waiting to acquire work queue mutex. %.1lfms spent polling.\n", workQueueMutexBlock_us/1e3, pollingTime*1e3);
    }
}

void CleanUpWork(std::vector<WorkItem *> work[kNumWorkTypes], const char *threadFileName, const char *mode, bool useCache)
{
    if ((threadFileName != NULL) && (strlen(threadFileName) > 0))
    {
        FILE *threadFile = fopen(threadFileName, mode);
        for (int w = 0; w < kNumWorkTypes; w++)
            for (size_t i = 0; i < work[w].size(); i++)
                fprintf(threadFile, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%zd\t%ld\t%ld\t%ld\n", work[w][i]->ranOnThread, w, work[w][i]->runStartTime, work[w][i]->runEndTime, work[w][i]->mutexWaitStartTime[0], work[w][i]->mutexWaitEndTime[0], work[w][i]->mutexWaitStartTime[1], work[w][i]->mutexWaitEndTime[1], work[w][i]->memoryUsage, (long)work[w][i]->DestArray(), (long)work[w][i]->SrcArray1(), (long)work[w][i]->SrcArray2());
        fclose(threadFile);
    }
    for (int w = 0; w < kNumWorkTypes; w++)
        for (size_t i = 0; i < work[w].size(); i++)
        {
            if (!work[w][i]->PersistentMemory())
                delete work[w][i];
        }
}

void ConvolvePart2(JPythonArray3D<RTYPE> projection, int bb, int aa, int Nnum, bool mirrorY, bool mirrorX, FHWorkItemBase *fftWorkItem, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray3D<TYPE> yAxisMultipliers, JPythonArray3D<TYPE> accum, std::vector<JMutex*> &accumMutex, fftwf_plan plan, std::vector<WorkItem *> work[kNumWorkTypes], int backwards, int cc, int &mCounter, int &cCounter)
{
    // Note that (in contrast to the python function of the same name) this function does not actually do the work,
    // it just sets up the WorkItems that will be run later.
    ALWAYS_ASSERT(projection.Dims(0) == accum.Dims(0));
    for (int i = 0; i < projection.Dims(0); i++)
    {
        ConvolveWorkItem *workConvolve = new ConvolveWorkItem(projection[i], bb, aa, Nnum, fftWorkItem, mirrorX, xAxisMultipliers, yAxisMultipliers, accum[i], accumMutex[i], plan, cc, cCounter);
        work[kWorkConvolve].push_back(workConvolve);
    }
    cCounter++;
    if (mirrorY)
    {
        MirrorWorkItem *workCalcMirror = new MirrorWorkItem(fftWorkItem, xAxisMultipliers[kXAxisMultiplierMirrorY], backwards, cc, mCounter);
        work[kWorkMirrorY].push_back(workCalcMirror);
        for (int i = 0; i < projection.Dims(0); i++)
        {
            ConvolveWorkItem *workConvolveMirror = new ConvolveWorkItem(projection[i], bb, Nnum-aa-1, Nnum, workCalcMirror, mirrorX, xAxisMultipliers, yAxisMultipliers, accum[i], accumMutex[i], plan, cc, cCounter);
            work[kWorkConvolve].push_back(workConvolveMirror);
        }
        mCounter++;
        cCounter++;
    }
}

typedef std::map<int,FHWorkItemBase *> CacheMapType;
CacheMapType gFHCache;
char *gCacheIdentifier = NULL;

extern "C" PyObject *ClearFHCache(PyObject *self, PyObject *args)
{
    // Free the objects referred to within the map
    for (CacheMapType::iterator i = gFHCache.begin(); i != gFHCache.end(); ++i)
    {
        i->second->CleanUpAllocations();
        delete i->second;
    }
    // Clear the map itself;
    gFHCache.clear();
    Py_RETURN_NONE;
}

extern "C" PyObject *EnableFHCachingWithIdentifier(PyObject *self, PyObject *args)
{
    // Enables caching of F(H) - but great care needed not to overrun the available memory!
    // To ensure we don't cache results from a different PSF, the caller should pass in an identifier,
    // which could for example be the path to the underlying PSF file.
    const char *cacheIdentifier;
    if (!PyArg_ParseTuple(args, "s", &cacheIdentifier))
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    
    if (gCacheIdentifier != NULL)
    {
        if (strcmp(gCacheIdentifier, cacheIdentifier))
            ClearFHCache(self, Py_None);
        delete gCacheIdentifier;
    }
    gCacheIdentifier = new char[strlen(cacheIdentifier)+1];
    strcpy(gCacheIdentifier, cacheIdentifier);
    Py_RETURN_NONE;
}

extern "C" PyObject *DisableFHCaching(PyObject *self, PyObject *args)
{
    if (gCacheIdentifier != NULL)
        delete gCacheIdentifier;
    gCacheIdentifier = NULL;
    ClearFHCache(self, Py_None);
    Py_RETURN_NONE;
}

#define CACHE_OR_CREATE(BK, C, B, A, WORKTYPE, VAR, TYPE, PARAMS)           \
{                                                                       \
    VAR = NULL;                                                         \
    int cacheIndex = BK + 2*(A + B*HtsFull.Dims(1) + C*HtsFull.Dims(0)*HtsFull.Dims(1));  \
    if (useCache)                                                       \
    {                                                                   \
        CacheMapType::iterator iter = gFHCache.find(cacheIndex);        \
        if (iter != gFHCache.end())                                     \
            VAR = iter->second;                                         \
    }                                                                   \
    if (VAR == NULL)                                                    \
    {                                                                   \
        VAR = new TYPE PARAMS;                                          \
        work[WORKTYPE].push_back(VAR);                                  \
        if (useCache)                                                   \
        {                                                               \
            VAR->SetPersistentMemory();                                 \
            gFHCache.insert(std::pair<int,FHWorkItemBase*>(cacheIndex, VAR));       \
        }                                                               \
    }                                                                   \
}

#if 0
/*  I had hoped to use FFTW wisdom to speed things up on the first run.
    However, there seems to be a problem where it refuses to load a previous wisdom file.
    That seems to only be a problem when running under python (weird...).
    I have just had to give up on this one - without delving into the fftw source code
    I have no idea what to do.  */
class LocalUseFFTWisdom
{
    const char *filename;
  public:
    LocalUseFFTWisdom(const char *_filename = NULL)
    {
        if (_filename == NULL)
            _filename = "/Users/jonny/light-field-flow.txt";
        filename = _filename;
        int res = fftwf_import_wisdom_from_filename(filename);
        if (res == 0)
            printf("No wisdom found - file will be created once we have done the FFT planning\n");
        else
            printf("Successfully imported wisdom\n");
    }
    ~LocalUseFFTWisdom()
    {
        int res = fftwf_export_wisdom_to_filename(filename);
        CHECK(res != 0);
        printf("Saved to %s\n", filename);
    }
};
#endif

extern "C" PyObject *ProjectForZList(PyObject *self, PyObject *args)
{
    try
    {
        PyObject    *workList;
        const char  *cacheIdentifier;
        int         numTimepoints;
        bool        useCache = false;
        if (!PyArg_ParseTuple(args, "O!zi", &PyList_Type, &workList, &cacheIdentifier, &numTimepoints))
        {
            return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
        }

        if (cacheIdentifier != NULL)
        {
            if (gCacheIdentifier == NULL)
            {
                PyErr_Format(PyErr_NewException((char*)"exceptions.ValueError", NULL, NULL), "Call EnableFHCachingWithIdentifier first, with this new identifier.");
                return NULL;
            }
            if (strcmp(cacheIdentifier, gCacheIdentifier))
            {
                PyErr_Format(PyErr_NewException((char*)"exceptions.ValueError", NULL, NULL), "Cache identifier does not match the one previously set. Call EnableFHCachingWithIdentifier for the new identifier if you want to reset the cache.");
                return NULL;
            }
            useCache = true;
        }
        long numZPlanes = PyList_Size(workList);
        
        std::vector<WorkItem *> work[kNumWorkTypes];
        std::vector<JMutex*> allMutexes;
        std::vector<fftwf_plan> allPlans;
        PyObject *resultList = PyList_New(numZPlanes);
        for (int cc = 0; cc < numZPlanes; cc++)
        {
            PyObject *planeInfo = PyList_GetItem(workList, cc);
            PyArrayObject *_projection, *_HtsFull, *_xAxisMultipliers, *_yAxisMultipliers;
            int Nnum, backwards, fshapeY, fshapeX, rfshapeY, rfshapeX;
            if (!PyArg_ParseTuple(planeInfo, "O!O!iiiiiiO!O!",
                                  &PyArray_Type, &_projection,
                                  &PyArray_Type, &_HtsFull,
                                  &Nnum, &backwards, &fshapeY, &fshapeX, &rfshapeY, &rfshapeX,
                                  &PyArray_Type, &_xAxisMultipliers,
                                  &PyArray_Type, &_yAxisMultipliers))
            {
                return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
            }
            if ((useCache) && (gFHCache.size() > 0))
            {
                /*  Check that the cached FFT arrays are compatible with what we are going to be working with.
                    We use gCacheIdentifier to track whether the *contents* of the PSF have changed, but even if the
                    PSF is unchanged then the *dimensions* of the FFT array will be different if the *camera* arrays are different.
                    Here is the only place it is convenient to check that (for each z plane)
                 */
                for (auto iter = gFHCache.begin(); iter != gFHCache.end(); ++iter)
                {
                    if ((iter->second->fftResult != NULL) && (iter->second->cc == cc) && (iter->second->backwards == backwards) &&
                        ((iter->second->fftResult->Dims(0) != fshapeY) || (iter->second->fftResult->Dims(1) != fshapeX)))
                    {
                        PyErr_Format(PyErr_NewException((char*)"exceptions.ValueError", NULL, NULL),
                                     "FFT cache appears to be stale and needs clearing. Fourier space dimensions (%dx%d) for plane %d do not match cached dimensions(%dx%d)",
                                     fshapeY, fshapeX, cc, iter->second->fftResult->Dims(0), iter->second->fftResult->Dims(1));
                        return NULL;
                    }
                }
            }
            
            JPythonArray3D<RTYPE> projection(_projection);
            JPythonArray4D<RTYPE> HtsFull(_HtsFull);
            JPythonArray2D<TYPE> xAxisMultipliers(_xAxisMultipliers);
            JPythonArray3D<TYPE> yAxisMultipliers(_yAxisMultipliers);
            
            if (!(projection.FinalDimensionUnitStride() && HtsFull.FinalDimensionUnitStride() && xAxisMultipliers.FinalDimensionUnitStride() && yAxisMultipliers.FinalDimensionUnitStride()))
            {
                PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Input arrays must have unit stride in the final dimension");
                return NULL;
            }
        
            npy_intp output_dims[3] = { projection.Dims(0), rfshapeY, rfshapeX };
            PyArrayObject *_accum = (PyArrayObject *)PyArray_ZEROS(3, output_dims, NPY_CFLOAT, 0);
            JPythonArray3D<TYPE> accum(_accum);
            PyList_SetItem(resultList, cc, (PyObject *)_accum);  // Steals reference
            TrackAllocation(&accum);

#if 0
            PySys_WriteStdout("Set up work items for z=%d\n", cc);
            PySys_WriteStdout("fshape=%d,%d. rfshape=%d,%d\n", fshapeY, fshapeX, rfshapeY, rfshapeX);
            PySys_WriteStdout("Memory usage: %.3lfGB\n", gMemoryUsage/1e9);
#endif
            
            // Set up the work items describing the complete projection operation for this z plane
            fftwf_plan plan = NULL;
            std::vector<JMutex*> accumMutex(projection.Dims(0));
            for (size_t i = 0; i < accumMutex.size(); i++)
                accumMutex[i] = new JMutex;
            allMutexes.insert(allMutexes.end(), accumMutex.begin(), accumMutex.end());
            int fhCounter = 0, fhtCounter = 0, mCounter = 0, cCounter = 0;
            for (int bb = 0; bb < HtsFull.Dims(0); bb++)
            {
                for (int aa = bb; aa < int(Nnum+1)/2; aa++)
                {
                    int cent = int(Nnum/2);
                    bool mirrorX = (bb != cent);
                    bool mirrorY = (aa != cent);
                    bool transpose = ((aa != bb) && (aa != (Nnum-bb-1)));
                    
                    FHWorkItemBase *f1;
                    CACHE_OR_CREATE(backwards, cc, bb, aa, kWorkFFT, f1, FHWorkItem, (HtsFull[bb][aa], fshapeY, fshapeX, backwards, false, cc, fhCounter++))
                    
                    FHWorkItemBase *f2 = NULL;
                    if (transpose)
                    {
                        if (fshapeY == fshapeX)
                        {
                            // We do not currently support the transpose here, although that can speed things up in certain circumstances (see python code in projector.convolve()).
                            // The only scenario where we could gain (by avoiding recalculating the FFT) is if the image array is square.
                            CACHE_OR_CREATE(backwards, cc, aa, bb, kWorkTranspose, f2, TransposeWorkItem, (f1, backwards, cc, fhtCounter++))
                        }
                        else
                        {
                            // There is no easy way to calculate FFT(h^T) when the padded array is non-square.
                            // (If there was, then I think that in general FFTs of padded arrays could be computed very fast!)
                            CACHE_OR_CREATE(backwards, cc, aa, bb, kWorkFFT, f2, FHWorkItem, (HtsFull[bb][aa], fshapeY, fshapeX, backwards, true, cc, fhCounter++))
                        }
                    }
                    
                    if (plan == NULL)
                        plan = ConvolveWorkItem::GetFFTPlan(f1, Nnum);
                    
                    /*  It is not totally obvious whether I should increment cCounter and mCounter just at the end of all this, or after each chunk.
                        Since its purpose is to separate out work that is contending for the same mutex, I think I should be doing it after each chunk.
                        But note that there is only any need for any of that re-sorting when the number of timepoints is low - so possibly
                        what I should be doing is just not sorting things at all, in the case where the number of timepoints is significantly larger than
                        the number of threads...?   
                        For my 30-timepoint benchmark, there is no consistent change to performance in the 4-threaded case if I disabled the sorting
                        This suggests it doesn't seem to have an impact on the cases where there is no lock contention in the first place. 
                        Good to know it doesn't have a big detrimental impact.
                     */
                    ConvolvePart2(projection, bb, aa, Nnum, mirrorY, mirrorX, f1, xAxisMultipliers, yAxisMultipliers, accum, accumMutex, plan, work, backwards, cc, mCounter, cCounter);
                    if (transpose)
                    {
                        // Note that my,mx (and bb,aa) have been swapped here, which is necessary following the transpose.
                        ConvolvePart2(projection, aa, bb, Nnum, mirrorX, mirrorY, f2, xAxisMultipliers, yAxisMultipliers, accum, accumMutex, plan, work, backwards, cc, mCounter, cCounter);
                    }
                }
            }
            allPlans.push_back(plan);
        }
        //PySys_WriteStdout("Sorting. Memory usage: %.3lfGB\n", gMemoryUsage/1e9);

        /*  Some sort of sorting is necessary to avoid mutex contention.
            However, sorting in this manner slows down a single-timepoint reconstruction by about 10%.
            I presume this is due to poor cache usage, or the need to allocate more blocks of memory simultaneously.
            TODO: I should investigate a better approach (which might include giving individual threads larger chunks
                  of work to do, so an individual thread does less jumping around)  */
        // Calculate how many z coordinates we need to batch up in order to avoid mutex contention
        // when processing the work items in the standard order to minimise memory usage
        gMutexContentionFactor = 1 + int(gNumThreadsToUse / numTimepoints);
        //PySys_WriteStdout("Z block factor %d (%d, %d)\n", gMutexContentionFactor, gNumThreadsToUse, numTimepoints);
        for (int w = 0; w < kNumWorkTypes; w++)
            std::stable_sort(work[w].begin(), work[w].end(), WorkItem::Compare);
        
#if 1
        /*  Generate a report on what we *think* mutex contention will be like.
            For each work item, look at how far back we have to go to find a work item that uses the same mutex.
            If that distance is consistently larger than the number of cores, we shouldn't have issues with mutex contention
        */
        FILE *mutexReport = fopen("mutex_report_static.txt", "w");
        ALWAYS_ASSERT(mutexReport != NULL);
        for (size_t i = 0; i < work[kWorkConvolve].size(); i++)
        {
            const JMutex *thisMutex = ((ConvolveWorkItem*)work[kWorkConvolve][i])->AccumMutex();
            int distance = 0;
            int j;
            for (j = i-1; j >= 0; j--)
            {
                distance++;
                if (thisMutex == ((ConvolveWorkItem*)work[kWorkConvolve][j])->AccumMutex())
                    break;
            }
            if (j >= 0)
                fprintf(mutexReport, "%zd\t%d\t%p\n", i, distance, thisMutex);
            else
                fprintf(mutexReport, "%zd\t%d\n", i, 0);
        }
        fclose(mutexReport);
#endif
        
        // Do the actual hard work (parallelised)
        //PySys_WriteStdout("Running. Memory usage: %.3lfGB\n", gMemoryUsage/1e9);
        RunWork(work);
        CleanUpWork(work, gThreadFileName, "w", useCache);
        for (size_t i = 0; i < allMutexes.size(); i++)
            delete allMutexes[i];
        for (size_t i = 0; i < allPlans.size(); i++)
            fftwf_destroy_plan(allPlans[i]);
        
        // At this point we are tracking memory usage of the result variables.
        // They belong to the caller after this, so we need to stop tracking
        // their memory usage ourselves. Otherwise our memory usage will grow
        // (misleadingly) as we are called multiple times.
        for (long i = 0; i < PyList_Size(resultList); i++)
        {
            PyArrayObject *_accum = (PyArrayObject *)PyList_GetItem(resultList, i);
            JPythonArray3D<TYPE> accum(_accum);
            TrackAllocation(&accum, true);
        }

        return resultList;
    }
    catch (const std::invalid_argument& e)
    {
        // Presumably an error with python arrays not matching expectations.
        // The python exception will have already been set, so we just have to return NULL.
        // At the moment, if this happens we will leak some memory, but we shouldn't be leaking huge amounts
        // because exceptions will occur before we actually start doing the computational work.
        return NULL;
    }
}

extern "C" PyObject *ProjectForZ(PyObject *self, PyObject *args)
{
    // Convenience function to project for a single z plane.
    // However, ProjectForZList should be used for optimal parallelism
    PyObject *planeList = PyList_New(1);
    Py_INCREF(args);                            // So that the reference can be stolen!
    PyList_SetItem(planeList, 0, args);         // Steals reference
    
    PyObject *argTuple = PyTuple_New(2);
    PyTuple_SetItem(argTuple, 0, planeList);         // Steals reference
    PyTuple_SetItem(argTuple, 1, Py_None);
    PyObject *resultList = ProjectForZList(self, argTuple);
    PyObject_Free(argTuple);
    if (resultList == NULL)
    {
        // An error occurred, presumably when parsing the planeList.
        // The error will be a bit confusing (since it refers to the parameters taken by "the function"),
        // but I'll return it and the user will hopefully be able to figure it out!
        return NULL;
    }
    PyObject *result = PyList_GetItem(resultList, 0);
    Py_INCREF(result);                          // So that we can return it
    PyObject_Free(resultList);
    return result;
}

extern "C" PyObject *InverseRFFTList(PyObject *self, PyObject *args)
{
    try
    {
        PyObject *workList;
        if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &workList))
            return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
        long numZPlanes = PyList_Size(workList);
        PyObject *resultList = PyList_New(numZPlanes);
        
        std::vector<WorkItem *> work[kNumWorkTypes];
        std::vector<JMutex*> allMutexes;
        std::vector<fftwf_plan> allPlans;
        for (int cc = 0; cc < numZPlanes; cc++)
        {
            PyObject *planeInfo = PyList_GetItem(workList, cc);
            PyArrayObject *_fab;
            int fshapeY, fshapeX;
            if (!PyArg_ParseTuple(planeInfo, "O!ii",
                                  &PyArray_Type, &_fab,
                                  &fshapeY, &fshapeX))
            {
                return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
            }
            
            JPythonArray3D<TYPE> fab(_fab);
            
            npy_intp output_dims[3] = { fab.Dims(0), fshapeY, fshapeX };
            PyArrayObject *_result = (PyArrayObject *)PyArray_EMPTY(3, output_dims, NPY_FLOAT, 0);
            PyList_SetItem(resultList, cc, (PyObject *)_result);  // Steals reference
            JPythonArray3D<RTYPE> result(_result);
            ALWAYS_ASSERT(!(((size_t)result.Data()) & 0xF));        // Check alignment. Just plain malloc seems to give sufficient alignment.
            
            int ifCounter = 0;
            for (int i = 0; i < fab.Dims(0); i++)
            {
                IFFTWorkItem *f = new IFFTWorkItem(fab[i], result[i], fshapeY, fshapeX, cc, ifCounter++);
                work[kWorkFFT].push_back(f);
            }
        }
        RunWork(work);
        CleanUpWork(work, gThreadFileName, "a", false);
        return resultList;
    }
    catch (const std::invalid_argument& e)
    {
        // Presumably an error with python arrays not matching expectations.
        // The python exception will have already been set, so we just have to return NULL.
        // At the moment, if this happens we will leak some memory, but we shouldn't be leaking huge amounts
        // because exceptions will occur before we actually start doing the computational work.
        return NULL;
    }
}

extern "C" PyObject *InverseRFFT(PyObject *self, PyObject *args)
{
    // This is old code that should be retired, or made to call through to InverseRFFTList
    PyArrayObject *_mat;
    int inputShapeY, inputShapeX;
    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyArray_Type, &_mat,
                          &inputShapeY, &inputShapeX))
    {
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    }

    JPythonArray2D<TYPE> mat(_mat);

    npy_intp paddedInputDims[2] = { inputShapeY, int(inputShapeX/2)+1 };
    JPythonArray2D<TYPE> paddedMatrix(NULL, paddedInputDims, NULL);
    
    npy_intp output_dims[2] = { inputShapeY, inputShapeX };
    PyArrayObject *_result = (PyArrayObject *)PyArray_EMPTY(2, output_dims, NPY_FLOAT, 0);
    JPythonArray2D<RTYPE> result(_result);
    
    // We do need to define the plan *before* the data is initialized, especially if using FFTW_MEASURE (which will overwrite the contents of the buffers)
    
    ALWAYS_ASSERT(!(((size_t)result.Data()) & 0xF));        // Check alignment. Just plain malloc seems to give sufficient alignment.
    
    fftwf_plan_with_nthreads(gNumThreadsToUse);

    int dims[2] = { result.Dims(0), result.Dims(1) };
    int inFullShape[2] = { paddedMatrix.Dims(0), paddedMatrix.Strides(0) };
    int outFullShape[2] = { result.Dims(0), result.Strides(0) };
    ALWAYS_ASSERT(mat.Strides(1) == 1);     // We assume contiguous
    ALWAYS_ASSERT(paddedMatrix.Strides(1) == 1);     // We assume contiguous
    ALWAYS_ASSERT(result.Strides(1) == 1);  // We assume contiguous
    fftwf_plan plan = fftwf_plan_many_dft_c2r(2/*2D FFT*/, dims, 1/*howmany*/,
                                     (fftwf_complex *)paddedMatrix.Data(), inFullShape,
                                     1/*stride*/, 0/*unused*/,
                                     result.Data(), outFullShape,
                                     1/*stride*/, 0/*unused*/,
                                     gFFTPlanMethod);
    fftwf_plan_with_nthreads(1);

    result.SetZero();
    // TODO: need to confirm whether I use result.Dims() or the dims of something else, if I consider a case with extra padding.
    // The answer seems to be result.Dims since my code works like this, but I should check for sure and write a definitive comment here.
    float inverseTotalSize = 1.0f / (float(result.Dims(0)) * float(result.Dims(1)));
    ALWAYS_ASSERT(paddedMatrix.Dims(0) >= mat.Dims(0)); // This assertion is because we don't support cropping the input matrix, only padding
    ALWAYS_ASSERT(paddedMatrix.Dims(1) >= mat.Dims(1));
    paddedMatrix.SetZero();     // Inefficient, but I will do this for now. TODO: update this code to only zero out the (small number of) values we won't be overwriting in the loop
    for (int y = 0; y < mat.Dims(0); y++)
    {
        JPythonArray1D<TYPE> _mat = mat[y];
        JPythonArray1D<TYPE> _paddedMatrix = paddedMatrix[y];
        for (int x = 0; x < mat.Dims(1); x++)
            _paddedMatrix[x] = _mat[x] * inverseTotalSize;
    }
    
    // Compute the full 2D FFT (i.e. not just the RFFT)
    fftwf_execute_dft_c2r(plan, (fftwf_complex *)paddedMatrix.Data(), result.Data());

    fftwf_destroy_plan(plan);
    return PyArray_Return(_result);
}

extern "C" PyObject *MemoryBenchmark(PyObject *self, PyObject *args)
{
    int numElements, numThreads, benchmarkType;
    if (!PyArg_ParseTuple(args, "ii", &numThreads, &numElements))
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    
    // Time how long it takes to churn through a simple memory-access benchmark task
    std::vector<WorkItem *> work[kNumWorkTypes];
    for (int i = 0; i < numThreads; i++)
    {
        BenchmarkWorkItem *b = new BenchmarkWorkItem(numElements, 0);
        work[kWorkFFT].push_back(b);
    }
    double times[kNumBenchmarkTypes];
    int temp = gNumThreadsToUse;
    gNumThreadsToUse = numThreads;
    for (benchmarkType = 0; benchmarkType < kNumBenchmarkTypes; benchmarkType++)
    {
        for (int i = 0; i < numThreads; i++)
        {
            BenchmarkWorkItem *b = ((BenchmarkWorkItem*)work[kWorkFFT][i]);
            b->benchmarkType = benchmarkType;
            b->Reset();
        }
        double t0 = GetTime();
        RunWork(work);
        double t1 = GetTime();
        times[benchmarkType] = t1-t0;
        printf("%d(x%d) took %.6lf\n", benchmarkType, numThreads, t1-t0);
    }
    gNumThreadsToUse = temp;
    CleanUpWork(work, NULL, NULL, false);
    
    ALWAYS_ASSERT(kNumBenchmarkTypes == 8); // As a reminder to update the return value if this changes
    return Py_BuildValue("(dddddddd)", times[0], times[1], times[2], times[3], times[4], times[5], times[6], times[7]);
}

// This mess here is to ensure I can compile under both python 2 and python 3
#if PY_MAJOR_VERSION >= 3
const char *MyImportNumpy2(void)
{
    /*  In the case of an error, the definition of import_array() returns (and in python 3 it actually returns NULL.
     I don't want it to force anything to return, I just want to catch that situation and assert.
     As a result, I have to write this wrapper function, and the caller can assert that the return value is non-NULL. */
    import_array()
    return "ok";        // Return something that is not NULL
}

void MyImportNumpy(void)
{
    const char *importResult = MyImportNumpy2();
    ALWAYS_ASSERT(importResult != NULL);
}
#else
void MyImportNumpy(void)
{
    import_array();
}
#endif

void *TestMe(void)
{
    /*  Function to be called as part of a standalone C program,
        when I want to test, and more importantly profile, the C code. 
        I haven't worked out how to get Instruments to recognise my symbols when running as an actual module loaded into python,
        so profiling of this C code is only really possible when running as a standalong C program (processing dummy data).
     */
     
    // For reasons I don't yet understand, I seem to have to do this python initialization in the same source file where I am using the numpy arrays.
    // As a result, I cannot do this setup from my test function main.cpp, and have to embed it in this module here...
    const char *anacondaFolder = "/Users/jonny/opt/anaconda3";
    // Remember to set up LD_LIBRARY_PATH under Scheme/Environment variables, when running under Xcode.
    
#if PY_MAJOR_VERSION < 3
    Py_SetPythonHome((char *)anacondaFolder);
#else
    wchar_t *tempString = Py_DecodeLocale(anacondaFolder, NULL);
    Py_SetPythonHome(tempString);
    // Do not free tempString, because the docs say that the memory must remain valid for the whole time the program is running.
    // As a result, I effectively just leak the memory intentionally!
    //    *do not call * PyMem_RawFree(tempString);
#endif
    
    unsetenv("PATH");
    setenv("PATH", anacondaFolder, 1);
    unsetenv("PYTHONPATH");
    setenv("PYTHONPATH", anacondaFolder, 1);
    Py_Initialize();
    MyImportNumpy();
    
    fftwf_init_threads();
    
    for (int n = 0; n < 2; n++)
    {
        PyObject *pArgs = PyTuple_New(10);
        ALWAYS_ASSERT(pArgs != NULL);
        
    #if 1
        // Largest use-case:
        //    (1, 450, 675) (8, 8, 391, 391) 840 1065 840 533 (16, 1065) (2, 15, 840)
        npy_intp hdims[4] = { 8, 8, 391, 391 };
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(15));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(1));
        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(840));
        PyTuple_SetItem(pArgs, 5, PyLong_FromLong(1065));
        PyTuple_SetItem(pArgs, 6, PyLong_FromLong(840));
        PyTuple_SetItem(pArgs, 7, PyLong_FromLong(533));
        npy_intp xdims[2] = { 16, 1065 };
        npy_intp ydims[3] = { 2, 15, 840 };
    #else
        // Smallest use-case:
        //    (1, 450, 675) (8, 8, 61, 61) 510 735 510 368 (16, 735) (2, 15, 510)
        npy_intp hdims[4] = { 8, 8, 61, 61 };
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(15));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(1));
        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(510));
        PyTuple_SetItem(pArgs, 5, PyLong_FromLong(735));
        PyTuple_SetItem(pArgs, 6, PyLong_FromLong(510));
        PyTuple_SetItem(pArgs, 7, PyLong_FromLong(368));
        npy_intp xdims[2] = { 16, 735 };
        npy_intp ydims[3] = { 2, 15, 510 };
    #endif

//        npy_intp pdims[3] = { 1, 450, 675 };
        npy_intp pdims[3] = { 30, 450, 675 };
        PyObject *projection = PyArray_ZEROS(3, pdims, NPY_FLOAT, 0);
        PyObject *HtsFull = PyArray_ZEROS(4, hdims, NPY_FLOAT, 0);
        PyObject *xAxisMultipliers = PyArray_ZEROS(2, xdims, NPY_CFLOAT, 0);
        PyObject *yAxisMultipliers = PyArray_ZEROS(3, ydims, NPY_CFLOAT, 0);
        PyTuple_SetItem(pArgs, 0, projection);
        PyTuple_SetItem(pArgs, 1, HtsFull);
        PyTuple_SetItem(pArgs, 8, xAxisMultipliers);
        PyTuple_SetItem(pArgs, 9, yAxisMultipliers);
        
        PyObject *result = ProjectForZ(NULL, pArgs);
        
        if (result != NULL)
            Py_DECREF(result);
        else
            PyErr_Print();
        
        Py_DECREF(pArgs);       // Should also release the arrays (and other objects) that I added to the tuple
    }

    printf("Done\n");
    return NULL;
}


/* Define a methods table for the module */

static PyMethodDef plf_methods[] = {
    {"ProjectForZ", ProjectForZ, METH_VARARGS},
    {"ProjectForZList", ProjectForZList, METH_VARARGS},
    {"InverseRFFT", InverseRFFT, METH_VARARGS},
    {"InverseRFFTList", InverseRFFTList, METH_VARARGS},
    {"SetStatsFile", SetStatsFile, METH_VARARGS},
    {"SetThreadFileName", SetThreadFileName, METH_VARARGS},
    {"GetNumThreadsToUse", GetNumThreadsToUse, METH_NOARGS},
    {"SetNumThreadsToUse", SetNumThreadsToUse, METH_VARARGS},
    {"GetPlanningMode", GetPlanningMode, METH_NOARGS},
    {"SetPlanningMode", SetPlanningMode, METH_VARARGS},
    {"EnableFHCachingWithIdentifier", EnableFHCachingWithIdentifier, METH_VARARGS},
    {"DisableFHCaching", DisableFHCaching, METH_VARARGS},
    {"ClearFHCache", ClearFHCache, METH_NOARGS},
    {"MemoryBenchmark", MemoryBenchmark, METH_VARARGS},
	{NULL,NULL} };

/* initialisation - register the methods with the Python interpreter */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef py_light_field =
{
    PyModuleDef_HEAD_INIT,
    "py_light_field", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    plf_methods
};

PyMODINIT_FUNC PyInit_py_light_field(void)
{
    import_array();
    return PyModule_Create(&py_light_field);
}

#else

extern "C" void initpy_light_field(void)
{
    (void) Py_InitModule("py_light_field", plf_methods);
    import_array();
}

#endif
