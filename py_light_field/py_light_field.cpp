#include <unistd.h>
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
const int kMaxThreads = 128;
int gNumThreadsToUse = NumActualProcessorsAvailable();        // But can be modified using API call

template<class TYPE> void SetImageWindowForPythonWindow(ImageWindow<TYPE> &imageWindow, JPythonArray2D<TYPE> &pythonWindow)
{
    imageWindow.baseAddr = pythonWindow.Data();
    imageWindow.width = pythonWindow.Dims(1);
    imageWindow.height = pythonWindow.Dims(0);
    imageWindow.elementsPerRow = pythonWindow.Strides(0);
}

typedef complex<float> TYPE;
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

double GetTime(void)
{
    // Standard BSD function
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void PauseFor(double secs)
{
    // Block for the specified number of seconds before returning
    struct timeval timeout;
    timeout.tv_sec = (int)secs;
    timeout.tv_usec = (int)((secs - (int)secs) * 1000000);
    select(0, NULL, NULL, NULL, &timeout);
}

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

extern "C" PyObject *GetNumThreadsInUse(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", gNumThreadsToUse);
}

extern "C" PyObject *SetNumThreadsInUse(PyObject *self, PyObject *args)
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

double gFFTWPlanTime = 0, gFFTWInitializeTime1 = 0, gFFTWInitializeTime2 = 0, gFFTExecuteTime = 0, gFFTWTime2 = 0, gTotalHTime = 0, gSpecialTime = 0, gMirrorTime = 0;

extern "C" PyObject *PrintStats(PyObject *self, PyObject *args)
{
    double total = gTotalHTime + gSpecialTime + gMirrorTime;
    printf("Times: Total H %.2lf (plan %.2lf, initialize %.2lf+%.2lf, execute %.2lf). Special %.2lf (of which FFT %.2lf). Mirror %.2lf\n", gTotalHTime, gFFTWPlanTime, gFFTWInitializeTime1, gFFTWInitializeTime2, gFFTExecuteTime, gSpecialTime, gFFTWTime2, gMirrorTime);
    printf("(plan %.1lf%%, initialize %.1lf%%, execute %.1lf%%)\n", gFFTWPlanTime/gTotalHTime*100, (gFFTWInitializeTime1+gFFTWInitializeTime2)/gTotalHTime*100, gFFTExecuteTime/gTotalHTime*100);
    printf("(H %.1lf%%, special %.1lf%%)\n", gTotalHTime/total*100, gSpecialTime/total*100);
    Py_RETURN_NONE;
}

extern "C" PyObject *ResetStats(PyObject *self, PyObject *args)
{
    gFFTWPlanTime = 0;
    gFFTWInitializeTime1 = 0;
    gFFTWInitializeTime2 = 0;
    gFFTExecuteTime = 0;
    gFFTWTime2 = 0;
    gSpecialTime = 0;
    gTotalHTime = 0;
    gMirrorTime = 0;
    Py_RETURN_NONE;
}

void MirrorYArray(JPythonArray2D<TYPE> &fHtsFull, JPythonArray1D<TYPE> &mirrorYMultiplier, JPythonArray2D<TYPE> &result)
{
    // Given F(H), return F(mirrorY(H)), where mirrorY(H) is the vertical reflection of H.
    int height = fHtsFull.Dims(0);
    int width = fHtsFull.Dims(1);
    for (int x = 0; x < width; x++)
        result[0][x] = conj(fHtsFull[0][x]) * mirrorYMultiplier[x];
    for (int y = 1; y < height; y++)
    {
        // Note that I tried writing hand-optimized SSE code and the longhand C code was faster.
        // That was surprising, but the C code seems to be being auto-vectorized rather well already.
        // Oh well, that saves me faffing around doing it myself!
        JPythonArray1D<TYPE> _result = result[y];
        JPythonArray1D<TYPE> _fHtsFull = fHtsFull[height-y];
        for (int x = 0; x < width; x++)
            _result[x] = conj(_fHtsFull[x]) * mirrorYMultiplier[x];
    }
}

extern "C" PyObject *mirrorXY(PyObject *self, PyObject *args, bool x)
{
    // inputs
    PyArrayObject *a, *_multiplier;
    
    // parse the input arrays from *args
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &a,
                          &PyArray_Type, &_multiplier))
    {
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    }
    if ((PyArray_TYPE(a) != NPY_CFLOAT) ||
        (PyArray_TYPE(_multiplier) != NPY_CFLOAT))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unsuitable array types %d %d passed in", (int)PyArray_TYPE(a), (int)PyArray_TYPE(_multiplier));
        return NULL;
    }
    
    JPythonArray2D<TYPE> aa(a);
    ImageWindow<TYPE> fHtsFull;
    SetImageWindowForPythonWindow(fHtsFull, aa);
    
    JPythonArray1D<TYPE> multiplier(_multiplier);
    
    int height = aa.Dims(0);
    int width = aa.Dims(1);
    PyArrayObject *r;
    if (x)
    {
        // result = np.empty(fHtsFull.shape, dtype=fHtsFull.dtype)
        npy_intp output_dims[2] = { aa.Dims(0), aa.Dims(1) };
        r = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_CFLOAT);
        JPythonArray2D<TYPE> rr(r);
        ImageWindow<TYPE> result;
        SetImageWindowForPythonWindow(result, rr);
        
        // result[:,0] = fHtsFull[:,0].conj()*temp
        for (int y = 0; y < height; y++)
            result.SetXY(0, y, conj(fHtsFull.PixelXY(0,y)) * multiplier[y]);
        // for i in range(1,fHtsFull.shape[1]):
        //     result[:,i] = (fHtsFull[:,fHtsFull.shape[1]-i].conj()*temp)
        for (int y = 0; y < height; y++)
        {
            int x = 1;
#if 0
            // This would be where I would write my own vectorized code.
            // However, the experience with the y case (below) is that actually the auto-vectorized code
            // is better than anything I can write by hand myself! (Which is no bad thing - saves me the effort!)
            for (; x+2 <= width; x++) {}
#endif
            
            for (; x < width; x++)
                result.SetXY(x, y, conj(fHtsFull.PixelXY(width-x,y)) * multiplier[y]);
        }
    }
    else
    {
        r = (PyArrayObject *)PyArray_SimpleNew(2, aa.Dims(), NPY_CFLOAT);
        JPythonArray2D<TYPE> result(r);
        MirrorYArray(aa, multiplier, result);
    }
    return PyArray_Return(r);
}

extern "C" PyObject *mirrorX(PyObject *self, PyObject *args)
{
    return mirrorXY(self, args, true);
}

extern "C" PyObject *mirrorY(PyObject *self, PyObject *args)
{
    return mirrorXY(self, args, false);
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
    ac[0] += fap[0] * conj(fbu[0]) * emy;
    for (int x = 1; x < lim; x++)
    {
        ac[x] += fap[x] * conj(fbu[trueLength-x]) * emy;
    }
}

void CalculateRowBoth(TYPE *ac, const TYPE *fap1, const TYPE *fap2, const TYPE *fbu, const TYPE emy1, const TYPE emy2, int lim, int trueLength)
{
    ac[0] += fap1[0] * fbu[0] * emy1;
    ac[0] += fap2[0] * conj(fbu[0]) * emy2;
    for (int x = 1; x < lim; x++)
    {
        ac[x] += fap1[x] * fbu[x] * emy1;
        ac[x] += fap2[x] * conj(fbu[trueLength-x]) * emy2;
    }
}

class WorkItem
{
private:  // To ensure callers go through our accessor functions
    bool        complete;
    WorkItem    *dependency;
    int         dependencyCount;
public:
    int         cc, order;
    double      runStartTime, runEndTime, mutexWaitStartTime[2], mutexWaitEndTime[2];
    int         ranOnThread;
    
    WorkItem(int _cc, int _order) : complete(false), dependency(NULL), dependencyCount(0), cc(_cc), order(_order)
    {
        mutexWaitStartTime[0] = mutexWaitEndTime[0] = 0;
        mutexWaitStartTime[1] = mutexWaitEndTime[1] = 0;
    }
    virtual ~WorkItem() { ALWAYS_ASSERT(dependencyCount == 0); }
    virtual void Run(void) = 0;
    virtual void CleanUpAllocations(void) = 0;
    void RunComplete(void)
    {
        // TODO: care is needed here about thread safety, but for now I think I am ok to do this without a mutex, since I am just polling
        // in the case where a thread is blocked waiting on another piece of work.
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
        if (newVal == 0)
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
    static int Compare(WorkItem *a, WorkItem *b)        // Note that the pointers are because the std::vector itself is a vector of pointers
    {
        return a->order < b->order;
    }
};

class FHWorkItemBase : public WorkItem
{
public:
    int                     fshapeY, fshapeX;
    JPythonArray2D<TYPE>    *fftResult;
    
    FHWorkItemBase(int _fshapeY, int _fshapeX, int _cc, int _order) : WorkItem(_cc, _order), fshapeY(_fshapeY), fshapeX(_fshapeX), fftResult(NULL)
    {
    }
    
    void AllocateResultArray(void)
    {
        // Allocate an array to hold the results
        npy_intp dims[2] = { fshapeY, fshapeX };
        npy_intp strides[2] = { int((fshapeX + 15)/16)*16, 1 };     // Ensure 16-byte alignment of each row, which seems to make things *slightly* faster
        fftResult = new JPythonArray2D<TYPE>(NULL, dims, strides);
        ALWAYS_ASSERT(!(((size_t)fftResult->Data()) & 0xF));        // Check base alignment. In fact, just plain malloc seems to give sufficient alignment.
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
        delete fftResult;
        fftResult = NULL;
    }
};

class FHWorkItem : public FHWorkItemBase
{
public:
    JPythonArray2D<RTYPE>   Hts;
    bool                    transpose;
    fftwf_plan              plan, plan2;
    
    
    FHWorkItem(JPythonArray2D<RTYPE> _Hts, int fshapeY, int fshapeX, bool _transpose, int _cc, int _order)
        : FHWorkItemBase(fshapeY, fshapeX, _cc, _order), Hts(_Hts), transpose(_transpose)
    {
        /*  Set up the FFT plan.
            The complication here is that we cannot afford to allocate memory for every 
            FFT array simultaneously, as we would exhause all the available memory.
            Because of this we temporarily allocate an array for FFTW planning purposes.
            I am imagining this will never be a bottleneck, but I will want to keep an eye on how long the setup time takes.  */
        fftwf_plan_with_nthreads(1);
        AllocateResultArray();      // Just temporarily, for FFT planning! We will delete it at the end of this function
        int nx[1] = { fftResult->Dims(1) };
        // Compute the horizontal 1D FFTs, for only the nonzero rows
        // Note that we manually split up the 2D FFT into horizontal and vertical 1D FFTs, to enable us to skip the rows that we know are all zeroes.
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
        delete fftResult; fftResult = NULL;
    }
    
    virtual ~FHWorkItem()
    {
        fftwf_destroy_plan(plan);
        fftwf_destroy_plan(plan2);
    }
    
    void Run(void)
    {
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
    
};

class MirrorWorkItem : public FHWorkItemBase
{
public:
    FHWorkItem            *sourceFFTWorkItem;
    JPythonArray1D<TYPE>  mirrorYMultiplier;
    
    MirrorWorkItem(FHWorkItem *_sourceFFTWorkItem, JPythonArray1D<TYPE> _mirrorYMultiplier, int _cc, int _order)
        : FHWorkItemBase(_sourceFFTWorkItem->fshapeY, _sourceFFTWorkItem->fshapeX, _cc, _order), sourceFFTWorkItem(_sourceFFTWorkItem), mirrorYMultiplier(_mirrorYMultiplier)
    {
        AddDependency(sourceFFTWorkItem);
    }
    void Run(void)
    {
        AllocateResultArray();
        MirrorYArray(*sourceFFTWorkItem->fftResult, mirrorYMultiplier, *fftResult);
        RunComplete();
    }
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
};

enum
{
    kWorkFFT = 0,
    kWorkMirrorY,
    kWorkConvolve,
    kNumWorkTypes
};

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
        while (1)
        {
            WorkItem *workItem = NULL;
            // Pick a work item to run.
            double t1 = GetTime();
            bool polled = false;
            {
            repeat:
                LocalGetMutex lgm(workQueueMutex, workQueueMutexBlock_us);
                // Run anything that is not blocked, prioritising the convolution work
                for (int w = kNumWorkTypes - 1; w >= 0; w--)
                {
                    if (workCounter[w] < work[w]->size())
                    {
                        workItem = (*work[w])[workCounter[w]];
                        if (workItem->CanRun())
                        {
                            // First item of work[w] is not blocked - we should run it
                            workCounter[w]++;
                            break;
                        }
                        else
                            workItem = NULL;
                    }
                }
                if (workItem == NULL)
                {
                    // All work types are either complete or are blocked.
                    // We should wait for something to unblock.
                    // We *MUST* prioritise the early work types (which in practice means the mirror),
                    // because otherwise we could end up deadlocked.
                    for (int w = 0; w < kNumWorkTypes; w++)
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
        ALWAYS_ASSERT(gNumThreadsToUse <= kMaxThreads);
        printf("Running semi-parallelised\n");
        for (int w = 0; w < kNumWorkTypes; w++)
        {
            printf("Run work (%d)\n", w);
            ThreadInfo threadInfo { 0, {0, 0, 0}, &workQueueMutex, &workQueueMutexBlock_us, &pollingTime, {&work[w], new std::vector<WorkItem *>(), new std::vector<WorkItem *>()} };     // 'new' leaks, but this is only temporary code anyway
            pthread_t threads[kMaxThreads];
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
        ALWAYS_ASSERT(gNumThreadsToUse <= kMaxThreads);
        ThreadInfo threadInfo { 0, {0, 0, 0}, &workQueueMutex, &workQueueMutexBlock_us, &pollingTime, {&work[0], &work[1], &work[2]} };
        pthread_t threads[kMaxThreads];
        for (int i = 0; i < gNumThreadsToUse; i++)
            pthread_create(&threads[i], NULL, ThreadFunc, &threadInfo);
        for (int i = 0; i < gNumThreadsToUse; i++)
            pthread_join(threads[i], NULL);
//        printf("%.1lfms spent waiting to acquire work queue mutex. %.1lfms spent polling.\n", workQueueMutexBlock_us/1e3, pollingTime*1e3);
    }
}

void ConvolvePart2(JPythonArray3D<RTYPE> projection, int bb, int aa, int Nnum, bool mirrorY, bool mirrorX, FHWorkItem *fftWorkItem, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray3D<TYPE> yAxisMultipliers, JPythonArray3D<TYPE> accum, std::vector<JMutex*> &accumMutex, fftwf_plan plan, std::vector<WorkItem *> work[kNumWorkTypes], int cc, int &mCounter, int &cCounter)
{
    // Note that this function does not actually do the work, it just sets up the WorkItems that will be run later.
    ALWAYS_ASSERT(projection.Dims(0) == accum.Dims(0));
    for (int i = 0; i < projection.Dims(0); i++)
    {
        ConvolveWorkItem *workConvolve = new ConvolveWorkItem(projection[i], bb, aa, Nnum, fftWorkItem, mirrorX, xAxisMultipliers, yAxisMultipliers, accum[i], accumMutex[i], plan, cc, cCounter);
        work[kWorkConvolve].push_back(workConvolve);
    }
    cCounter++;
    if (mirrorY)
    {
        MirrorWorkItem *workCalcMirror = new MirrorWorkItem(fftWorkItem, xAxisMultipliers[kXAxisMultiplierMirrorY], cc, mCounter);
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

extern "C" PyObject *ProjectForZList(PyObject *self, PyObject *args)
{
    // For now this is just a placeholder that calls through to ProjectForZ, but ultimately I intend to do all the work in one massive batch.
    // Doing that will help reduce lock contention when we only have a few timepoints to process.
    PyObject *workList;
#if 0//TESTING
    // PyArg_ParseTuple doesn't seem to work when I use it on my own synthesized tuple.
    // I don't know why that is, but this code exists as a workaround for that problem.
    // -> Actually, the parsing code seems to work now. I've disabled this, but left it in case the problem returns!
    workList = PyTuple_GetItem(args, 0);
#else
    if (!PyArg_ParseTuple(args, "O!",
                          &PyList_Type, &workList))
    {
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    }
#endif
    long numZPlanes = PyList_Size(workList);

    double t0 = GetTime();
    std::vector<WorkItem *> work[kNumWorkTypes];
    std::vector<JMutex*> allMutexes;
    std::vector<fftwf_plan> allPlans;
    PyObject *resultList = PyList_New(numZPlanes);
    for (int cc = 0; cc < numZPlanes; cc++)
    {
        PyObject *planeInfo = PyList_GetItem(workList, cc);
        PyArrayObject *_projection, *_HtsFull, *_xAxisMultipliers, *_yAxisMultipliers;
        int Nnum, fshapeY, fshapeX, rfshapeY, rfshapeX;
    #if 0//TESTING
        // PyArg_ParseTuple doesn't seem to work when I use it on my own synthesized tuple.
        // I don't know why that is, but this code exists as a workaround for that problem.
        // -> Actually, the parsing code seems to work now. I've disabled this, but left it in case the problem returns!
        _projection = (PyArrayObject *)PyTuple_GetItem(planeInfo, 0);
        _HtsFull = (PyArrayObject *)PyTuple_GetItem(planeInfo, 1);
        Nnum = (int)PyLong_AsLong(PyTuple_GetItem(planeInfo, 2));
        fshapeY = (int)PyLong_AsLong(PyTuple_GetItem(planeInfo, 3));
        fshapeX = (int)PyLong_AsLong(PyTuple_GetItem(planeInfo, 4));
        rfshapeY = (int)PyLong_AsLong(PyTuple_GetItem(planeInfo, 5));
        rfshapeX = (int)PyLong_AsLong(PyTuple_GetItem(planeInfo, 6));
        _xAxisMultipliers = (PyArrayObject *)PyTuple_GetItem(planeInfo, 7);
        _yAxisMultipliers = (PyArrayObject *)PyTuple_GetItem(planeInfo, 8);
    #else
        if (!PyArg_ParseTuple(planeInfo, "O!O!iiiiiO!O!",
                              &PyArray_Type, &_projection,
                              &PyArray_Type, &_HtsFull,
                              &Nnum, &fshapeY, &fshapeX, &rfshapeY, &rfshapeX,
                              &PyArray_Type, &_xAxisMultipliers,
                              &PyArray_Type, &_yAxisMultipliers))
        {
            return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
        }
    #endif
        
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

        // Set up the work items describing the complete projection operation for this z plane
        fftwf_plan plan = NULL;
        std::vector<JMutex*> accumMutex(projection.Dims(0));
        for (size_t i = 0; i < accumMutex.size(); i++)
            accumMutex[i] = new JMutex;
        allMutexes.insert(allMutexes.end(), accumMutex.begin(), accumMutex.end());
        int fhCounter = 0, mCounter = 0, cCounter = 0;
        for (int bb = 0; bb < HtsFull.Dims(0); bb++)
        {
            for (int aa = bb; aa < int(Nnum+1)/2; aa++)
            {
                int cent = int(Nnum/2);
                bool mirrorX = (bb != cent);
                bool mirrorY = (aa != cent);
                // We do not currently support the transpose here, although that can speed things up in certain circumstances (see python code in projector.convolve()).
                // The only scenario where we could gain (by avoiding recalculating the FFT) is if the image array is square.
                // At the moment, we process the case with the transpose, but simply by recalculating the appropriate FFT(H) from scratch
                bool transpose = ((aa != bb) && (aa != (Nnum-bb-1)));

                FHWorkItem *f1 = new FHWorkItem(HtsFull[bb][aa], fshapeY, fshapeX, false, cc, fhCounter++);
                work[kWorkFFT].push_back(f1);

                FHWorkItem *f2 = NULL;
                if (transpose)
                {
                    f2 = new FHWorkItem(HtsFull[bb][aa], fshapeY, fshapeX, true, cc, fhCounter++);
                    work[kWorkFFT].push_back(f2);
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
                ConvolvePart2(projection, bb, aa, Nnum, mirrorY, mirrorX, f1, xAxisMultipliers, yAxisMultipliers, accum, accumMutex, plan, work, cc, mCounter, cCounter);
                if (transpose)
                {
                    // Note that my,mx (and bb,aa) have been swapped here, which is necessary following the transpose.
                    ConvolvePart2(projection, aa, bb, Nnum, mirrorX, mirrorY, f2, xAxisMultipliers, yAxisMultipliers, accum, accumMutex, plan, work, cc, mCounter, cCounter);
                }
            }
        }
        allPlans.push_back(plan);
    }
    for (int w = 0; w < kNumWorkTypes; w++)
        std::stable_sort(work[w].begin(), work[w].end(), WorkItem::Compare);
    
    // Do the actual hard work (parallelised)
    TimeStruct before;
    double t1 = GetTime();
    RunWork(work);
    double t2 = GetTime();
    TimeStruct after;
    // Clean up work items
    // TODO: it is only at this point that I free up any memory at all! I will run out. I need to track dependencies and free memory once all dependencies have completed.
    FILE *threadFile = fopen("threads.txt", "w");
    for (int w = 0; w < kNumWorkTypes; w++)
        for (size_t i = 0; i < work[w].size(); i++)
        {
            fprintf(threadFile, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", work[w][i]->ranOnThread, w, work[w][i]->runStartTime, work[w][i]->runEndTime, work[w][i]->mutexWaitStartTime[0], work[w][i]->mutexWaitEndTime[0], work[w][i]->mutexWaitStartTime[1], work[w][i]->mutexWaitEndTime[1]);
            delete work[w][i];
        }
    fclose(threadFile);
    for (size_t i = 0; i < allMutexes.size(); i++)
        delete allMutexes[i];
    for (size_t i = 0; i < allPlans.size(); i++)
        fftwf_destroy_plan(allPlans[i]);
    double t3 = GetTime();
    double utime = TimeStruct::Secs(after._self.ru_utime)-TimeStruct::Secs(before._self.ru_utime);
    double stime = TimeStruct::Secs(after._self.ru_stime)-TimeStruct::Secs(before._self.ru_stime);
//    printf("ProjectForZ took %.3lf %.3lf %.3lf. User work %.3lf system %.3lf. Parallelism %.2lf\n", t1-t0, t2-t1, t3-t2, utime, stime, (utime+stime)/(t2-t1));
    
    return resultList;
}

extern "C" PyObject *ProjectForZ(PyObject *self, PyObject *args)
{
    // Convenience function to project for a single z plane.
    // However, ProjectForZList should be used for optimal parallelism
    PyObject *planeList = PyList_New(1);
    Py_INCREF(args);                            // So that the reference can be stolen!
    PyList_SetItem(planeList, 0, args);         // Steals reference
    PyObject *planeTuple = PyTuple_New(1);
    PyTuple_SetItem(planeTuple, 0, planeList);         // Steals reference
    PyObject *resultList = ProjectForZList(self, planeTuple);
    PyObject_Free(planeTuple);
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

extern "C" PyObject *InverseRFFT(PyObject *self, PyObject *args)
{
    PyArrayObject *_mat;
    int inputShapeY, inputShapeX;
    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyArray_Type, &_mat,
                          &inputShapeY, &inputShapeX))
    {
        return NULL;    // PyArg_ParseTuple already sets an appropriate PyErr
    }

    double t0 = GetTime();
    JPythonArray2D<TYPE> mat(_mat);

    npy_intp paddedInputDims[2] = { inputShapeY, int(inputShapeX/2)+1 };
    JPythonArray2D<TYPE> paddedMatrix(NULL, paddedInputDims, NULL);
    
    npy_intp output_dims[2] = { inputShapeY, inputShapeX };
    PyArrayObject *_result = (PyArrayObject *)PyArray_EMPTY(2, output_dims, NPY_FLOAT, 0);
    JPythonArray2D<RTYPE> result(_result);
    
    // We do need to define the plan *before* the data is initialized, especially if using FFTW_MEASURE (which will overwrite the contents of the buffers)
    
    ALWAYS_ASSERT(!(((size_t)result.Data()) & 0xF));        // Check alignment. Just plain malloc seems to give sufficient alignment.
    double t1 = GetTime();
    
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

    double t2 = GetTime();
    result.SetZero();
    double t2a = GetTime();
    // TODO: need to confirm whether I use result.Dims() or the dims of something else, if I consider a case with extra padding.
    // The answer seems to be result.Dims since my code works like this, but I should check for sure and write a definitive comment here.
    float inverseTotalSize = 1.0f / (float(result.Dims(0)) * float(result.Dims(1)));
    ALWAYS_ASSERT(paddedMatrix.Dims(0) >= mat.Dims(0)); // This assertion is because we don't support cropping the input matrix, only padding
    ALWAYS_ASSERT(paddedMatrix.Dims(1) >= mat.Dims(1));
    paddedMatrix.SetZero();     // Inefficient, but I will do this for now. TODO: update this code to only zero out the (small number of) values we won't be overwriting in the loop
    for (int y = 0; y < mat.Dims(0); y++)
    {
        auto _mat = mat[y];
        auto _paddedMatrix = paddedMatrix[y];
        for (int x = 0; x < mat.Dims(1); x++)
            _paddedMatrix[x] = _mat[x] * inverseTotalSize;
    }
    double t3 = GetTime();
    
    // Compute the full 2D FFT (i.e. not just the RFFT)
    fftwf_execute_dft_c2r(plan, (fftwf_complex *)paddedMatrix.Data(), result.Data());

    double t4 = GetTime();
    gFFTWPlanTime += t2-t1;
    gFFTWInitializeTime1 += t2a-t2;
    gFFTWInitializeTime2 += t3-t2a;
    gFFTExecuteTime += t4-t3;
    fftwf_destroy_plan(plan);
    double t5 = GetTime();
    
    //printf("iFFT times %.2lf plan %.2lf init {%.2lf, %.2lf} ex %.2lf %.2lf\n", (t1-t0)*1e6, (t2-t1)*1e6, (t2a-t2)*1e6, (t3-t2a)*1e6, (t4-t3)*1e6, (t5-t4)*1e6);
    
    return PyArray_Return(_result);
}

void *TestMe(void)
{
    /*  Function to be called as part of a standalone C program,
        when I want to test, and more importantly profile, the C code. 
        I haven't worked out how to get Instruments to recognise my symbols when running as an actual module loaded into python,
        so profiling of this C code is only really possible when running as a standalong C program (processing dummy data).
     */
     
    // For reasons I don't yet understand, I seem to have to do this python initialization in the same source file where I am using the numpy arrays.
    // As a result, I cannot do this setup from my test code, and have to embed it in this module here...
    const char *anacondaFolder = "/Users/jonny/anaconda";
    // Remember to set up LD_LIBRARY_PATH under Scheme/Environment variables, when running under Xcode.
    
    wchar_t *tempString = Py_DecodeLocale(anacondaFolder, NULL);
    Py_SetPythonHome(tempString);
    
    unsetenv("PATH");
    setenv("PATH", anacondaFolder, 1);
    unsetenv("PYTHONPATH");
    setenv("PYTHONPATH", anacondaFolder, 1);
    Py_Initialize();
    import_array()
    
    fftwf_init_threads();
    
    for (int n = 0; n < 2; n++)
    {
        PyObject *pArgs = PyTuple_New(9);
        ALWAYS_ASSERT(pArgs != NULL);
        
    #if 1
        // Largest use-case:
        //    (1, 450, 675) (8, 8, 391, 391) 840 1065 840 533 (16, 1065) (2, 15, 840)
        npy_intp hdims[4] = { 8, 8, 391, 391 };
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(15));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(840));
        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(1065));
        PyTuple_SetItem(pArgs, 5, PyLong_FromLong(840));
        PyTuple_SetItem(pArgs, 6, PyLong_FromLong(533));
        npy_intp xdims[2] = { 16, 1065 };
        npy_intp ydims[3] = { 2, 15, 840 };
    #else
        // Smallest use-case:
        //    (1, 450, 675) (8, 8, 61, 61) 510 735 510 368 (16, 735) (2, 15, 510)
        npy_intp hdims[4] = { 8, 8, 61, 61 };
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(15));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(510));
        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(735));
        PyTuple_SetItem(pArgs, 5, PyLong_FromLong(510));
        PyTuple_SetItem(pArgs, 6, PyLong_FromLong(368));
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
        PyTuple_SetItem(pArgs, 7, xAxisMultipliers);
        PyTuple_SetItem(pArgs, 8, yAxisMultipliers);
        
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
	{"mirrorX", mirrorX, METH_VARARGS},
    {"mirrorY", mirrorY, METH_VARARGS},
    {"ProjectForZ", ProjectForZ, METH_VARARGS},
    {"ProjectForZList", ProjectForZList, METH_VARARGS},
    {"InverseRFFT", InverseRFFT, METH_VARARGS},
    {"PrintStats", PrintStats, METH_NOARGS},
    {"ResetStats", ResetStats, METH_NOARGS},
    {"SetStatsFile", SetStatsFile, METH_VARARGS},
    {"GetNumThreadsInUse", GetNumThreadsInUse, METH_NOARGS},
    {"SetNumThreadsInUse", SetNumThreadsInUse, METH_VARARGS},
    {"GetPlanningMode", GetPlanningMode, METH_NOARGS},
    {"SetPlanningMode", SetPlanningMode, METH_VARARGS},
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
