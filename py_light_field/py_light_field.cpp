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

/* Define the FFTW planning strategy we will use. Run times:
    FFTW_ESTIMATE took 5.46
    FFTW_MEASURE took 5.01   (after 2s the first time, and <1ms on the later times)
    FFTW_PATIENT took 4.58   (after 29s the first time)
   In other words, there is a further gain available from FFTW_PATIENT, but it's too slow to be convenient until I really am running in anger with large datasets
 */
const int fftPlanMethod = FFTW_MEASURE;
/*  Define the number of threads we will use for parallel processing.
    Frustratingly, I haven't found a good way (in C or Python) to identify that we are running with hyperthreaded CPUs.
    Performance is best when *not* using extra hyperthreads, so I just divide the reported number of processors by 2.
    TODO: if this is run on a machine where no hyperthreading is available, it will not use the optimum number of processors.
    An ideal solution might be to actually have a function that runs timing tests on dummy data and decides how many processors to use,
    but that is probably overkill for now!  */
const int kMaxThreads = 16;
const int gNumThreadsToUse = int(sysconf(_SC_NPROCESSORS_ONLN) / 2);

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

extern "C" PyObject *SetStatsFile(PyObject *self, PyObject *args)
{
    const char *path;
    int append;
    if (!PyArg_ParseTuple(args, "si", &path, &append))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse string!");
        return NULL;
    }
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
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse array!");
        return NULL;
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

class WorkItem
{
public:
    virtual ~WorkItem() { }
    virtual void Run(void) = 0;
};

class FFTWorkItem : public WorkItem
{
public:
    JPythonArray2D<TYPE>    *fftResult;
    
    FFTWorkItem(int fshapeY, int fshapeX)
    {
        // Allocate an array to hold the results
        npy_intp dims[2] = { fshapeY, fshapeX };
        npy_intp strides[2] = { int((fshapeX + 15)/16)*16, 1 };     // Ensure 16-byte alignment of each row, which seems to make things *slightly* faster
        // TODO: eventually we cannot allocate like this when the work item is constructed - that will more than fill the available memory! I will need to change things so that this is allocated later (and make sure no dependencies rely on it from their own constructors)
        fftResult = new JPythonArray2D<TYPE>(NULL, dims, strides);
    }
    
    virtual ~FFTWorkItem()
    {
        delete fftResult;
    }
};

class FHWorkItem : public FFTWorkItem
{
public:
    JPythonArray2D<RTYPE>   Hts;
    bool                    transpose;
    fftwf_plan              plan, plan2;
    
    
    FHWorkItem(JPythonArray2D<RTYPE> _Hts, int fshapeY, int fshapeX, bool _transpose) : FFTWorkItem(fshapeY, fshapeX), Hts(_Hts), transpose(_transpose)
    {
        // Note that we manually split up the 2D FFT into horizontal and vertical 1D FFTs, to enable us to skip the rows that we know are all zeroes.
        int nx[1] = { fftResult->Dims(1) };
        ALWAYS_ASSERT(!(((size_t)fftResult->Data()) & 0xF));        // Check alignment. In fact, just plain malloc seems to give sufficient alignment.
        fftwf_plan_with_nthreads(1);
        // Compute the horizontal 1D FFTs, for only the nonzero rows
        plan = fftwf_plan_many_dft(1, nx, Hts.Dims(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   (fftwf_complex *)fftResult->Data(), NULL,
                                   fftResult->Strides(1), fftResult->Strides(0),
                                   FFTW_FORWARD, fftPlanMethod);
        // Compute the vertical 1D FFTs
        int ny[1] = { fftResult->Dims(0) };
        plan2 = fftwf_plan_many_dft(1, ny, fftResult->Dims(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    (fftwf_complex *)fftResult->Data(), NULL,
                                    fftResult->Strides(0), fftResult->Strides(1),
                                    FFTW_FORWARD, fftPlanMethod);
    }
    
    virtual ~FHWorkItem()
    {
        fftwf_destroy_plan(plan);
        fftwf_destroy_plan(plan2);
    }
    
    void Run(void)
    {
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
        
        // Compute the full 2D FFT (i.e. not just the RFFT)
        fftwf_execute(plan);
        fftwf_execute(plan2);
    }
    
};

class MirrorWorkItem : public FFTWorkItem
{
public:
    FHWorkItem            *sourceFFTWorkItem;
    JPythonArray1D<TYPE>  mirrorYMultiplier;
    
    MirrorWorkItem(FHWorkItem *_sourceFFTWorkItem, JPythonArray1D<TYPE> _mirrorYMultiplier) : FFTWorkItem(_sourceFFTWorkItem->fftResult->Dims(0), _sourceFFTWorkItem->fftResult->Dims(1)),
                                                                                                sourceFFTWorkItem(_sourceFFTWorkItem), mirrorYMultiplier(_mirrorYMultiplier)
    {
    }
    void Run(void)
    {
        MirrorYArray(*sourceFFTWorkItem->fftResult, mirrorYMultiplier, *fftResult);
    }
};

class ConvolveWorkItem : public WorkItem
{
    JPythonArray2D<RTYPE> projection;
    int                   bbUnmirrored, aaUnmirrored, Nnum;
    FFTWorkItem           *fftWorkItem_unXmirrored;
    bool                  mirrorX;
    JPythonArray2D<TYPE>  xAxisMultipliers;
    JPythonArray3D<TYPE>  yAxisMultipliers3;
    JPythonArray2D<TYPE>  accum;
    
    fftwf_plan            plan;
    JPythonArray2D<TYPE>  *fftArray;

public:
    ConvolveWorkItem(JPythonArray2D<RTYPE> _projection, int _bb, int _aa, int _Nnum, FFTWorkItem *_fftWorkItem_unXmirrored, bool _mirrorX, JPythonArray2D<TYPE> _xAxisMultipliers, JPythonArray3D<TYPE> _yAxisMultipliers3, JPythonArray2D<TYPE> _accum)
                       : projection(_projection), bbUnmirrored(_bb), aaUnmirrored(_aa), Nnum(_Nnum), fftWorkItem_unXmirrored(_fftWorkItem_unXmirrored), mirrorX(_mirrorX), xAxisMultipliers(_xAxisMultipliers), yAxisMultipliers3(_yAxisMultipliers3), accum(_accum),
                         plan(NULL), fftArray(NULL)
    {
        // Define the FFTW plan that we will use as part of our convolution task
        int fullYSize = fftWorkItem_unXmirrored->fftResult->Dims(0);
        int fullXSize = fftWorkItem_unXmirrored->fftResult->Dims(1);
        npy_intp smallDims[2] = { fullYSize/Nnum, fullXSize/Nnum };
        fftArray = new JPythonArray2D<TYPE>(smallDims);
        // A reminder that this planning may modify the data in fftArray, especially if using FFTW_MEASURE.
        // That's fine, becasue we have not filled it in yet.
        plan = fftwf_plan_dft_2d((int)smallDims[0], (int)smallDims[1], (fftwf_complex *)fftArray->Data(), (fftwf_complex *)fftArray->Data(), FFTW_FORWARD, fftPlanMethod);
    }
    
    virtual ~ConvolveWorkItem()
    {
        delete fftArray;
        fftwf_destroy_plan(plan);
    }
    
    void special_fftconvolve_part1(JPythonArray2D<TYPE> &result, JPythonArray1D<TYPE> expandXMultiplier, int bb, int aa)
    {
        // Select the pixels indexed by bb,aa, and pad to the size we need based on what we will eventually tile up to the same shape as fHtsFull_unXmirrored
        fftArray->SetZero();    // Note that we may not be setting every element, I think, due to padding - so we do need to set to zero initially
        for (int y = bb, y2 = 0; y < projection.Dims(0); y += Nnum, y2++)
            for (int x = aa, x2 = 0; x < projection.Dims(1); x += Nnum, x2++)
                (*fftArray)[y2][x2] = projection[y][x];
        
        // Compute the full 2D FFT (i.e. not just the RFFT)
        fftwf_execute(plan);
        
        // Tile the result up to the length that is implied by expandXMultiplier (using that length saves us figuring out the length for ourselves)
        int outputLength = expandXMultiplier.Dims(0);
        for (int y = 0; y < fftArray->Dims(0); y++)
        {
            JPythonArray1D<TYPE> _result = result[y];
            JPythonArray1D<TYPE> _fftArray = (*fftArray)[y];
            for (int x = 0, inputX = 0; x < outputLength; x++, inputX++)
            {
                if (inputX == fftArray->Dims(1))     // Modulo operator is surprisingly time-consuming - it is faster to do this test instead
                    inputX = 0;
                _result[x] = _fftArray[inputX] * expandXMultiplier[x];
            }
        }
    }
    
    void ConvolvePart4(int bb, int aa, bool mirrorX, JPythonArray2D<TYPE> yAxisMultipliers)
    {
        // This plays the role that special_fftconvolve2 does in the python code.
        // We take advantage of the fact that we have been passed fHTsFull, to tell us what the padded array dimensions should be for the FFT.
        
        npy_intp output_dims[2] = { fftArray->Dims(0), xAxisMultipliers.Dims(1) };
        JPythonArray2D<TYPE> partialFourierOfProjection(output_dims);
        special_fftconvolve_part1(partialFourierOfProjection, xAxisMultipliers[kXAxisMultiplierExpandXStart+aa], bb, aa);
        
        for (int y = 0; y < accum.Dims(0); y++)
        {
            int ya = y % partialFourierOfProjection.Dims(0);
            // Note that the array accessors take time, so should be lifted out of CalculateRow.
            // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
            // What I'd really like to do is to profile this code and see how well it does, because there may be other things I can improve.
            if (!mirrorX)
                CalculateRow(&accum[y][0], &partialFourierOfProjection[ya][0], &(*fftWorkItem_unXmirrored->fftResult)[y][0], yAxisMultipliers[bb][y], accum.Dims(1));
            else
                CalculateRowMirror(&accum[y][0], &partialFourierOfProjection[ya][0], &(*fftWorkItem_unXmirrored->fftResult)[y][0], yAxisMultipliers[bb][y], accum.Dims(1), fftWorkItem_unXmirrored->fftResult->Dims(1));
        }
    }
    
    void Run(void)
    {
        ConvolvePart4(bbUnmirrored, aaUnmirrored, false, yAxisMultipliers3[kYAxisMultiplierNoMirror]);
        if (mirrorX)
            ConvolvePart4(Nnum-bbUnmirrored-1, aaUnmirrored, true, yAxisMultipliers3[kYAxisMultiplierMirrorX]);
    }
    
};

#if 0
struct ThreadInfo
{
    int imageCounter;
    int numThreads;
    double idle[kMaxThreads];
    double t1[kMaxThreads];
    double t2[kMaxThreads];
    double t3[kMaxThreads];
    JPythonArray3D<RTYPE> *projection;
    int bb;
    int aa;
    int Nnum;
    bool mirrorX;
    JPythonArray2D<TYPE> *fHTsFull_unmirrored;
    JPythonArray2D<TYPE> *xAxisMultipliers;
    JPythonArray2D<TYPE> *yAxisMultipliers;
    JPythonArray3D<TYPE> *accum;
    fftwf_plan plan;
    
    bool WorkAvailable(void) { ALWAYS_ASSERT(accum != NULL); return (imageCounter < accum->Dims(0)); }
    bool WorkInProgress(void)
    {
        bool anyBusy = false;
        for (int i = 0; i < numThreads; i++)
            if (!idle[i])
                anyBusy = true;
        return anyBusy;
    }
};

pthread_t gThreads[kMaxThreads];  // Since this is global, it will start as all NULL.
int gNumThreadsRunning = 0;
int gThreadIDCounter;
ThreadInfo *gThreadInfo;
JMutex workThreadMutex;
pthread_cond_t	workAvailableSignal, workCompletedSignal;

void *ThreadFunc(void *params)
{
    int *threadIDCounter = (int *)params;
    int thisThreadID = __sync_fetch_and_add(threadIDCounter, 1);
    while (1)
    {
        ThreadInfo *t = NULL;
        int i;
        {
            LocalGetMutex lgm(&workThreadMutex);
            t = gThreadInfo;
            if (!t->WorkAvailable())
                t->t3[thisThreadID] = GetTime();
            while (!t->WorkAvailable())
            {
                // We are out of work. Wait until fresh work appears
                t->idle[thisThreadID] = true;
                pthread_cond_broadcast(&workCompletedSignal);
                workThreadMutex.BlockWaitingForSignal(&workAvailableSignal, __LINE__, true);
                t = gThreadInfo;
                t->t1[thisThreadID] = GetTime();
            }
            i = t->imageCounter++;
        }

        JPythonArray3D<RTYPE> *projection = t->projection;
        int bb = t->bb;
        int aa = t->aa;
        int Nnum = t->Nnum;
        bool mirrorX = t->mirrorX;
        JPythonArray2D<TYPE> *fHTsFull_unmirrored = t->fHTsFull_unmirrored;
        JPythonArray2D<TYPE> *xAxisMultipliers = t->xAxisMultipliers;
        JPythonArray2D<TYPE> *yAxisMultipliers = t->yAxisMultipliers;
        JPythonArray3D<TYPE> *accum = t->accum;
        fftwf_plan plan = t->plan;
        ALWAYS_ASSERT(plan != NULL);

        JPythonArray2D<TYPE> partialFourierOfProjection = PartialFourierOfInputArray((*projection)[i], bb, aa, fHTsFull_unmirrored->Dims(0), fHTsFull_unmirrored->Dims(1), Nnum, (*xAxisMultipliers)[kXAxisMultiplierExpandXStart+aa], plan);
        if (t->t2[thisThreadID] == 0)
            t->t2[thisThreadID] = GetTime();
        for (int y = 0; y < accum->Dims(1); y++)
        {
            int ya = y % partialFourierOfProjection.Dims(0);
            // Note that the array accessors take time, so should be lifted out of CalculateRow.
            // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
            // What I'd really like to do is to profile this code and see how well it does, because there may be other things I can improve.
            if (!mirrorX)
                CalculateRow(&(*accum)[i][y][0], &partialFourierOfProjection[ya][0], &(*fHTsFull_unmirrored)[y][0], (*yAxisMultipliers)[bb][y], accum->Dims(2));
            else
                CalculateRowMirror(&(*accum)[i][y][0], &partialFourierOfProjection[ya][0], &(*fHTsFull_unmirrored)[y][0], (*yAxisMultipliers)[bb][y], accum->Dims(2), fHTsFull_unmirrored->Dims(1));
        }
    }
    return NULL;
}

void RunThreads(ThreadInfo *thisThreadInfo)
{
    int numThreadsToUse = thisThreadInfo->numThreads;
    ALWAYS_ASSERT(numThreadsToUse <= kMaxThreads);
    ALWAYS_ASSERT(numThreadsToUse == gNumThreadsToUse);     // TODO: still need to know whether this is the number of threads we actually launched
    
    LocalGetMutex lgm(&workThreadMutex);
    
    if (gNumThreadsRunning == 0)
    {
        int result = pthread_cond_init(&workAvailableSignal, NULL);
        ALWAYS_ASSERT(result == 0);
        result = pthread_cond_init(&workCompletedSignal, NULL);
        ALWAYS_ASSERT(result == 0);
    }
    if (gNumThreadsRunning != numThreadsToUse)
    {
        for (int i = 0; i < kMaxThreads; i++)
            if (gThreads[i] != NULL)
                pthread_kill(gThreads[i], -9);
        gThreadIDCounter = 0;
        for (int i = 0; i < numThreadsToUse; i++)
            pthread_create(&gThreads[i], NULL, ThreadFunc, &gThreadIDCounter);
        gNumThreadsRunning = numThreadsToUse;
    }
    gThreadInfo = thisThreadInfo;
    pthread_cond_broadcast(&workAvailableSignal);
    while (thisThreadInfo->WorkInProgress())
        workThreadMutex.BlockWaitingForSignal(&workCompletedSignal, __LINE__, false);
}
#endif

enum
{
    kWorkFFT = 0,
    kWorkMirrorY,
    kWorkConvolve,
    kNumWorkTypes
};

void ConvolvePart2(JPythonArray2D<RTYPE> projection, int bb, int aa, int Nnum, bool mirrorY, bool mirrorX, FHWorkItem *fftWorkItem, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray3D<TYPE> yAxisMultipliers, JPythonArray2D<TYPE> accum, std::vector<WorkItem *> work[kNumWorkTypes])
{
    ConvolveWorkItem *workConvolve = new ConvolveWorkItem(projection, bb, aa, Nnum, fftWorkItem, mirrorX, xAxisMultipliers, yAxisMultipliers, accum);
    work[kWorkConvolve].push_back(workConvolve);
    if (mirrorY)
    {
        MirrorWorkItem *workCalcMirror = new MirrorWorkItem(fftWorkItem, xAxisMultipliers[kXAxisMultiplierMirrorY]);
        work[kWorkMirrorY].push_back(workCalcMirror);
        ConvolveWorkItem *workConvolveMirror = new ConvolveWorkItem(projection, bb, Nnum-aa-1, Nnum, workCalcMirror, mirrorX, xAxisMultipliers, yAxisMultipliers, accum);
        work[kWorkConvolve].push_back(workConvolveMirror);
    }
}

void RunWork(std::vector<WorkItem *> work[kNumWorkTypes])
{
    // Initially doing this single-threaded (and in the correct order of work types)
    for (int w = 0; w < kNumWorkTypes; w++)
        for (size_t i = 0; i < work[w].size(); i++)
            work[w][i]->Run();
}
             
extern "C" PyObject *ProjectForZ(PyObject *self, PyObject *args)
{
    PyArrayObject *_projection, *_HtsFull, *_xAxisMultipliers, *_yAxisMultipliers;
    int Nnum, fshapeY, fshapeX, rfshapeY, rfshapeX;
#if TESTING
    // PyArg_ParseTuple doesn't seem to work when I use it on my own synthesized tuple.
    // I don't know why that is, but this code exists as a workaround for that problem.
    _projection = (PyArrayObject *)PyTuple_GetItem(args, 0);
    _HtsFull = (PyArrayObject *)PyTuple_GetItem(args, 1);
    Nnum = (int)PyLong_AsLong(PyTuple_GetItem(args, 2));
    fshapeY = (int)PyLong_AsLong(PyTuple_GetItem(args, 3));
    fshapeX = (int)PyLong_AsLong(PyTuple_GetItem(args, 4));
    rfshapeY = (int)PyLong_AsLong(PyTuple_GetItem(args, 5));
    rfshapeX = (int)PyLong_AsLong(PyTuple_GetItem(args, 6));
    _xAxisMultipliers = (PyArrayObject *)PyTuple_GetItem(args, 7);
    _yAxisMultipliers = (PyArrayObject *)PyTuple_GetItem(args, 8);
#else
    if (!PyArg_ParseTuple(args, "O!O!iiiiiO!O!",
                          &PyArray_Type, &_projection,
                          &PyArray_Type, &_HtsFull,
                          &Nnum, &fshapeY, &fshapeX, &rfshapeY, &rfshapeX,
                          &PyArray_Type, &_xAxisMultipliers,
                          &PyArray_Type, &_yAxisMultipliers))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
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

    // Set up the work items describing this complete projection operation
    std::vector<WorkItem *> work[kNumWorkTypes];
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

            FHWorkItem *f1 = new FHWorkItem(HtsFull[bb][aa], fshapeY, fshapeX, false);
            work[kWorkFFT].push_back(f1);

            ALWAYS_ASSERT(projection.Dims(0) == accum.Dims(0));
            for (int i = 0; i < projection.Dims(0); i++)
            {
                ConvolvePart2(projection[i], bb, aa, Nnum, mirrorY, mirrorX, f1, xAxisMultipliers, yAxisMultipliers, accum[i], work);

                if (transpose)
                {
                    FHWorkItem *f2 = new FHWorkItem(HtsFull[bb][aa], fshapeY, fshapeX, true);
                    work[kWorkFFT].push_back(f2);
                    // Note that my,mx (and bb,aa) have been swapped here, which is necessary following the transpose.
                    ConvolvePart2(projection[i], aa, bb, Nnum, mirrorX, mirrorY, f2, xAxisMultipliers, yAxisMultipliers, accum[i], work);
                }
            }
        }
    }
    // Do the actual hard work (parallelised)
    RunWork(work);
    // Clean up work items
    for (int w = 0; w < kNumWorkTypes; w++)
        for (size_t i = 0; i < work[w].size(); i++)
            delete work[w][i];
    
    return PyArray_Return(_accum);
}

extern "C" PyObject *InverseRFFT(PyObject *self, PyObject *args)
{
    PyArrayObject *_mat;
    int inputShapeY, inputShapeX;
    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyArray_Type, &_mat,
                          &inputShapeY, &inputShapeX))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
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
                                     fftPlanMethod);
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
    {"InverseRFFT", InverseRFFT, METH_VARARGS},
    {"PrintStats", PrintStats, METH_NOARGS},
    {"ResetStats", ResetStats, METH_NOARGS},
    {"SetStatsFile", SetStatsFile, METH_VARARGS},
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
