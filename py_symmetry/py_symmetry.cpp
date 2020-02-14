#include "common/jAssert.h"
#include "common/VectorFunctions.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jPythonCommon.h"
#include "common/PIVImageWindow.h"
#include <fftw3.h>

// Note: the TESTING macro is set when we build within Xcode, but not when we build using setup.py

const int kMaxThreads = 16;
const int gNumThreadsToUse = 8;
const int fftPlanMethod = FFTW_MEASURE;
/* TODO: decide which planning method to use.
    FFTW_ESTIMATE took 5.46
    FFTW_MEASURE took 5.01   (after 2s the first time, and <1ms on the later times)
    FFTW_PATIENT took 4.58   (after 29s the first time) */

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

PyArrayObject *MirrorYArray(JPythonArray2D<TYPE> fHtsFull, JPythonArray1D<TYPE> mirrorYMultiplier)
{
    // Given F(H), return F(mirrorY(H)), where mirrorY(H) is the vertical reflection of H.
    PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(2, fHtsFull.Dims(), NPY_CFLOAT);
    JPythonArray2D<TYPE> result(r);
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
    return r;
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
        r = MirrorYArray(aa, multiplier);
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

void special_fftconvolve_part1(JPythonArray2D<RTYPE> inputArray, JPythonArray2D<TYPE> result, JPythonArray1D<TYPE> expandXMultiplier, int bb, int aa, int Nnum, int fullYSize, int fullXSize, fftwf_plan inPlan = NULL)
{
    // Select the pixels indexed by bb,aa, and pad to the size we need based on what we will eventually tile up to (fullYSize x fullXSize)
    npy_intp smallDims[2] = { fullYSize/Nnum, fullXSize/Nnum };
    JPythonArray2D<TYPE> fftArray(smallDims);
    
    double t1 = GetTime();
    fftwf_plan planToUse = inPlan;
    if (planToUse == NULL)
        planToUse = fftwf_plan_dft_2d((int)smallDims[0], (int)smallDims[1], (fftwf_complex *)fftArray.Data(), (fftwf_complex *)fftArray.Data(), FFTW_FORWARD, fftPlanMethod);     // This does need to be done before the data is initialized, especially if using FFTW_MEASURE
    double t2 = GetTime();
    fftArray.SetZero();

    for (int y = bb, y2 = 0; y < inputArray.Dims(0); y += Nnum, y2++)
        for (int x = aa, x2 = 0; x < inputArray.Dims(1); x += Nnum, x2++)
            fftArray[y2][x2] = inputArray[y][x];
    double t3 = GetTime();

    // Compute the full 2D FFT (i.e. not just the RFFT)
    if (inPlan != NULL)
    {
        // Adapt it to our data
        fftwf_execute_dft(planToUse, (fftwf_complex *)fftArray.Data(), (fftwf_complex *)fftArray.Data());
    }
    else
        fftwf_execute(planToUse);
    double t4 = GetTime();
    gFFTWTime2 += t4-t3;
    if (inPlan == NULL)
        fftwf_destroy_plan(planToUse);
    double t5 = GetTime();

    // Tile the result up to the length that is implied by expandXMultiplier (using that length saves us figuring out the length for ourselves)
    int outputLength = expandXMultiplier.Dims(0);
    for (int y = 0; y < smallDims[0]; y++)
    {
        auto _result = result[y];
        auto _fftArray = fftArray[y];
        for (int x = 0, inputX = 0; x < outputLength; x++, inputX++)
        {
            if (inputX == smallDims[1])     // Modulo operator is surprisingly time-consuming - it is faster to do this test instead
                inputX = 0;
            _result[x] = _fftArray[inputX] * expandXMultiplier[x];
        }
    }
    double t6 = GetTime();
//    printf("Times %.2lf %.2lf %.2lf %.2lf %.2lf \n", (t2-t1)*1e6, (t3-t2)*1e6, (t4-t3)*1e6, (t5-t4)*1e6, (t6-t5)*1e6);
}

JPythonArray3D<TYPE> PartialFourierOfInputArray(JPythonArray3D<RTYPE> inputArray, int bb, int aa, int fullYSize, int fullXSize, int Nnum, JPythonArray1D<TYPE> expandXMultiplier)
{
    // Allocate an array to hold the results (after tiling expansion along the X direction).
    // I do not use a real python array (just my own class, allocating its own data) to avoid the potential overheads of the actual python runtime code.
    npy_intp output_dims[3] = { inputArray.Dims(0), fullYSize / Nnum, expandXMultiplier.Dims(0) };
    JPythonArray3D<TYPE> result = JPythonArray3D<TYPE>(output_dims);
    for (int n = 0; n < inputArray.Dims(0); n++)
        special_fftconvolve_part1(inputArray[n], result[n], expandXMultiplier, bb, aa, Nnum, fullYSize, fullXSize);
    return result;
}

JPythonArray2D<TYPE> PartialFourierOfInputArray(JPythonArray2D<RTYPE> inputArray, int bb, int aa, int fullYSize, int fullXSize, int Nnum, JPythonArray1D<TYPE> expandXMultiplier, fftwf_plan plan)
{
    // Allocate an array to hold the results (after tiling expansion along the X direction).
    // I do not use a real python array (just my own class, allocating its own data) to avoid the potential overheads of the actual python runtime code.
    npy_intp output_dims[2] = { fullYSize / Nnum, expandXMultiplier.Dims(0) };
    JPythonArray2D<TYPE> result(output_dims);
    special_fftconvolve_part1(inputArray, result, expandXMultiplier, bb, aa, Nnum, fullYSize, fullXSize, plan);
    return result;
}

extern "C" PyObject *special_fftconvolve_part1(PyObject *self, PyObject *args)
{
    // TODO: I have disabled this as I have been working on the Convolve() function as a replacement,
    // but I should reinstate these after updating them to match my recent code and API changes.
#if 0
    // Parameters: inputArray, bb, aa, Nnum, in2Shape [asserted to be 2d], expandXMultiplier
    // Obtains fshape by calling convolutionShape, passing in2Shape
    // Calls special_rfftn (partial=True)
    // Returns fa[partial] and fshape (which is later used to get expand2multiplier)

    // Parse the input arrays from *args
    PyArrayObject *_inputArray, *_expandXMultiplier;
    int bb, aa, Nnum, fullYSize, fullXSize;
    if (!PyArg_ParseTuple(args, "O!iiiiiO!",
                          &PyArray_Type, &_inputArray,
                          &bb, &aa, &Nnum, &fullYSize, &fullXSize,
                          &PyArray_Type, &_expandXMultiplier))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
    }
    
    JPythonArray1D<TYPE> expandXMultiplier(_expandXMultiplier);
    PyArrayObject *_result;
    if (PyArray_NDIM(_inputArray) == 3)
    {
        JPythonArray3D<RTYPE> inputArray(_inputArray);
        _result = PartialFourierOfInputArray(inputArray, bb, aa, fullYSize, fullXSize, Nnum, expandXMultiplier);
    }
    else
    {
        JPythonArray2D<RTYPE> inputArray(_inputArray);
        npy_intp output_dims[2] = { fullYSize / Nnum, expandXMultiplier.Dims(0) };
        _result = (PyArrayObject *)PyArray_EMPTY(2, output_dims, NPY_CFLOAT, 0);
        JPythonArray2D<TYPE> result(_result);
        special_fftconvolve_part1(inputArray, result, expandXMultiplier, bb, aa, Nnum, fullYSize, fullXSize);
    }
    return PyArray_Return(_result);
#endif
    return NULL;
}

extern "C" PyObject *special_fftconvolve(PyObject *self, PyObject *args)
{
    // TODO: I have disabled this as I have been working on the Convolve() function as a replacement,
    // but I should reinstate these after updating them to match my recent code and API changes.
#if 0
    PyArrayObject *_accum, *_fa_partial, *_fb_unmirrored, *_expandMultiplier, *_mirrorXMultiplier;
    int validWidth;
    
    // Parse the input arrays from *args
    if (!PyArg_ParseTuple(args, "OO!O!O!iO!",
                          &_accum,
                          &PyArray_Type, &_fa_partial,
                          &PyArray_Type, &_fb_unmirrored,
                          &PyArray_Type, &_expandMultiplier,
                          &validWidth,
                          &PyArray_Type, &_mirrorXMultiplier))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
    }

    if ((PyArray_TYPE(_fa_partial) != NPY_CFLOAT) ||
        (PyArray_TYPE(_fb_unmirrored) != NPY_CFLOAT))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unsuitable array types %d %d passed in", (int)PyArray_TYPE(_fa_partial), (int)PyArray_TYPE(_fb_unmirrored));
        return NULL;
    }

    JPythonArray3D<TYPE> fa_partial(_fa_partial);
    JPythonArray2D<TYPE> fb_unmirrored(_fb_unmirrored);
    JPythonArray1D<TYPE> expandMultiplier(_expandMultiplier);
    JPythonArray1D<TYPE> *mirrorXMultiplier = NULL;
    if (_mirrorXMultiplier != (PyArrayObject *)Py_None)
    {
        mirrorXMultiplier = new JPythonArray1D<TYPE>(_mirrorXMultiplier);
    }
    if (!gPythonArraysOK)
    {
        printf("Returning NULL due to python error\n");
        return NULL;        // When flag was cleared, a PyErr should have been set up with the details.
    }

    if ((fa_partial.Strides(2) != 1) ||
        (fb_unmirrored.Strides(1) != 1) ||
        (expandMultiplier.Strides(0) != 1) ||
        ((mirrorXMultiplier != NULL) && (mirrorXMultiplier->Strides(0) != 1)))
    {
        delete mirrorXMultiplier;
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "One of the input arrays does not have unit stride (%zd %zd %zd)", fa_partial.Strides(2), fb_unmirrored.Strides(1), expandMultiplier.Strides(0));
        return NULL;
    }

    // Create a new array for accum, if we were not passed in an existing one
    npy_intp output_dims[3] = { fa_partial.Dims(0), fb_unmirrored.Dims(0), validWidth };
    PyArrayObject *__accum = _accum;
    if (__accum == (PyArrayObject *)Py_None)
        __accum = (PyArrayObject *)PyArray_ZEROS(3, output_dims, NPY_CFLOAT, 0);
    else
        Py_INCREF(__accum);      // Although I haven't found a definite statement to this effect, I suspect I need to incref on an input variable because pyarray_return presumably decrefs it?
    JPythonArray3D<TYPE> accum(__accum);
    
    special_fftconvolve(fa_partial, fb_unmirrored, expandMultiplierY, mirrorXMultiplier);
    
    if (mirrorXMultiplier != NULL)
        delete mirrorXMultiplier;
    return PyArray_Return(__accum);
#endif
    return NULL;
}

struct ThreadInfo
{
    int imageCounter;
    double t1[kMaxThreads];
    double t2[kMaxThreads];
    JPythonArray3D<RTYPE> projection;
    int bb;
    int aa;
    int Nnum;
    bool mirrorX;
    JPythonArray2D<TYPE> fHTsFull_unmirrored;
    JPythonArray2D<TYPE> xAxisMultipliers;
    JPythonArray2D<TYPE> yAxisMultipliers;
    JPythonArray3D<TYPE> accum;
    fftwf_plan plan;
};

void RunThreads(void *(*threadFunc)(void *), void *parms, int numThreadsToUse)
{
    pthread_t			ourThreads[kMaxThreads];
    ALWAYS_ASSERT(numThreadsToUse <= kMaxThreads);
    for (long i = 0; i < numThreadsToUse; i++)
        pthread_create(&ourThreads[i], NULL, threadFunc, parms);
    for (long i = 0; i < numThreadsToUse; i++)
        pthread_join(ourThreads[i], NULL);
}

void *ThreadFunc(void *params)
{
    ThreadInfo *t = (ThreadInfo *)params;
    JPythonArray3D<RTYPE> projection = t->projection;
    int bb = t->bb;
    int aa = t->aa;
    int Nnum = t->Nnum;
    bool mirrorX = t->mirrorX;
    JPythonArray2D<TYPE> fHTsFull_unmirrored = t->fHTsFull_unmirrored;
    JPythonArray2D<TYPE> xAxisMultipliers = t->xAxisMultipliers;
    JPythonArray2D<TYPE> yAxisMultipliers = t->yAxisMultipliers;
    JPythonArray3D<TYPE> accum = t->accum;
    fftwf_plan plan = t->plan;
    ALWAYS_ASSERT(plan != NULL);

    int thisThreadID = 0;
    while (1)
    {
        int i = __sync_fetch_and_add(&t->imageCounter, 1);
        if (i < 8)
        {
            thisThreadID = i;
            t->t1[i] = GetTime();
        }
        if (i >= accum.Dims(0))
            break;
        JPythonArray2D<TYPE> partialFourierOfProjection = PartialFourierOfInputArray(projection[i], bb, aa, fHTsFull_unmirrored.Dims(0), fHTsFull_unmirrored.Dims(1), Nnum, xAxisMultipliers[kXAxisMultiplierExpandXStart+aa], plan);
        for (int y = 0; y < accum.Dims(1); y++)
        {
            int ya = y % partialFourierOfProjection.Dims(0);
            // Note that the array accessors take time, so should be lifted out of CalculateRow.
            // By directly accessing the pointers, I bypass range checks and implicitly assume contiguous arrays (the latter makes a difference to speed).
            // What I'd really like to do is to profile this code and see how well it does, because there may be other things I can improve.
            if (!mirrorX)
                CalculateRow(&accum[i][y][0], &partialFourierOfProjection[ya][0], &fHTsFull_unmirrored[y][0], yAxisMultipliers[bb][y], accum.Dims(2));
            else
                CalculateRowMirror(&accum[i][y][0], &partialFourierOfProjection[ya][0], &fHTsFull_unmirrored[y][0], yAxisMultipliers[bb][y], accum.Dims(2), fHTsFull_unmirrored.Dims(1));
        }
    }
    t->t2[thisThreadID] = GetTime();
    return NULL;
}

void ConvolvePart4(JPythonArray3D<RTYPE> projection, int bb, int aa, int Nnum, bool mirrorX, JPythonArray2D<TYPE> fHTsFull_unmirrored, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray2D<TYPE> yAxisMultipliers, JPythonArray3D<TYPE> accum)
{
    // This plays the role that special_fftconvolve2 does in the python code.
    // We take advantage of the fact that we have been passed fHTsFull, to tell us what the padded array dimensions should be for the FFT.
    double t0 = GetTime();
    
    t0 = GetTime();

    // Define the FFTW plan that we will use in each of our multiple wor threads that will perform this task
    int fullYSize = fHTsFull_unmirrored.Dims(0);
    int fullXSize = fHTsFull_unmirrored.Dims(1);
    npy_intp smallDims[2] = { fullYSize/Nnum, fullXSize/Nnum };
    JPythonArray2D<TYPE> fftArray(smallDims);
    fftwf_plan plan = fftwf_plan_dft_2d((int)smallDims[0], (int)smallDims[1], (fftwf_complex *)fftArray.Data(), (fftwf_complex *)fftArray.Data(), FFTW_FORWARD, fftPlanMethod);     // This does need to be done before the data is initialized, especially if using FFTW_MEASURE
    
    ThreadInfo threadInfo = { 0, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, projection, bb, aa, Nnum, mirrorX, fHTsFull_unmirrored, xAxisMultipliers, yAxisMultipliers, accum, plan};
    RunThreads(ThreadFunc, &threadInfo, gNumThreadsToUse);
  #if 0
    double t3 = GetTime();
    double totalDead = 0, totalWork = 0;
    for (int i = 0; i < numThreadsToUse; i++)
    {
        double deadStart = (threadInfo.t1[i]-t0)*1e3;
        double deadEnd = (t3-threadInfo.t2[i])*1e3;
        double work = (threadInfo.t2[i]-threadInfo.t1[i])*1e3;
        totalDead += deadStart + deadEnd;
        totalWork += work;
        printf("Thread %d start %.3lf end %.3lf (took %.3lf). Dead time %.3lf\n", i, deadStart, (threadInfo.t2[i]-t0)*1e3, work, deadEnd);
    }
    printf("Total dead %.3lf, total work %.3lf (%.1lfx load)\n", totalDead, totalWork, totalWork / ((t3-t0)*1e3));
  #endif
    gSpecialTime += GetTime()-t0;
}

void ConvolvePart3(JPythonArray3D<RTYPE> projection, int bb, int aa, int Nnum, JPythonArray2D<TYPE> fHtsFull, bool mirrorX, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray3D<TYPE> yAxisMultipliers, JPythonArray3D<TYPE> accum)
{
    
    ConvolvePart4(projection, bb, aa, Nnum, false, fHtsFull, xAxisMultipliers, yAxisMultipliers[kYAxisMultiplierNoMirror], accum);
    if (mirrorX)
        ConvolvePart4(projection, Nnum-bb-1, aa, Nnum, mirrorX, fHtsFull, xAxisMultipliers, yAxisMultipliers[kYAxisMultiplierMirrorX], accum);
}

void ConvolvePart2(JPythonArray3D<RTYPE> projection, int bb, int aa, int Nnum, bool mirrorY, bool mirrorX, JPythonArray2D<TYPE> fHtsFull, JPythonArray2D<TYPE> xAxisMultipliers, JPythonArray3D<TYPE> yAxisMultipliers, JPythonArray3D<TYPE> accum)
{
    ConvolvePart3(projection, bb, aa, Nnum, fHtsFull, mirrorX, xAxisMultipliers, yAxisMultipliers, accum);
    if (mirrorY)
    {
        double t0 = GetTime();
        PyArrayObject *fHtsFull_mirror = MirrorYArray(fHtsFull, xAxisMultipliers[kXAxisMultiplierMirrorY]);
        gMirrorTime += GetTime()-t0;
        // TODO: it is probably inefficient to be working with a PyArrayObject here (reliant on python's memory management etc),
        // but it's probably the easiest way to do it if I want to keep the ability to call MirrorYArray as a standalone function.
        ConvolvePart3(projection, bb, Nnum-aa-1, Nnum, JPythonArray2D<TYPE>(fHtsFull_mirror), mirrorX, xAxisMultipliers, yAxisMultipliers, accum);
        Py_DECREF(fHtsFull_mirror);
    }
}

extern "C" PyObject *Convolve(PyObject *self, PyObject *args)
{
    PyArrayObject *_projection, *_fHtsFull, *_xAxisMultipliers, *_yAxisMultipliers, *_accum;
    int bb, aa, Nnum;
    if (!PyArg_ParseTuple(args, "O!O!iiiO!O!O!",
                          &PyArray_Type, &_projection,
                          &PyArray_Type, &_fHtsFull,
                          &bb, &aa, &Nnum,
                          &PyArray_Type, &_xAxisMultipliers,
                          &PyArray_Type, &_yAxisMultipliers,
                          &PyArray_Type, &_accum))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse input parameters!");
        return NULL;
    }
    
    JPythonArray3D<RTYPE> projection(_projection);
    JPythonArray2D<TYPE> fHtsFull(_fHtsFull);
    JPythonArray2D<TYPE> xAxisMultipliers(_xAxisMultipliers);
    JPythonArray3D<TYPE> yAxisMultipliers(_yAxisMultipliers);
    JPythonArray3D<TYPE> accum(_accum);
    
    if (!(projection.FinalDimensionUnitStride() && fHtsFull.FinalDimensionUnitStride() && xAxisMultipliers.FinalDimensionUnitStride() && yAxisMultipliers.FinalDimensionUnitStride() && accum.FinalDimensionUnitStride()))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Input arrays must have unit stride in the final dimension");
        return NULL;
    }

    int cent = int(Nnum/2);
    bool mirrorX = (bb != cent);
    bool mirrorY = (aa != cent);
    
    // Perform the convolution
    ConvolvePart2(projection, bb, aa, Nnum, mirrorY, mirrorX, fHtsFull, xAxisMultipliers, yAxisMultipliers, accum);

    // Although I haven't found a definite statement to this effect, I suspect I need to incref on an input variable because pyarray_return presumably decrefs it?
    // TODO: I probably don't need to return it at all (since it was passed in to us), but it feels to me like it's tidier and clearer that way.
    Py_INCREF(_accum);
    return PyArray_Return(_accum);
}

JPythonArray2D<TYPE> CalcFH(JPythonArray2D<RTYPE> Hts, int fshapeY, int fshapeX, bool transpose)
{
    // Allocate an array to hold the results
    npy_intp dims[2] = { fshapeY, fshapeX };
    npy_intp strides[2] = { int((fshapeX + 15)/16)*16, 1 };     // Ensure 16-byte alignment of each row, which seems to make things *slightly* faster
    double t0 = GetTime();
    JPythonArray2D<TYPE> result(NULL, dims, strides);
    
    double t1 = GetTime();
    // We do need to define the plan *before* the data is initialized, especially if using FFTW_MEASURE (which will overwrite the contents of the buffers)

    // Note that we manually split up the 2D FFT into horizontal and vertical 1D FFTs, to enable us to skip the rows that we know are all zeroes.
    int nx[1] = { result.Dims(1) };
    ALWAYS_ASSERT(!(((size_t)result.Data()) & 0xF));        // Check alignment. Just plain malloc seems to give sufficient alignment.
    // Compute the horizontal 1D FFTs, for only the nonzero rows
    fftwf_plan_with_nthreads(gNumThreadsToUse);
    fftwf_plan plan = fftwf_plan_many_dft(1, nx, Hts.Dims(0),
                                             (fftwf_complex *)result.Data(), NULL,
                                             result.Strides(1), result.Strides(0),
                                             (fftwf_complex *)result.Data(), NULL,
                                             result.Strides(1), result.Strides(0),
                                             FFTW_FORWARD, fftPlanMethod);
    // Compute the vertical 1D FFTs
    int ny[1] = { result.Dims(0) };
    fftwf_plan plan2 = fftwf_plan_many_dft(1, ny, result.Dims(1),
                                            (fftwf_complex *)result.Data(), NULL,
                                            result.Strides(0), result.Strides(1),
                                            (fftwf_complex *)result.Data(), NULL,
                                            result.Strides(0), result.Strides(1),
                                            FFTW_FORWARD, fftPlanMethod);
    fftwf_plan_with_nthreads(1);
    double t2 = GetTime();
    result.SetZero();
    double t2a = GetTime();
    assert(Hts.Dims(0) == Hts.Dims(1));     // Sanity check - this should be true. If it is not, then among other things our whole transpose logic gets messed up.
    if (transpose)
        for (int y = 0; y < Hts.Dims(0); y++)
        {
            auto _result = result[y];
            for (int x = 0; x < Hts.Dims(1); x++)
                _result[x] = Hts[x][y];
        }
    else
        for (int y = 0; y < Hts.Dims(0); y++)
        {
            auto _result = result[y];
            auto _Hts = Hts[y];
            for (int x = 0; x < Hts.Dims(1); x++)
                _result[x] = _Hts[x];
        }
    double t3 = GetTime();
    
    // Compute the full 2D FFT (i.e. not just the RFFT)
    fftwf_execute(plan);
    fftwf_execute(plan2);
    double t4 = GetTime();
    gFFTWPlanTime += t2-t1;
    gFFTWInitializeTime1 += t2a-t2;
    gFFTWInitializeTime2 += t3-t2a;
    gFFTExecuteTime += t4-t3;
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(plan2);
    double t5 = GetTime();
    
    //printf("FFT times %.2lf plan %.2lf init {%.2lf, %.2lf} ex %.2lf %.2lf\n", (t1-t0)*1e6, (t2-t1)*1e6, (t2a-t2)*1e6, (t3-t2a)*1e6, (t4-t3)*1e6, (t5-t4)*1e6);
    
    return result;
}

extern "C" PyObject *ProjectForZ(PyObject *self, PyObject *args)
{
    PyArrayObject *_projection, *_HtsFull, *_xAxisMultipliers, *_yAxisMultipliers;
    int Nnum, fshapeY, fshapeX, rfshapeY, rfshapeX;
    double t1 = GetTime();
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
    
    double t2 = GetTime();
    for (int bb = 0; bb < HtsFull.Dims(0); bb++)
    {
        for (int aa = bb; aa < int(Nnum+1)/2; aa++)
        {
            int cent = int(Nnum/2);
            bool mirrorX = (bb != cent);
            bool mirrorY = (aa != cent);
            // TODO: we do not currently support the transpose here, although that can speed things up in certain circumstances (see projector.convolve()).
            // The only scenario where we could gain (by avoiding recalculating the FFT) is if the image array is square.
            // At the moment, we process the case with the transpose, but simply by recalculating the appropriate FFT(H) from scratch
            bool transpose = ((aa != bb) && (aa != (Nnum-bb-1)));

            double t0 = GetTime();
            JPythonArray2D<TYPE> fHtsFull = CalcFH(HtsFull[bb][aa], fshapeY, fshapeX, false);
            gTotalHTime += GetTime() - t0;
            ConvolvePart2(projection, bb, aa, Nnum, mirrorY, mirrorX, fHtsFull, xAxisMultipliers, yAxisMultipliers, accum);

            if (transpose)
            {
                t0 = GetTime();
                JPythonArray2D<TYPE> fHtsFull2 = CalcFH(HtsFull[bb][aa], fshapeY, fshapeX, true);
                gTotalHTime += GetTime() - t0;
                // Note that mx,my have been swapped here, which is necessary following the transpose. And bb,aa have been as well.
                ConvolvePart2(projection, aa, bb, Nnum, mirrorX, mirrorY, fHtsFull2, xAxisMultipliers, yAxisMultipliers, accum);
            }
        }
    }
    PyObject *ret = PyArray_Return(_accum);
    
    double t3 = GetTime();
#if TESTING
    printf("Took %lf (+ test setup %lf)\n", t3-t2, t2-t1);
    PrintStats(NULL, NULL);
    ResetStats(NULL, NULL);
#endif
    
    return ret;
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

static PyMethodDef symm_methods[] = {
	{"mirrorX", mirrorX, METH_VARARGS},
    {"mirrorY", mirrorY, METH_VARARGS},
    {"special_fftconvolve", special_fftconvolve, METH_VARARGS},
    {"special_fftconvolve_part1", special_fftconvolve_part1, METH_VARARGS},
    {"Convolve", Convolve, METH_VARARGS},
    {"ProjectForZ", ProjectForZ, METH_VARARGS},
    {"InverseRFFT", InverseRFFT, METH_VARARGS},
    {"PrintStats", PrintStats, METH_NOARGS},
    {"ResetStats", ResetStats, METH_NOARGS},
	{NULL,NULL} };



/* initialisation - register the methods with the Python interpreter */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef py_symmetry =
{
    PyModuleDef_HEAD_INIT,
    "py_symmetry", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symm_methods
};

PyMODINIT_FUNC PyInit_py_symmetry(void)
{
    import_array();
    return PyModule_Create(&py_symmetry);
}

#else

extern "C" void initpy_symmetry(void)
{
    (void) Py_InitModule("py_symmetry", symm_methods);
    import_array();
}

#endif
