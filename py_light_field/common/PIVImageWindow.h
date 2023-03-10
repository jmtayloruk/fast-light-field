//
//  PIVImageWindow.h
//  Spim Interface
//
//  Created by Jonny Taylor on 30/10/2016.
//
//

#ifndef __Spim_Interface__PIVImageWindow__
#define __Spim_Interface__PIVImageWindow__

#include <cctype>
#if __OBJC__
	#import "jCocoaImageUtils.h"
#else
	typedef void NSBitmapImageRep;
#endif
#include "GeometryObjectsC.h"
#include <complex>
#include "jCoord.h"

template<class TYPE> struct ImageWindow
{
	TYPE *baseAddr;
    bool baseAddrAllocated;
	int width, height, elementsPerRow;
	NSBitmapImageRep *retainedBitmap;
	
	ImageWindow(void) : baseAddr(NULL), baseAddrAllocated(false), width(0), height(0), elementsPerRow(0), retainedBitmap(NULL)
	{
	}
	
	ImageWindow(int x, int y) : width(x), height(y), elementsPerRow(x)
	{
		retainedBitmap = NULL;
		baseAddr = new TYPE[width*height];
        baseAddrAllocated = true;
	}
	
#if __OBJC__
	ImageWindow(const NSImage *srcImage)
	{
		Construct(RawBitmapFromImage(srcImage));
	}
	
	ImageWindow(NSBitmapImageRep *srcBitmap)
	{
		Construct(srcBitmap);
	}
	
	void Construct(NSBitmapImageRep *srcBitmap)
	{
        ALWAYS_ASSERT(srcBitmap != nil);
		retainedBitmap = [srcBitmap retain];
		width = (int)retainedBitmap.pixelsWide;
		height = (int)retainedBitmap.pixelsHigh;
		ALWAYS_ASSERT(retainedBitmap.bitsPerPixel == 8*sizeof(TYPE));
		elementsPerRow = (int)(retainedBitmap.bytesPerRow / sizeof(TYPE));
		baseAddr = (TYPE*)retainedBitmap.bitmapData;
        baseAddrAllocated = false;
	}
#endif
	
	void ReallocateImageWithPadding(int paddingX, int paddingY)
	{
		// Add black padding around the perimeter of the image.
		// If we previously had a view into an actual NSBitmap, we will now instead have a padded copy
		size_t newSize = (width+2*paddingX) * (height+2*paddingY);
		TYPE *newBaseAddr = new TYPE[newSize];
		memset(newBaseAddr, 0, newSize * sizeof(TYPE));
		for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
				newBaseAddr[(j+paddingY)*(width+2*paddingX) + i+paddingX] = PixelXY(i, j);
		
#if __OBJC__
		if (retainedBitmap != NULL)
			[retainedBitmap release];
        retainedBitmap = NULL;
#endif
        if (baseAddrAllocated)
            delete[] baseAddr;
		baseAddr = newBaseAddr;
        baseAddrAllocated = true;
		width = width + 2*paddingX;
		height = height + 2*paddingY;
        elementsPerRow = width;
	}
	
    template<class SRC_TYPE> void AllocateCopyOf(const ImageWindow<SRC_TYPE> &srcWindow)
    {
        ALWAYS_ASSERT(baseAddr == NULL);
        ALWAYS_ASSERT(retainedBitmap == NULL);
        width = srcWindow.width;
        height = srcWindow.height;
        elementsPerRow = width;
        retainedBitmap = NULL;
		if (baseAddrAllocated)
			delete[] baseAddr;
        baseAddr = new TYPE[width*height];
        baseAddrAllocated = true;
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
                SetXY(i, j, srcWindow.PixelXY(i, j));
    }
		
    template<class SRC_TYPE> void MipWith(const ImageWindow<SRC_TYPE> &otherWindow)
    {
		ALWAYS_ASSERT(width == otherWindow.width);
		ALWAYS_ASSERT(height == otherWindow.height);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
                SetXY(i, j, MAX(PixelXY(i, j), otherWindow.PixelXY(i, j)));
    }
	
	ImageWindow(const ImageWindow<TYPE> &inVal) : baseAddr(NULL), baseAddrAllocated(false), width(0), height(0), elementsPerRow(0), retainedBitmap(NULL)
	{
		AllocateCopyOf(inVal);
	}
	
	ImageWindow<TYPE> &operator=(ImageWindow<TYPE> &inVal)
	{
		AllocateCopyOf(inVal);
		return *this;
	}

	virtual ~ImageWindow()
	{
#if __OBJC__
		if (retainedBitmap != NULL)
			[retainedBitmap release];
#endif
        if (baseAddrAllocated)
            delete[] baseAddr;
	}
    
    void SetXY(int x, int y, TYPE val)
    {
        baseAddr[y*elementsPerRow+x] = val;
    }
    
    TYPE PixelXY(int x, int y) const
    {
        return baseAddr[y*elementsPerRow+x];
    }
    
    TYPE *PixelXYAddr(int x, int y) const
    {
        return &baseAddr[y*elementsPerRow+x];
    }
    
    double SumIntensity(void)
    {
        double total = 0.0;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                total += PixelXY(x, y);
        return total;
    }
    
    void SubtractMean(void)
    {
        ALWAYS_ASSERT(sizeof(TYPE) == 8);       // Crude check that we are of double type. Not sure I trust this otherwise (can only subtract an integer value...)
        double mean = SumIntensity() / (height * width);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
                SetXY(i, j, PixelXY(i, j) - mean);
    }
	
    void ZeroAll(void)
    {
        for (int i = 0; i < width*height; i++)
            baseAddr[i] = 0;
    }
	
	coord2 CalculateFlowPeak(void) const;
    IntegerPoint CalculateFlowPeakInteger(void) const;
    double CalculateSNR(int threshold) const;
    
    void InitWithGaussianAt(double x, double y, double amp, double w)
    {
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                double val = amp * exp(-(SQUARE(x-i) + SQUARE(y-j)) / SQUARE(w));
                SetXY(i, j, val);
            }
    }
    
    bool TileOffsetValid(int i, int j, int tileWidth, int tileHeight, int tileOffsetX, int tileOffsetY)
	{
		return ((i >= 0) && (j >= 0) &&
				(i*tileOffsetX+tileWidth <= width) && (j*tileOffsetY+tileHeight <= height));
	}
	
    ImageWindow<TYPE> &operator+=(ImageWindow<TYPE> &other)
    {
        ALWAYS_ASSERT(other.width == width);
        ALWAYS_ASSERT(other.height == height);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
                SetXY(i, j, PixelXY(i, j) + other.PixelXY(i, j));
        return (*this);
    }
    
	void GetWindowOffset(ImageWindow<TYPE> &result, int i, int j, int tileOffsetX, int tileOffsetY, int tileWidth, int tileHeight, int windowWidth, int windowHeight) const
	{
		int extraDeltaX = (tileWidth - windowWidth)/2;
        int extraDeltaY = (tileHeight - windowHeight)/2;
		result.baseAddr = baseAddr + (j*tileOffsetY+extraDeltaY)*elementsPerRow + (i*tileOffsetX+extraDeltaX);
        result.width = windowWidth;
        result.height = windowHeight;
		result.elementsPerRow = elementsPerRow;
	}
	
#if __OBJC__
	NSBitmapImageRep *Bitmap(double gain = 1.0)
	{
		NSBitmapImageRep *result = [[NSBitmapImageRep alloc]
									initWithBitmapDataPlanes:NULL
									pixelsWide:width
									pixelsHigh:height
									bitsPerSample:8*sizeof(TYPE)
									samplesPerPixel:1
									hasAlpha:NO
									isPlanar:NO
									colorSpaceName:NSCalibratedWhiteColorSpace
									bytesPerRow:width*sizeof(TYPE)
									bitsPerPixel:0];
        ALWAYS_ASSERT(result != nil);
		TYPE *destData = (TYPE *)result.bitmapData;
        ALWAYS_ASSERT(destData != nil);
		
        if (gain != 1.0)
        {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    destData[y*width+x] = (TYPE)LIMIT(this->baseAddr[y*this->elementsPerRow+x] * gain, 0.0, ((1<<(8*sizeof(TYPE)))-1.0));
        }
        else
        {
            for (int y = 0; y < height; y++)
                memcpy(destData + y*width, this->baseAddr + y*this->elementsPerRow, width * sizeof(TYPE));
        }
		
		return [result autorelease];
	}
	
	NSBitmapImageRep *NormalizedBitmap(double normalization = 0)
	{
		NSBitmapImageRep *result = [[NSBitmapImageRep alloc]
									initWithBitmapDataPlanes:NULL
									pixelsWide:width
									pixelsHigh:height
									bitsPerSample:16
									samplesPerPixel:1
									hasAlpha:NO
									isPlanar:NO
									colorSpaceName:NSCalibratedWhiteColorSpace
									bytesPerRow:width*2
									bitsPerPixel:0];
		unsigned short *destData = (unsigned short *)[result bitmapData];
		
		TYPE maxVal = 0;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				maxVal = MAX(maxVal, PixelXY(x, y));
		if (normalization == 0)
			normalization = maxVal;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				destData[y*width+x] = (unsigned short)(PixelXY(x, y) / maxVal * 65535);
		
		return [result autorelease];
	}
#endif
};

enum
{
	kCorrelationSAD = 1,
	kCorrelationSSD,
	kCorrelationDCC
};

template<int correlationType, class TYPE> void CrossCorrelateImageWindows(ImageWindow<TYPE> &window1, ImageWindow<TYPE> &window2, ImageWindow<double> &result);

template<class TYPE> void CrossCorrelateImageWindows(ImageWindow<TYPE> &window1, ImageWindow<TYPE> &window2, ImageWindow<double> &result, int correlationType)
{
	switch (correlationType)
	{
		case kCorrelationSAD:
			CrossCorrelateImageWindows<kCorrelationSAD>(window1, window2, result);
			break;
		case kCorrelationSSD:
			CrossCorrelateImageWindows<kCorrelationSSD>(window1, window2, result);
			break;
		case kCorrelationDCC:
			CrossCorrelateImageWindows<kCorrelationDCC>(window1, window2, result);
			break;
		default:
			ALWAYS_ASSERT(0);
	}
}

#endif /* defined(__Spim_Interface__PIVImageWindow__) */
