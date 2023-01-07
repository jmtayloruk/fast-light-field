import numpy as np
import tifffile
import scipy.interpolate
import sys, glob, os
import argparse
from pathlib import Path

def RectifyImage(IMG_BW, xCenter_m, yCenter_m, dx, Nnum, Crop, XcutLeft, XcutRight, YcutUp, YcutDown):
    ''' 
        JT: I have translated this function directly from the Prevedel et al matlab code
        This isn't how I would have written it, but my main priority is to ensure identical
        output to their code. For 8-bit input it matches exactly. For 16-bit input the Matlab
        code outputs an 8-bit rectified output, but my code will generate a 16-bit output.
        
        As always the translation is complicated by the differences in Matlab vs Python array indexing
        Variables suffixed _m are one-based array indexes (Matlab indexing),
        and variables suffixed _p are zero-based array indexes (Python indexing).
        
        IMG_BW         ndarray     Input image (which may be a 2D or 3D)
        xCenter_m      int         X coordinate (matlab indexing) of the origin for the center lenslet
        yCenter_m      int         Y coordinate (matlab indexing) of the origin for the center lenslet
                                   (although I don't think the coordinates need to be for the center lenslet,
                                    they can be for any lenslet in fact)
        dx             float       Lenslet period in the raw input image
        Nnum           float       Desired sampling (pixels across a lenslet) for the rectified image
                                   Must be an odd number
        Crop           bool        Should the rectified image be cropped?
        XcutLeft       int         Number of lenslets to crop from the left boundary of the rectified image
        XcutRight, YcutUp, YcutDown   Equivalents for the other boundaries. All must be positive integers.
    '''
    if Nnum - 2*(Nnum//2) != 1:
        raise ValueError('Nnum must be an odd integer')
    M = Nnum    # Matlab code uses M, but their GUI uses N and their deconvolution code uses Nnum
                # so for my own sanity I have made my function parameter be Nnum
    dy = dx
    Mdiff = M//2

    # Resample the image
    # Note the ±1e-10 is a hack to get np.arange to do "up to and including" the endpoint to match Matlab behaviour
    Xresample_m = np.append(np.arange(xCenter_m+1, 1-1e-10, -dx/M)[::-1],
                            np.arange((xCenter_m+1)+dx/M, IMG_BW.shape[-1]+1e-10, dx/M))
    Yresample_m = np.append(np.arange(yCenter_m+1, 1-1e-10, -dy/M)[::-1],
                            np.arange((yCenter_m+1)+dy/M, IMG_BW.shape[-2]+1e-10, dy/M))
    _y_p = np.arange(IMG_BW.shape[-2])
    _x_p = np.arange(IMG_BW.shape[-1])
    X_p,Y_p = np.meshgrid(_x_p, _y_p) # redundant?
    Xq_m,Yq_m = np.meshgrid(Xresample_m, Yresample_m) # can be made redundant?

    XqCenterInit_m = np.where(Xq_m[0,:] == (xCenter_m+1))[0][0] - Mdiff + 1   # Final +1 makes the result a matlab index
    XqInit_m = XqCenterInit_m -  M * (XqCenterInit_m//M) + M
    XqEnd_m = M * ((Xq_m.shape[1]-XqInit_m+1)//M)
    YqCenterInit_m = np.where(Yq_m[:,0] == (yCenter_m+1))[0][0] - Mdiff + 1   # Final +1 makes the result a matlab index 
    YqInit_m = YqCenterInit_m -  M* (YqCenterInit_m//M) + M
    YqEnd_m = M * ((Yq_m.shape[0]-YqInit_m+1)//M)

    XresampleQ_m = Xresample_m[XqInit_m-1:]
    YresampleQ_m = Yresample_m[YqInit_m-1:]
    Yqq_p,Xqq_p = np.meshgrid(YresampleQ_m-1, XresampleQ_m-1) # redundant?

    interpolator = scipy.interpolate.RectBivariateSpline(_y_p, _x_p, IMG_BW, kx=1, ky=1);
    IMG_RESAMPLE = interpolator(YresampleQ_m-1, XresampleQ_m-1)
    IMG_RESAMPLE_crop1 = IMG_RESAMPLE[0:M*((IMG_RESAMPLE.shape[0]-YqInit_m)//M),
                                      0:M*((IMG_RESAMPLE.shape[1]-XqInit_m)//M)]  # JT: not sure if I need a ±1 on XqInit_m

    # Crop the right portion
    if Crop:
        XsizeML = IMG_RESAMPLE_crop1.shape[1]//M # Expected to be integer multiple anyway, but cast to int
        YsizeML = IMG_RESAMPLE_crop1.shape[0]//M # Expected to be integer multiple anyway, but cast to int
        if (XcutLeft + XcutRight) >= XsizeML:
            raise ValueError('X-cut range is larger than the x-size of image')
        if (YcutUp + YcutDown) >= YsizeML:
            raise ValueError('Y-cut range is larger than the y-size of image')

        Xrange_p = range(XcutLeft, XsizeML-XcutRight)
        Yrange_p = range(YcutUp, YsizeML-YcutDown)
        IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1[(Yrange_p[0])*M : (Yrange_p[-1]+1)*M,
                                                (Xrange_p[0])*M : (Xrange_p[-1]+1)*M]
    else:
        IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1

    return IMG_RESAMPLE_crop2

def RectifyImageFiles(files, outputDir, **kwargs):
    '''
        Run rectification for a list of image files, saving the results to outputDir
        
        files        list(str)   List of input file paths (which may contain wildcards)
        outputDir    str         Output file path (img.tif will be saved to outputDir/img_X3.tif if Nnum=3)
        kwargs                   Other arguments as required for RectifyImage
    '''
    for g in files:
        paths = sorted(glob.glob(g))
        if len(paths) == 0:
            print(f"WARNING: no file found matching {g}")
        for p in paths:
            print(f"Processing {p}")
            inputImage = tifffile.imread(p)
            if len(inputImage.shape) == 2:
                rect = RectifyImage(inputImage, **kwargs)
                dirPath, imPath = os.path.split(p)
                inputImage = tifffile.imread(p)
                fn,ext = os.path.splitext(imPath)
                destFilePath = os.path.join(outputDir, f"{fn}_N{kwargs['Nnum']}{ext}")
                Path(outputDir).mkdir(parents=True, exist_ok=True)
                tifffile.imsave(destFilePath, np.around(rect).astype(inputImage.dtype))
            else:
                print("  WARNING: SKIPPING FILE because it is not a single 2D image. In line with Prevedel et al's code, we require each timepoint to be in a separate .tif file")

def RectifyWithArgList(raw_args):
    '''
        Run rectification based on command-line arguments (run script with --help for argument list)
    '''
    parser = argparse.ArgumentParser(description="Rectify raw light field images ready for deconvolution",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=False)
    # Some gymnastics to have required and optional arguments (https://stackoverflow.com/questions/24180527/argparse-required-arguments-listed-under-optional-arguments)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    # Add help argument back in
    optional.add_argument('-h',
                          '--help',
                          action='help',
                          default=argparse.SUPPRESS,
                          help='show this help message and exit')
    parser.add_argument("files", nargs='+', help="List of file(s) to rectify (which may contain wildcards)")
        #parser.add_argument("settings", help="path to .json file containing settings")
    required.add_argument("-x", "--x-center", dest="xCenter_m", type=float, required=True, help="X origin of center lenslet array (matlab indexing of raw image pixels)")
    required.add_argument("-y", "--y-center", dest="yCenter_m", type=float, required=True, help="Y origin of center lenslet array (matlab indexing of raw image pixels)")
    required.add_argument("-d", "--dx", dest="dx", type=float, required=True, help="Period of lenslet array in raw image pixels")
    required.add_argument("-n", "--Nnum", dest="Nnum", type=int, required=True, help="Resampled period of lenslet array (odd integer)")
    optional.add_argument("-c", "--crop", dest="Crop", action="store_true", help="Crop rectified image")
    optional.add_argument("-l", "--crop-left", dest="XcutLeft", type=int, default=0, help="Number of lenslets to crop from the left boundary of the rectified image")
    optional.add_argument("-r", "--crop-right", dest="XcutRight", type=int, default=0, help="Number of lenslets to crop from the right boundary of the rectified image")
    optional.add_argument("-t", "--crop-top", dest="YcutUp", type=int, default=0, help="Number of lenslets to crop from the top boundary of the rectified image")
    optional.add_argument("-b", "--crop-bottom", dest="YcutDown", type=int, default=0, help="Number of lenslets to crop from the bottom boundary of the rectified image")
    optional.add_argument("-o", "--output-dir", dest="outputDir", default="Data/02_Rectified", help="Directory in which to save rectified images")

    args = parser.parse_args(raw_args)
    RectifyImageFiles(**vars(args))


if __name__ == "__main__":
    RectifyWithArgList(sys.argv[1:])
