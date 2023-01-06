## Introduction
This document briefly describes how to install and use the fast-light-field module to deconvolve light field microscopy images. This module gives output equivalent to the Matlab code of Prevedel et al, but runs ~10x as fast.

Inputs: one or more TIFF files containing light field microscopy camera images. For now, these **must** be *rectified* (i.e. they must have been preprocessed to ensure they have exactly Nmax pixels across each lenslet footprint). This rectification can be performed using Prevedel’s Matlab script.

Outputs: one or more 3D TIFF files containing volume reconstructions of your light field microscopy camera images.

These instructions assume a basic familiarity with the command line, and with python - and that you already have a basic python 3 (v3.5 or higher) installation on your computer.

To obtain the fast-light-field Python code, run:

```
git clone https://jmtaylor@bitbucket.org/jmtaylor/fast-light-field.git
```

## Installing
Once you have downloaded the fast-light-field directory, run the following command-line operations:

```
# Start in the fast-light-field directory
cd fast-light-field

# If you have cloned directly from the git repository, run the following commands to configure the git submodules
git submodule init
git submodule update

# Now you have a choice of whether you want to use a python virtual environment or not.
# If you don’t know what that is, just skip these next two commands
# IMPORTANT: if you do set up a virtual environment, it is vital that you omit the “--user” flag from all subsequent commands,
# or you may get hard-to-diagnose errors
python3 -m venv lff-venv
. ./lff-venv/bin/activate

# Install dependencies
python3 -m pip install --user -r requirements_minimum.txt
# Check that this has run successfully, printing “Successfully installed” followed by a list of module names. 
# There may be earlier error messages that can probably be ignored, but if it does not make it to “Successfully installed” 
# then something has gone wrong that needs fixing. See “troubleshooting”, below

# Optional, if you want GPU support
# Note that the actual cuda library also needs to be installed first (see cuda documentation).
# This step seems to be awkward, and I have a few possible suggestions under “troubleshooting”, below
python3 -m pip install --user cupy pycuda pynvml

# Get everything set up for the fast-light-field code (this will take 5-10 minutes)
python3 setup.py --user build self-test

# If all is well, you will get a lot of output on the command line, but ultimately it should end with a green line reading “== Self-tests complete (passed 24/24) ==“. 
# If anything goes wrong, please just drop me an email.

# Optional: run this command to get some benchmark measurements of performance on your system (takes about 5-10 minutes first time round, about 2 minutes for subsequent runs)
# Make sure your computer is otherwise idle - if it is doing something else while this runs, the benchmark numbers will be degraded.
# I would be very interested to see the output if you want to send it to me by email
python3 setup.py benchmark
```

### Troubleshooting - general python modules
Unfortunately, installing the required python modules can be temperamental, even for a relatively simple project like this. If you encounter error messages while installing the dependencies, you may need to tweak things a bit. Drop me an email if you run into problems you can’t sort. Some issues I have encountered:

numpy: `Python version >= 3.7 required`. The solution is to manually run something like `python3 -m pip install --user numpy==1.18.5` to install an older version of numpy that is compatible with your version of python.

imagecodecs: `imagecodecs/opj_color.c:45:22: fatal error: openjpeg.h: No such file or directory`. Solution is to manually run something like `python3 -m pip install --user tifffile==2019.7.2` to install an older version of tifffile.

After running these manual fixes, rerun the “install dependencies” command from the original install instructions (above) and hopefully it will succeed.

Error: `No module named xxx`. Are you running in a virtual environment? If so, you probably forgot to remote the `--user` flag from one of the commands you ran.

I have occasionally seen problems where skimage crashes when called from our code. I do not understand the root cause of that, but I think I fixed it by reinstalling skimage.

### Troubleshooting - cupy module
Unfortunately cupy seems rather fiddly to install. We have found that the following helps get past errors:
```
    python3 -m pip install --user wheel
    python3 -m pip install --user cupy==8.2.0
```

## Deconvolving light field images
Before you use this deconvolution code, you need to use the Prevedel Matlab code to rectify your light field images according to your experimental parameters - I have not yet got round to porting that (small) code to python. Once you have got your rectified images, you should just be able to run this deconvolution code. At the moment the code expects every image to be in a separate (2D) tiff file (which is the output format generated by Prevedel’s rectification script). 

## Calling script from the command line
You invoke my deconvolution script with a command like:
```
python3 deconvolve_time_series.py --psf "PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-156to156zspacing4Nnum19lambda520n1.33.mat" --dest DeconvolvedOutput --timepoints RectifiedInputData/*.tif
```

`DeconvolvedOutput` is the name of a folder into which your deconvolved images will be saved

`RectifiedInputData` is the name of a folder in which your rectified images can be found. This example deconvolves every image in the folder, but you can of course replace the `*.tif` to be more specific, and/or list more than one individual file you want to deconvolve

In this example there is a folder `PSFmatrix` which will contain the light field PSF. If the matrix file you specify does not exist, my code will automatically generate the file according to the parameters in the filename. These are the same as in Prevedel’s code:

	M - magnification
	NA - numerical aperture
	MLPitch - microlens pitch (µm)
	fml - microlens focal length (µm)
	from/to/zspacing - z stack dimensions (µm)
	Nnum - number of pixels in your rectified image,
			 across each microlens footprint
	lambda - wavelength (nm)
	n - refractive index of medium

	
Note that the matrix filename must follow the exact format shown: it should start with `fdnormPSFMatrix_`. It is also important not to have any trailing zeroes in the parameters, e.g. you should write `…lambda520n1.mat` not `…lambda520n1.0.mat`.

## Calling code as a python module
Although the entire fast-light-field project is not currently packaged as a formal python module of its own, you can still import individual .py files (e.g. hMatrix and lfdeconv) and call the functions from your own python code. The APIs are not yet formally documented, but the code in `deconvolve_time_series.py` etc provides a suitable demonstration of usage.

## Best performance
- Note that the first iteration will take a bit longer, as the code self-calibrates for best performance, but subsequent iterations will be faster.
- The code will run fastest if given significant numbers (e.g. 32) of individual timepoint images to deconvolve in parallel. 
- For most systems, GPU-based deconvolution will be fastest. The GPU-based code is not quite as extensively tested - it should be correct, but has not been tested on many different platforms, so let me know if you encounter any errors). Run on GPU by specifying `-m gpu` on the command line. 
- The command line option `-cacheFH` is a specialised option that is strictly for small reconstruction problems only. It will give faster performance (especially for small batch sizes) but has a dramatically higher memory requirement (may well exhaust available RAM).
- The CPU-based code should be able to handle very large volume reconstructions, but the GPU code may run out of memory (depending on the specs of your GPU).
- The code will use all available cores on the CPU, but does not currently farm out work across a cluster of computers (handle this manually by running different jobs on different computers)
- The code will not make use of multiple GPUs - let me know if you have this setup and we can work together to get this working.