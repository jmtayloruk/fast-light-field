## Introduction
This document briefly describes how to install and use the fast-light-field module to deconvolve light field microscopy images. 

The mathematical basis of our fast light field deconvolution algorithms is described in an upcoming manuscript.
Our *implementation* of this as fast Python computer code for light field deconvolution (i.e. this repository) has benefitted from the Matlab code written and publicly posted by Prevedel, Yoon *et al* in conjuction with their publication:
"Simultaneous whole-animal 3D-imaging of neuronal activity using light field microscopy", R. Prevedel, Y.-G. Yoon *et al*, Nature Methods **11** 727-730 (2014).
Our computer code for deconvolution uses the same notation as Prevedel *et al* but is basically reimplemented from scratch.
The code for PSF generation is Python translation of their code, with bug fixes and performance improvements.
The code for rectification is a direct Python translation of their code.

Our code gives output equivalent to the Matlab code of Prevedel *et al*, but runs 8-35x as fast.

Inputs: one or more TIFF files containing **rectified** light field microscopy camera images. See below for instructions for rectification.

Outputs: one or more 3D TIFF files containing volume reconstructions of your light field microscopy camera images.

These instructions assume a basic familiarity with the command line, and with python - and that you already have a basic python 3 (v3.5 or higher) installation on your computer.

To obtain the fast-light-field Python code, run:

```
git clone https://github.com/jmtayloruk/fast-light-field.git
```

## Installing
Once you have downloaded the fast-light-field directory, run the following command-line operations:

```
# Start in the fast-light-field directory
cd fast-light-field

# OPTIONAL: now you have a choice of whether you want to use a python virtual environment or not.
# If you don’t know what that is, just skip these next two commands
# IMPORTANT: if you do set up a virtual environment, it is vital that you omit the “--user” flag from all subsequent commands,
# or you may get hard-to-diagnose errors
python3 -m venv flf-venv
. ./flf-venv/bin/activate

# Install dependencies
# You should check that this next command has run successfully - it should “Successfully installed” followed by a list of module names. 
# There may be earlier error messages that can probably be ignored, but if it does not make it to “Successfully installed” 
# then something has gone wrong that needs fixing. See “troubleshooting”, below
python3 -m pip install --user -r requirements_minimum.txt

# OPTIONAL: if you want GPU support, run this next command
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

pyfftw: `Could not find the FFTW header 'fftw3.h'`. Are you installing on an Apple M1 machine? 
If so, [follow these instructions](https://github.com/andrej5elin/howto_fftw_apple_silicon) to manually install pyfftw.

pyfftw: if you encounter any other issues, then actually this is installation is not required as part of critical high-performance code.
Edit `requirements_minimum.txt` to remove it, change the `if False` in `myfft.py` to `if True`.

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

### Rectification
First you must ensure your images are **rectified** (integer sampling across the footprint of each lenslet).
To rectify your images, you can run the script `rectify.py` (run `rectify.py --help` for usage instructions).
The script parameters match those in Prevedel *et al*'s GUI, and the output of my script will be bit-for-bit identical for 8-bit input data.
The only slight difference in behaviour is that my script generates 16-bit output data if given 16-bit input data (where Prevedel *et al* generate 8-bit output data).

To determine the correct rectification parameters, we refer the reader to [http://graphics.stanford.edu/software/LFDisplay](http://graphics.stanford.edu/software/LFDisplay), as Prevedel *et al* do in their supplementary material.

Once you have got your rectified images, follow the instructions below to run the fast deconvolution code.
At the moment the code expects every image to be in a separate (2D) tiff file (which is the output format generated by the rectification scripts). 

### Calling script from the command line
You invoke my Python deconvolution script with a command like:
```
python3 deconvolve_time_series.py --psf "PSFmatrix/fdnormPSFMatrix_M22.2NA0.5MLPitch125fml3125from-156to156zspacing4Nnum19lambda520n1.33.mat" --dest DeconvolvedOutput --timepoints RectifiedInputData/*.tif
```

`DeconvolvedOutput` is the name of a folder into which your deconvolved images will be saved

`RectifiedInputData` is the name of a folder in which your rectified images can be found. This example deconvolves every image in the folder, but you can of course replace the `*.tif` to be more specific, and/or list more than one individual file you want to deconvolve

In this example there is a folder `PSFmatrix` which will contain the light field PSF. If the matrix file you specify does not exist, my code will automatically generate the file according to the parameters in the filename. These are the same as in Prevedel *et al*’s code:

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

### Calling code as a python module
Although the entire fast-light-field project is not currently packaged as a formal python module of its own, you can still import individual .py files (e.g. hMatrix and lfdeconv) and call the functions from your own python code. The APIs are not yet formally documented, but the code in `deconvolve_time_series.py` etc provides a suitable demonstration of usage.

### Best performance
- Note that the first iteration will take a bit longer (potentially a *lot* longer on the GPU), as the code self-calibrates for best performance, but subsequent iterations will be faster.
- The code will run fastest if given significant numbers (e.g. 32) of individual timepoint images to deconvolve in parallel. 
- For most systems, GPU-based deconvolution will be fastest. The GPU-based code is not quite as extensively tested - it should be correct, but has not been tested on many different platforms, so let me know if you encounter any errors). Run on GPU by specifying `-m gpu` on the command line. 
- The command line option `-cacheFH` is a specialised option that is strictly for small reconstruction problems only. It will give faster performance (especially for small batch sizes) but has a dramatically higher memory requirement (may well exhaust available RAM).
- The CPU-based code should be able to handle very large volume reconstructions, but the GPU code may run out of memory (depending on the specs of your GPU).
- The code will use all available cores on the CPU, but does not currently farm out work across a cluster of computers (handle this manually by running different jobs on different computers)
- The code will not make use of multiple GPUs - let me know if you have this setup and we can work together to get this working.

## Benchmarking

As mentioned in the installation instructions, you can get some quick-ish benchmark measurements by running `python3 setup.py benchmark`.
For more in-depth benchmarks on realistically-sized tasks (as presented in our manuscript), see `benchmarking.ipynb`.

## Integrating with Matlab code

I have a separate repository [https://github.com/jmtayloruk/prevedel-yoon-matlab-light-field](https://github.com/jmtayloruk/prevedel-yoon-matlab-light-field)
that documents and demonstrates how to call my fast code from Matlab, if you have existing analysis pipelines written as Matlab scripts.
This requires some additional RAM, but allows the full speed of my implementation to be accessed from Matlab code.
