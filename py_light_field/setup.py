from distutils.core import setup, Extension
import numpy	# So we can work out where the numpy headers live!
import platform
import os, sys

# TODO: I don't know what the right way is to express dependencies in a setup.py file, but this here is not good practice.
# I should improve it
# TODO: Currently if we run 'setup.py clean' then this *builds* fftw!

# Check dependencies have been built
if not os.path.exists('fftw-3.3.8/.libs/libfftw3f.a'):
    print('=== NOTE: building FFTW. This may take some time ===')
    if not os.path.exists('fftw-3.3.8'):
        os.system('gzip -d < fftw-3.3.8.tar.gz | tar -x')
    os.system('cd fftw-3.3.8; ./configure CFLAGS=-fPIC --enable-float --enable-threads; make -j$(nproc); cd ..')
if not os.path.exists('fftw-3.3.8/.libs/libfftw3f.a'):
    raise RuntimeError('An error occurred while building FFTW')
    exit()

# Work out if we should be building a 32 or 64 bit library
# Apparently this "can be a bit fragile" on OS X:
# http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
# but I'll try it and see if it works out ok for now.
archInfo = platform.architecture()
if (archInfo[0] == '32bit'):
	ARCH = ['-march=native', '-arch', 'i386']
elif ('macOS' in platform.platform()) and (platform.processor() == 'arm'):
    ARCH = ['-arch', 'apple-a12']
else:
	ARCH = ['-march=native', '-arch', 'x86_64']

# Determine if the -arch parameter is actually even available on this platform,
# by running a dummy gcc command that includes that option
# If it is not, then we will not include any arch-related options at all for gcc.
theString = 'gcc ' + ARCH[0] + ' ' + ARCH[1] + ' -E -dM - < /dev/null > /dev/null 2>&1'
result = os.system(theString)
if (result != 0):
	ARCH = []

BUILD_MODULES = []

# Note: using -Ofast (and/or -ffast-math -funsafe-math-optimizations) does not seem to offer any additional speed gain beyond -O3.
# Note: -flax-vector-conversions is required to compile on some architectures (e.g. beag-shuil).
# I need to improve my VectorTypes.h, but I'm not even using that in this code, so that's a problem for another day!
py_light_field = Extension('py_light_field',
	include_dirs = ['/usr/local/include', numpy.get_include()],
	sources = ['py_light_field.cpp', 'common/jPythonArray.cpp', 'common/jPythonCommon.cpp', 'common/jMutex.cpp', 'common/jAssert.cpp', 'common/DebugPrintf_Unix.cpp'],
	extra_link_args = ARCH + ['-Lfftw-3.3.8/.libs', '-Lfftw-3.3.8/threads/.libs', 'fftw-3.3.8/.libs/libfftw3f.a', 'fftw-3.3.8/threads/.libs/libfftw3f_threads.a'],
	extra_compile_args = ['-O3', '-flax-vector-conversions', '-std=c++11'] + ARCH
)
BUILD_MODULES.append(py_light_field)

setup (name='py_light_field',
       version='1.0.0',
       description='Utility module used by fast-light-field, providing optimised C-based projection operations',
       author='Jonathan Taylor, University of Glasgow',
       url='https://github.com/jmtayloruk/fast-light-field',
       ext_modules = BUILD_MODULES,
       install_requires=[])
