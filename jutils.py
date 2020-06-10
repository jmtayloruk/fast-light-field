import resource, psutil
import numpy as np

def CheckComparison(arrayA, arrayB, maxAcceptableDifference, description="Test result", shouldBe=None):
    # Checks that arrayA and arrayB agree to within a reasonable absolute tolerance
    comparison = np.max(np.abs(arrayA - arrayB))
    if shouldBe is None:
        shouldBe = "< %f"%maxAcceptableDifference
    print("%s (should be %s): %f" % (description, shouldBe, comparison))
    if (comparison > maxAcceptableDifference):
        print(" -> WARNING: disagreement detected")
    else:
        print(" -> OK")

def noProgressBar(work, desc=None, leave=True, **kwargs):
    # Dummy drop-in function to be used in place of 'tqdm' when we want to suppress a progress bar
    return work

def cpuTime(kind):
    rus = resource.getrusage(resource.RUSAGE_SELF)
    ruc = resource.getrusage(resource.RUSAGE_CHILDREN)
    if (kind == 'self'):
        return np.array([rus.ru_utime, rus.ru_stime])
    elif (kind == 'children'):
        return np.array([ruc.ru_utime, ruc.ru_stime])
    else:
        return np.array([rus.ru_utime+ruc.ru_utime, rus.ru_stime+ruc.ru_stime])

def isnotebook():
    # Returns True if the current code is running within a Jupyter notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# This next bit creates the object tqdm_alias.
# If we are running in a notebook, the object maps to tqdm_notebook
# Otherwise it maps to vanilla tqdm
# Callers can import the correct one into their own environment via:
#   from jutils import tqdm_alias as tqdm
if isnotebook():
    from tqdm import tqdm_notebook as tqdm_alias
else:
    from tqdm import tqdm as tqdm_alias

def PhysicalCoreCount():
    # On my mac pro, this cpu_count() call correctly returns 8 (psutil version 5.2.2).
    # I need to check if it works correctly on all machines - I thought on my laptop it gave the wrong number.
    return psutil.cpu_count(logical=False)
