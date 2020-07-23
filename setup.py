# This is not actually a standard distutils setup.py script.
# Here we build and install the submodules, and optionally run the self-test routines
import sys, os

if __name__ == "__main__":
    if ('init' in sys.argv):
        # Set up git submodules
        for cmd in ['git submodule init', 'git submodule update']:
            ret = os.system(cmd)
            if (ret != 0):
                print('Command "%s" failed' % cmd)
                exit(ret)
    
    if ('build' in sys.argv):
        # Build custom python modules
        for subfolder in ['light-field-integrands', 'py_light_field']:
            print('Build %s' % subfolder)
            ret = os.system('cd %s; python setup.py install' % subfolder)
            if (ret != 0):
                print('Failed to build \"%s\" - terminating setup process' % subfolder)
                exit(ret)

    if ('self-test' in sys.argv):
        # Run self-tests
        print('=== RUNNING SELF-TESTS ===')
        print('This will take several minutes to complete')
        import lfdeconv
        import projector as proj
        try:
            import cupy as cp
            gpuAvailable = True
        except:
            print('No GPU support detected')
            gpuAvailable = False

        # Tests that verify fast implementations against my slow reference python implementation
        testOutcomes = proj.selfTest()

        # Tests that compare against reconstructions from Prevedel's MATLAB
        for cacheFH in [[], ['cacheFH']]:
            args = ['basic', 'full', 'parallel', 'parallel-threading'] + cacheFH
            testOutcomes += lfdeconv.main(args)
            if gpuAvailable:
                testOutcomes += lfdeconv.main(args, projectorClass=proj.Projector_gpuHelpers)

        print('== Self-tests complete (passed %d/%d) ==' % (testOutcomes[0], testOutcomes[1]))
