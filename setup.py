# This is not actually a standard distutils setup.py script.
# Here we build and install the submodules, and optionally run the self-test routines
import sys, os, datetime
import benchmark

if __name__ == "__main__":
    args = sys.argv
    if (len(args) == 1): # We test against 1 because argv[0] is the script path
        print('No arguments passed - will run full setup')
        # For now I do not do the init stage by default, since the submodules are not publicly accessible.
        # I just need to provide people with the pre-downloaded git module.
        args = ['build', 'self-test', 'benchmark']
    
    if ('init' in args):
        # Set up git submodules
        for cmd in ['git submodule init', 'git submodule update']:
            ret = os.system(cmd)
            if (ret != 0):
                print('Command "%s" failed' % cmd)
                exit(ret)
    
    if ('build' in args):
        # Build custom python modules
        if '--user' in args:
            userFlag = '--user'
        else:
            userFlag = ''
        for subfolder in ['light-field-integrands', 'py_light_field']:
            print('Build %s' % subfolder)
            ret = os.system('cd %s; python3 setup.py install %s' % (subfolder, userFlag))
            if (ret != 0):
                print('Failed to build \"%s\" - terminating setup process' % subfolder)
                exit(ret)

    if ('self-test' in args):
        # Run self-tests
        print('\033[1;32m=== RUNNING SELF-TESTS ===\033[0m')
        print('This will take several minutes to complete')
        import lfdeconv
        import projector as proj
        try:
            import cupy as cp
            gpuAvailable = True
        except ImportError:
            print('No GPU support detected')
            gpuAvailable = False

        # Tests that verify fast implementations against my slow reference python implementation
        testOutcomes = proj.selfTest(verbose=False)

        # Tests that compare against reconstructions from Prevedel's MATLAB
        for cacheFH in [[], ['cacheFH']]:
            args = ['basic', 'full', 'parallel', 'parallel-threading'] + cacheFH
            testOutcomes += lfdeconv.main(args)
            if gpuAvailable:
                testOutcomes += lfdeconv.main(args, projectorClass=proj.Projector_gpuHelpers)

        if testOutcomes[0] == testOutcomes[1]:
            print('\033[1;32m')
        else:
            print('\033[1;31m')
        print('== Self-tests complete (passed %d/%d) ==' % (testOutcomes[0], testOutcomes[1]))
        print('\033[0m')

    if ('benchmark' in args):
        # Run benchmarks
        result = benchmark.main()
        with open('benchmarks.txt', 'a') as f:
            f.write("%s: %s\n" % (datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), result))
