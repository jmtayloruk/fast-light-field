import tifffile
import numpy as np

def LoadLightFieldTiff(imagePath, bigger=(1,1)):
    # Load the tiff file from disk
    image = tifffile.imread(imagePath)
    assert(len(image.shape) == 2)
    
    # For compatibility with Prevedel's matlab code, I transpose the image we have loaded.
    # For convenience, throughout my code I index x and y in the same order Prevedel does.
    # Given that Matlab indexes arrays in the opposite order to Python, that would seem on the
    # face of it to cause problems, that would require us to transpose the image when we load it here
    # (and transpose it back again after we have completed the reconstruction).
    # However I am 99% certain that symmetries in the light field PSF mean that we can get away without
    # doing any of that transposing - hence why I have disabled this next line of code.
    #image = image.transpose().copy()
    
    # Convert to float32 and return.
    # If caller specifies 'bigger', we tile the image up to a larger size.
    # This is purely just a convenient way of generating larger images for performance stress-testing of my code.
    return np.tile(image.astype('float32'), bigger)
