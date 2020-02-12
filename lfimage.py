import tifffile
import numpy as np

def LoadLightFieldTiff(imagePath, bigger=(1,1)):
    # Load the tiff file from disk
    image = tifffile.imread(imagePath)
    assert(len(image.shape) == 2)
    # For compatibility with Prevedel's matlab code, I transpose the image we have loaded.
    # This allows me to index it in the same order Prevedel does (given that Matlab indexes arrays in the opposite order to Python)
    # TODO: that said, this is a bit of a confusing and unhelpful way to deal with that issue.
    # I should probably handle it better somehow - but for now I am leaving all the code as-is since it works!
    # The copy is to ensure contiguous strides in the final dimension, which my C code expects and requires
    image = image.transpose().copy()
    # Convert to float32 and return.
    # If caller specifies 'bigger', we tile the image up to a larger size.
    # This is purely just a convenient way of generating larger images for performance stress-testing of my code.
    return np.tile(image.astype('float32'), bigger)
