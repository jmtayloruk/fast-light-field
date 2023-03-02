I have been puzzling over the fact that the Prevedel et al Matlab code uses a modified Richardson-Lucy algorithm that calculates the error in object space rather than image space.
I have not come across that anywhere else. Broxton describes a conventional R-L reconstruction.

I find that if I use a vanilla R-L deconvolution then I get major (additional) artefacts in the native focal plane, at the "corners" between individual lenslet footprints.
This seems to be caused by background light in the camera image, that lies outside the footprint of the lenslets where the PSF implies we would be very unlikely to get any photons.
As a consequence it is hard to find any object that matches the image, and we end up iterating towards an object that has these prominent artefacts,
because that's the only way we can put any energy into the regions of H(x) that "should" be dark.

A workaround is to define a "trust mask" that is 1 for image pixels where we would expect signal, and 0 for pixels that are expected to be very dim (in the absence of background/scatter/etc).
My code then effectively ignores those regions when it comes to calculating the error term and back-propagating it.

I am still curious as to the rationale behind the Prevedel et al modified R-L, which I have not seen anywhere else. I am sticking to that code as the default behaviour for my implementation.
Young-Gyu Yoon tells me he "I don't think it was a novel attempt to calculate the error in the back-projected x domain (but I cannot recall a specific paper that discusses this)".
Certainly I (JT) haven't managed to find any other discussion of this approach in the literature.
