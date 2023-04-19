import numpy as np

""" Relative imports """
from ..misc import wavelength_from_energy

def calculate_fresnel_number(energy,pixel_size,sample_detector_distance,magnification=1,source_sample_distance=0):
    wavelength = wavelength_from_energy(energy) # meters
    if magnification != 1:
        magnification = (source_sample_distance+source_sample_distance)/source_sample_distance
    return -(pixel_size**2) / (wavelength * sample_detector_distance * magnification)

def Propagate(img, fresnel_number): # Probe propagation
        """ Frunction for free space propagation of the probe in the Fraunhoffer regime

        See paper `Memory and CPU efficient computation of the Fresnel free-space propagator in Fourier optics simulations <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28750&id=420820>`_.
        Args:
                img (array): probe
                fresnel_number (float): Fresnel number

        Returns:
                [type]: [description]
        """    
        hs = img.shape[-1] // 2
        ar = np.arange(-hs, hs) / float(2 * hs)
        xx, yy = np.meshgrid(ar, ar)
        g = np.exp(-1j * np.pi / fresnel_number * (xx ** 2 + yy ** 2))

        return np.fft.ifft2(np.fft.fft2(img) * np.fft.fftshift(g))


    
