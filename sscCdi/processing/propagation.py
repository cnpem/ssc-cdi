# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################



import numpy as np
import cupy as cp
from tqdm import tqdm

""" Relative imports """
from ..misc import wavelength_meters_from_energy_keV

def fresnel_propagator_cone_beam(wavefront, wavelength, pixel_size, sample_to_detector_distance, source_to_sample_distance = 0.0):
    """ Wavefront propagator in the Fresnel Regime by the angular spectrum method (ASM).

    If a source_to_sample_distance is given, calculates magnification and the equivalent parallel beam configuration

    Args:
        wavefront: 2d array containing your wavefront/beam
        wavelength: wavelength in meters
        pixel_size: matrix pixel size in meters
        sample_to_detector_distance: distance between sample and detectior in meters.
        source_to_sample_distance (float, optional): distance between source and sample in meters. Defaults to 0.

    Returns:
        2d array: propagated wave
    """    


    np = cp.get_array_module(wavefront) # make code agnostic to cupy and numpy
    
    K = 2*np.pi/wavelength # wavenumber
    z2 = sample_to_detector_distance
    z1 = source_to_sample_distance
    
    if z1 != 0:
        M = 1 + (z2/z1)
    else:
        M = 1
    
    FT = np.fft.fftshift(np.fft.fft2(wavefront))

    ny, nx = wavefront.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx,d = pixel_size/M))#*2*np.pi 2*np.pi factor to calculate angular frequencies 
    fy = np.fft.fftshift(np.fft.fftfreq(ny,d = pixel_size/M))#*2*np.pi
    FX, FY = np.meshgrid(fx,fy)
    # kernel = np.exp(-1j*(z2/M)/(2*K)*(FX**2+FY**2)) # if using angular frequencies. Formula as in Paganin equation 1.28
    kernel = np.exp(-1j*np.pi*wavelength*(z2/M)*(FX**2+FY**2)) # if using standard frequencies. Formula as in Goodman, Fourier Optics, equation 4.21

    wave_parallel = np.fft.ifft2(np.fft.ifftshift(FT * kernel))*np.exp(1j*K*z2/M)

    if z1 != 0:
        # gamma_M = 1 - 1/M
        # y, x = np.indices(wavefront.shape)
        # y = (y - y.shape[0]//2)*pixel_size/M
        # x = (x - x.shape[1]//2)*pixel_size/M
        wave_cone = wave_parallel * (1/M) #* np.exp(1j*gamma_M*K*z2) * np.exp(1j*gamma_M*K*(x**2+y**2)/(2*z2)) # Need to check the commented phase terms, which are part of the full form for the Fresnel Scaling theorem (i.e. without calculating absolute value)
        return wave_cone
    else:
        return wave_parallel
        

def calculate_fresnel_number(energy,pixel_size,sample_detector_distance,source_sample_distance=0):
    """
    Calculate fresnel number in magnification scenario. 

    Args:
        energy: energy in keV
        pixel_size: object pixel size
        sample_detector_distance: sample to detector distance in meters
        magnification (int, optional): magnification of the optical system. If 1, no magnification is used. Defaults to 1.
        source_sample_distance (int, optional): source to sample distance in meters. Defaults to 0.

    Returns:
        (float): Fresnel number 
    """

    if source_sample_distance != 0:
        magnification = (source_sample_distance+sample_detector_distance)/source_sample_distance
    wavelength = wavelength_meters_from_energy_keV(energy) # meters
    return -(pixel_size**2) / (wavelength * sample_detector_distance * magnification)        
