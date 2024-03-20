
import cupy as cp
import numpy as np

from ..processing.propagation import fresnel_propagator_cone_beam

""" COMMON FUNCTIONS FOR DIFFERENT PTYCHO ENGINES"""

def update_exit_wave_multiprobe_cupy(wavefront_modes,measurement,inputs):
    wavefront_modes = propagate_multiprobe_cupy(wavefront_modes,inputs)
    wavefront_modes_at_detector = Fspace_update_multiprobe_cupy(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_multiprobe_cupy(wavefront_modes_at_detector,inputs,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_multiprobe_cupy(wavefront_modes,inputs = {},backpropagate=False):

    if inputs["source_distance"] is None:
        if backpropagate == False:
            for m, mode in enumerate(wavefront_modes): #TODO: worth propagating in parallel?
                wavefront_modes[m] = cp.fft.fftshift(cp.fft.fft2(mode))
        else:
            for m in range(wavefront_modes.shape[0]):
                wavefront_modes[m] = cp.fft.ifft2(cp.fft.ifftshift(wavefront_modes[m]))
    else:
        if backpropagate == False:
            z2 = 1*inputs["detector_distance"]
            z1 = inputs["source_distance"]
        else:
            if inputs["source_distance"] !=0:
                z2 = -1*inputs["detector_distance"]
                z1 = -inputs["source_distance"]
            else:
                z2 = -1*inputs["detector_distance"]
                z1 = inputs["source_distance"] # should be 0 here

        for m, mode in enumerate(wavefront_modes): 
            wavefront_modes[m] = fresnel_propagator_cone_beam(mode,inputs["wavelength"],inputs["detector_pixel_size"],z2,z1)
    
    return wavefront_modes

def Fspace_update_multiprobe_cupy(wavefront_modes,measurement,epsilon=0.001):
    
    total_wave_intensity = cp.zeros_like(wavefront_modes[0])

    for mode in wavefront_modes:
        total_wave_intensity += cp.abs(mode)**2
    total_wave_intensity = cp.sqrt(total_wave_intensity)
    
    updated_wavefront_modes = wavefront_modes
    for m, mode in enumerate(wavefront_modes): #TODO: worth updating in parallel?
        updated_wavefront_modes[m][measurement>=0] = cp.sqrt(measurement[measurement>=0])*mode[measurement>=0]/(total_wave_intensity[measurement>=0]+epsilon)
    
    return updated_wavefront_modes

def calculate_recon_error_Fspace_cupy(diffractions_patterns,wavefronts,experiment_params):

    error_numerator = 0
    error_denominator = 0
    for DP, wave in zip(diffractions_patterns,wavefronts):
        wave_at_detector = propagate_beam_cupy(wave, experiment_params,propagator='fourier')
        intensity = cp.abs(wave_at_detector)**2
        
        error_numerator += cp.sum(cp.abs(DP-intensity))
        error_denominator += cp.sum(cp.abs(DP))

    return error_numerator/error_denominator 

def calculate_recon_error_Fspace(diffractions_patterns,wavefronts,experiment_params):

    error_numerator = 0
    error_denominator = 0
    for DP, wave in zip(diffractions_patterns,wavefronts):
        wave_at_detector = propagate_beam_cupy(wave, experiment_params,propagator='fourier')
        intensity = np.abs(wave_at_detector)**2
        
        error_numerator += np.sum(np.abs(DP-intensity))
        error_denominator += np.sum(np.abs(DP))

    return error_numerator/error_denominator 

def propagate_beam_cupy(wavefront, experiment_params,propagator='fourier'):
    

    """ Propagate a wavefront using fresnel ou fourier propagator

    Args:
        wavefront : the wavefront to propagate
        dx : pixel spacing of the wavefront input
        wavelength : wavelength of the illumination
        distance : distance to propagate
        propagator (str, optional): 'fresenel' or 'fourier'. Defaults to 'fresnel'.

    Returns:
        output: propagated wavefront
    """    
    
    dx, wavelength,distance = experiment_params 
    
    if propagator == 'fourier':
        if distance > 0:
            output = cp.fft.fftshift(cp.fft.fft2(wavefront))
        else:
            output = cp.fft.ifft2(cp.fft.ifftshift(wavefront))
    
    elif propagator == 'fresnel':
    
        ysize, xsize = wavefront.shape
        x_array = cp.linspace(-xsize/2,xsize/2-1,xsize)
        y_array = cp.linspace(-ysize/2,ysize/2-1,ysize)

        fx = x_array/(xsize)
        fy = y_array/(ysize)

        FX,FY = cp.meshgrid(fx,fy)
        # Calculate approx phase distribution for each plane wave component
        w = FX**2 + FY**2 
        # Compute FFT
        F = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(wavefront)))
        # multiply by phase-shift and inverse transform 
        a = cp.exp(-1j*cp.pi*( distance*wavelength/dx**2)*w)
        output = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(F*a)))
    return output