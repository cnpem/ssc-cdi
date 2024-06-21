
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

def get_magnitude_error(diffractions_patterns,wavefronts,inputs):

    error_numerator = 0
    error_denominator = 0
    for DP, wave in zip(diffractions_patterns,wavefronts):
        DP = np.squeeze(DP)
        
        wave_at_detector = propagate_multiprobe_cupy(np.expand_dims(wave,axis=0).copy(), inputs)
        intensity = cp.abs(wave_at_detector)[0]
        intensity[DP<0] = -1
        error_numerator += cp.sum((DP-intensity)**2) 
        error_denominator += cp.sum(DP+1)

    return error_numerator/error_denominator/np.prod(diffractions_patterns[0].shape)

    # for DP, wave in zip(diffractions_patterns,wavefronts):
    #     wave_at_detector = propagate_multiprobe_cupy(np.expand_dims(wave,axis=0).copy(), inputs)
    #     intensity = cp.abs(wave_at_detector)**2
        
    #     error_numerator += poisson_log_likelihood(DP, intensity)

    # return error_numerator


def apply_probe_support(probe_modes,probe_support,distance_focus_sample,wavelength,obj_pixel):
    if distance_focus_sample == 0:
        probe_modes = probe_modes*probe_support
    else:
        for i, mode in enumerate(probe_modes): # propagate each mode back to focus
            probe_modes[i] = fresnel_propagator_cone_beam(mode,wavelength,obj_pixel,-distance_focus_sample)
        probe_modes = probe_modes*probe_support
        for i, mode in enumerate(probe_modes): # propagate each mode back to sample plane
            probe_modes[i] = fresnel_propagator_cone_beam(mode,wavelength,obj_pixel,distance_focus_sample)
    return probe_modes


def poisson_log_likelihood(y, lambda_pred):
    """
    Calculate the negative Poisson log likelihood.
    
    Parameters:
    y : array-like
        Observed counts.
    lambda_pred : array-like
        Predicted mean counts from the model.
    
    Returns:
    float
        Negative Poisson log likelihood.
    """
    np = cp.get_array_module(y)  

    # Ensuring y and lambda_pred are numpy arrays
    y = np.array(y)
    lambda_pred = np.array(lambda_pred)
    
    # Calculate each component of the log likelihood
    log_likelihood = y * np.log(lambda_pred) - lambda_pred - np.log(np.arange(1, y.max() + 1)).sum()
    
    # Sum the log likelihoods and take the negative
    nll = -np.sum(log_likelihood)
    
    return nll/np.sum(y)

def gaussian_log_likelihood(y, mu, sigma2=0.1):
    """
    Calculate the negative Gaussian log likelihood.
    
    Parameters:
    y : array-like
        Observed values.
    mu : array-like
        Predicted mean values.
    sigma2 : float
        Variance of the Gaussian distribution.
    
    Returns:
    float
        Negative Gaussian log likelihood.
    """

    np = cp.get_array_module(y)  

    # Ensure y and mu are numpy arrays
    y = np.array(y)
    mu = np.array(mu)
    
    # Number of observations
    N = len(y)
    
    # Calculate each component of the log likelihood
    log_likelihood = -0.5 * N * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * np.sum((y - mu) ** 2)
    
    # Take the negative of the log likelihood
    nll = -log_likelihood
    
    return nll