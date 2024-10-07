# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


try:
    import cupy as cp
    # Check if a GPU is available
    cp.cuda.Device(0).compute_capability  # Access the first GPU (0-indexed)
    np = cp  # np will be an alias for cupy
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    # Fallback to NumPy if GPU is not available or cupy is not installed
    import numpy as np

from ..processing.propagation import fresnel_propagator

""" COMMON FUNCTIONS FOR DIFFERENT PTYCHO ENGINES"""

def update_exit_wave(wavefront_modes,measurement,detector_distance,wavelength,detector_pixel_size,propagator,free_LLK_mask,epsilon=0.001):
    wavefront_modes_at_detector = propagate_wavefronts(wavefront_modes,detector_distance,wavelength,detector_pixel_size,propagator) # propagate to detector plane
    updated_wavefront_modes_at_detector = update_wavefronts(wavefront_modes_at_detector.copy(),measurement,epsilon) # update wavefronts. # copy to use wavefront_modes_at_detector in errror calaculation
    wavefront_modes = propagate_wavefronts(updated_wavefront_modes_at_detector,-detector_distance,wavelength,detector_pixel_size,propagator) # propagate back to sample plane

    errors = calculate_errors(measurement, wavefront_modes_at_detector ,free_LLK_mask)

    return wavefront_modes, errors

def propagate_wavefronts(wavefront_modes,detector_distance,wavelength,detector_pixel_size,propagator='fraunhoffer'):

    if propagator == 'fraunhoffer':
        if detector_distance > 0: 
            wavefront_modes = cp.fft.fftshift(cp.fft.fft2(wavefront_modes,axes=(1,2)),axes=(1,2))
        else:
            wavefront_modes = cp.fft.ifft2(cp.fft.ifftshift(wavefront_modes,axes=(1,2)),axes=(1,2))

    elif propagator == 'fresnel':
        for m, mode in enumerate(wavefront_modes): 
            wavefront_modes[m] = fresnel_propagator(mode,wavelength,detector_pixel_size,detector_distance)
    else:
        raise ValueError('Invalid propagator type. Choose between "fraunhoffer" and "fresnel"')
    
    return wavefront_modes

def update_wavefronts(wavefront_modes,measurement,epsilon=0.001):
    
    total_wave_intensity = cp.zeros_like(wavefront_modes[0])

    for mode in wavefront_modes:
        total_wave_intensity += cp.abs(mode)**2
    total_wave_intensity = cp.sqrt(total_wave_intensity)
    
    for m, mode in enumerate(wavefront_modes): 
        wavefront_modes[m][measurement>=0] = cp.sqrt(measurement[measurement>=0])*mode[measurement>=0]/(total_wave_intensity[measurement>=0]+epsilon)
    
    return wavefront_modes

def calculate_errors(measurement, wavefronts_at_detector,free_llk_mask):

    total_wave_intensity = np.sum(cp.abs(wavefronts_at_detector)**2,axis=0)
    valid_data_mask = measurement > 0

    r_factor_numerator, r_factor_denominator = calculate_rfactor(measurement, total_wave_intensity,free_llk_mask,valid_data_mask)

    nmse_numerator = calculate_nmserror(measurement, total_wave_intensity,free_llk_mask,valid_data_mask)

    poisson_likelihood_error = calculate_poisson_likelihood(measurement, total_wave_intensity,free_llk_mask,valid_data_mask)

    all_errors = [r_factor_numerator, r_factor_denominator, nmse_numerator, poisson_likelihood_error]

    return all_errors

def calculate_nmserror(measurement, total_wave_intensity,free_llk_mask,valid_data_mask):
        mask = free_llk_mask*valid_data_mask
        error_numerator = np.sum(np.abs(mask*((measurement-total_wave_intensity)/(measurement)))**2)/(measurement.shape[0]*measurement.shape[1])
        return error_numerator


def calculate_rfactor(measurement, estimated_intensity,free_llk_mask,valid_data_mask):
    """
    R-factor defined as:  R = sum( | sqrt(DP) - estimate| ) / sum( np.sqrt(DP) )
    """

    mask = free_llk_mask*valid_data_mask
    error_numerator = np.sum(np.abs(np.sqrt(mask*measurement)-np.sqrt(mask*estimated_intensity)))
    error_denominator = np.sum(np.sqrt(measurement))

    return error_numerator, error_denominator

def calculate_poisson_likelihood(measurement,estimated_intensity,free_llk_mask,valid_data_mask,epsilon=1e-10):
    """
    LLK_error = sum ( n*log(lambda)-lambda ) , where n is the measurement and lambda is the estimated intensity
    """
    integrated_intensity = np.sum(measurement*valid_data_mask)

    estimated_intensity = estimated_intensity/integrated_intensity # normalize the estimated intensity
    measurement = measurement/integrated_intensity # normalize the measurement
    return np.sum((valid_data_mask*estimated_intensity-valid_data_mask*measurement*np.log(valid_data_mask*estimated_intensity)))


def apply_probe_support(probe_modes,probe_support,distance_focus_sample,wavelength,obj_pixel):
    if distance_focus_sample == 0:
        probe_modes = probe_modes*probe_support
    else:
        for i, mode in enumerate(probe_modes): # propagate each mode back to focus
            probe_modes[i] = propagate_wavefronts(mode,wavelength,obj_pixel,-distance_focus_sample)
        probe_modes = probe_modes*probe_support
        for i, mode in enumerate(probe_modes): # propagate each mode back to sample plane
            probe_modes[i] = propagate_wavefronts(mode,wavelength,obj_pixel,distance_focus_sample)
    return probe_modes


# def poisson_log_likelihood(y, lambdas):
#     """
#     Calculate the Poisson log-likelihood error.

#     Parameters:
#     y (array-like): Observed counts
#     lambdas (array-like): Predicted counts (mean of the Poisson distribution)

#     Returns:
#     float: Negative Poisson log-likelihood error
#     """
#     # Ensure inputs are numpy arrays
#     y = np.array(y)
#     lambdas = np.array(lambdas)
    
#     # Prevent log(0) by ensuring lambda values are greater than 0
#     lambdas = np.clip(lambdas, a_min=1e-10, a_max=None)

#     # Calculate Poisson log-likelihood
#     log_likelihood = np.sum(lambdas - y * np.log(lambdas))
    
#     return log_likelihood


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


def create_random_binary_mask(Y, X, N):
    """
    Create a binary mask of dimensions (Y, X) with N randomly placed 1s.
    
    Parameters:
    Y (int): Number of rows (height of the mask).
    X (int): Number of columns (width of the mask).
    N (int): Number of points to be set to 1.
    
    Returns:
    np.array: Binary mask of dimensions (Y, X) with N values equal to 1.
    """
    # Ensure that N is not greater than the total number of pixels (Y*X)
    if N > Y * X:
        raise ValueError("N cannot be greater than the total number of pixels (Y*X)")
    
    # Create a zero-filled mask
    mask = np.zeros((Y, X), dtype=np.int32)
    
    # Generate N unique random indices
    indices = np.random.choice(Y * X, N, replace=False)
    
    # Reshape the array and set values directly
    mask = mask.reshape(-1)  # Reshape to 1D
    mask[indices] = 1        # Set the chosen indices to 1
    mask = mask.reshape((Y, X))  # Reshape back to (Y, X)
    
    return mask