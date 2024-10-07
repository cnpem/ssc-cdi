# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import sys
import cupy as cp
from .engines_common import update_exit_wave, apply_probe_support, create_random_binary_mask
from ..misc import extract_values_from_all_slices, get_random_2D_indices

def PIE_python(diffraction_patterns, positions, object_guess, probe_guess, inputs):
    """"
    Implementation of rPIE and mPIE algorithms 
    """

    r_o = inputs["regularization_object"]
    r_p = inputs["regularization_probe"]
    s_o = inputs["step_object"]
    s_p = inputs["step_probe"]
    f_o = inputs["friction_object"]
    f_p = inputs["friction_probe"]
    m_counter_limit = inputs["momentum_counter"]
    n_of_modes = inputs["incoherent_modes"]
    iterations = inputs["iterations"]
    obj_pixel = inputs['object_pixel']
    wavelength = inputs['wavelength']
    detector_distance = inputs['detector_distance']
    distance_focus_sample  = inputs['distance_sample_focus']
    detector_pixel_size = inputs["detector_pixel_size"]
    propagator = inputs['regime']
    free_log_likelihood = inputs['free_log_likelihood']
    probe_support  = inputs["probe_support_array"] 
    fourier_power_bound = inputs['fourier_power_bound']

    try:
        import cupy as cp

        # Check if a GPU is available
        cp.cuda.Device(0).compute_capability  # Access the first GPU (0-indexed)
        print("Using CuPy (GPU)")
        np = cp  # np will be an alias for cupy

        print('Transfering data to GPU...')

        object_guess = cp.array(object_guess) # convert from numpy to cupy
        probe_guess  = cp.array(probe_guess)
        positions    = cp.array(positions)
        diffraction_patterns = cp.array(diffraction_patterns)
        probe_support = cp.array(probe_support)
        obj = cp.ones((n_of_modes,object_guess.shape[0],object_guess.shape[1]),dtype=complex)

    except (ImportError, cp.cuda.runtime.CUDARuntimeError):
        # Fallback to NumPy if GPU is not available or cupy is not installed
        import numpy as np
        print("Using NumPy (CPU)")

    if free_log_likelihood > 0:
        print('free_log_likelihood>0! Reconstruction will use FREE error metrics!')
        free_indices = get_random_2D_indices(free_log_likelihood,diffraction_patterns[0].shape[0],diffraction_patterns[0].shape[1])
        free_data = extract_values_from_all_slices(diffraction_patterns,free_indices)
        slice_indices = np.arange(diffraction_patterns.shape[0])
        diffraction_patterns[slice_indices[:, None], free_indices[0], free_indices[1]] = -1 # remove free data from diffraction patterns. Free data shall be used only for error estimation
    else:
        free_data, free_indices = None, None

    obj[:] = object_guess # object matrix repeats for each slice; each slice will operate with a different probe mode

    offset = probe_guess.shape

    if inputs["incoherent_modes"] > 1:
        probe_modes = cp.empty((inputs["incoherent_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[0] = probe_guess # first mode is guess
        for mode in range(1,inputs["incoherent_modes"]): # remaining modes are random
            probe_modes[mode] = cp.random.rand(*probe_guess.shape)
    elif inputs["incoherent_modes"] == 1:
        probe_modes = cp.empty((inputs["incoherent_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[:] = probe_guess
    else:
        sys.exit('Please select the correct amount of modes: ',inputs["incoherent_modes"])

    probe_velocity = cp.zeros_like(probe_modes,dtype=complex)
    obj_velocity   = cp.zeros_like(obj,dtype=complex)
    
    momentum_counter = 0
    error = cp.zeros((iterations,4))
    for iteration in range(iterations):
        
        temporary_obj, temporary_probe = obj.copy(), probe_modes.copy()
        
        for j in cp.random.permutation(len(diffraction_patterns)):
            py, px = positions[:,1][j],  positions[:,0][j]

            obj_box = obj[:,py:py+offset[0],px:px+offset[1]]

            """ Wavefront at object exit plane """
            wavefront_modes = obj_box*probe_modes

            """ Propagate + Update + Backpropagate """
            updated_wavefront_modes, all_errors = update_exit_wave(wavefront_modes.copy(),diffraction_patterns[j],detector_distance,wavelength,detector_pixel_size,propagator,free_data,free_indices,fourier_power_bound=fourier_power_bound,epsilon=0.001) #copy so it doesn't work as a pointer!
            
            error_r_factor_num, error_r_factor_den, error_nmse, error_llk = all_errors
            error[iteration,0] += error_r_factor_num
            error[iteration,1] += error_r_factor_den
            error[iteration,2] += error_nmse
            error[iteration,3] += error_llk

            obj[:,py:py+offset[0],px:px+offset[1]] , probe_modes = update_object_and_probe(obj_box[0],probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p)

            if inputs["use_mPIE"] == True: # momentum addition                                                                                      
                momentum_counter,obj_velocity,probe_velocity,temporary_obj,temporary_probe,obj,probe_modes = momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,temporary_obj,temporary_probe,obj,probe_modes,f_o,f_p,m_counter_limit,momentum_type="")

        probe_modes = apply_probe_support(probe_modes,probe_support,distance_focus_sample,wavelength,obj_pixel)


        print('\r', end='')
        print(f'\tIteration {iteration+1}/{iterations} \t Errors: R-factor={error[iteration,0]/error[iteration,1]:.2e}; MSE={error[iteration,2]:.2e}; Poisson LLK={error[iteration,3]:.2e}',end='')

    print('\n')    

    error[:,0] = error[:,0]/error[:,1] # R-factor calculation
    error = np.delete(error, 1, axis=1) # delete denominator column of R-factor, not needed anymore   

    if np == cp: # if using gpus
        return obj[0].get(), probe_modes.get(), error.get(), positions.get()
    else:
        return obj[0], probe_modes, error, positions
    
def update_object_and_probe(obj,probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p):

    """ 
    s: step constant
    r: regularization constant
    """
    
    def get_denominator_p(obj,reg_p):
        power = cp.abs(obj)**2
        denominator = (1-reg_p)*power+ reg_p*cp.max(power)
        return denominator  

    def get_denominator_o(probe_modes,reg_o):
        
        total_probe_power = cp.zeros_like(cp.abs(probe_modes[0]))
        for mode in probe_modes:
            total_probe_power += cp.abs(mode)**2    
            
        denominator = (1-reg_o)*total_probe_power + reg_o*cp.max(total_probe_power)
        
        return denominator  

    # r_o,r_p,s_o,s_p,_,_,_ = mPIE_params

    # Pre-calculating to avoid repeated operations
    denominator_object = get_denominator_o(probe_modes,r_o)
    probe_modes_conj = probe_modes.conj()
    Delta_wavefront_modes =  updated_wavefront_modes - wavefront_modes

    obj = obj + s_o * cp.sum(probe_modes_conj*Delta_wavefront_modes,axis=0) / denominator_object # object update

    obj_conj = obj.conj()
    denominator_probe  = get_denominator_p(obj,r_p)
    for m in range(probe_modes.shape[0]): # P_(i+1) = P_(i) + s_p * DeltaP_(i)
        probe_modes[m] = probe_modes[m] + s_p * obj_conj*Delta_wavefront_modes[m] / denominator_probe # probe update


    return obj, probe_modes

def momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,O_aux,P_aux,obj,probe,friction_object,friction_probe,m_counter_limit,momentum_type=""):
    

    momentum_counter += 1    
    if momentum_counter == m_counter_limit : 

        probe_velocity = friction_probe*probe_velocity + (probe - P_aux) # equation 19 in the paper
        obj_velocity   = friction_object*obj_velocity  + (obj - O_aux)  

        if momentum_type == "Nesterov": # equation 21
            obj = obj + friction_object*obj_velocity
            probe = probe + friction_object*probe_velocity 
        else: # equation 20     
            obj = O_aux + obj_velocity
            probe = P_aux + probe_velocity 

        O_aux = obj
        P_aux = probe            
        momentum_counter = 0
    
    return momentum_counter,obj_velocity,probe_velocity,O_aux,P_aux,obj,probe
