# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import cupy as cp
from .engines_common import update_exit_wave, apply_probe_support, create_random_binary_mask
from ..misc import extract_values_from_all_slices, get_random_2D_indices

def RAAR_python(diffraction_patterns,positions,obj,probe,inputs):

    try:
        import cupy as cp
        # Check if a GPU is available
        cp.cuda.Device(0).compute_capability  # Access the first GPU (0-indexed)
        print("Using CuPy (GPU)")
        np = cp  # np will be an alias for cupy

        print('Transfering data to GPU...')
        diffraction_patterns = cp.array(diffraction_patterns) 
        positions = cp.array(positions)
        obj = cp.array(obj)
        probe = cp.array(probe)
    except (ImportError, cp.cuda.runtime.CUDARuntimeError):
        # Fallback to NumPy if GPU is not available or cupy is not installed
        import numpy as np
        print("Using NumPy (CPU)")

    iterations = inputs['iterations']
    beta       = inputs['beta']
    regularization_obj    = inputs['regularization_obj']
    regularization_probe  = inputs['regularization_probe']
    obj_pixel  = inputs['object_pixel']
    wavelength = inputs['wavelength']
    n_of_modes = inputs["incoherent_modes"]
    probe_support  = inputs["probe_support_array"] 
    distance_focus_sample  = inputs['distance_sample_focus']
    detector_distance = inputs['detector_distance']
    detector_pixel_size = inputs['detector_pixel_size']
    free_log_likelihood = inputs['free_log_likelihood']
    propagator = inputs['regime']
    fourier_power_bound = inputs['fourier_power_bound']
    clip_object_magnitude = inputs['clip_object_magnitude']

    if free_log_likelihood > 0:
        print('free_log_likelihood>0! Reconstruction will use FREE error metrics!')
        free_indices = get_random_2D_indices(free_log_likelihood,diffraction_patterns[0].shape[0],diffraction_patterns[0].shape[1])
        free_data = extract_values_from_all_slices(diffraction_patterns,free_indices)
        slice_indices = np.arange(diffraction_patterns.shape[0])
        diffraction_patterns[slice_indices[:, None], free_indices[0], free_indices[1]] = -1 # remove free data from diffraction patterns. Free data shall be used only for error estimation
    else:
        free_data, free_indices = None, None

    if probe_support is None:
        probe_support = cp.ones_like(probe)
    else:
        probe_support = cp.array(probe_support)

    obj_matrix = cp.ones((n_of_modes,obj.shape[-2],obj.shape[-1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = cp.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = cp.empty((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    for i in range(0,probe_modes.shape[0]):
        if i == 0:
            probe_modes[i] = probe
        else:
            probe_modes[:] = cp.random.rand(*probe.shape)
    
    for index, (posx, posy) in enumerate(positions):
        obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        wavefronts[index] = probe_modes*obj_box
        
    print('Object shape:',obj_matrix.shape)
    print('Probe shape:',probe_modes.shape)
    print('Diffraction patterns shape:',diffraction_patterns.shape)
    print('Wavefronts shape:',wavefronts.shape)
    print('Positions shape:',positions.shape)



    error = cp.zeros((iterations,4))
    for iteration in range(0,iterations):
        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
      
            wavefronts[index], all_errors = RAAR_wavefront_update(wavefronts[index],diffraction_patterns,probe_modes,obj_box,beta,index,detector_distance,wavelength,detector_pixel_size,propagator,free_data, free_indices,fourier_power_bound)

            error_r_factor_num, error_r_factor_den, error_nmse_num, error_llk = all_errors
            error[iteration,0] += error_r_factor_num
            error[iteration,1] += error_r_factor_den
            error[iteration,2] += error_nmse_num
            error[iteration,3] += error_llk

        probe_modes, single_obj = update_object_and_probe(wavefronts,obj_matrix[0],probe_modes,positions,regularization_obj,regularization_probe) # Update Object and Probe. Projection in Real space (consistency condition)

        if clip_object_magnitude:
            single_obj = cp.clip(cp.abs(single_obj),0,1)*cp.exp(1j*cp.angle(single_obj))

        obj_matrix[:] = single_obj # update all obj slices to be the same;

        probe_modes = apply_probe_support(probe_modes,probe_support,distance_focus_sample,wavelength,obj_pixel)

        print('\r', end='')
        print(f'\tIteration {iteration+1}/{iterations} \t Errors: R-factor={error[iteration,0]/error[iteration,1]:.2e}; MSE={error[iteration,2]:.2e}; Poisson LLK={error[iteration,3]:.2e}',end='')

    print('\n')    

    error[:,0] = error[:,0]/error[:,1] # R-factor calculation
    error = np.delete(error, 1, axis=1) # delete denominator column of R-factor, not needed anymore

    if np == cp: # if using gpus
        return obj_matrix[0].get(), probe_modes.get(), error.get()
    else:
        return obj_matrix[0], probe_modes, error

def RAAR_wavefront_update(wavefront,diffraction_patterns,probe_modes,obj_box,beta,index,detector_distance,wavelength,detector_pixel_size,propagator,free_data=None, free_data_indices=None,fourier_power_bound=0):
    """
    RAAR update function:
    psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
    psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi
    psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
    """
    epsilon = 1e-3 # to avoid division by zero; #TODO: test with different values!
    psi_after_Pr = probe_modes*obj_box
    psi_after_reflection_Rspace = 2*psi_after_Pr-wavefront
    psi_after_projection_Fspace, all_errors = update_exit_wave(psi_after_reflection_Rspace,diffraction_patterns[index],detector_distance,wavelength,detector_pixel_size,propagator,free_data=free_data, free_data_indices=free_data_indices,fourier_power_bound=fourier_power_bound,epsilon=epsilon) # Projection in Fourier space
    updated_wavefront =  beta*(wavefront + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 
    return updated_wavefront, all_errors


def AP_wavefront_update(wavefront,diffraction_patterns,probe_modes,obj_box,beta,index,detector_distance,wavelength,detector_pixel_size,propagator,free_data=None, free_data_indices=None,fourier_power_bound=0):
    """
    Alternating Projection update function:
    """
    psi_after_Pr = probe_modes*obj_box
    updated_wavefront, all_errors = update_exit_wave(psi_after_Pr,diffraction_patterns[index],detector_distance,wavelength,detector_pixel_size,propagator,free_data=free_data, free_data_indices=free_data_indices,fourier_power_bound=fourier_power_bound,epsilon=epsilon) # Projection in Fourier space
    return updated_wavefront, all_errors

def update_object_and_probe(wavefronts,obj,probes,positions,regularization_obj,regularization_probe):
    probes = update_probe(wavefronts, obj, probes.shape,positions, epsilon=regularization_probe) 
    obj   = update_object(wavefronts, probes, obj.shape, positions,epsilon=regularization_obj)
    return probes, obj

def update_object(wavefronts, probe, object_shape, positions,epsilon):

    modes,m,n = probe.shape
    k,l = object_shape

    probe_sum  = cp.zeros((k,l),dtype=complex)
    wave_sum   = cp.zeros((k,l),dtype=complex)
    probe_intensity  = cp.abs(probe)**2
    probe_conj = cp.conj(probe)

    for mode in range(modes):
        for index, (posx, posy) in enumerate((positions)):
            probe_sum[posy:posy + m , posx:posx+n] += probe_intensity[mode]
            wave_sum[posy:posy + m , posx:posx+n]  += probe_conj[mode]*wavefronts[index,mode] 

    obj = wave_sum/(probe_sum + epsilon)

    return obj

def update_probe(wavefronts, obj, probe_shape,positions, epsilon=0.01):
    
    l,m,n = probe_shape

    object_sum = cp.zeros((m,n),dtype=complex)
    wave_sum = cp.zeros((l,m,n),dtype=complex)
    
    obj_intensity = cp.abs(obj)**2
    obj_conj = cp.conj(obj)
    
    for index, (posx, posy) in enumerate(positions):
        object_sum += obj_intensity[posy:posy + m , posx:posx+n] 
        for mode in range(l):
            wave_sum[mode] += obj_conj[posy:posy + m , posx:posx+n]*wavefronts[index,mode]

    probes = wave_sum/(object_sum + epsilon) # epsilon to avoid division by zero. 

    return probes


