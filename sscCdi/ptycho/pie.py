import sys
import cupy as cp
from .common import update_exit_wave_multiprobe_cupy, get_magnitude_error, apply_probe_support

from .. import log_event, event_start, event_stop

@log_event
def PIE_multiprobe_loop(diffraction_patterns, positions, object_guess, probe_guess, inputs):

    # TODO: write numpy/cupy agnostic code for use both with cpus or gpus

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
    fresnel_regime = inputs["fresnel_regime"]
    probe_support  = inputs["probe_support_array"] 

    if fresnel_regime == True:
        pass
    else:
        inputs['source_distance'] = None

    object_guess = cp.array(object_guess) # convert from numpy to cupy
    probe_guess  = cp.array(probe_guess)
    positions    = cp.array(positions)
    diffraction_patterns = cp.array(diffraction_patterns)
    probe_support = cp.array(probe_support)

    obj = cp.ones((n_of_modes,object_guess.shape[0],object_guess.shape[1]),dtype=complex)
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

    wavefronts = cp.empty((len(diffraction_patterns),probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)

    probe_velocity = cp.zeros_like(probe_modes,dtype=complex)
    obj_velocity   = cp.zeros_like(obj,dtype=complex)
    
    momentum_counter = 0
    error = cp.zeros((iterations,1))
    for i in range(iterations):
        
        temporary_obj, temporary_probe = obj.copy(), probe_modes.copy()
        
        for j in cp.random.permutation(len(diffraction_patterns)):
            py, px = positions[:,1][j],  positions[:,0][j]

            obj_box = obj[:,py:py+offset[0],px:px+offset[1]]

            """ Wavefront at object exit plane """
            wavefront_modes = obj_box*probe_modes

            wavefronts[j] = wavefront_modes[0] # save mode 0 wavefront to calculate recon error
 
            """ Propagate + Update + Backpropagate """
            updated_wavefront_modes, _ = update_exit_wave_multiprobe_cupy(wavefront_modes.copy(),diffraction_patterns[j],inputs) #copy so it doesn't work as a pointer!
            
            obj[:,py:py+offset[0],px:px+offset[1]] , probe_modes = PIE_update_func_multiprobe(obj_box[0],probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p)

            if inputs["use_mPIE"] == True: # momentum addition                                                                                      
                momentum_counter,obj_velocity,probe_velocity,temporary_obj,temporary_probe,obj,probe_modes = momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,temporary_obj,temporary_probe,obj,probe_modes,f_o,f_p,m_counter_limit,momentum_type="")

        probe_modes = apply_probe_support(probe_modes,probe_support,distance_focus_sample,wavelength,obj_pixel)

        iteration_error = get_magnitude_error(diffraction_patterns,wavefronts,inputs)

        print('\r', end='')
        print(f'\tIteration {i+1}/{iterations} \tError: {iteration_error:.2e}')

        error[i] = iteration_error
   
    print('\n')    

    return obj[0].get(), probe_modes.get(), error.get()

def PIE_update_func_multiprobe(obj,probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p):

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
