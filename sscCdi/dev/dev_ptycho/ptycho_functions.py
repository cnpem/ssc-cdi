import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import time
import h5py, os
import random
import tqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
import threading

from skimage.registration import phase_cross_correlation
from numpy.fft import fft2, fftshift, ifftshift, ifft2

random.seed(0)

""" RAAR + probe decomposition   """

def RAAR_multiprobe_cupy(diffraction_patterns,positions,obj,probe,inputs, probe_support = None):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    # Numpy to Cupy
    diffraction_patterns = cp.array(diffraction_patterns)
    positions = cp.array(positions)
    obj = cp.array(obj)
    probe = cp.array(probe)

    if probe_support is None:
        probe_support = cp.ones_like(probe)
    else:
        probe_support = cp.array(probe_support)

    obj_matrix = cp.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = cp.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = cp.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        wavefronts[index] = probe_modes*obj_box
        
    error = []
    for iteration in range(0,iterations):
        """
        RAAR update function:
        psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
        psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
        psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
        """

        t1 = time.time()
        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
            
            psi_after_Pr = probe_modes*obj_box
            psi_after_reflection_Rspace = 2*psi_after_Pr-wavefronts[index]
            psi_after_projection_Fspace, _ = update_exit_wave_multiprobe_cupy(psi_after_reflection_Rspace.copy(),diffraction_patterns[index]) # Projection in Fourier space

            wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 
        t2 =time.time()
        print(t2-t1)
        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR_cupy(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = calculate_recon_error_Fspace_cupy(diffraction_patterns,wavefronts,(dx,wavelength,distance)).get()
        if iteration%10==0:
            print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0].get(), probe_modes.get(), error

def projection_Rspace_multiprobe_RAAR_cupy(wavefronts,obj,probes,positions,epsilon):
    probes = RAAR_multiprobe_update_probe_cupy(wavefronts, obj, probes.shape,positions, epsilon=epsilon) 
    obj   = RAAR_multiprobe_update_object_cupy(wavefronts, probes, obj.shape, positions,epsilon=epsilon)
    return probes, obj

def RAAR_multiprobe_update_object_cupy(wavefronts, probe, object_shape, positions,epsilon):

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

def RAAR_multiprobe_update_probe_cupy(wavefronts, obj, probe_shape,positions, epsilon=0.01):
    
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


""" parallel RAAR + probe decompositon """

def RAAR_multiprobe_parallel(diffraction_patterns,positions,obj,probe,inputs, probe_support = None, processes=32):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    if probe_support is None:
        probe_support = np.ones_like(probe)
    else:
        probe_support = np.array(probe_support)

    obj_matrix = np.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = np.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = np.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        
    error = []
    for iteration in range(0,iterations):

        t1 = time.time()
        # wavefronts = update_wavefronts_parallel(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta,processes)
        wavefronts = update_wavefronts_parallel2(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta)
        t2 = time.time()
        print(t2-t1)

        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = 0 # calculate_recon_error_Fspace(diffraction_patterns,wavefronts,(dx,wavelength,distance))
        if iteration%10==0:
            print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0], probe_modes, error

def update_wavefronts_parallel2(obj_matrix, probe_modes, wavefronts0, diffraction_patterns0,positions,beta):
    t1=time.time()
    global wavefronts,projected_wavefronts,diffraction_patterns

    wavefronts,diffraction_patterns = wavefronts0, diffraction_patterns0

    shapey,shapex = probe_modes.shape[1], probe_modes.shape[2]

    projected_wavefronts = np.empty_like(wavefronts)

    for index, (posx, posy) in enumerate(positions):
        projected_wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]

    indexes = [i for i in range(wavefronts.shape[0])]
    
    t2=time.time()

    processes = []
    for index in indexes:
        # process = multiprocessing.Process(target=update_wavefront2,args=(index, beta))
        process = threading.Thread(target=update_wavefront2,args=(index, beta))
        process.start()
        processes.append(process)

    t3=time.time()

    for p in processes: # wait for processes to finish
        p.join()

    t4=time.time()
    print(t4-t2,t2-t1)

    return wavefronts

def update_wavefront2(index,beta):
    """
    RAAR update function:
    psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
    psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
    psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
    """
    global wavefronts,projected_wavefronts,diffraction_patterns
    # t1 = time.time()
    psi_after_reflection_Rspace = 2*projected_wavefronts[index]-wavefronts[index]
    psi_after_projection_Fspace, _ = update_exit_wave_multiprobe(psi_after_reflection_Rspace,diffraction_patterns[index]) # Projection in Fourier space
    wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*projected_wavefronts[index] 
    # t2 = time.time()
    # print('a',t2-t1)
def update_wavefronts_parallel(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta, processes):
    t0 = time.time()

    shapey,shapex = probe_modes.shape[1], probe_modes.shape[2]

    # update_wavefront_partial = partial(update_wavefront,beta)

    projected_wavefronts = np.empty_like(wavefronts)
    for index, (posx, posy) in enumerate(positions):
        projected_wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]

    with ThreadPoolExecutor(max_workers=processes) as executor:
        t1 = time.time()
        results = executor.map(update_wavefront,wavefronts, projected_wavefronts, diffraction_patterns)
        t2 = time.time()
        for counter, result in enumerate(results):
            wavefronts[counter,:,:] = result
        t3 = time.time()

    print(t3-t2,t2-t1,t1-t0)

    return wavefronts

def update_wavefront(wavefront,psi_after_Pr,data,beta=1):
    """
    RAAR update function:
    psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
    psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
    psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
    """
    psi_after_reflection_Rspace = 2*psi_after_Pr-wavefront
    psi_after_projection_Fspace, _ = update_exit_wave_multiprobe(psi_after_reflection_Rspace.copy(),data) # Projection in Fourier space
    wavefront = beta*(wavefront + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 
    return wavefront


def RAAR_multiprobe(diffraction_patterns,positions,obj,probe,inputs, probe_support = None):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    if probe_support is None:
        probe_support = np.ones_like(probe)
    else:
        probe_support = np.array(probe_support)

    obj_matrix = np.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = np.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = np.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        wavefronts[index] = probe_modes*obj_box
        
    error = []
    for iteration in range(0,iterations):
        """
        RAAR update function:
        psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
        psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
        psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
        """

        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
            psi_after_Pr = probe_modes*obj_box

            psi_after_reflection_Rspace = 2*psi_after_Pr-wavefronts[index]
            psi_after_projection_Fspace, _ = update_exit_wave_multiprobe(psi_after_reflection_Rspace.copy(),diffraction_patterns[index]) # Projection in Fourier space

            wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 

        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = calculate_recon_error_Fspace(diffraction_patterns,wavefronts,(dx,wavelength,distance)).get()
        if iteration%10==0:
            print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0], probe_modes, error

def update_exit_wave_multiprobe(wavefront_modes,measurement):
    wavefront_modes = propagate_farfield_multiprobe(wavefront_modes)
    wavefront_modes_at_detector = Fspace_update_multiprobe(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_farfield_multiprobe(wavefront_modes_at_detector,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_farfield_multiprobe(wavefront_modes,backpropagate=False):
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): #TODO: worth propagating in parallel?
            wavefront_modes[m] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mode)))
    else:
        for m in range(wavefront_modes.shape[0]):
            wavefront_modes[m] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(wavefront_modes[m])))
    return wavefront_modes

def update_exit_wave_multiprobe(wavefront_modes,measurement):
    wavefront_modes = propagate_farfield_multiprobe(wavefront_modes)
    wavefront_modes_at_detector = Fspace_update_multiprobe(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_farfield_multiprobe(wavefront_modes_at_detector,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_farfield_multiprobe(wavefront_modes,backpropagate=False):
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): #TODO: worth propagating in parallel?
            wavefront_modes[m] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mode)))
    else:
        for m in range(wavefront_modes.shape[0]):
            wavefront_modes[m] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(wavefront_modes[m])))
    return wavefront_modes

def Fspace_update_multiprobe(wavefront_modes,measurement,epsilon=0.001):
    
    total_wave_intensity = np.zeros_like(wavefront_modes[0])

    for mode in wavefront_modes:
        total_wave_intensity += np.abs(mode)**2
    total_wave_intensity = np.sqrt(total_wave_intensity)
    
    updated_wavefront_modes = wavefront_modes
    for m, mode in enumerate(wavefront_modes): #TODO: worth updating in parallel?
        updated_wavefront_modes[m][measurement>=0] = np.sqrt(measurement[measurement>=0])*mode[measurement>=0]/(total_wave_intensity[measurement>=0]+epsilon)
    
    return updated_wavefront_modes

def projection_Rspace_multiprobe_RAAR(wavefronts,obj,probes,positions,epsilon):
    probes = RAAR_multiprobe_update_probe(wavefronts, obj, probes.shape,positions, epsilon=epsilon) 
    obj   = RAAR_multiprobe_update_object(wavefronts, probes, obj.shape, positions,epsilon=epsilon)
    return probes, obj

def RAAR_multiprobe_update_object(wavefronts, probe, object_shape, positions,epsilon):

    modes,m,n = probe.shape
    k,l = object_shape

    probe_sum  = np.zeros((k,l),dtype=complex)
    wave_sum   = np.zeros((k,l),dtype=complex)
    probe_intensity  = np.abs(probe)**2
    probe_conj = np.conj(probe)

    for mode in range(modes):
        for index, (posx, posy) in enumerate((positions)):
            probe_sum[posy:posy + m , posx:posx+n] += probe_intensity[mode]
            wave_sum[posy:posy + m , posx:posx+n]  += probe_conj[mode]*wavefronts[index,mode] 

    obj = wave_sum/(probe_sum + epsilon)

    return obj

def RAAR_multiprobe_update_probe(wavefronts, obj, probe_shape,positions, epsilon=0.01):
    
    l,m,n = probe_shape

    object_sum = np.zeros((m,n),dtype=complex)
    wave_sum = np.zeros((l,m,n),dtype=complex)
    
    obj_intensity = np.abs(obj)**2
    obj_conj = np.conj(obj)
    
    for index, (posx, posy) in enumerate(positions):
        object_sum += obj_intensity[posy:posy + m , posx:posx+n] 
        for mode in range(l):
            wave_sum[mode] += obj_conj[posy:posy + m , posx:posx+n]*wavefronts[index,mode]

    probes = wave_sum/(object_sum + epsilon) # epsilon to avoid division by zero. 

    return probes


"""  mPIE + probe decomposition   """

def PIE_multiprobe_loop(diffraction_patterns, positions, object_guess, probe_guess, inputs):

    r_o = inputs["regularization_object"]
    r_p = inputs["regularization_probe"]
    s_o = inputs["step_object"]
    s_p = inputs["step_probe"]
    f_o = inputs["friction_object"]
    f_p = inputs["friction_probe"]
    m_counter_limit = inputs["momentum_counter"]
    n_of_modes = inputs["n_of_modes"]
    iterations = inputs["iterations"]
    experiment_params =  (inputs['object_pixel'], inputs['wavelength'],inputs['distance'])

    object_guess = cp.array(object_guess) # convert from numpy to cupy
    probe_guess  = cp.array(probe_guess)
    positions    = cp.array(positions)
    diffraction_patterns = cp.array(diffraction_patterns)

    obj = cp.ones((n_of_modes,object_guess.shape[0],object_guess.shape[1]),dtype=complex)
    obj[:] = object_guess # object matrix repeats for each slice; each slice will operate with a different probe mode

    offset = probe_guess.shape

    if inputs["n_of_modes"] > 1:
        probe_modes = cp.empty((inputs["n_of_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[0] = probe_guess # first mode is guess
        for mode in range(1,inputs["n_of_modes"]): # remaining modes are random
            probe_modes[mode] = cp.random.rand(*probe_guess.shape)
    elif inputs["n_of_modes"] == 1:
        probe_modes = cp.empty((inputs["n_of_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[:] = probe_guess
    else:
        sys.exit('Please select the correct amount of modes: ',inputs["n_of_modes"])

    wavefronts = cp.empty((len(diffraction_patterns),probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)

    probe_velocity = cp.zeros_like(probe_modes,dtype=complex)
    obj_velocity   = cp.zeros_like(obj,dtype=complex)
    
    momentum_counter = 0
    error_list = []
    for i in range(iterations):
        
        temporary_obj, temporary_probe = obj.copy(), probe_modes.copy()
        
        for j in cp.random.permutation(len(diffraction_patterns)):  
            py, px = positions[:,1][j],  positions[:,0][j]

            obj_box = obj[:,py:py+offset[0],px:px+offset[1]]

            """ Wavefront at object exit plane """
            wavefront_modes = obj_box*probe_modes

            wavefronts[j] = wavefront_modes[0] # save mode 0 wavefront to calculate recon error
 
            """ Propagate + Update + Backpropagate """
            updated_wavefront_modes, _ = update_exit_wave_multiprobe_cupy(wavefront_modes.copy(),diffraction_patterns[j]) #copy so it doesn't work as a pointer!
            
            obj[:,py:py+offset[0],px:px+offset[1]] , probe_modes = PIE_update_func_multiprobe(obj_box[0],probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p)

            if inputs["use_mPIE"] == True: # momentum addition                                                                                      
                momentum_counter,obj_velocity,probe_velocity,temporary_obj,temporary_probe,obj,probe_modes = momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,temporary_obj,temporary_probe,obj,probe_modes,f_o,f_p,m_counter_limit,momentum_type="")

        iteration_error = calculate_recon_error_Fspace_cupy(diffraction_patterns,wavefronts,experiment_params).get()
        if i%10==0:
            print(f'\tIteration {i}/{iterations} \tError: {iteration_error:.2e}')
        error_list.append(iteration_error) # error in fourier space 

    return obj.get(), probe_modes.get(), error_list

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

""" GENERAL """

def update_exit_wave_multiprobe_cupy(wavefront_modes,measurement):
    wavefront_modes = propagate_farfield_multiprobe_cupy(wavefront_modes)
    wavefront_modes_at_detector = Fspace_update_multiprobe_cupy(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_farfield_multiprobe_cupy(wavefront_modes_at_detector,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_farfield_multiprobe_cupy(wavefront_modes,backpropagate=False):
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): #TODO: worth propagating in parallel?
            wavefront_modes[m] = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(mode)))
    else:
        for m in range(wavefront_modes.shape[0]):
            wavefront_modes[m] = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(wavefront_modes[m])))
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
            output = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(wavefront)))
        else:
            output = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(wavefront)))            
    
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
        F = fftshift(fft2(fftshift(wavefront)))
        # multiply by phase-shift and inverse transform 
        a = cp.exp(-1j*cp.pi*( distance*wavelength/dx**2)*w)
        output = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(F*a)))
    return output


""" DEV: Position Correction """

def update_beta(positions1,positions2, beta):
    
    k = np.corrcoef(positions1,positions2)[0,1]

    if np.isnan(k).any():
        print('Skipping')
    else:
        threshold1 = +0.3
        threshold2 = -0.3
        
        if k > threshold1:
            beta = beta*1.1 # increase by 10%
        elif k < threshold2:
            beta = beta*0.9 #reduce by 10%
        else:
            pass # keep same value
        
    return beta

def get_illuminated_mask(probe,probe_threshold):
    probe = np.abs(probe)
    mask = np.where(probe > np.max(probe)*probe_threshold, 1, 0)
    return mask

def position_correction(i, obj,previous_obj,probe,position_x,position_y, betas, probe_threshold=0.5, upsampling=100):

    beta_x,beta_y = betas

    illumination_mask = get_illuminated_mask(probe,probe_threshold)

    obj = obj*illumination_mask
    previous_obj = previous_obj*illumination_mask

    relative_shift, error, diffphase = phase_cross_correlation(obj, previous_obj, upsample_factor=upsampling)

    # if 0 :
    #     threshold = 5
    #     if np.abs(beta_y*relative_shift[0]) > threshold or np.abs(beta_x*relative_shift[1]) > threshold:
    #         new_position = np.array([position_x,position_y])
    #     else:
    #         new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    #         # new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # else:
    
    # new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # new_position = np.array([position_x + beta_x*relative_shift[0], position_y + beta_y*relative_shift[1]])

    if i == 0:
        print(position_x, beta_x*relative_shift[1],'\t',position_y,beta_y*relative_shift[0],relative_shift)

    return new_position, relative_shift, illumination_mask

def position_correction2(i,updated_wave,measurement,obj,probe,px,py,offset,betas,experiment_params):
    """ Position correct of the gradient of intensities """ 
    
    beta_x, beta_y = betas
    
    
    # Calculate intensity difference
    updated_intensity_at_detector = np.abs(updated_wave)**2
    intensity_diff = (updated_intensity_at_detector-measurement).flatten()
    
    # Calculate wavefront gradient
    obj_dy = np.roll(obj,1,axis=0)
    obj_dx = np.roll(obj,1,axis=1)
    
    obj_box     = obj[py:py+offset[0],px:px+offset[1]]
    obj_dy_box  = obj_dy[py:py+offset[0],px:px+offset[1]]
    obj_dx_box  = obj_dx[py:py+offset[0],px:px+offset[1]]
    
    wave_at_detector    = propagate_beam(obj_box*probe,    experiment_params,propagator='fourier')
    wave_at_detector_dy = propagate_beam(obj_dy_box*probe, experiment_params,propagator='fourier')
    wave_at_detector_dx = propagate_beam(obj_dx_box*probe, experiment_params,propagator='fourier')

    obj_pxl = experiment_params[0]
    wavefront_gradient_x = (wave_at_detector-wave_at_detector_dx)/obj_pxl
    wavefront_gradient_y = (wave_at_detector-wave_at_detector_dy)/obj_pxl
   
    # Calculate intensity gradient
    intensity_gradient_x = 2*np.real(wavefront_gradient_x*np.conj(wave_at_detector))
    intensity_gradient_y = 2*np.real(wavefront_gradient_y*np.conj(wave_at_detector))
    
    
    # Solve linear system
    A_matrix = np.column_stack((intensity_gradient_x.flatten(),intensity_gradient_y.flatten()))
    A_transpose = np.transpose(A_matrix)
    relative_shift = np.linalg.pinv(A_transpose@A_matrix)@A_transpose@intensity_diff

    # Update positions
    # new_positions = np.array([px - beta_x*relative_shift[0], py - beta_y*relative_shift[1]])
    new_positions = np.array([py - beta_y*relative_shift[1],px - beta_x*relative_shift[0]])
    
    if i == 0:
        print(px, beta_x*relative_shift[1],'\t',py,beta_y*relative_shift[0],relative_shift)
    
    return new_positions

def plot_positions_and_errors(data_folder,dataname,offset,PIE_positions=[],positions_story=[]):
    
    import os, json
    
    metadata = json.load(open(os.path.join(data_folder,dataname,'mdata.json')))
    distance = metadata['/entry/beamline/experiment']['distance']*1e-3
    energy = metadata['/entry/beamline/experiment']['energy']
    pixel_size = metadata['/entry/beamline/detector']['pimega']['pixel size']*1e-6
    wavelength, wavevector = calculate_wavelength(energy)
    
    diffraction_patterns = np.load(os.path.join(data_folder,dataname,f"0000_{dataname}_001.hdf5.npy"))

    n_pixels = diffraction_patterns.shape[1]
    obj_pixel_size = wavelength*distance/(n_pixels*pixel_size)
    
    _,_,measured = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}",obj_pixel_size,offset,0)
    _,_,true = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}_without_error",obj_pixel_size,offset,0)
    
    colors = np.linspace(0,positions.shape[0]-1,positions.shape[0])
    fig, ax = plt.subplots(dpi=150)
    ax.legend(["True" , "Measured", "Corrected", "Path"],loc=(1.05,0.84))    
    ax.scatter(measured[:,1],measured[:,0],marker='o',c='red')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    if positions_story != []:
        for i in range(PIE_positions.shape[0]):
            y = positions_story[:,i,1]
            x = positions_story[:,i,0]
            ax.scatter(y,x,color='blue',s=2,marker=',',alpha=0.2)
    if PIE_positions != []:
        ax.scatter(PIE_positions[:,1],PIE_positions[:,0],marker='x',color='blue')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    ax.scatter(true[:,1],true[:,0],marker='*',color='green')#,c=colors,cmap='jet')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid()