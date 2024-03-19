import cupy as cp
from .ptychography import update_exit_wave_multiprobe_cupy, calculate_recon_error_Fspace_cupy


def RAAR_multiprobe_cupy(diffraction_patterns,positions,obj,probe,inputs):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["incoherent_modes"]
    fresnel_regime = inputs["fresnel_regime"]
    probe_support  = inputs["probe_support"] 

    if fresnel_regime == True:
        pass
    else:
        inputs['source_distance'] = None

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

        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
            
            psi_after_Pr = probe_modes*obj_box
            
            psi_after_reflection_Rspace = 2*psi_after_Pr-wavefronts[index]
            psi_after_projection_Fspace, _ = update_exit_wave_multiprobe_cupy(psi_after_reflection_Rspace,diffraction_patterns[index],inputs) # Projection in Fourier space
            wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 

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