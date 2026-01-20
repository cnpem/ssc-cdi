# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################

import numpy as np
import sys
from .engines_common import update_exit_wave, apply_probe_support, create_random_binary_mask
from ..misc import extract_values_from_all_slices, get_random_2D_indices
from .ptychography import call_ptychography

def position_correction_fresnel(diffraction_patterns, recon_positions, recon_object, recon_probe, iterations, inputs):
    import cupy as cp
    d_DPs = cp.array(diffraction_patterns)
    corrected_positions  = recon_positions.copy()
    inputs['algorithms']['1']['iterations'] = 50
    inputs['n_of_positions_to_remove'] = recon_positions.shape[0]//2
    inputs['positions_unit'] = 'pixels'
    inputs['distance_sample_focus'] = 0
    inputs['magnification'] = 1 

    for iteration in range(iterations):
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") 
        print(f"Iteration {iteration} of {n_iteration}")
        recon_obj, recon_probe, positions_recon, inputs, error_ite = call_ptychography(inputs, diffraction_patterns, corrected_positions, initial_obj=recon_object, initial_probe=recon_probe,plot=False)
        error_global = np.append(error_global, error_ite)
        d_probe = cp.array(recon_probe)
        corrected_positions = positions_recon.copy()
        for index_position in range(positions_new.shape[0]):
            radius_search = int(5*(n_iteration - iteration)/n_iteration) + 1
            offset_x, offset_y = np.meshgrid(np.arange(-radius_search, radius_search + 1), np.arange(-radius_search, radius_search + 1))
            offset_x  = offset_x.flatten()
            offset_y = offset_y.flatten()
            index_random_positions = np.arange(len(offset_x))
            np.random.shuffle(index_random_positions)
            index_random_positions = index_random_positions[:8 + len(offset_x)//4]
            offset_x  = offset_x[index_random_positions]
            offset_y = offset_y[index_random_positions]

            xmin = np.repeat(int(corrected_positions[index_position,1]), len(offset_x)) + offset_x 
            xmax = np.repeat(int(corrected_positions[index_position,1] + recon_probe.shape[2]) , len(offset_x)) + offset_x 
            ymin = np.repeat(int(corrected_positions[index_position,0]), len(offset_x)) + offset_y 
            ymax = np.repeat(int(corrected_positions[index_position,0] + recon_probe.shape[1]) , len(offset_x)) + offset_y

            for i in range(len(xmax)):
                if(xmin[i] > 0 and xmax[i] < obj.shape[1] and ymin[i] > 0 and ymax[i] < obj.shape[0]):
                    obj_local = cp.array(recon_obj[ymin[i]:ymax[i], xmin[i]:xmax[i]])
                    wave = obj_local*d_probe
                    wave = fresnel_propagator(wave, wavelength = inputs["wavelength"] , 
                                            pixel_size= inputs["detector_pixel_size"], 
                                            sample_to_detector_distance= inputs["detector_distance"])
                    wavefront = cp.sum(cp.abs(wave)**2, axis = 0)
                    error[i] = cp.sqrt(cp.sum((wavefront - d_DPs[index_position])**2)/cp.sum((d_DPs[index_position])**2))
           
            index_min = cp.asnumpy(cp.argmin(error))
            corrected_positions[index_position] = np.array([ymin[index_min], xmin[index_min]])

        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    return recon_object, recon_probe, position
def position_correction_python(diffraction_patterns, recon_positions, recon_object, recon_probe, inputs):
    
    ## Array to store cropped diffraction patterns
    cut_obj = np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2])) + 1j*np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2]))
    cut_obj_off = np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2])) + 1j*np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2]))

    ## Use randomized arrays
    random_searches = False
    
    ## Variable to store annealing radii
    radius_begin = 0
    radius_end = 5
    
    ## Array to store offsets, in pixels, to try
    pos_offx_original = init_offsets(radius_end)[0]
    pos_offy_original = init_offsets(radius_end)[1] 
    
    ## Array to store final error for each position
    error = np.zeros(cut_obj.shape[0])

    ## Run the following loop for each measured position
    for i in range(cut_obj.shape[0]):

        ## First calculate the FFT and error for the current position
        ## Define limits to crop a window with the probe size around a measured position
        x_window_begin = int(recon_positions[i,1])
        x_window_end = int(recon_positions[i,1] + recon_probe.shape[2])
        y_window_begin = int(recon_positions[i,0])
        y_window_end = int(recon_positions[i,0] + recon_probe.shape[1])

        ## Cut the regions delimitated
        cut_obj[i] = recon_obj[y_window_begin:y_window_end,x_window_begin:x_window_end]

        ## Find the spectrum of the window
        obj_propagated = np.fft.fft2(cut_obj[i]*np.sum(recon_probe,0))
        obj_propagated = np.fft.fftshift(obj_propagated)

        ## Calculate the error (r-factor)
        error_original = np.sqrt(np.sum((np.abs(obj_propagated)**2 - diff_patterns[i])**2))
        error_min = error_original
        print("Original error:  ", error_original, "Positions:  ", recon_positions[i])
        
        ## Defines the number of global iterations of the algorithm (given by the radius decrease in the random case)
        if random_searches == True:
            number_of_iterations = radius_end
        else:
            number_of_iterations = 1
        
        ## Decrease radius in each iteration
        for y in reversed(range(number_of_iterations)):
            
            if random_searches == True:
                ## Create random offsets array from shuffling and cropping the original array
                indexes = np.arange(0,len(pos_offx_original[radius_begin:1+8*radius_end]),1)
                permutated_indexes = np.random.permutation(indexes)
                permutated_indexes = radius_begin*np.ones(len(permutated_indexes)) + permutated_indexes
                pos_offx = np.concatenate([pos_offx_original[int(i)] for i in permutated_indexes])
                pos_offy = np.concatenate([pos_offy_original[int(i)] for i in permutated_indexes])
            else:
                ## Use the whole array in order
                pos_offx = pos_offx_original
                pos_offy = pos_offy_original

            ## Now add small offsets and recalculate the previous variables to check if there is a better position nearby
            number_of_searches = len(pos_offx) # defines the number of positions to be searched during execution
            for j in range(number_of_searches):
                x_window_begin_off = int(recon_positions[i,1] + pos_offx[j])
                x_window_end_off = int(recon_positions[i,1] + recon_probe.shape[2] + pos_offx[j])
                y_window_begin_off = int(recon_positions[i,0] + pos_offy[j])
                y_window_end_off = int(recon_positions[i,0] + recon_probe.shape[1] + pos_offy[j])

                ## Checks if the window is appropriate
                if (y_window_begin_off < 0 or x_window_begin_off < 0 or y_window_end_off < 0 or x_window_end_off < 0):
                    continue

                cut_obj_off[i] = recon_obj[y_window_begin_off:y_window_end_off,x_window_begin_off:x_window_end_off]

                obj_propagated_off = np.fft.fft2(cut_obj_off[i]*np.sum(recon_probe,0))
                obj_propagated_off = np.fft.fftshift(obj_propagated_off)

                error_current = np.sqrt(np.sum((np.abs(obj_propagated_off)**2 - diff_patterns[i])**2))

                ## Check whether the new position improves the error metric
                if (error_current < error_min):
                    recon_positions[i,1] += pos_offx[j]
                    recon_positions[i,0] += pos_offy[j]
                    error_min = error_current

            error[i] = error_min
            print("Final error:  ", error_min, "Position: ", recon_positions[i])
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
    return recon_object, recon_probe, error, recon_positions

## Function to calculate offsets array
## Copied from ssc-cdi/-/blob/dev-poscor/cuda/src/ptycho/engines_common.cu
def init_offsets (maximum_n_neighborhoods):
    
    n_pos_neighbors = (2*maximum_n_neighborhoods+1)*(2*maximum_n_neighborhoods+1) -1
    pos_offx = np.zeros((n_pos_neighbors+1,1))
    pos_offy = np.zeros((n_pos_neighbors+1,1))

    idx = 1
	
    for curr_amplitude in range(1, maximum_n_neighborhoods+1):

        for j in range(-curr_amplitude, curr_amplitude+1):
            pos_offx[idx] = -curr_amplitude
            pos_offy[idx] = j
            idx = idx + 1

        for i in range(-curr_amplitude+1, curr_amplitude):
            pos_offx[idx] =  i
            pos_offy[idx] = -curr_amplitude
            idx = idx + 1
            pos_offx[idx] =  i
            pos_offy[idx] = curr_amplitude
            idx = idx + 1

        for j in range(-curr_amplitude+1, curr_amplitude+1):
            pos_offx[idx] = curr_amplitude
            pos_offy[idx] = j
            idx = idx + 1
    return pos_offx, pos_offy