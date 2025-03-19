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

def position_correction_python(diffraction_patterns, recon_positions, recon_object, recon_probe, inputs):
    
        ## Array to store cropped diffraction patterns
    cut_obj = np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2])) + 1j*np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2]))
    cut_obj_off = np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2])) + 1j*np.zeros((recon_positions.shape[0], recon_probe.shape[1], recon_probe.shape[2]))

    ## Array to store offsets in pixels to try
    pos_offx = np.array([ 0, 1, -1, 0, 0, -1, -1, 1, 1]);
    pos_offy = np.array([ 0, 0, 0, 1, -1, -1, 1, -1, 1]);

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
        error_max = error_original
        print("Error original:  ", error_original, "Positions:  ", recon_positions[i])

        #plt.imshow(np.abs(obj_propagated))
        #plt.show()


        ## Now add small offsets and recalculate the previous variables to check if there is a better position nearby
        for j in range(len(pos_offx)):
            x_window_begin_off = int(recon_positions[i,1] + pos_offx[j])
            x_window_end_off = int(recon_positions[i,1] + recon_probe.shape[2] + pos_offx[j])
            y_window_begin_off = int(recon_positions[i,0] + pos_offy[j])
            y_window_end_off = int(recon_positions[i,0] + recon_probe.shape[1] + pos_offy[j])

            cut_obj_off[i] = recon_obj[y_window_begin_off:y_window_end_off,x_window_begin_off:x_window_end_off]

            obj_propagated_off = np.fft.fft2(cut_obj_off[i]*np.sum(recon_probe,0))
            obj_propagated_off = np.fft.fftshift(obj_propagated_off)

            error_current = np.sqrt(np.sum((np.abs(obj_propagated_off)**2 - diff_patterns[i])**2))

            ## Check wether the new position improves the error metric
            if (error_current < error_max):
                recon_positions[i,1] += pos_offx[j]
                recon_positions[i,0] += pos_offy[j]
                error_max = error_current

        print("Error final:  ", error_max, "Position: ", recon_positions[i])
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
    return recon_object, recon_probe, error, recon_positions