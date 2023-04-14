from sys import argv
import numpy as np
import os, time, json

""" Sirius Scientific Computing Imports """
import sscCdi

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for SIRIUS' CATERETE beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

np.random.seed(1)  # define seed for generation of the same random values

if __name__ == '__main__':

    t0 = time.time()

    input_dict = json.load(open(argv[1]))  # open input_dict file containing desired inputs

    print("\nCreating folders...")
    input_dict = sscCdi.caterete.cat_ptycho_processing.define_paths(input_dict)

    print("Reading files of interest...")
    filepaths, filenames = sscCdi.caterete.cat_ptycho_processing.get_files_of_interest(input_dict)

    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """

    t1 = time.time()
    print('Starting restoration...')
    restoration_dict_list, restored_data_info_list = sscCdi.caterete.cat_restoration.restoration_CAT(input_dict) # restoration of all frames; restored DPs saved at output temporary folder
    t2 = time.time()

    object,probe, input_dict, probe_positions = sscCdi.caterete.cat_ptycho_processing.cat_ptychography(input_dict,restoration_dict_list,restored_data_info_list)
    t3 = time.time()

    print('Finished reconstruction!\n')

    t4 = time.time()
    """ ===================== Post-processing ===================== """

    print("Post-processing data...")
    object = sscCdi.caterete.cat_ptycho_processing.crop_sinogram(object,input_dict,probe_positions)
    
    if input_dict['phase_unwrap'] != []: # Apply phase unwrap to data 
        print('\tUnwrapping sinogram...')
        phase = sscCdi.caterete.unwrap_in_parallel(object, input_dict["phase_unwrap"]) 
        sscCdi.misc.save_variable(input_dict,phase, flag = 'object_unwrapped')

    # if input_dict["FRC"] != []:
        # print('\tCalculating Fourier Ring Correlation...')
        # if input_dict['phase_unwrap'] != []: # if unwrapping, FRC is calculated on phase image
            # img = phase[input_dict["FRC"][0]]
        # else: # else, on the absorption image
            # img = np.abs(object)[input_dict["FRC"][0]] 
        # sscCdi.caterete.cat_ptycho_processing.calculate_FRC(img, input_dict)

    t5 = time.time()
    """ ===================== Save and preview data ===================== """
    object, probe = sscCdi.misc.save_volume_from_parts(input_dict)
    
    print('\nSaving Object of shape: ',object.shape)
    sscCdi.misc.save_variable(input_dict, object,flag='object')
  
    print('\nSaving Probe of shape: ',probe.shape)
    sscCdi.misc.save_variable(input_dict,probe,flag='probe')

    sscCdi.misc.save_json_logfile(input_dict) 
    sscCdi.misc.delete_temporary_folders(input_dict)

    t6 = time.time()
    time_elapsed_restauration = t2 - t1
    time_elapsed_ptycho = t3 - t2
    print('\n')
    print(f'Restauration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes ({100*(time_elapsed_restauration)/(t6 - t0):.0f}%)')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes ({100*(time_elapsed_ptycho)/(t6 - t0):.0f}%)')
    print(f'Post-processing time:  {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes ({100*(t5 - t4)/(t6 - t0):.0f}%)')
    print(f'Save time:             {t6 - t5:.2f} seconds = {(t6 - t5) / 60:.2f} minutes ({100*(t6 - t5)/(t6 - t0):.0f}%)')
    print(f'Total time:            {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')
