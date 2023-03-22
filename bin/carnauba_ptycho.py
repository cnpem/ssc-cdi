from sys import argv
import numpy as np
import os, time, json

""" Sirius Scientific Computing Imports """
import sscCdi

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for SIRIUS' CARNAUBA beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

np.random.seed(1)  # define seed for generation of the same random values

if __name__ == '__main__':

    t0 = time.time()

    input_dict = json.load(open(argv[1]))  # open input_dict file containing desired inputs

    print("Creating folders...")
    input_dict = sscCdi.ptycho.ptycho_processing.define_paths(input_dict)

    print("Reading files of interest...")
    filepaths, filenames = sscCdi.ptycho.ptycho_processing.get_files_of_interest(input_dict)

    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """

    t1 = time.time()
    print('Starting restoration...')
    DPs, input_dict = sscCdi.caterete.cnb_restoration.restoration_CNB(input_dict) # restoration of all frames; restored DPs saved at output temporary folder
    t2 = time.time()

    print('Starting ptychography...')
    # object,probe, input_dict = sscCdi.ptycho.ptycho_processing.cat_ptychography(input_dict,restoration_dict_list,restored_data_info_list)
    t3 = time.time()

    print('\tFinished reconstruction!')

    t4 = time.time()
    """ ===================== Post-processing ===================== """

    cropped_sinogram = sscCdi.ptycho.ptycho_processing.crop_sinogram(object,input_dict)
    
    if input_dict['phase_unwrap'][0]: # Apply phase unwrap to data 
        print('Unwrapping sinogram...')
        phase, absol = sscCdi.ptycho.ptycho_processing.apply_phase_unwrap(cropped_sinogram, input_dict) # phase = np.angle(object), absol = np.abs(object)
        cropped_sinogram = absol*np.exp(-1j*phase)
        sscCdi.ptycho.ptycho_processing.save_variable(cropped_sinogram, os.path.join(input_dict['ReconsPath'],'unwrapped_object_' + input_dict["acquisition_folders"][0]))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)

    print('Calculating Fourier Ring Correlation...')
    input_dict = sscCdi.ptycho.ptycho_processing.calculate_FRC(cropped_sinogram, input_dict)

    t5 = time.time()
    """ ===================== Save and preview data ===================== """

    if input_dict["LogfilePath"] != "":  sscCdi.misc.misc.save_json_logfile(input_dict["LogfilePath"], input_dict) # overwrite logfile with new information
            
    print('Saving Object...')
    sscCdi.ptycho.ptycho_processing.save_variable(cropped_sinogram  , os.path.join(input_dict['ReconsPath'],input_dict["acquisition_folders"][0]) + '_object')

    print('Saving Probe...')
    sscCdi.ptycho.ptycho_processing.save_variable(probe, os.path.join(input_dict['ReconsPath'], input_dict["acquisition_folders"][0]) + '_probe' )

    for i in range(phase.shape[0]):
        sscCdi.ptycho.ptycho_processing.preview_ptycho(input_dict, phase, absol, probe, frame=i)

    t6 = time.time()
    time_elapsed_restauration = t2 - t1
    time_elapsed_ptycho = t3 - t2
    print('\n')
    print(f'Restauration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes')
    print(f'Post-processing time:  {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Save time:             {t6 - t5:.2f} seconds = {(t6 - t5) / 60:.2f} minutes')
    print(f'Total time:            {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')