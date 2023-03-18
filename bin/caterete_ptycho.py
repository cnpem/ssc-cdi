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

    jason = json.load(open(argv[1]))  # open jason file containing desired inputs

    jason = sscCdi.ptycho.ptycho_processing.define_paths(jason)

    filepaths, filenames = sscCdi.ptycho.ptycho_processing.get_files_of_interest(jason)

    t1 = time.time()
    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """

    restoration_dict, restored_data_info = sscCdi.caterete.cat_restoration.restoration_cuda_parallel(jason) # difpads is a list of size = len(Aquisition_folders)
    t2 = time.time()

    #TODO: call ptycho
    object,probe, jason = sscCdi.ptycho.ptycho_processing.cat_ptychography(jason,restoration_dict,restored_data_info)
    t3 = time.time()

    if len(object) > 1: # Concatenate if object is a list of multiple elements. Each element is a ndarray of recons performed together
        object = np.concatenate(object, axis = 0)
        probe  = np.concatenate(probe, axis = 0)
    else: # If one folder, get the first (and only) item on list
        object = object[0]
        probe  = probe[0]

    print('Finished Ptycho reconstruction!')

    t4 = time.time()
    """ ===================== Post-processing ===================== """

    cropped_sinogram = sscCdi.ptycho.ptycho_processing.crop_sinogram(object,jason)
    
    if jason['phase_unwrap'][0]: # Apply phase unwrap to data 
        print('Unwrapping sinogram...')
        phase,absol = sscCdi.ptycho.ptycho_processing.apply_phase_unwrap(cropped_sinogram, jason) # phase = np.angle(object), absol = np.abs(object)
        cropped_sinogram = absol*np.exp(-1j*phase)
        sscCdi.ptycho.ptycho_processing.save_variable(cropped_sinogram, os.path.join(jason['ReconsPath'],'unwrapped_object_' + jason["acquisition_folders"][0]))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)


    jason = sscCdi.ptycho.ptycho_processing.calculate_FRC(cropped_sinogram, jason)

    t5 = time.time()
    """ ===================== Save and preview data ===================== """

    if jason["LogfilePath"] != "":  sscCdi.misc.misc.save_json_logfile(jason["LogfilePath"], jason) # overwrite logfile with new information
            
    print('Saving Object!')
    sscCdi.ptycho.ptycho_processing.save_variable(cropped_sinogram  , os.path.join(jason['ReconsPath'],jason["acquisition_folders"][0]) + '_object')

    print('Saving Probe!')
    sscCdi.ptycho.ptycho_processing.save_variable(probe, os.path.join(jason['ReconsPath'], jason["acquisition_folders"][0]) + '_probe' )

    for i in range(phase.shape[0]):
        sscCdi.ptycho.ptycho_processing.preview_ptycho(jason, phase, absol, probe, frame=i)

    t6 = time.time()
    time_elapsed_restauration = t2 - t1
    time_elapsed_ptycho = t3 - t2
    print('\n')
    print(f'Restauration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes')
    print(f'Post-processing time:  {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Save time:             {t6 - t5:.2f} seconds = {(t6 - t5) / 60:.2f} minutes')
    print(f'Total time:            {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')