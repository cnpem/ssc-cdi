from sys import argv
import os
from time import time
import json
import numpy as np

import sscCdi

np.random.seed(1)  # define seed for generation of the same random values

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for Sirius/caterete beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    t0 = time()

    jason = json.load(open(argv[1]))  # open jason file containing desired inputs

    jason = sscCdi.caterete.ptycho_processing.define_paths(jason)

    sscCdi.caterete.ptycho_processing.create_output_directories(jason) # create all output directories of interest
    
    t1 = time()
    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """

    filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(jason)
    
    if len(filenames) > 1 and jason['SerialRestauration'] == False: # 3D batch restauration form (computationally faster, but not memory safe)
        difpads,_ , jason = sscCdi.caterete.ptycho_restoration.restauration_cat_3d(jason) # difpads is a list of size = len(Aquisition_folders)
        t2 = time()
        time_elapsed_restauration = t2 - t1
        object,probe,background, jason = sscCdi.caterete.ptycho_processing.cat_ptycho_3d(difpads,jason) 
        t3 = time()
        time_elapsed_ptycho = t3 - t2
    else: # serial reconstruction, either of single or multiple 2D frames
        object,probe,background,time_elapsed_restauration,time_elapsed_ptycho,jason  = sscCdi.caterete.ptycho_processing.cat_ptycho_serial(jason)  # restauration happens inside!

    if len(object) > 1: # Concatenate if object is a list of multiple elements. Each element is a ndarray of recons performed together
        object = np.concatenate(object, axis = 0)
        probe  = np.concatenate(probe, axis = 0)
        background = np.concatenate(background, axis = 0)
    else: # If one folder, get the first (and only) item on list
        object = object[0]
        probe  = probe[0]
        background = background[0]

    print('Finished Ptycho reconstruction!')

    t4 = time()
    """ ===================== Post-processing ===================== """

    cropped_sinogram = sscCdi.caterete.ptycho_processing.crop_sinogram(object,jason)
    
    if jason['Phaseunwrap'][0]: # Apply phase unwrap to data 
        print('Unwrapping sinogram...')
        phase,absol = sscCdi.caterete.ptycho_processing.apply_phase_unwrap(cropped_sinogram, jason) # phase = np.angle(object), absol = np.abs(object)
        cropped_sinogram = absol*np.exp(-1j*phase)
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram, os.path.join(jason['ReconsPath'],'unwrapped_object_' + jason["Acquisition_Folders"][0]))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)


    jason = sscCdi.caterete.ptycho_processing.calculate_FRC(cropped_sinogram, jason)

    t5 = time()
    """ ===================== Save and preview data ===================== """

    if jason["LogfilePath"] != "":  sscCdi.caterete.misc.save_json_logfile(jason["LogfilePath"], jason) # overwrite logfile with new information
            
    if jason['SaveObj']:
        print('Saving Object!')
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram  , os.path.join(jason['ReconsPath'],jason["Acquisition_Folders"][0]) + '_object')

    if jason['SaveProbe']:
        print('Saving Probe!')
        sscCdi.caterete.ptycho_processing.save_variable(probe, os.path.join(jason['ReconsPath'], jason["Acquisition_Folders"][0]) + '_probe' )

    for i in range(phase.shape[0]):
        sscCdi.caterete.ptycho_processing.preview_ptycho(jason, phase, absol, probe, frame=i)

    t6 = time()
    print('\n')
    print(f'Restauration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes')
    print(f'Post-processing time:  {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Save time:             {t6 - t5:.2f} seconds = {(t6 - t5) / 60:.2f} minutes')
    print(f'Total time:            {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')