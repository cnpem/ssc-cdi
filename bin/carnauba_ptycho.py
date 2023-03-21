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

    input_dict = json.load(open(argv[1]))  # open input_dict file containing desired inputs
    beamline = input_dict['beamline']

    input_dict = sscCdi.caterete.ptycho_processing.define_paths(input_dict)

    sscCdi.caterete.ptycho_processing.create_output_directories(input_dict) # create all output directories of interest
    
    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """
    t1 = time()

    filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(input_dict)
    if len(filenames) > 1 and input_dict['SerialRestauration'] == False: # 3D batch restauration form (computationally faster, but not memory safe)
        difpads,_ , input_dict = sscCdi.caterete.ptycho_restoration.restoration_3d(input_dict) # difpads is a list of size = len(Aquisition_folders)
        t2 = time()
        time_elapsed_restauration = t2 - t1
        object,probe,background, input_dict = sscCdi.caterete.ptycho_processing.cat_ptycho_3d(difpads,input_dict) 
        t3 = time()
        time_elapsed_ptycho = t3 - t2
    else: # serial reconstruction, either of single or multiple 2D frames
        object,probe,background,time_elapsed_restauration,time_elapsed_ptycho,input_dict  = sscCdi.caterete.ptycho_processing.cat_ptycho_serial(input_dict)  # restauration happens inside!

    if len(object) > 1: # Concatenate if object is a list of multiple elements. Each element is a ndarray of recons performed together
        object = np.concatenate(object, axis = 0)
        probe  = np.concatenate(probe, axis = 0)
        background = np.concatenate(background, axis = 0)
    else: # If one folder, get the first (and only) item on list
        object = object[0]
        probe  = probe[0]
        background = background[0]

    print('Finished Ptycho reconstruction!')

    """ ===================== Post-processing ===================== """
    t4 = time()

    cropped_sinogram = sscCdi.caterete.ptycho_processing.crop_sinogram(object,input_dict)

    
    if input_dict['Phaseunwrap'][0]: # Apply phase unwrap to data 
        print('Unwrapping sinogram...')
        phase,absol = sscCdi.caterete.ptycho_processing.apply_phase_unwrap(cropped_sinogram, input_dict) # phase = np.angle(object), absol = np.abs(object)
        cropped_sinogram = absol*np.exp(-1j*phase)
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram, os.path.join(input_dict['ReconsPath'],'unwrapped_object_' + input_dict["Acquisition_Folders"][0]))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)


    # input_dict = sscCdi.caterete.ptycho_processing.calculate_FRC(cropped_sinogram, input_dict)

    t5 = time()

    """ ===================== Save and preview data ===================== """

    if input_dict["LogfilePath"] != "":  sscCdi.caterete.misc.save_json_logfile(input_dict["LogfilePath"], input_dict) # overwrite logfile with new information
            
    if input_dict['SaveObj']:
        print('Saving Object!')
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram  , os.path.join(input_dict['ReconsPath'], 'object_' + input_dict["Acquisition_Folders"][0]))

    if input_dict['SaveProbe']:
        print('Saving Probe!')
        sscCdi.caterete.ptycho_processing.save_variable(probe, os.path.join(input_dict['ReconsPath'], 'probe_' + input_dict["Acquisition_Folders"][0]) )

    for i in range(phase.shape[0]):
        sscCdi.caterete.ptycho_processing.preview_ptycho(input_dict, phase, absol, probe, frame=i)

    t6 = time()
    print('\n')
    print(f'Restauration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes')
    print(f'Post-processing time:  {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Save time:             {t6 - t5:.2f} seconds = {(t6 - t5) / 60:.2f} minutes')
    print(f'Total time:            {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')

    

