from sys import argv
import os
import time
import json
import numpy as np

import sscCdi

np.random.seed(1)  # define seed for generation of the same random values

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for Sirius/caterete beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def cat_ptycho_3d(difpads,jason):
    t2 = time()

    scans_string, positions_string = jason['scans_string']. jason['positions_string']

    sinogram = []
    probe = []
    background = [] 

    '''
        BEGIN MAIN PTYCHO RUN
    '''  
    count = -1
    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(jason['ProposalPath'] , acquisitions_folder,scans_string), look_for_extension=".hdf5")
        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)
        
        total_frames = len(filenames)
        print('\nFilenames: ', filenames)
        args = [jason, filenames, filepaths, jason['ProposalPath'] , acquisitions_folder, scans_string, positions_string]

        # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
        if count == 0:
            object_shapey, object_shapex, maxroi, hsize, object_pixel_size, jason =sscCdi.caterete.ptycho_processing.set_object_shape(difpads[count],args)
            jason["object_pixel"] = object_pixel_size
            args[0] = jason # update args

        params = (args,maxroi,hsize,(object_shapey,object_shapex),total_frames)

        threads = len(jason['GPUs'])
        
        # Main ptycho iteration on ALL frames in threads
        sinogram3d ,probe3d, background3d = sscCdi.caterete.ptycho_processing.ptycho3d_batch(difpads[count], threads, params)

        sinogram.append(sinogram3d)
        probe.append(probe3d)
        background.append(background3d)
    
    t3 = time()

    '''
        END MAIN PTYCHO RUN
    '''
    return sinogram,probe,background,t2,t3, jason


def cat_ptycho_serial(jason):

    scans_string, positions_string = jason['scans_string']. jason['positions_string']

    sinogram_list = []
    probe_list    = []
    background_list      = []
    
    counter = 0
    first_iteration = True
    first_of_folder = True
    for acquisitions_folder in jason['Acquisition_Folders']:  

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(jason['ProposalPath'] , acquisitions_folder,scans_string), look_for_extension=".hdf5")

        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)  

        for measurement_file, measurement_filepath in zip(filenames, filepaths):   
            
            arguments = (args,acquisitions_folder,measurement_file,measurement_filepath,len(filenames))

            difpads,_ , jason = sscCdi.caterete.ptycho_restauration.restauration_cat_2d(arguments,jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads'],first_run=first_iteration) # Restauration of 2D Projection (difpads - real, is a ndarray of size (1,:,:,:))

            arg = [jason, [measurement_file], [measurement_filepath], jason['ProposalPath'] , jason["Acquisition_Folders"][0], scans_string, positions_string]

            if first_iteration: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                object_shapey, object_shapex, maxroi, hsize, object_pixel_size, jason = sscCdi.caterete.ptycho_processing.set_object_shape(difpads,arg)
                jason["object_pixel"] = object_pixel_size
                arg[0] = jason # update args
                first_iteration = False
            
            params = (arg,maxroi,hsize,(object_shapey,object_shapex),len(filenames))

            object_dummy = np.zeros((1,object_shapey,object_shapex),dtype = complex) # build 3D Sinogram
            probe_dummy  = np.zeros((1,1,difpads.shape[-2],difpads.shape[-1]),dtype = complex)
            background_dummy    = np.zeros((1,difpads.shape[-2],difpads.shape[-1]))
            
            t2 = time() 
            object2d, probe2d, background2d = sscCdi.caterete.ptycho_processing.ptycho_main(difpads, object_dummy, probe_dummy, background_dummy, params, 0, 1, jason['GPUs'])   # Main ptycho iteration on ALL frames in threads
            t3 = time()

            if first_of_folder:
                object = object2d
                probe  = probe2d
                background    = background2d
                first_of_folder = False
            else:
                object = np.concatenate((object,object2d), axis = 0)
                probe  = np.concatenate((probe,probe2d),  axis = 0)
                background = np.concatenate((background,background2d),    axis = 0)
            counter +=1

        first_of_folder = True

        sinogram_list.append(object)
        probe_list.append(probe)
        background_list.append(background)

    return sinogram_list, probe_list, background_list, t2, t3,jason



if __name__ == '__main__':

    t0 = time()

    jason = json.load(open(argv[1]))  # open jason file containing desired inputs


    jason = sscCdi.caterete.ptycho_processing.define_paths(jason)

    sscCdi.caterete.ptycho_processing.create_output_directories(jason) # create all output directories of interest
    
    """ =========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D ===================== """
    t1 = time()

    jason['SerialRestauration'] = True

    filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(jason)
    if len(filenames) > 1 and jason['SerialRestauration'] == False: # 3D batch restauration form (computationally faster, but not memory safe)
        difpads,_ , jason = sscCdi.caterete.ptycho_restauration.restauration_cat_3d(jason) # difpads is a list of size = len(Aquisition_folders)
        object,probe,background, t2,t3, jason  =  cat_ptycho_3d(difpads,jason) 
    else: # serial reconstruction, either of single or multiple 2D frames
        object,probe,background,t2,t3,jason  = cat_ptycho_serial(jason)  # restauration happens inside!

    if len(object) > 1: # Concatenate if object is a list of multiple elements. Each element is a ndarray of recons performed together
        object = np.concatenate(object, axis = 0)
        probe  = np.concatenate(probe, axis = 0)
        background    = np.concatenate(background, axis = 0)
    else: # If one folder, get the first (and only) item on list
        object = object[0]
        probe  = probe[0]
        background    = background[0]

    print('Finished Ptycho reconstruction!')

    cropped_sinogram = sscCdi.caterete.ptycho_processing.crop_sinogram(object,jason)

    t4 = time()
    
    if jason['Phaseunwrap'][0]: # Apply phase unwrap to data 
        print('Unwrapping sinogram...')
        phase,absol = sscCdi.caterete.ptycho_processing.apply_phase_unwrap(cropped_sinogram, jason) # phase = np.angle(object), absol = np.abs(object)
        cropped_sinogram = absol*np.exp(-1j*phase)
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram, os.path.join(jason['ObjPath'],'unwrapped_object_' + jason["Acquisition_Folders"][0]))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)

    t5 = time()

    jason = calculate_FRC(cropped_sinogram, jason)

    if jason["LogfilePath"] != "":  sscCdi.caterete.misc.save_json_logfile(jason["LogfilePath"], jason) # overwrite logfile with new information
            
    if jason['SaveObj']:
        print('Saving Object!')
        sscCdi.caterete.ptycho_processing.save_variable(cropped_sinogram  , os.path.join(jason['ObjPath'], 'object_' + jason["Acquisition_Folders"][0]))

    if jason['SaveProbe']:
        print('Saving Probe!')
        sscCdi.caterete.ptycho_processing.save_variable(probe, os.path.join(jason['ProbePath'], 'probe_' + jason["Acquisition_Folders"][0]))

    for i in range(phase.shape[0]):
        sscCdi.caterete.ptycho_processing.preview_ptycho(jason, phase, absol, probe, frame=i)

    t6 = time()
    print(f'\nElapsed time for restauration of all difpads: {t2 - t1:.2f} seconds = {(t2 - t1) / 60:.2f} minutes')
    print(f'Ptychography time: {t3 - t2:.2f} seconds = {(t3 - t2) / 60:.2f} minutes')
    print(f'Auto Crop object time: {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes')
    print(f'Phase unwrap object time: {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Total time: {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')