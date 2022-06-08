from distutils import core
import sscCdi
from sys import argv
import os
import time
import h5py
import json
import numpy as np

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from sscCdi.caterete.ptycho_restauration import *
from sscCdi.caterete.unwrap import *

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for Sirius/caterete beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def preview_ptycho(jason, phase, absol, probe, frame = 0):
    if jason['Preview']:  # Preview Reconstruction:
        # # '''
        # plt.figure()
        # plt.scatter(probe_positionsi[:, 0], probe_positionsi[:, 1])
        # plt.scatter(datapack['rois'][:, 0, 0], datapack['rois'][:, 0, 1])
        # plt.savefig(jason['PreviewFolder'] + '/scatter_2d.png', format='png', dpi=300)
        # plt.clf()
        # plt.close()
        # # '''
        # print('probe shape', probe[frame].shape)
        # print('phase shape', phase[frame].shape)
        # Show probe:
        plotshow([abs(Prop(p, jason['f1'])) for p in probe[frame]] + [p for p in probe[frame]], file=jason['PreviewFolder'] + '/probe_2d_' + str(frame), nlines=2)

        # Show object:
        # ango = np.angle(sinogram[frame])
        # abso = np.clip(abs(sinogram[frame]), 0.0, np.max(abs(sinogram[frame][hsize:maxroi, hsize:maxroi])))
        # abso = abs(sinogram[frame])

        plotshow([phase[frame], absol[frame]], subplot_title=['Phase', 'Magnitude'], file=jason['PreviewFolder'] + '/object_2d_' + str(frame), cmap='gray', nlines=1)
        

def cat_ptycho_3d(difpads,args):

    jason, ibira_datafolder, scans_string, positions_string = args

    sinogram = []
    probe = []
    bkg = [] 

    '''
        BEGIN MAIN PTYCHO RUN
    '''  
    count = -1
    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)
        
        total_frames = len(filenames)
        print('\nFilenames in cat_ptycho_3d: ', filenames)
        args = [jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string, positions_string]

        # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
        if count == 0:
            object_shapey, object_shapex, maxroi, hsize, object_pixel_size, jason = set_object_shape(difpads[count],args)
            jason["object_pixel"] = object_pixel_size
            args[0] = jason # update args

        params = (args,maxroi,hsize,(object_shapey,object_shapex),total_frames)

        threads = len(jason['GPUs'])
        
        # Main ptycho iteration on ALL frames in threads
        sinogram3d ,probe3d, bkg3d = ptycho3d_batch(difpads[count], threads, params)

        sinogram.append(sinogram3d)
        probe.append(probe3d)
        bkg.append(bkg3d)


    '''
        END MAIN PTYCHO RUN
    '''
    return sinogram,probe,bkg


def cat_ptycho_serial(args):

    jason, ibira_datafolder, scans_string, positions_string = args

    sinogram_list = []
    probe_list    = []
    bkg_list      = []
    
    first_run = True
    for acquisitions_folder in jason['Acquisition_Folders']:  

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")

        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)  

        for measurement_file, measurement_filepath in zip(filenames, filepaths):   
            
            arguments = (args,acquisitions_folder,measurement_file,measurement_filepath,len(filenames))

            difpads,_ , jason = restauration_cat_2d(arguments,jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads'],first_run=first_run) # Restauration of 2D Projection (difpads - real, is a ndarray of size (1,:,:,:))

            arg = [jason, [measurement_file], [measurement_filepath], ibira_datafolder, acquisition_folder, scans_string, positions_string]

            if first_run: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                object_shapey, object_shapex, maxroi, hsize, object_pixel_size, jason = set_object_shape(difpads,arg)
                jason["object_pixel"] = object_pixel_size
                arg[0] = jason # update args
                params = (arg,maxroi,hsize,(object_shapey,object_shapex),len(filenames))

            sinogram = np.zeros((1,object_shapey,object_shapex),dtype = complex) # build 3D Sinogram
            probe    = np.zeros((1,1,difpads.shape[-2],difpads.shape[-1]),dtype = complex)
            bkg      = np.zeros((1,difpads.shape[-2],difpads.shape[-1]))
            
            t2 = time() 
            sinogram, probe2d, bkg2d = ptycho_main(difpads, sinogram, probe, bkg, params, 0, 1, jason['GPUs'])   # Main ptycho iteration on ALL frames in threads
            t3 = time()

            if first_run == True:
                object = sinogram
                probe  = probe2d
                bkg    = bkg2d
                first_run = False
            else:
                object = np.concatenate((object,sinogram), axis = 0)
                probe  = np.concatenate((probe,probe2d),  axis = 0)
                bkg    = np.concatenate((bkg,bkg2d),    axis = 0)

    sinogram_list.append(object)
    probe_list.append(probe)
    bkg_list.append(bkg)

    return sinogram_list, probe_list, bkg_list, t2, t3,jason




#TODO: if you put one frame, it will be done by 2d code and it will be set as frame 0, always

if __name__ == '__main__':

    t0 = time()

    jason = json.load(open(argv[1]))  # Open jason file

    np.random.seed(jason['Seed'])  # define seed for generation of the same random values

    if 'PreviewGCC' not in jason: jason['PreviewGCC'] = [False,""] # flag to save previews of interest only to GCC, not to the beamline user
    
    #=========== Set Parameters and Folders =====================
    ibira_datafolder = jason['ProposalPath'] 
    acquisition_folder = jason["Acquisition_Folders"][0]
    print('ibira_datafolder   : ', ibira_datafolder)
    print('acquisition_folder : ', acquisition_folder)
 
    if jason["PreviewGCC"][0] == True: # path convention for GCC users
        if 'LogfilePath' not in jason: jason['LogfilePath'] = ''
        jason["PreviewGCC"][1]  = os.path.join(jason["PreviewGCC"][1],acquisition_folder)
        jason["PreviewFolder"]  = os.path.join(jason["PreviewGCC"][1],'preview')
        jason["SaveDifpadPath"] = os.path.join(jason["PreviewGCC"][1],'difpads')
        jason["ObjPath"]        = os.path.join(jason["PreviewGCC"][1],'reconstruction')
        jason["ProbePath"]      = os.path.join(jason["PreviewGCC"][1],'reconstruction')
        jason["BkgPath"]        = os.path.join(jason["PreviewGCC"][1],'reconstruction')
    else:
        beamline_outputs_path = os.path.join(ibira_datafolder.rsplit('/',3)[0], 'proc','recons',acquisition_folder) # standard folder chosen by CAT for their outputs
        jason["LogfilePath"]    = beamline_outputs_path
        jason["PreviewFolder"]  = beamline_outputs_path
        jason["SaveDifpadPath"] = beamline_outputs_path
        jason["ObjPath"]        = beamline_outputs_path
        jason["ProbePath"]      = beamline_outputs_path
        jason["BkgPath"]        = beamline_outputs_path

    create_output_directories(jason) # create all output directories of interest

    if jason['InitialObj'] in jason and jason['InitialObj']   != "": jason['InitialObj']   = os.path.join(jason['ObjPath'],   jason['InitialObj']) # append initialObj filename to path
    if jason['InitialObj'] in jason and jason['InitialProbe'] != "": jason['InitialProbe'] = os.path.join(jason['ProbePath'], jason['InitialProbe'])
    if jason['InitialObj'] in jason and jason['InitialBkg']   != "": jason['InitialBkg']   = os.path.join(jason['BkgPath'],   jason['InitialBkg'])

    if 'OldFormat' not in jason: # flag to indicate if we are working with old or new input file format. Old format will be deprecated in the future.

        scans_string = 'scans'
        positions_string = 'positions'

        images_folder    = os.path.join(acquisition_folder,'images')
        positions_folder = os.path.join(ibira_datafolder,acquisition_folder,'positions')
        scans_folder     = os.path.join(ibira_datafolder,acquisition_folder,'scans')

        input_dict = json.load(open(os.path.join(ibira_datafolder,acquisition_folder,'mdata.json')))
        jason["Energy"] = input_dict['/entry/beamline/experiment']["energy"]
        jason["DetDistance"] = input_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
        jason["RestauredPixelSize"] = input_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
        jason["DetectorExposure"].append(input_dict['/entry/beamline/detector']['pimega']["exposure time"])


        jason["EmptyFrame"] = os.path.join(ibira_datafolder,images_folder,'empty.hdf5')
        jason["FlatField"]  = os.path.join(ibira_datafolder,images_folder,'flat.hdf5')
        jason["Mask"]       = os.path.join(ibira_datafolder,images_folder,'mask.hdf5')

    else:
        scans_string = ''
        positions_string = ''
        flatfield = np.load(jason["FlatField"])
        empty = np.asarray(h5py.File(jason['EmptyFrame'], 'r')['/entry/data/data']).squeeze().astype(np.float32)
    
    filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisition_folder,scans_string), look_for_extension=".hdf5")

    if jason['Projections'] != []:
        filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths, filenames)
    
    args = (jason, ibira_datafolder, scans_string, positions_string)

      #=========== MAIN PTYCHO RUN: RESTAURATION + PTYCHO 3D and 2D =====================
    t1 = time()

    serial = True #jason['Serial3D'] # Here is the bool variable to choose serial version of 3D or not.

    if len(filenames) > 1 and serial == False: # 3D batch form (computational time is faster, but not memory safe)
        difpads,_ , jason = restauration_cat_3d(args,jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads']) # Restauration of ALL Projections (difpads - real, is a list of size len(Aquisition_folders))
        t2 = time()
        object,probe,bkg  =  cat_ptycho_3d(difpads,args) # Ptycho of ALL Projections (object - complex, probe - complex, bkg - real, are a list of size len(Aquisition_folders))
        t3 = time()
    else: # serial reconstruction, either of single or multiple 2D frames
        object,probe,bkg,t2,t3,jason  = cat_ptycho_serial(args)  # restauration happens inside

    if len(object) > 1: # Concatenate if object is a list because divided into more than one folder (All projections in each folder are resolved together, and put on a list of size len(Aquisition_folders))
        object = np.concatenate(object, axis = 0)
        probe  = np.concatenate(probe, axis = 0)
        bkg    = np.concatenate(bkg, axis = 0)
    else: # If one folder, get the first (and only) item on list
        object = object[0]
        probe  = probe[0]
        bkg    = bkg[0]

    print('Finished Ptycho reconstruction!')

    cropped_sinogram = crop_sinogram(object, jason)

    t4 = time()
    
    if jason['Phaseunwrap'][0]: # Apply phase unwrap to data
        print('Unwrapping sinogram...')
        phase,absol = apply_phase_unwrap(cropped_sinogram, jason) # phase = np.angle(object), absol = np.abs(object)
        save_variable2(phase, os.path.join(jason['ObjPath'],'unwrapped_phase_' + acquisition_folder))
        save_variable2(absol, os.path.join(jason['ObjPath'],'unwrapped_magnitude_' + acquisition_folder))
    else:
        print("Extracting phase and magnitude...")
        phase = np.angle(cropped_sinogram)
        absol = np.abs(cropped_sinogram)

    t5 = time()

    calculate_FRC(cropped_sinogram, jason)

    if jason["LogfilePath"] != "":  sscCdi.caterete.misc.save_json_logfile(jason["LogfilePath"], jason) # overwrite logfile with new information
            
    if jason['SaveObj']:
        print('Saving Object!')
        save_variable(cropped_sinogram, os.path.join(jason['ObjPath'], 'object_' + acquisition_folder))

    if jason['SaveProbe']:
        print('Saving Probe!')
        save_variable(probe, os.path.join(jason['ProbePath'], 'probe_' + acquisition_folder))

    preview_ptycho(jason, phase, absol, probe, frame=0)

    t6 = time()
    print(f'\nElapsed time for restauration of all difpads: {t2 - t1:.2f} seconds = {(t2 - t1) / 60:.2f} minutes')
    print(f'Ptycho batch total time: {t3 - t2:.2f} seconds = {(t3 - t2) / 60:.2f} minutes')
    print(f'Auto Crop object time: {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes')
    print(f'Phase unwrap object time: {t5 - t4:.2f} seconds = {(t5 - t4) / 60:.2f} minutes')
    print(f'Total time: {t6 - t0:.2f} seconds = {(t6 - t0) / 60:.2f} minutes')