import numpy as np
import os
import matplotlib.pyplot as plt
import json
import h5py
import sys

from sscIO import io
from sscPimega import pi540D, opt540D
from sscPimega import misc as miscPimega

from ..jupyter import slide_and_play

""" Sirius Scientific Computing Imports """
from sscPimega import pi540D, opt540D

""" sscCdi relative imports"""
from ..misc import read_hdf5, list_files_in_folder, select_specific_angles
from ..processing.restoration import restore_IO_SharedArray

def Geometry(distance,susp,fill):
    """ Get sscPimega detector geometry for certain distance and corresponding dictionary of input params

    Args:
        distance (_type_): _description_
        susp (int): parameter to ignore border pixels of each pimega chip
        fill (bool): fill blank information via interpolation after restoration

    Returns:
        geo: detector geometry variable
        params (dict): input additional parameters for restoration
    """
    params = {'geo':'nonplanar','opt':True,'mode':'virtual', 'fill': fill, 'susp': susp }
    project = pi540D.dictionary540D( distance, params ) 
    geo = pi540D.geometry540D( project )
    return geo, params

def restoration_ptycho_CAT(input_dict):
    """ Restore diffraction patterns and saves them in temporary folder

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json

    Returns:
        dic_list: list of dictionaries, each containing the necessary information for reading restored DPs 
        restored_data_info_list: list containing unique IDs for saved restored DPs
    """
    
    if input_dict["detector"] == "540D": 
        detector_size = 3072

    dic_list = []
    restored_data_info_list = []
    for acquisitions_folder in input_dict['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        print(f'\tRestoration of folder {acquisitions_folder}')

        filepaths0, filenames0 = list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")
        
        if input_dict['projections'] != []:
            filepaths, filenames = select_specific_angles(input_dict['projections'], filepaths0,  filenames0)
            print(f"\tUsing {len(filenames)} of {len(filenames0)} angle(s)")
        else:
            filepaths, filenames  = filepaths0, filenames0 
        
        input_dict["filepaths"], input_dict["filenames"] = filepaths, filenames

        geometry, params = Geometry(input_dict["detector_distance"]*1000,susp=input_dict["suspect_border_pixels"],fill=input_dict["fill_blanks"]) # distance in milimeters

        if input_dict["using_direct_beam"]:
            print("\t Using direct beam to find center: ",input_dict["DP_center"])
            input_dict["DP_center"][1], input_dict["DP_center"][0] = opt540D.mapping540D( input_dict["DP_center"][1], input_dict["DP_center"][0], input_dict["detector_distance"]*1000, params)
            print("\t\t New center: ",input_dict["DP_center"])

        dic = {} # dictionary for restoration function
        dic['path']     = filepaths
        dic['outpath']  = input_dict["temporary_DPs"]
        dic['order']    = "yx" 
        dic['rank']     = "ztyx" # order of axis
        dic['dataset']  = "entry/data/data"
        dic['gpus']     = input_dict["GPUs"]
        dic['init']     = 0
        dic['final']    = -1 # -1 to use all DPs
        dic['saving']   = 1  # save or not
        dic['timing']   = 0  # print timers 
        dic['blocksize']= 10

        input_dict["DP_center"][1],input_dict["DP_center"][0] = opt540D.mapping540D( input_dict["DP_center"][1], input_dict["DP_center"][0], pi540D.dictionary540D(input_dict["detector_distance"]*1000, params )) # change from raw to restored coordinates
        dic['center'] = (input_dict["DP_center"][1],input_dict["DP_center"][0]) # [1400,1400]

        if input_dict["detector_ROI_radius"] < 0:
            dic['roi'] = min(min(input_dict["DP_center"][1],detector_size-input_dict["DP_center"][1]),min(input_dict["DP_center"][0],detector_size-input_dict["DP_center"][0])) # get the biggest size possible such that the restored difpad is still squared
        else:
            dic['roi'] = input_dict["detector_ROI_radius"] # integer
        if input_dict["debug"]: 
            print(dic)
            dic['flat'] = np.ones([3072, 3072]) 
            dic['mask'] = np.zeros([3072, 3072])
        else:
            dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # np.ones([3072, 3072])
            dic['mask'] = read_hdf5(input_dict["mask"])[()][0, 0, :, :]      # np.zeros([3072, 3072])
        if os.path.isfile(input_dict["empty"]):
            dic['empty'] = read_hdf5(input_dict["empty"])[()][0, 0, :, :] 
        else:
            dic['empty'] = np.zeros_like(dic['flat'])
        dic['daxpy']    = [0,np.zeros([3072,3072])] 
        dic['geometry'] = geometry

        if len(filepaths) == 1:
            dic['path'] = dic['path'][0]
            os.system(f"h5clear -s {dic['path']}")
            restored_data_info = pi540D.ioSet_Backward540D( dic )
        else:
            os.system(f"h5clear -s {dic['path']}")
            restored_data_info = pi540D.ioSetM_Backward540D( dic )
            
        dic_list.append(dic)
        restored_data_info_list.append(restored_data_info)

    return dic_list, restored_data_info_list


def restoration_CAT(input_dict,method = 'IO'):
    
    if input_dict['using_direct_beam']: # if center coordinates are obtained from dbeam image at raw diffraction pattern; distance in mm
        input_dict['DP_center'][1], input_dict['DP_center'][0] = opt540D.mapping540D( input_dict['DP_center'][1], input_dict['DP_center'][0], pi540D.dictionary540D(input_dict["distance"], {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual'} ))
        print(f"Corrected center position: cy={input_dict['DP_center'][0]} cx={input_dict['DP_center'][1]}")

    """ Get detector geometry from distance """
    geometry, _ = Geometry(input_dict["distance"]*1000,susp=input_dict['suspect_border_pixels'],fill=input_dict['fill_blanks'])

    if input_dict['detector'] == '540D':
        detector_size = 3072

    if method == "CUDA":
        dic = {} # dictionary for restoration function
        dic['path']     = input_dict["data_path"]
        dic['outpath']  = input_dict["save_path"]
        dic['order']    = "yx" 
        dic['rank']     = "ztyx" # order of axis
        dic['dataset']  = "entry/data/data"
        dic['gpus']     = input_dict["GPUs"]
        dic['init']     = 0
        dic['final']    = -1 # -1 to use all DPs
        dic['saving']   = 1  # save or not
        dic['timing']   = 0  # print timers 
        dic['blocksize']= 10
        dic['center'] = (input_dict["DP_center"][1],input_dict["DP_center"][0]) # [1400,1400]
      
        if input_dict["detector_ROI_radius"] < 0:
            dic['roi'] = min(min(input_dict["DP_center"][1],detector_size-input_dict["DP_center"][1]),min(input_dict["DP_center"][0],detector_size-input_dict["DP_center"][0])) # get the biggest size possible such that the restored difpad is still squared
        elif input_dict["detector_ROI_radius"] == 0:
            dic['roi'] = detector_size//2
        else:
            dic['roi'] = input_dict["detector_ROI_radius"] # integer

        if input_dict["flatfield"] != '':    
            dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] 
        else:
            dic['flat'] = np.ones([detector_size, detector_size])

        if input_dict["mask"] != '':    
            dic['mask'] = read_hdf5(input_dict["mask"])[()][0, 0, :, :]
        else:
            dic['mask'] = np.zeros([detector_size, detector_size])

        if input_dict["empty_path"] != '':    
            dic['empty'] = read_hdf5(input_dict["empty_path"])[()][0, 0, :, :] 
        else:
            dic['empty'] = np.zeros([detector_size, detector_size])

        if input_dict["subtraction_path"] != '':
            subtraction_mask = read_hdf5(input_dict["subtraction_path"])[()][0, 0, :, :]*dic['flat'] # apply flat to subtraction measurement
            dic['daxpy'] = [-1,subtraction_mask]
        else:
            dic['daxpy'] = [0,np.zeros([3072,3072])] 

        dic['geometry'] = geometry

        if len(input_dict["data_path"]) == 1:
            dic['path'] = dic['path'][0]
            print("Restoring data...")
            restored_data_info = pi540D.ioSet_Backward540D( dic )
            print("Reading restored data...")
            DPs = pi540D.ioGet_Backward540D( dic, restored_data_info[0],restored_data_info[1])
            print("Cleaning temporary data...")
            pi540D.ioClean_Backward540D( dic, restored_data_info[0] )
        else:
            print("Restoring data...")
            restored_data_info = pi540D.ioSetM_Backward540D( dic )
            print("Reading restored data...")
            for file_number in range(len(input_dict["data_path"])):
                if file_number == 0:
                    DPs = pi540D.ioGetM_Backward540D( dic, restored_data_info, file_number) # read restored DPs from temporary folder
                else:
                    DPs = np.concatenate((DPs,pi540D.ioGetM_Backward540D( dic, restored_data_info, file_number)),axis=0)# read restored DPs from temporary folder
            print("Cleaning temporary data...")
            pi540D.ioCleanM_Backward540D( dic, restored_data_info )

        DPs = DPs.astype(np.float32)

    elif method == "IO":
    
        data_path = input_dict['data_path'][0]

        """ Restore data """
        os.system(f"h5clear -s {data_path}") # gambiarra because file is not closed at the backend!
        DPs = restore_IO_SharedArray(input_dict, geometry, data_path,method="IO")

    print(f"Output data shape {DPs.shape}. Type: {DPs.dtype}")
    print(f"Dataset size: {sys.getsizeof(DPs)/(1e6):.2f} MBs = {sys.getsizeof(DPs)/(1e9):.2f} GBs")

    if input_dict["save_path"] != '':
        if not os.path.exists(input_dict['save_path']):
            os.makedirs(input_dict['save_path'])
        print("Saving data at: ",input_dict['save_path'])
        h5f = h5py.File(os.path.join(input_dict['save_path'],input_dict["data_path"][0].rsplit('/',2)[-1]), 'w')
        h5f.create_dataset('entry/data/data', data=DPs)
        h5f.close()

    return DPs

