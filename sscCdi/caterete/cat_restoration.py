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
    """ 
    Get sscPimega detector geometry for certain distance and corresponding dictionary of input params

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
    return geo, project

def flatfield_forward_restoration(input_dict: dict):
    """
    Generates forward flat field restoration
    
    Args:
        input_dict (dict): 
        
    Returns:
        flat_forward (nunpy array): array with new flat
    """
    
    flat_backward = np.load(input_dict["flatfield"])
    geometry, project = Geometry(
        input_dict["detector_distance"]*1000,
        susp = input_dict["suspect_border_pixels"],
        fill = input_dict["fill_blanks"]
    ) # distance in milimeters
    
    flat_forward = pi540D.forward540D(flat_backward,  geometry)
        
    return flat_forward

def restoration_ptycho_CAT(input_dict):
    """ 
    Restore diffraction patterns from CAT beamline using CUDA restoration and saves them in temporary folder

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json

    Returns:
        dic_list: list of dictionaries, each containing the necessary information for reading restored DPs 
        restored_data_info_list: list containing unique IDs for saved restored DPs
        filepaths: list containing paths of the required projections
        filenames: list containing names of the required projections
        folders_name: list containing the folders with the required projections
        folders_number: list containig the number of the folders with the required projections
    """
    
    if input_dict["detector"] == "540D": 
        detector_size = 3072

    filepaths_list = []
    filenames_list = []
    folder_numbers_list = []
    folder_names_list = []

    for folder_number, acquisitions_folder in enumerate(input_dict['acquisition_folders']):
        filepaths, filenames = list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")
        filepaths_list.extend(filepaths)
        filenames_list.extend(filenames)
        folder_numbers_list.extend([folder_number]*len(filenames))
        folder_names_list.extend([acquisitions_folder]*len(filenames))

    if input_dict['projections'] != []:
        filepaths, filenames, folders_name, folders_number = select_specific_angles(input_dict['projections'], filepaths_list,  filenames_list, folder_names_list, folder_numbers_list)
        print(f"\tUsing {len(filenames)} of {len(filenames_list)} angle(s)")
    else:
        filepaths, filenames, folders_number, folders_name  = filepaths_list, filenames_list, folder_numbers_list, folder_names_list
    
    input_dict["filepaths"], input_dict["filenames"] = filepaths, filenames

    geometry, project = Geometry(input_dict["detector_distance"]*1000,susp=input_dict["suspect_border_pixels"],fill=input_dict["fill_blanks"]) # distance in milimeters

    if input_dict["using_direct_beam"]:
        print("\t Using direct beam to find center: ",input_dict["DP_center"])
        input_dict["DP_center"][1], input_dict["DP_center"][0] = opt540D.mapping540D( input_dict["DP_center"][1], input_dict["DP_center"][0], project)
        print("\t\t New center: ",input_dict["DP_center"])

    dic = {} # dictionary for restoration function
    dic['path']     = filepaths
    dic['outpath']  = input_dict["temporary_output"]
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
    else:
        dic['roi'] = input_dict["detector_ROI_radius"] # integer
    if input_dict["debug"]: 
        print(dic)
        dic['flat'] = np.ones([3072, 3072]) 
        dic['mask'] = np.zeros([3072, 3072])
    else:
        flat_path = input_dict["flatfield"]
        flat_type = flat_path.rsplit(".")[-1]

        if flat_type == "npy":
            dic["flat"] = flatfield_forward_restoration(input_dict)
        else:
            dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # np.ones([3072, 3072])
        
        dic['mask'] = read_hdf5(input_dict["mask"])[()][0, 0, :, :]      

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
        for i, filepath in enumerate(dic['path']):
            if i == 0:
                print("Closing open hdf5 files with h5clear -s")
            os.system(f"h5clear -s {filepath}")
        restored_data_info = pi540D.ioSetM_Backward540D( dic )

    return dic, restored_data_info, filepaths, filenames, folders_name, folders_number


def restoration_CAT(input_dict,method = 'IO'):
    """
    Function to perform restoration either via CUDA or IO-SharedArray approaches and saves diffraction patterns

    Args:
        method: IO or CUDA to select restoration method
        input_dict['data_path']: list of absolute paths to HDF5 diffractiom data patterns
        input_dict['save_path']: path to output folder where restored diffraction pattern will be stored
        input_dict["detector_distance"]: sample-detector distance in meters 
        input_dict["suspect_border_pixels"]: number of pixels to ignore at the chip's border
        input_dict['fill_blanks']: fill blank lines that appear from restoration via interpolation 
        input_dict['detector']: detector model: 540D or 135D
        input_dict['DP_center']: [row,column] coordinates for the DP center
        input_dict['detector_ROI_radius']: radius
        input_dict['using_direct_beam']: if True, will convert DP center coordinates to restored coordinates
        input_dict['GPUs']: number of GPUs for parallel CUDA restoration
        input_dict['flatfield']: path to flatfield HDF5 file
        input_dict['mask']: path to bad pixel mask HDF5 file
        input_dict['empty_path']: path to empty mask HDF5 file
        input_dict['subtraction_path']: path to background mask HDF5 file
    """

    geometry, project = Geometry(input_dict["detector_distance"]*1000,susp=input_dict['suspect_border_pixels'],fill=input_dict['fill_blanks'])

    if input_dict['using_direct_beam']: # if center coordinates are obtained from dbeam image at raw diffraction pattern; distance in mm
            input_dict['DP_center'][1], input_dict['DP_center'][0] = opt540D.mapping540D( input_dict['DP_center'][1], input_dict['DP_center'][0], pi540D.dictionary540D(input_dict["detector_distance"]*1000, {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual'} ))
            print(f"Corrected center position: cy={input_dict['DP_center'][0]} cx={input_dict['DP_center'][1]}")


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

        if "flatfield" not in input_dict:
            dic['flat'] = np.ones([detector_size, detector_size])
        elif input_dict["flatfield"] != '':    
            # dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] 
            flat_path = input_dict["flatfield"]
            flat_type = flat_path.rsplit(".")[-1]

            if flat_type == "npy":
                dic["flat"] = flatfield_forward_restoration(input_dict)
            else:
                dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # np.ones([3072, 3072])
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

