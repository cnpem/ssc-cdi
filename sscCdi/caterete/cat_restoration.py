import numpy as np
import os, sys

""" Sirius Scientific Computing Imports """
from sscPimega import pi540D, opt540D

""" sscCdi relative imports"""
from ..misc import read_hdf5, list_files_in_folder, select_specific_angles

def Geometry(distance,susp=3,fill=False,scale=0.98):
    params = {'geo':'nonplanar','opt':True,'mode':'virtual', 'fill': fill, 'susp': susp }
    project = pi540D.dictionary540D( distance, params ) 
    geo = pi540D.geometry540D( project )
    return geo, params

def restoration_CAT(input_dict):
    
    #TODO: estimate size of output DP after restoration; abort if using bertha and total size > 100GBs

    if input_dict["detector"] == "540D": detector_size = 3072

    dic_list = []
    restored_data_info_list = []
    for acquisitions_folder in input_dict['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        print(f'\tRestoration of folder {acquisitions_folder}')

        filepaths0, filenames0 = list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")
        
        if input_dict['projections'] != []:
            filepaths, filenames = select_specific_angles(input_dict['projections'], filepaths0,  filenames0)
            print(f"\tUsing {len(filenames)} of {len(filenames0)} angle(s)")

        geometry, params = Geometry(input_dict["detector_distance"]*1000,susp=input_dict["suspect_border_pixels"],fill=input_dict["fill_blanks"]) # distance in milimeters

        if input_dict["direct_beam_path"] != "":
            print("\t Using direct beam to find center: ",input_dict["DP_center"])
            input_dict["DP_center"][1], input_dict["DP_center"][0] = opt540D.mapping540D( input_dict["DP_center"][1], input_dict["DP_center"][0], input_dict["detector_distance"]*1000, params)
            print("\t\t New center: ",input_dict["DP_center"])

        dic = {}
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
        dic['center']   = (input_dict["DP_center"][1],input_dict["DP_center"][0]) # [1400,1400]
        if input_dict["detector_ROI_radius"] < 0:
            dic['roi'] = min(min(input_dict["DP_center"][1],detector_size-input_dict["DP_center"][1]),min(input_dict["DP_center"][0],detector_size-input_dict["DP_center"][0])) # get the biggest size possible such that the restored difpad is still squared
        else:
            dic['roi'] = input_dict["detector_ROI_radius"] # integer
        if input_dict["debug"]: 
            print(dic)
            dic['flat'] = np.ones([3072, 3072]) 
            dic['mask'] = np.zeros([3072, 3072])
        else:
            dic['flat'] = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # np.ones([3072, 3072]) #
            dic['mask'] = read_hdf5(input_dict["mask"])[()][0, 0, :, :] # np.zeros([3072, 3072])
        if os.path.isfile(input_dict["empty"]):
            dic['empty'] = read_hdf5(input_dict["empty"])[()][0, 0, :, :] 
        else:
            dic['empty'] = np.zeros_like(dic['flat']) # OBSOLETE! empty is not used anymore;      
        dic['daxpy']    = [0,np.zeros([3072,3072])] 
        dic['geometry'] = geometry

        if len(filepaths) == 1:
            dic['path'] = dic['path'][0]
            restored_data_info = pi540D.ioSet_Backward540D( dic )
        else:
            restored_data_info = pi540D.ioSetM_Backward540D( dic )

        dic_list.append(dic)
        restored_data_info_list.append(restored_data_info)

    return dic_list, restored_data_info_list
