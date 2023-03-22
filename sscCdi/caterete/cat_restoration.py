import numpy as np
import os, sys, h5py, time
from scipy import ndimage
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

""" Sirius Scientific Computing Imports """
import sscCdi
from sscIO import io
from sscPimega import pi540D

""" sscCdi relative imports"""
from ..misc import read_hdf5

def Geometry(L,susp=3,scale=0.98,fill=False):
    project = pi540D.dictionary540D( L, {'geo':'nonplanar','opt':True,'mode':'virtual', 'fill': fill, 'susp': susp } ) 
    geo = pi540D.geometry540D( project )
    return geo

def restoration_CAT(input_dict):
    
    ibira_datafolder, scans_string  = input_dict['data_folder'],input_dict['scans_string']

    #TODO: estimate size of output DP after restoration; abort if using bertha and total size > 100GBs

    dic_list = []
    restored_data_info_list = []
    for acquisitions_folder in input_dict['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if input_dict['projections'] != []:
            filepaths, filenames = sscCdi.misc.misc.select_specific_angles(input_dict['projections'], filepaths,  filenames)
            print(f"\tSelected a total of {len(filenames)} projections")

        params = (input_dict, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        distance = input_dict["detector_distance"]*1000 # distance in milimeters
        geometry = Geometry(distance)
        params   = {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual' ,'susp': input_dict["suspect_border_pixels"]}
        project  = pi540D.dictionary540D(distance, params )
        geometry = pi540D.geometry540D( project )

        dic = {}
        dic['path']     = filepaths
        dic['outpath']  = input_dict["temporary_output"]
        dic['order']    = "yx" 
        dic['rank']     = "ztyx" # order of axis
        dic['dataset']  = "entry/data/data"
        dic['nGPUs']    = len(input_dict["GPUs"])
        dic['GPUs']     = input_dict["GPUs"]
        dic['init']     = 0
        dic['final']    = -1 # -1 to use all DPs
        dic['saving']   = 1  # save or not
        dic['timing']   = 0  # print timers 
        dic['blocksize']= 10
        dic['geometry'] = geometry
        dic['roi']      = input_dict["detector_ROI_radius"] # 512
        dic['center']   = input_dict["DP_center"] # [1400,1400]
        dic['flat']     = read_hdf5(input_dict["FlatField"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
        dic['empty']    = read_hdf5(input_dict['EmptyFrame']).squeeze().astype(np.float32) # numpy.zeros([3072,3072]) 
        
        restored_data_info = pi540D.ioSetM_Backward540D( dic )

        dic_list.append(dic)
        restored_data_info_list.append(restored_data_info)

    return dic_list, restored_data_info_list
