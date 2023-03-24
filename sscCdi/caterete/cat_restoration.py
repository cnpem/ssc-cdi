import numpy as np
import os, sys

""" Sirius Scientific Computing Imports """
from sscPimega import pi540D

""" sscCdi relative imports"""
from ..misc import read_hdf5, list_files_in_folder, select_specific_angles

def Geometry(L,susp=3,scale=0.98,fill=False):
    project = pi540D.dictionary540D( L, {'geo':'nonplanar','opt':True,'mode':'virtual', 'fill': fill, 'susp': susp } ) 
    geo = pi540D.geometry540D( project )
    return geo

def restoration_CAT(input_dict):
    
    #TODO: estimate size of output DP after restoration; abort if using bertha and total size > 100GBs

    dic_list = []
    restored_data_info_list = []
    for acquisitions_folder in input_dict['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")
        
        if input_dict['projections'] != []:
            filepaths, filenames = select_specific_angles(input_dict['projections'], filepaths,  filenames)
            print(f"\tSelected a total of {len(filenames)} projections")

        params = (input_dict, filenames, filepaths, input_dict['data_folder'], acquisitions_folder, input_dict['scans_string'])

        geometry = Geometry(input_dict["detector_distance"]*1000) # distance in milimeters
        params   = {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual' ,'susp': input_dict["suspect_border_pixels"]}
        project  = pi540D.dictionary540D(input_dict["detector_distance"]*1000, params )
        geometry = pi540D.geometry540D( project )

        dic = {}
        # dic['path']     = ["/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/data/ptycho2d/SS61/scans/0000_SS61_001.hdf5"] #filepaths
        # dic['outpath']  = "/ibira/lnls/labs/tepui/home/yuri.tonin/test/"
        # dic['order']    = "yx" 
        # dic['rank']     = "ztyx" # order of axis
        # dic['dataset']  = "entry/data/data"
        # dic['gpus']     = [5] # input_dict["GPUs"]
        # dic['init']     = 0
        # dic['final']    = -1 # -1 to use all DPs
        # dic['saving']   = 1  # save or not
        # dic['timing']   = 0  # print timers 
        # dic['blocksize']= 10
        # dic['roi']      = 256 # input_dict["detector_ROI_radius"] # 512
        # dic['center']   = [1400,1400] #input_dict["DP_center"] # [1400,1400]
        # print(dic)
        # dic['daxpy']    = [0,np.zeros([3072,3072])] 
        # dic['flat']     = np.ones([3072, 3072]) # read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
        # dic['mask']     = np.zeros([3072, 3072]) # read_hdf5(input_dict["mask"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
        # dic['empty']    = np.zeros([3072, 3072]) #np.zeros_like(dic['flat']) # OBSOLETE! empty is not used anymore;
        # dic['geometry'] = geometry

        dic['path'] = "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/data/ptycho2d/SS61/scans/0000_SS61_001.hdf5"
        dic['outpath'] = "/home/ABTLUS/yuri.tonin/temp/"
        dic['order'] = "yx"
        dic['rank'] = "ztyx"
        dic['dataset'] = "entry/data/data"
        dic['gpus'] = [5] 
        dic['init'] = 0
        dic['final'] = -1
        dic['saving'] = 1
        dic['timing'] = 0
        dic['blocksize']= 10
        dic['center'] = [1000, 1000]
        dic['roi'] = 256
        print(dic)
        dic['daxpy'] = [0, np.zeros([3072,3072])] 
        dic['daxpy'] = [0, np.zeros([3072,3072])] 
        dic['flat'] = np.ones([3072,3072])
        dic['empty'] = np.zeros([3072,3072])
        dic['mask'] = np.zeros([3072,3072])

        if len(filepaths) == 1:
            print("Restoration of single file")
            # dic['path'] = dic['path'][0]
            print(dic['path'])
            restored_data_info = pi540D.ioSet_Backward540D( dic )
        else:
            restored_data_info = pi540D.ioSetM_Backward540D( dic )
        print(restored_data_info)
        sys.exit("Saindo")

        dic_list.append(dic)
        restored_data_info_list.append(restored_data_info)

    return dic_list, restored_data_info_list
