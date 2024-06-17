
import h5py, sys
import numpy as np
import os
from numpy import loadtxt

from ...misc import create_directory_if_doesnt_exist

def read_data(dic):
    """ Read data from ptychography or plane-wave CDI measurement.

    Args:
        dic (dict): dictionary of inputs for CAT beamline

    Returns:
        object: volume containg sinogram of the object either from ptychography or plane-wave CDI
        angles: array containing rotation angle for each frame of the sinogram
    """
    
    if dic["recon_method"] == 'ptycho':
        file = h5py.File(dic["sinogram_path"], 'r')
        object = file['recon/object']
        angles = file['recon/angles']

        angles = angles[:,[0,2]]

        if 'angles' in dic: # read angles directly from 
            data = np.loadtxt(dic['angles'])
            angles_file = []
            for frame, angle in enumerate(data):
                angles_file.append([frame,True,angle*np.pi/180,angle])

            angles_file = angles_file[0:object.shape[0]]

    elif dic["recon_method"] == "pwcdi":
        
        object = np.load(dic["sinogram_path"])
        object = select_contrast(dic, object)

        # angles = np.load(dic["angles_path"])
        angles = loadtxt(dic["angles_path"])*np.pi/180
        numbering = np.linspace(0,angles.shape[0]-1,angles.shape[0],dtype=int)
        angles = np.vstack((numbering,angles)).T

    object = select_contrast(dic,object)

    return object, angles

def select_contrast(dic, data):
    """ Loads either magnitude or phase data from complex-valued array

    Args:
        dic (dict): dictionary of inputs for CAT beamline. Should contain "contrast_type" key for selecting desired contrast
        data (ndarray): array of complex values

    Returns:
        obj (ndarray): real valued array contaning only magnitude or phase of data
    """

    if dic["contrast_type"] == "phase":
        obj = np.angle(data)
    elif dic["contrast_type"] == "magnitude":
        obj = np.abs(data)
    elif dic["contrast_type"] == "complex":
        obj = np.asarray(data)
    else:
        sys.exit("Please select the correct contrast type: magnitude or phase")
    return obj

def define_CAT_tomo_paths(dic):
    """ Defines all the path required for the remaining parts of the code; adds them to the dicitionary and creates necessary folders

    Args:
        dic (dict): dictionary of inputs  
            keys:
                "output_folder": folder containing dataset path
                "contrast_type": contrast type
                "output_folder": path for output files

    Returns:
        dic (dict): updated dictionary of inputs  
            updated keys:
                "filename"
                "temp_folder"
                "ordered_angles_filepath"
                "projected_angles_filepath"
                "ordered_object_filepath"
                "cropped_sinogram_filepath"
                "pre_aligned_sinogram_filepath"
                "equalized_sinogram_filepath"
                "unwrapped_sinogram_filepath"
                "wiggle_sinogram_filepath"
                "wiggle_cmas_filepath"
                "reconstruction_filepath"
                "eq_reconstruction_filepath"
    """

    dic["filename"]    = os.path.basename(os.path.normpath(dic["output_folder"]))
    dic["temp_folder"] = os.path.join(dic["output_folder"],'temp')
    dic["ordered_angles_filepath"]       = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles.h5')
    dic["projected_angles_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles_projected.h5')
    dic["ordered_object_filepath"]       = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_object.h5')
    dic["cropped_sinogram_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_cropped_sinogram.h5')
    dic["cc_aligned_sinogram_filepath"]  = os.path.join(dic["temp_folder"],f'{dic["filename"]}_cc_aligned_sinogram.h5')
    dic["vmf_aligned_sinogram_filepath"]  = os.path.join(dic["temp_folder"],f'{dic["filename"]}_vmf_aligned_sinogram.h5')
    dic["equalized_sinogram_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_equalized_sinogram.h5')
    dic["unwrapped_sinogram_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_unwrapped_sinogram.h5')
    dic["wiggle_sinogram_filepath"]      = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_sinogram.h5')
    dic["wiggle_cmas_filepath"]          = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_ctr_mass.h5')
    dic["reconstruction_filepath"]       = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo.h5')
    dic["eq_reconstruction_filepath"]    = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo_equalized.h5')

    create_directory_if_doesnt_exist(dic["output_folder"])
    create_directory_if_doesnt_exist(dic["temp_folder"])

    return dic