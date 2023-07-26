
import h5py, sys
import numpy as np
from numpy import loadtxt

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