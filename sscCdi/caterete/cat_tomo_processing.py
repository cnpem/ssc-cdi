
import h5py
import numpy as np
from numpy import loadtxt

from ..tomo.tomo_processing import select_contrast

def read_data(dic):
    
    if dic["recon_method"] == 'ptycho':
        file = h5py.File(dic["sinogram_path"], 'r')
        object = file['recon/object']
        angles = file['recon/angles']

        angles = angles[:,[0,2]]

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
    if dic["contrast_type"] == "phase":
        obj = np.angle(data)
    elif dic["contrast_type"] == "magnitude":
        obj = np.abs(data)
    else:
        sys.exit("Please select the correct contrast type: magnitude or phase")
    return obj