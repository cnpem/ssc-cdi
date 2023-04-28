
import h5py
import numpy as np

def read_data(dic):
    if dic["recon_method"] == 'ptycho':
        file = h5py.File(dic["sinogram_path"], 'r')
        object = file['recon/object']
        angles = file['recon/angles']

        angles = angles[:,[0,2]]

    elif dic["recon_method"] == "pwcdi":
        object = np.load(dic["sinogram_path"])
        angles = np.load(dic["angles_path"])
    
        #TODO: adjust angle to right format

    return object, angles