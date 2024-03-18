import numpy as np
import h5py, os

import sscCdi, sscPimega, sscResolution, sscRaft, sscRadon

""" sscCdi relative imports"""
from ..ptycho.ptychography import call_GB_ptychography, set_object_shape, set_object_pixel_size, call_ptychography
from ..misc import add_to_hdf5_group, wavelength_from_energy

def mogno_ptychography(input_dict,DPs):

    probe_positions = read_mogno_probe_positions(input_dict,DPs.shape)

    input_dict = set_object_shape(input_dict,DPs.shape,probe_positions) # add object shape to input_dict

    sinogram = np.zeros((1,input_dict["object_shape"][0],input_dict["object_shape"][1]),dtype=np.complex64) # first dimension to be expanded in the future for multiple angles
    probes   = np.zeros((1,input_dict["incoherent_modes"],DPs.shape[-2],DPs.shape[-1]),dtype=np.complex64)
    
    sinogram[0, :, :], probes[0, :, :, :], error, _ = call_ptychography(input_dict,DPs,probe_positions)

    add_to_hdf5_group(input_dict["hdf5_output"],'log','error',np.array(error))

    return sinogram, probes, input_dict

def get_datetime(name):
    """
    Get custom str with acquisition name and current datetime to use as filename

    Args:
        name (str): name of current acquisition
    Returns:
        datetime (str): filename with current date and time
    """    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%y%m%d-%Hh%Mm")
    datetime = dt_string + "_" + name.split('.')[0]
    return datetime 

def read_mogno_probe_positions(input_dict,sinogram_shape):
    """
    Read probe positions and convert from meters to pixels

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "object_padding":
                "object_pixel":
        sinogram_shape (array): tuple with sinogram size

    Returns:
        positions_pixels (array): array with probe positions in pixels
    """    
    positions_mm = read_position_metadata(input_dict)
    input_dict = set_object_pixel_size(input_dict,sinogram_shape[1]) 
    positions_pixels = convert_probe_mogno_positions_meters_to_pixels(input_dict["object_padding"],input_dict["object_pixel"], positions_mm)

    return positions_pixels

def convert_probe_mogno_positions_meters_to_pixels(offset_topleft, dx, probe_positions):
    """
    Convert probe positions from meter to pixels

    Args:
        offset_topleft (int): border offset in the corners
        dx (float): pixel size
        probe_positions (array): probe positions in meters

    Returns:
        probe_positions: probe positions in pixels
    """    

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] =  probe_positions[:, 0] / pixel_size  
    probe_positions[:, 1] =  probe_positions[:, 1] / pixel_size 
    
    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    return probe_positions



