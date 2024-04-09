import numpy as np

def convert_probe_mogno_positions_meters_to_pixels(probe_positions,pixel_size,offset_topleft=50):
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



