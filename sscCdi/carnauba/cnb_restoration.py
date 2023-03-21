import numpy as np
import h5py
from scipy import ndimage, signal
from time import time
from PIL.Image import open as tifOpen
import os

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from scipy import ndimage

from sscIO import io
import sscCdi
from sscPimega import pi135D

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def apply_empty_acquisition(difpads, input_dict):

    print('Appling empty acquisition...')
    empty_acquisition_dir = input_dict['empty_acquisition_directory']

    empty_acquisition = np.asarray(h5py.File(empty_acquisition_dir, 'r')['/entry/data/data'])[:,0,:,:]
    empty_acquisition = empty_acquisition[1:999, :,:]
    
    empty = np.mean(empty_acquisition, axis = 0)
    difpads = np.where(empty == 0, difpads, -1)
    
    return difpads

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def cnb_preprocessing_linear_correction(raw_difpads, input_dict):

    positionspath = os.path.join(input_dict["ProposalPath"], str(input_dict['Proposal']), 'proc', input_dict["BeamlineParameters_Filename"])

    p = h5py.File(positionspath, 'r')

    try:
        acq_time = p['general_info/Acquisition time'][()]   
    except:  
        trajectory = p['general_info/Trajectory'][()]
        trajectory = trajectory.decode('utf-8')

        with open(input_dict["Trajectory_Path"] + 'trajetorias.txt') as trajectories:
            lines = trajectories.readlines()
        
        for line in lines:
            if trajectory == (line.split(' '))[-6]:
                print("\nTrajectory is set")
                acq_time = float(line.split(' ')[-2])
    
    if input_dict["Linearity_Function"]:
        print("\nApplying Linearity_Function")
        difpads = sscCdi.carnauba.cnb_processing.linearity_batch(input_dict, raw_difpads, acq_time)
    else:
        difpads = raw_difpads


    if input_dict['empty_acquisition']:
        difpads = apply_empty_acquisition(difpads, input_dict)
    
    return difpads

