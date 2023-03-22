import numpy as np
import h5py, os, time

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from sscPimega import pi135D

from .cnb_ptycho_processing import linearity_batch
from ..processing.restoration import restore_IO_SharedArray

def geometry_CNB(susp):
    project = pi135D.get_detector_dictionary( -1,  {'geo':'planar','opt':True,'mode':'real', 'hexa': range(6)} ) 
    project['s'] = [susp,susp] 
    geometry = pi135D.geometry135D( project )
    return geometry

def restoration_CNB(input_dict):
    geometry = geometry_CNB(input_dict["suspect_border_pixels"])
    diffraction_patterns = restore_IO_SharedArray(input_dict, geometry)
    return diffraction_patterns

def apply_empty_acquisition(difpads, input_dict):

    print('Appling empty acquisition...')
    empty_acquisition_dir = input_dict['empty_acquisition_directory']

    empty_acquisition = np.asarray(h5py.File(empty_acquisition_dir, 'r')['/entry/data/data'])[:,0,:,:]
    empty_acquisition = empty_acquisition[1:999, :,:]
    
    empty = np.mean(empty_acquisition, axis = 0)
    difpads = np.where(empty == 0, difpads, -1)
    
    return difpads

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
        difpads = linearity_batch(input_dict, raw_difpads, acq_time)
    else:
        difpads = raw_difpads


    if input_dict['empty_acquisition']:
        difpads = apply_empty_acquisition(difpads, input_dict)
    
    return difpads

