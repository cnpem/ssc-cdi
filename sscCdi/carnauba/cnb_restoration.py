import numpy as np
import h5py, os, time

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from sscPimega import pi135D

from ..processing.restoration import restore_IO_SharedArray

def geometry_CNB(susp):
    project = pi135D.get_detector_dictionary( -1,  {'geo':'planar','opt':True,'mode':'real', 'hexa': range(6)} ) 
    project['s'] = [susp,susp] 
    geometry = pi135D.geometry135D( project )
    return geometry

def restoration_CNB(input_dict,hdf5_path):
    geometry = geometry_CNB(input_dict["suspect_border_pixels"])
    diffraction_patterns = restore_IO_SharedArray(input_dict, geometry,hdf5_path)
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


#============================================    LINEARITY CORRECTION =============================================================================#

def linearity_batch(input_dict, difpads, acq_time):
    
    name = str( uuid.uuid4())
    
    threads = 128
    
    try:
        sa.delete(name)
    except:
        pass
                
    corrected_difpads = sa.create(name,[difpads.shape[0], difpads.shape[1], difpads.shape[2]], dtype=np.float32)
    _params_ = ( corrected_difpads, difpads, threads, acq_time )

    print("\nBatching...")
    
    _build_batch_of_difpads_ ( _params_ )

    sa.delete(name)

    return corrected_difpads

#==============================================================================================================================================================#

def _build_batch_of_difpads_(params):

    print("\nBuilding...")

    total_frames = params[1].shape[0]
    threads      = params[2]
    
    b = int( np.ceil( total_frames/threads )  ) 
    
    processes = []
    for k in range( threads ):
        begin_ = k*b
        end_   = min( (k+1)*b, total_frames )
        gpu = [k]

        p = multiprocessing.Process(target=_worker_batch_difpads_, args=(params, begin_, end_, gpu))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
#==============================================================================================================================================================#
def _worker_batch_difpads_(params, idx_start, idx_end, gpu):
    
    corrected_difpads = params[0]
    difpads       = params[1]
    threads = params[2]
    acq_time = params[3]

    print("\nWorking...")
    
    
    _start_ = idx_start
    _end_   = idx_end

    corrected_difpads[_start_:_end_,:,:] = apply_linearity_correction( difpads[_start_:_end_,:,:], _start_, acq_time)
    
#==============================================================================================================================================================#

def apply_linearity_correction(difpad, _start_, acq_time):
    from scipy.special import lambertw  

    mu = 4.6e-7
    
    difpad = difpad/acq_time
    #if np.any(data > 8e5):
    #    raise ValueError("Counts/pixel.s is greater than 8e5")
    corrected = np.abs(-lambertw(-mu*difpad)/mu)
    corrected *= acq_time
    print(f'Image {_start_} done!')
    return corrected   
