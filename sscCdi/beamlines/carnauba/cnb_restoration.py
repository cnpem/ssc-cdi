import numpy as np
import h5py, os
import SharedArray as sa
import multiprocessing
import uuid

from sscPimega import pi135D

from ..processing.restoration import restore_IO_SharedArray

def restoration_CNB(input_dict):
    """Calls restoration algorithm

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "data_path": file location

    Returns:
        diffraction_patterns: restored diffraction patterns
    """    

    hdf5_path = input_dict["data_path"]
    geometry = geometry_CNB(input_dict)
    diffraction_patterns = restore_IO_SharedArray(input_dict, geometry,hdf5_path)
    return diffraction_patterns

def geometry_CNB(input_dict):
    """Get sscPimega detector geometry for certain distance and corresponding dictionary of input params

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "suspect_border_pixels": suspect border pixels for pimega restoration

    Returns:
        geometry (dict): detector geometry
    """    

    project = pi135D.dictionary135D( -1,  {'geo':'planar','opt':True,'mode':'real', 'hexa': range(6)} ) 
    susp = input_dict["suspect_border_pixels"]
    project['s'] = [susp,susp] 
    geometry = pi135D.geometry135D( project )
    return geometry

def cnb_preprocessing_linear_correction(input_dict,raw_DPs):
    """ Linear correction

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "Trajectory_Path":
        raw_DPs (array): raw diffraction patterns
    """    

    def get_acquisition_time(input_dict,p):
        try:
            acq_time = p['general_info/Acquisition time'][()]   
        except:  
            trajectory = p['general_info/Trajectory'][()]
            trajectory = trajectory.decode('utf-8')

            with open(input_dict["Trajectory_Path"]) as trajectories:
                lines = trajectories.readlines()
            
            for line in lines:
                if trajectory == (line.split(' '))[-6]:
                    print("\nTrajectory is set")
                    acq_time = float(line.split(' ')[-2])
        return acq_time

    positions = h5py.File(input_dict["beamline_parameters_path"], 'r')

    acquisition_time = get_acquisition_time(input_dict,positions)

    if input_dict["apply_linearity_correction"]:
        print("\nApplying linearity correction")
        DPs = linearity_batch(input_dict, raw_DPs, acquisition_time)
    else:
        DPs = raw_DPs

    # if input_dict['empty'] != "":
        # DPs = apply_empty_acquisition(DPs, input_dict)
    
    return DPs

def linearity_batch(input_dict, DPs, acq_time):
    """Calls linearity correction in parallel for multiple diffraction patterns

    Args:
        input_dict (dict)
        DPs (array)
        acq_time (float)
    """    
    
    def _build_batch_of_DPs_(params):

        total_frames = params[1].shape[0]
        threads      = params[2]
        
        b = int( np.ceil( total_frames/threads )  ) 
        
        processes = []
        for k in range( threads ):
            begin_ = k*b
            end_   = min( (k+1)*b, total_frames )
            gpu = [k]

            p = multiprocessing.Process(target=_worker_batch_DPs_, args=(params, begin_, end_, gpu))
            processes.append(p)
        
        for p in processes:
            p.start()

        for p in processes:
            p.join()
        
    def _worker_batch_DPs_(params, idx_start, idx_end, gpu):
        
        corrected_DPs = params[0]
        DPs       = params[1]
        threads = params[2]
        acq_time = params[3]

        _start_ = idx_start
        _end_   = idx_end

        corrected_DPs[_start_:_end_,:,:] = apply_linearity_correction( DPs[_start_:_end_,:,:], _start_, acq_time)
        

    def apply_linearity_correction(DP, _start_, acq_time):
        
        from scipy.special import lambertw  
        mu = 4.6e-7 # constant provided by Antonio Neto
        
        DP = DP/acq_time
        #if np.any(data > 8e5): 
        #    raise ValueError("Counts/pixel.s is greater than 8e5")
        corrected = np.abs(-lambertw(-mu*DP)/mu)
        corrected *= acq_time
        return corrected   

    name = str( uuid.uuid4())
    
    threads = input_dict["CPUs"]
    
    try:
        sa.delete(name)
    except:
        pass
                
    corrected_DPs = sa.create(name,[DPs.shape[0], DPs.shape[1], DPs.shape[2]], dtype=np.float32)
    _params_ = ( corrected_DPs, DPs, threads, acq_time )

    _build_batch_of_DPs_ ( _params_ )

    sa.delete(name)

    return corrected_DPs

def apply_empty_acquisition(DPs, input_dict):
    """
        Correction for empty field
    """    
    print('Applying empty detector correction...')
    empty_acquisition_dir = input_dict['empty_acquisition_directory']
    empty_acquisition = np.asarray(h5py.File(empty_acquisition_dir, 'r')['/entry/data/data'])[:,0,:,:]
    empty_acquisition = empty_acquisition[1:999, :,:]
    empty = np.mean(empty_acquisition, axis = 0)
    DPs = np.where(empty == 0, DPs, -1)
    return DPs