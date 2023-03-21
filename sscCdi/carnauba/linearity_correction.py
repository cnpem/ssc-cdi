import numpy as np
import SharedArray as sa
import multiprocessing
import uuid
import h5py

#==============================================================================================================================================================#

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

