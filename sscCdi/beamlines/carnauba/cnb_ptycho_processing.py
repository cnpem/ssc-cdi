import numpy as np
import h5py, os

import sscCdi

""" sscCdi relative imports"""

from ...ptycho.ptychography import call_GB_ptychography, set_object_shape, set_object_pixel_size
from ...misc import add_to_hdf5_group

def cnb_ptychography(input_dict,DPs):
    """Read restored diffraction data, read probe positions, calculate object parameters, calls ptychography and returns recostruction arrays

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "object_shape": added to the dictionary, object shape
                "incoherent_modes": list of incoherent modes
                "hdf5_output": output hdf5 file
        DPs (array): diffraction patterns

    Returns:
        sinogram: numpy array containing reconstructed frames
        probes: numpy array containing reconstructed probes
        input_dict (dict): updated input dictionary
    """    

    probe_positions = read_cnb_probe_positions(input_dict,DPs.shape)

    input_dict = set_object_shape(input_dict,DPs.shape,probe_positions) # add object shape to input_dict

    sinogram = np.zeros((1,input_dict["object_shape"][0],input_dict["object_shape"][1]),dtype=np.complex64) # first dimension to be expanded in the future for multiple angles
    probes   = np.zeros((1,input_dict["incoherent_modes"],DPs.shape[-2],DPs.shape[-1]),dtype=np.complex64)
    sinogram[0, :, :], probes[0, :, :, :], error = call_GB_ptychography(input_dict,DPs,probe_positions) # run ptycho

    add_to_hdf5_group(input_dict["hdf5_output"],'log','error',np.array(error))

    return sinogram, probes, input_dict

def define_paths(input_dict):
    """ Defines paths of interest for the ptychographic reconstruction and adds them to dictionary variable. Creates folders of interest and instantiates hdf5 output file

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "data_path": folders location
                "beamline_parameters_path": location of beamline parameters
    Returns:
        input_dict: updated input dictionary
            updated keys:
                "versions": versions of used packages
                "dataset_name": data set of file
                "output_path": location of output files
                "temporary_output": location of temporary files
                "energy": beamline energy
                "detector_distance": detector distance
                "restored_pixel_size": restored pixel size
                "detector_exposure": detector exposure
                "datetime": string with time and date to name files
                "hdf5_output": hdf5 output
    """
    
    #=========== Set Parameters and Folders =====================
    print('\tData path: ',input_dict['data_path'] )
 
    input_dict["versions"] = f"sscCdi={sscCdi.__version__}"

 
    input_dict["dataset_name"] = input_dict['data_path'].rsplit('/',1)[1].rsplit('.')[0]
    input_dict["output_path"] = input_dict["beamline_parameters_path"].rsplit('/',1)[0]
    print("\tOutput path:", input_dict["output_path"])

    input_dict["output_path"]  = os.path.join(input_dict["output_path"])
    input_dict["temporary_output"]  = os.path.join(input_dict["output_path"],'temp')

    data = h5py.File(input_dict["beamline_parameters_path"],'r')
    input_dict["energy"]               = data['beamline_parameters']['4CM Energy'][()]*1e-3 # keV
    input_dict["detector_distance"]    = data['beamline_parameters']['Distance PiMega'][()]*1e-3 # convert to meters
    input_dict["restored_pixel_size"]  = data['beamline_parameters']['PiMega Pixel Size'][()]*1e-6 # convert to microns
    input_dict["detector_exposure"]    = data['general_info']['Acquisition time'][()] # seconds
    data.close()

    input_dict["datetime"] = get_datetime(input_dict["dataset_name"])
    input_dict["hdf5_output"] = os.path.join(input_dict["output_path"],input_dict["datetime"]+".hdf5") # create output hdf5 file

    hdf5_output = h5py.File(input_dict["hdf5_output"], "w")
    hdf5_output.create_group("recon")
    hdf5_output.create_group("log")

    return input_dict

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

def read_cnb_probe_positions(input_dict,sinogram_shape):
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
    positions_pixels = convert_probe_positions_meters_to_pixels(input_dict["object_padding"],input_dict["object_pixel"], positions_mm)
    return positions_pixels

def convert_probe_positions_meters_to_pixels(offset_topleft, dx, probe_positions):
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

    probe_positions[:, 0] = 1E-3 * probe_positions[:, 0] / dx  # convert from mm to pixels
    probe_positions[:, 1] = 1E-3 * probe_positions[:, 1] / dx 
    
    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    return probe_positions

def read_position_metadata(input_dict):
    """
    Reads positions metadata

    Args:
        input_dict (dict): dictionary of inputs
            keys:
                "beamline_parameters_path": location of beamline parameters
    
    Returns:
        positions_mm (array): positions in meters
    """    

    def ParseTriggers(trigg):
        state = False
        highedge = []
        lowedge = []

        for k in range(trigg.size):
            if trigg[k] == True and state == False:
                highedge.append(k)
                state = True
            elif trigg[k] == False and state == True:
                lowedge.append(k)
                state = False

        return highedge, lowedge

    print('\nReading experiment parameters...')

    data = h5py.File(input_dict["beamline_parameters_path"],'r')

    highedge,lowedge = ParseTriggers(np.asarray(data['CRIO/Triggers']) > 0.5) #returns lists

    encoder_posx = np.asarray(data['/CRIO/Encoder Piezo Horizontal']) - 0.06 * np.asarray(data['/CRIO/Capacitive Sensor Rz HFM'])
    encoder_posy = np.asarray(data['/CRIO/Encoder Piezo Vertical']) - 0.15 * np.asarray(data['/CRIO/Capacitive Sensor Rx VFM'])
    
    data.close()

    posx = []
    posy = []

    for k in range(len(highedge)):
        s = np.s_[highedge[k]:lowedge[k]]
        posx.append( encoder_posx[s][0::30] )
        posy.append( encoder_posy[s][0::30] )

    posx = np.asarray(posx).astype(np.float32)
    posy = np.asarray(posy).astype(np.float32)

    posx -= posx.mean()
    posy -= posy.mean()

    positions_mm = np.asarray([posy,posx]).swapaxes(0,-1).swapaxes(0,1)
    positions_mm = positions_mm.mean(1)[:,None]
    positions_mm = np.reshape(positions_mm, (positions_mm.shape[0], 2))

    return positions_mm
