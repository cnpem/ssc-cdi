import numpy as np
import h5py, os


""" sscCdi relative imports"""
from ...ptycho.ptychography import set_object_shape, set_object_pixel_size, call_ptychography
from ...misc import wavelength_meters_from_energy_keV

def ema_ptychography(input_dict,DPs):
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

    probe_positions = read_ema_probe_positions(input_dict,DPs.shape)

    input_dict["object_shape"] = set_object_shape(input_dict["object_padding"],DPs.shape,probe_positions) # add object shape to input_dict

    sinogram = np.zeros((1,input_dict["object_shape"][0],input_dict["object_shape"][1]),dtype=np.complex64) # first dimension to be expanded in the future for multiple angles
    probes   = np.zeros((1,input_dict["incoherent_modes"],DPs.shape[-2],DPs.shape[-1]),dtype=np.complex64)
    sinogram[0, :, :], probes[0, :, :, :], positions,  input_dict, error = call_ptychography(input_dict,DPs,probe_positions)

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
                "detector_pixel_size": restored pixel size
                "detector_exposure": detector exposure
                "datetime": string with time and date to name files
                "hdf5_output": hdf5 output
    """
    
    #=========== Set Parameters and Folders =====================
    print('\tData path: ',input_dict['data_path'] )
 
    import sscCdi
    input_dict["versions"] = f"sscCdi={sscCdi.__version__}"

    input_dict["dataset_name"] = input_dict['data_path'].rsplit('/',1)[1].rsplit('.')[0]
    print("\tOutput path:", input_dict["output_path"])

    input_dict["output_path"]  = os.path.join(input_dict["output_path"])
    input_dict["temporary_output"]  = os.path.join(input_dict["output_path"],'temp')

    data = h5py.File(input_dict["beamline_parameters_path"],'r')
    
    input_dict["energy"]               = data['entry/info_exp/Energy(KeV)'][()] # keV
    input_dict["detector_pixel_size"]  = data['entry/info_exp/pixel(um)'][()]*1e-6 # convert to meters 

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

def read_ema_probe_positions(input_dict,sinogram_shape):
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

    if "wavenlegnth" not in input_dict:
        input_dict["wavelength"] = wavelength_meters_from_energy_keV(input_dict['energy'])
    
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

    data = h5py.File(input_dict["beamline_parameters_path"],'r')

    # getting probe positions
    bora_tx = data['entry/motors/bora-Tx'][()]
    bora_tz = data['entry/motors/bora-Tz'][()]
    
    x_positions = np.asarray(bora_tx).astype(np.float32)
    y_positions = np.asarray(bora_tz).astype(np.float32)

    initial_positions = np.asarray([y_positions,x_positions]).swapaxes(0,-1).swapaxes(0,1).T

    return initial_positions*input_dict["positions_unit_conversion"]

def crop_data(input_dict, diffraction_patterns):
    
    dp_center = input_dict["DP_center"]
    dp_radius = input_dict["DP_radius"]
    dp_shape  = diffraction_patterns[0,:,:].shape

    print(f"Center: {dp_center}")
    print(f"Given radius: {dp_radius}")
    print(f"Original shape : {dp_shape}")

    # cheking radius
    x_size_edge_left  = dp_center[0]
    x_size_edge_right = dp_shape[0] - dp_center[0]

    y_size_edge_left  = dp_center[1]
    y_size_edge_right = dp_shape[1] - dp_center[1]

    min_edge = min([x_size_edge_left, x_size_edge_right, y_size_edge_left, y_size_edge_right])
    if(dp_radius > min_edge):
        print(f"Given radius is greater than supported radius, new radius: {min_edge}")
        dp_radius = min_edge

    diffraction_patterns_cropped = diffraction_patterns[:,dp_center[0]-dp_radius:dp_center[0]+dp_radius,dp_center[1]-dp_radius:dp_center[1]+dp_radius]

    return diffraction_patterns_cropped



