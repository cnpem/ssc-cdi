import numpy as np
import h5py, os

import sscCdi, sscPimega, sscResolution, sscRaft, sscRadon

""" sscCdi relative imports"""
from ..ptycho.ptychography import call_GB_ptychography, set_object_shape
from ..caterete.cat_ptycho_processing import set_object_shape

def cnb_ptychography(input_dict,DPs):

    probe_positions = read_cnb_probe_positions(input_dict)

    input_dict = set_object_shape(input_dict,DPs[0].shape,probe_positions) # add object shape to input_dict

    sinogram = np.zeros((1,input_dict["object_shape"][0],input_dict["object_shape"][1])) # first dimension to be expanded in the future for multiple angles
    probes   = np.zeros((1,1,DPs.shape[-2],DPs.shape[-1]))
    sinogram[0, :, :], probes[0, :, :], error = call_GB_ptychography(input_dict,DPs,probe_positions) # run ptycho

    return sinogram, probes, input_dict

def define_paths(input_dict):
    """ Defines paths of interest for the ptychographic reconstruction and adds them to dictionary variable. Creates folders of interest and instantiates hdf5 output file

    Args:
        input_dict (dict): 

    Returns:
        input_dict: updated input dictionary
    """
    
    #=========== Set Parameters and Folders =====================
    print('\tData path: ',input_dict['data_path'] )
 
    input_dict["versions"] = f"sscCdi={sscCdi.__version__},sscPimega={sscPimega.__version__},sscResolution={sscResolution.__version__},sscRaft={sscRaft.__version__},sscRadon={sscRadon.__version__}"

    input_dict["output_path"] = input_dict["beamline_parameters_path"].rsplit('/',1)[0]
    print("\tOutput path:", input_dict["output_path"])

    input_dict["output_path"]  = os.path.join(input_dict["output_path"])
    input_dict["temporary_output"]  = os.path.join(input_dict["output_path"],'temp')

    data = h5py.File(input_dict["beamline_parameters_path"],'r')
    input_dict["energy"]               = data['beamline_parameters']['4CM Energy']
    input_dict["detector_distance"]    = data['beamline_parameters']['Distance Pimega']*1e-3 # convert to meters
    input_dict["detector_exposure"]    = data['general_info']['Acquisition time']
    data.close()

    input_dict["datetime"] = get_datetime(input_dict)
    input_dict["hdf5_output"] = os.path.join(input_dict["output_path"],input_dict["datetime"]+".hdf5") # create output hdf5 file

    hdf5_output = h5py.File(input_dict["hdf5_output"], "w")
    hdf5_output.create_group("recon")
    hdf5_output.create_group("log")

    return input_dict

def get_datetime(name):
    """ Get custom str with acquisition name and current datetime to use as filename
    """
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%y%m%d-%Hh%Mm")
    datetime = dt_string + "_" + name.split('.')[0]
    return datetime 

def read_cnb_probe_positions(input_dict, dx, objsize):
    positions_mm = read_position_metadata(input_dict)
    positions_pixels = convert_probe_positions_meters_to_pixels(positions_mm, dx, input_dict, objsize)
    return positions_pixels

def convert_probe_positions_meters_to_pixels(offset_topleft, dx, probe_positions):

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = 1E-3 * probe_positions[:, 0] / dx  # convert from mm to pixels
    probe_positions[:, 1] = 1E-3 * probe_positions[:, 1] / dx 
    
    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    return probe_positions

def read_position_metadata(input_dict):

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

    positions_mm = np.asarray([posx,posy]).swapaxes(0,-1).swapaxes(0,1)
    positions_mm = positions_mm.mean(1)[:,None]
    positions_mm = np.reshape(positions_mm, (positions_mm.shape[0], 2))

    return positions_mm

def cnb_probe():

    beam_params = GetBeamlineParams(input_dict)
    rois = np.asarray([beam_params['posx'],beam_params['posy']]).swapaxes(0,-1).swapaxes(0,1)
    rois = rois.mean(1)[:,None]
    rois = np.reshape(rois, (rois.shape[0], 2))

    object_shape, half_size, object_pixel_size, input_dict = set_cnb_object_shape(DPs,input_dict)
    objsize = object_shape[1]

    print('Objsize: ', objsize)
    
    probe_positions = read_cnb_probe_positions(input_dict, input_dict["object_pixel"], objsize)
    
    I0 = beam_params['I0']
    I1 = beam_params['I1']
    I0 = np.reshape(I0, (rois.shape[0], 1))
    I1 = np.reshape(I1, (rois.shape[0], 1))
    print('\nRois, I0 and I1 shape: ', rois.shape, I0.shape, I1.shape)
    probe_positions = np.concatenate((probe_positions, I0, I1), axis = 1)
    print('\nProbe positions shape: ', probe_positions.shape)   


