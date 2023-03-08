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

def read_cnb_probe_positions(jason, dx, objsize):
   
    positionspath = os.path.join(jason["ProposalPath"], str(jason['Proposal']), 'proc', jason["BeamlineParameters_Filename"])

    beam_params = GetBeamlineParams(jason)
    rois = np.asarray([beam_params['posx'],beam_params['posy']]).swapaxes(0,-1).swapaxes(0,1)
    rois = rois.mean(1)[:,None]
    rois = np.reshape(rois, (rois.shape[0], 2))

    probe_positions = convert_cnb_probe_positions(rois, dx, jason, objsize)
    return probe_positions

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def convert_cnb_probe_positions(positions, dx, jason, objsize):

    print('\nconvert_cnb_probe_positions')

    """Set probe positions considering maxroi and effective pixel size

    Args:
        difpads (3D array): measured diffraction patterns
        jason (json file): file with the setted parameters and directories for reconstruction
        probe_positions (array): each element is an 2-array with x and y probe positions
        offset_topleft (int, optional): [description]. Defaults to 20.

    Returns:
        object pixel size (float), maximum roi (int), probe positions (array)
    """    

    half_size = jason['DetectorROI']
    # Subtract the probe positions minimum to start at 0
    print('dx: ', dx)

    positions[:, 0] -= np.min(positions[:, 0])
    positions[:, 1] -= np.min(positions[:, 1])

    positions[:, 0] = 1E-3 * positions[:, 0] / dx  #shift probe positions to account for the padding
    positions[:, 1] = 1E-3 * positions[:, 1] / dx  #shift probe positions to account for the padding

    positions[:,0] += (objsize-positions[:,0].max()-2*half_size)*0.5
    positions[:,1] += (objsize-positions[:,1].max()-2*half_size)*0.5
   

    np.save(jason['PreviewGCC'][1] + '/Rois.npy', positions)

    return positions

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def GetBeamlineParams(jason):

    print('\nGetBeamlineParams')

    name = os.path.join(jason['ProposalPath'], str(jason['Proposal']), 'proc', jason['BeamlineParameters_Filename'])

    data = h5py.File(name,'r')
    highedge,lowedge = ParseTriggers(np.asarray(data['CRIO/Triggers']) > 0.5) #returns lists

    encoder_posx = np.asarray(data['/CRIO/Encoder Piezo Horizontal']) - 0.06 * np.asarray(data['/CRIO/Capacitive Sensor Rz HFM'])

    encoder_posy = np.asarray(data['/CRIO/Encoder Piezo Vertical']) - 0.15 * np.asarray(data['/CRIO/Capacitive Sensor Rx VFM'])
    
    #encoder_posx = scipy.signal.medfilt(encoder_posx)
    #encoder_posy = scipy.signal.medfilt(encoder_posy)
    
    measured_I0 = np.asarray(data['/CRIO/I0'])
    measured_I1 = np.asarray(data['/CRIO/I1'])
    measured_Photodiode_HFM = np.asarray(data['/CRIO/Photodiode HFM'])
    # energy = data['beamline_parameters/4CM Energy'][()]
    energy = jason['Energy']

    data.close()

    state = False
    posx = []
    posy = []
    I0 = []
    I1 = []
    Photodiode_HFM = []

    for k in range(len(highedge)):
        s = np.s_[highedge[k]:lowedge[k]]
        posx.append( encoder_posx[s][0::30] )
        posy.append( encoder_posy[s][0::30] )
        I0.append( measured_I0[s].mean() )
        I1.append( measured_I1[s].mean() )
        Photodiode_HFM.append( measured_Photodiode_HFM[s].sum() )

    posx = np.asarray(posx).astype(np.float32)
    posy = np.asarray(posy).astype(np.float32)

    posx -= posx.mean()
    posy -= posy.mean()

    beamline_params = {}
    beamline_params['I0'] = np.asarray(I0).astype(np.float32)
    beamline_params['I1'] = np.asarray(I1).astype(np.float32)
    beamline_params['posx'] = posx
    beamline_params['posy'] = posy
    beamline_params['energy'] = energy
    beamline_params['Photodiode_HFM'] = np.asarray(Photodiode_HFM).astype(np.float32)

    return beamline_params
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
