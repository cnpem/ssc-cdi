import numpy as np
import h5py, os
import uuid
import SharedArray as sa
import multiprocessing
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

""" Sirius Scientific Computing Imports """
import sscCdi
from sscPimega import pi540D

""" sscCdi relative imports"""
from ..ptycho.ptychography import  call_G_ptychography
from .cnb_restoration import restoration_CNB

def cnb_ptychography(input_dict,restoration_dict_list,restored_data_info_list,strategy="serial"):

    total_n_of_frames = 0
    for acquisitions_folder in input_dict['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon
        _, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")
        total_n_of_frames += len(filenames)

    if strategy == "serial":

        for folder_number, acquisitions_folder in enumerate(input_dict['acquisition_folders']):  # loop when multiple acquisitions were performed for a 3D recon
    
            filepaths, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")

            for file_number, (filepath,filename) in enumerate(zip(filepaths,filename)):

                frame = file_number + folder_number*len(filenames) # attribute singular value to each angle

                """ Read Diffraction Patterns for one angle """
                DPs = restoration_CNB(input_dict,filepath)
                
                if file_number == 0 and folder_number == 0: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                    object_shape, input_dict = sscCdi.caterete.ptycho.ptycho_processing.set_object_shape(DPs,input_dict, [filename], [filepath], acquisitions_folder)
                    sinogram = np.zeros((total_n_of_frames,object_shape[0],object_shape[1])) 
                    probes   = np.zeros((total_n_of_frames,1,DPs.shape[-2],DPs.shape[-1]))

                """ Read positions """
                probe_positions = read_cnb_probe_positions(input_dict, filename , DPs.shape[0])
                
                run_ptycho = np.any(probe_positions)  # check if probe_positions == null matrix. If so, won't run current iteration

                """ Call Ptycho """
                if not run_ptycho:
                    print(f'\t\t WARNING: Frame #{(folder_number,file_number)} being nulled because number of positions did not match number of diffraction pattern!')
                    input_dict['ignored_scans'].append((folder_number,file_number))
                    sinogram[frame, :, :]  = np.zeros((object_shape[0],object_shape[1])) # build 3D Sinogram
                    probes[frame, :, :, :] = np.zeros((1,DPs.shape[-2],DPs.shape[-1]))
                else:
                    sinogram[frame, :, :], probes[frame, :, :] = call_G_ptychography(input_dict,DPs,probe_positions) # run ptycho


    return sinogram, probes, input_dict



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def read_cnb_probe_positions(input_dict, dx, objsize):
   
    positionspath = os.path.join(input_dict["ProposalPath"], str(input_dict['Proposal']), 'proc', input_dict["BeamlineParameters_Filename"])

    beam_params = GetBeamlineParams(input_dict)
    rois = np.asarray([beam_params['posx'],beam_params['posy']]).swapaxes(0,-1).swapaxes(0,1)
    rois = rois.mean(1)[:,None]
    rois = np.reshape(rois, (rois.shape[0], 2))

    probe_positions = convert_cnb_probe_positions(rois, dx, input_dict, objsize)
    return probe_positions


def convert_cnb_probe_positions(positions, dx, input_dict, objsize):

    print('\nconvert_cnb_probe_positions')

    """Set probe positions considering maxroi and effective pixel size

    Args:
        difpads (3D array): measured diffraction patterns
        input_dict (json file): file with the setted parameters and directories for reconstruction
        probe_positions (array): each element is an 2-array with x and y probe positions
        offset_topleft (int, optional): [description]. Defaults to 20.

    Returns:
        object pixel size (float), maximum roi (int), probe positions (array)
    """    

    half_size = input_dict['DetectorROI']
    # Subtract the probe positions minimum to start at 0
    print('dx: ', dx)

    positions[:, 0] -= np.min(positions[:, 0])
    positions[:, 1] -= np.min(positions[:, 1])

    positions[:, 0] = 1E-3 * positions[:, 0] / dx  #shift probe positions to account for the padding
    positions[:, 1] = 1E-3 * positions[:, 1] / dx  #shift probe positions to account for the padding

    positions[:,0] += (objsize-positions[:,0].max()-2*half_size)*0.5
    positions[:,1] += (objsize-positions[:,1].max()-2*half_size)*0.5
   

    np.save(input_dict['PreviewGCC'][1] + '/Rois.npy', positions)

    return positions

#============================================    TRIGGERS  =============================================================================#

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


def GetBeamlineParams(input_dict):

    print('\nGetBeamlineParams')

    name = os.path.join(input_dict['ProposalPath'], str(input_dict['Proposal']), 'proc', input_dict['BeamlineParameters_Filename'])

    data = h5py.File(name,'r')
    highedge,lowedge = ParseTriggers(np.asarray(data['CRIO/Triggers']) > 0.5) #returns lists

    encoder_posx = np.asarray(data['/CRIO/Encoder Piezo Horizontal']) - 0.06 * np.asarray(data['/CRIO/Capacitive Sensor Rz HFM'])

    encoder_posy = np.asarray(data['/CRIO/Encoder Piezo Vertical']) - 0.15 * np.asarray(data['/CRIO/Capacitive Sensor Rx VFM'])
    
    #encoder_posx = scipy.signal.medfilt(encoder_posx)
    #encoder_posy = scipy.signal.medfilt(encoder_posy)
    
    measured_I0 = np.asarray(data['/CRIO/I0'])
    measured_I1 = np.asarray(data['/CRIO/I1'])
    measured_Photodiode_HFM = np.asarray(data['/CRIO/Photodiode HFM'])
    # energy = data['beamline_parameters/4CM energy'][()]
    energy = input_dict['energy']

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

def set_cnb_object_shape(difpads,input_dict):

    # Compute half size of diffraction patterns:
    half_size = int(difpads.shape[-1] // 2)
    
    # Compute/convert pixel size:
    dx, input_dict = set_object_pixel_size(input_dict,half_size)
    obj_magn = input_dict['object_magnification']

    beam_params = GetBeamlineParams(input_dict)

    largness_x =  1E-3*(beam_params['posx'].max() - beam_params['posx'].min()) / dx
    largness_y =  1E-3*(beam_params['posy'].max() - beam_params['posy'].min()) / dx
    largness = int(max(largness_x, largness_y));

    objsize = largness + obj_magn*difpads.shape[-1]
    probe_positions = read_cnb_probe_positions(input_dict, input_dict["object_pixel"], objsize)

    largness_x = probe_positions[:,0].max() - probe_positions[:,0].min()
    largness_y = probe_positions[:,1].max() - probe_positions[:,1].min()
    largness = int(max(largness_x, largness_y))
    
    return (objsize, objsize), half_size, dx, input_dict


def cnb_probe():

    beam_params = GetBeamlineParams(input_dict)
    rois = np.asarray([beam_params['posx'],beam_params['posy']]).swapaxes(0,-1).swapaxes(0,1)
    rois = rois.mean(1)[:,None]
    rois = np.reshape(rois, (rois.shape[0], 2))

    object_shape, half_size, object_pixel_size, input_dict = set_cnb_object_shape(difpads,input_dict)
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



def cnb_probe_support(probe, input_dict):
    from scipy.ndimage import gaussian_filter

    data_filename = input_dict['Data_Filename']
    output_folder_name = (data_filename.split("."))[0]
    output_folder = os.path.join('/ibira/lnls/beamlines/carnauba/apps/jupyter/00000000', 'proc', output_folder_name) # changes with control

    N = probe.shape[-1]
    # Compute probe support:
    h = int(probe.shape[0]/2)
    dx = 3.839600707343195e-08

    x = np.linspace(0,N-1,N) - N//2
    x = x / np.max(np.abs(x))
    X, Y = np.meshgrid(x,x)
    X = X - np.max(x)
    Y = Y - np.max(x)
    n = input_dict['n_parameter']
    R = input_dict['R_parameter']
    sigma = input_dict['sigma']
    border_px = input_dict['border_px']
    f1 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

    np.save(output_folder + '/support1.npy', f1)

    x = np.linspace(0,N-1,N) - N//2
    x = x / np.max(np.abs(x))
    X, Y = np.meshgrid(x,x)
    X = X + np.max(x)
    Y = Y + np.max(x)

    f2 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

    np.save(output_folder + '/support2.npy', f2)

    x = np.linspace(0,N-1,N) - N//2
    x = x / np.max(np.abs(x))
    X, Y = np.meshgrid(x,x)
    X = X - np.max(x)
    Y = Y + np.max(x)

    f3 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

    np.save(output_folder + '/support3.npy', f3)

    x = np.linspace(0,N-1,N) - N//2
    x = x / np.max(np.abs(x))
    X, Y = np.meshgrid(x,x)
    X = X + np.max(x)
    Y = Y - np.max(x)

    f4 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

    np.save(output_folder + '/support4.npy', f4)

    f = (f1 + f2 + f3 + f4)/4
    # f = np.where(f != 1, 0, 1)

    np.save(output_folder + '/support.npy', f)

    f[0:border_px,:] = 0
    f[:, 0:border_px] = 0
    f[f.shape[0] - border_px: f.shape[0], :] = 0
    f[:, f.shape[1] - border_px: f.shape[1]] = 0

    f = gaussian_filter(f, sigma)

    np.save(output_folder + '/blurred_support.npy', f)
    probesupp = np.asarray([f for k in range(probe.shape[0])])

    return probesupp


