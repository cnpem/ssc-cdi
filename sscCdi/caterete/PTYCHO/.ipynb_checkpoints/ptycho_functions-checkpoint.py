import numpy as np
import cupy as cp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import datetime
import time

from PIL import Image

from sscPimega import misc

import random
random.seed(0)

def set_object_pixel_size(jason,half_size):
    c = 299792458             # Speed of Light [m/s]
    planck = 4.135667662E-18  # Plank constant [keV*s]
    wavelength = planck * c / jason['Energy'] # meters
    jason["wavelength"] = wavelength
    
    # Convert pixel size:
    dx = wavelength * jason['DetDistance'] / ( jason['Binning'] * jason['RestauredPixelSize'] * half_size * 2)

    return dx, jason
    
def convert_probe_positions(dx, probe_positions, offset_topleft = 20):
    """Set probe positions considering maxroi and effective pixel size

    Args:
        difpads (3D array): measured diffraction patterns
        jason (json file): file with the setted parameters and directories for reconstruction
        probe_positions (array): each element is an 2-array with x and y probe positions
        offset_topleft (int, optional): [description]. Defaults to 20.

    Returns:
        object pixel size (float), maximum roi (int), probe positions (array)
    """    

    # Subtract the probe positions minimum to start at 0
    probe_positions[:, 0] -= np.min(probe_positions[:, 0])
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    offset_bottomright = offset_topleft #define padding width
    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / dx + offset_topleft #shift probe positions to account for the padding
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / dx + offset_topleft #shift probe positions to account for the padding

    return probe_positions, offset_bottomright


def set_object_shape(difpads,jason,filenames,filepaths,acquisitions_folder,offset_topleft = 20):

    ibira_datafolder    = jason['ProposalPath']
    positions_string    = jason['positions_string']

    # Pego a PRIMEIRA medida de posicao, supondo que ela nao tem erro
    measurement_file = filenames[0]
    measurement_filepath = filepaths[0]
    
    # Compute half size of diffraction patterns:
    half_size = difpads.shape[-1] // 2

    # Compute/convert pixel size:
    dx, jason = set_object_pixel_size(jason,half_size)

    probe_positions_file = os.path.join(acquisitions_folder, positions_string, measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
    probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)
    probe_positions, offset_bottomright = convert_probe_positions(dx, probe_positions, offset_topleft = offset_topleft)

    maxroi        = int(np.max(probe_positions)) + offset_bottomright
    object_shape  = 2 * half_size + maxroi
    print('Object shape:',object_shape)

    return (object_shape,object_shape), half_size, dx, jason
    
def apply_random_shifts_to_positions(positionsX,positionsY ):
        mu, sigma = 0, 1 # mean and standard deviation
        deltaX = np.random.normal(mu, sigma, positionsX.shape)
        deltaY = np.random.normal(mu, sigma, positionsY.shape)
        return positionsX+deltaX,positionsY+deltaY 

def get_positions_array(random_positions=False):
    positions = [2,16,32,64,96,126]
    # positions = [  2,  10,  18,  26,  34,  42,  50,  58,  66,  74,  82,  90,  98,  106, 114, 122]
    # positions = [2,   6,  10,  14,  18,  22,  26,  30,  34,  38,  42,  46,  50, 54,  58,  62,  66,  70,  74,  78,  82,  86,  90,  94,  98, 102, 106, 110, 114, 118, 122

    positionsX,positionsY = np.meshgrid(positions,positions)

    if random_positions == True:
        positionsX,positionsY = apply_random_shifts_to_positions(positionsX,positionsY)
        # print(positionsX,positionsY)
        
    if 1: # Plot positions map
        figure, ax = plt.subplots(dpi=100)
        ax.plot(positionsX,positionsY,'x',label='Original')
        ax.set_title('Positions') 
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    return positionsX.flatten(),positionsY.flatten()

def apply_invalid_regions(difpad):
    delta = 25
    for i in range(0,difpad.shape[0]):
        difpad[0:difpad.shape[0]:delta] = -1

    for i in range(0,difpad.shape[1]):
        difpad[:,0:difpad.shape[1]:delta] = -1
    return  difpad
    
def get_simulated_data(random_positions=False,use_bad_points=False):

    positionsX,positionsY = get_positions_array(random_positions)

    dimension = 128 # Must be < than object!
    
    """ Create Probe """
    x = np.linspace(-1,1,dimension)
    X,Y = np.meshgrid(x,x)
    probe = np.where(X**2 + Y**2 < 0.9,1,0)  # Probe

    """ Create object """
    phase = np.array( np.load('image.npy')) # Load Imagem
    phase = phase - np.min(phase)
    phase = 2*np.pi*phase/np.max(phase) - np.pi # rescale from 0 to 2pi

    magnitude = Image.open('bernardi.png' ).convert('L').resize(phase.shape)
    magnitude = magnitude/np.max(magnitude)
    
    model_object = np.abs(magnitude)*np.exp(-1j*phase)

    difpads = []
    for px,py in zip(positionsX,positionsY):

        """ Exit wave-field """
        W = model_object[py:py+dimension,px:px+dimension]*probe
    
        """ Propagation """
        difpad = np.fft.fft2(W)
        difpad = np.fft.fftshift(difpad)
        
        """ Measurement """
        difpad = np.absolute(difpad)**2
    
        if use_bad_points:# add invalid grid to data
            difpad = apply_invalid_regions(difpad)
        
        # misc.imshow(np.abs(difpad),(5,5),savename='difpadgrid.png')
        # plt.show()
        # plt.close()

        difpads.append(difpad)

    positions = np.hstack((np.array([positionsY]).T ,np.array([positionsX]).T)) # adjust positionsitions format for proper input
    difpads = np.asarray(difpads)
    
    return difpads, positions, model_object, probe


def propagate_beam(wavefront, experiment_params,propagator='fourier'):
    """ Propagate a wavefront using fresnel ou fourier propagator

    Args:
        wavefront : the wavefront to propagate
        dx : pixel spacing of the wavefront input
        wavelength : wavelength of the illumination
        distance : distance to propagate
        propagator (str, optional): 'fresenel' or 'fourier'. Defaults to 'fresnel'.

    Returns:
        output: propagated wavefront
    """    
    
    from numpy.fft import fft2, fftshift, ifftshift, ifft2

    dx, wavelength,distance = experiment_params 
    
    if propagator == 'fourier':
        if distance > 0:
            output = fftshift(fft2(fftshift(wavefront)))
        else:
            output = ifftshift(ifft2(ifftshift(wavefront)))            
    
    elif propagator == 'fresnel':
    
        ysize, xsize = wavefront.shape
        x_array = np.linspace(-xsize/2,xsize/2-1,xsize)
        y_array = np.linspace(-ysize/2,ysize/2-1,ysize)

        fx = x_array/(xsize)
        fy = y_array/(ysize)

        FX,FY = np.meshgrid(fx,fy)
        # Calculate approx phase distribution for each plane wave component
        w = FX**2 + FY**2 
        # Compute FFT
        F = fftshift(fft2(fftshift(wavefront)))
        # multiply by phase-shift and inverse transform 
        a = np.exp(-1j*np.pi*( distance*wavelength/dx**2)*w)
        output = ifftshift(ifft2(ifftshift(F*a)))

    return output

def calculate_recon_error(model,obj):
    error = np.sum(np.abs(model - obj))/model.size
    return error 


def RAAR_update_object(exit_waves, probe, object_shape, positions,epsilon=0.01):

    m,n = probe.shape
    k,l = object_shape

    probeSum  = np.zeros((k,l),dtype=complex)
    waveSum   = np.zeros((k,l),dtype=complex)
    probeInt  = np.abs(probe)**2
    conjProbe = np.conj(probe)

    for index, pos in enumerate((positions)):
        posy, posx = pos[0], pos[1]
        probeSum[posy:posy + m , posx:posx+n] = probeSum[posy:posy + m , posx:posx+n] + probeInt
        waveSum[posy:posy + m , posx:posx+n]  = waveSum[posy:posy + m , posx:posx+n]  + conjProbe*exit_waves[index] 

    object = waveSum/(probeSum + epsilon)

    return object


def RAAR_update_probe(exit_waves, obj, probe_shape,positions, epsilon=0.01):
    m,n = probe_shape

    objectSum = np.zeros((m,n),dtype=complex)
    waveSum = np.zeros((m,n),dtype=complex)
    objectInt = np.abs(obj)**2
    conjObject = np.conj(obj)

    for index, pos in enumerate((positions)):
        posy, posx = pos[0], pos[1]
        objectSum = objectSum + objectInt[posy:posy + m , posx:posx+n]
        waveSum = waveSum + conjObject[posy:posy + m , posx:posx+n]*exit_waves[index]

    probe = waveSum/(objectSum + epsilon)

    return probe

def update_exit_wave(wavefront,measurement,experiment_params,epsilon=0.01,propagator = 'fourier'):
    wave_at_detector = propagate_beam(wavefront, experiment_params,propagator=propagator)
    wave_at_detector = np.sqrt(measurement)*wave_at_detector/(np.abs(wave_at_detector)+epsilon)
    # wave_at_detector[measurement>=0] = (np.sqrt(measurement)*wave_at_detector/(np.abs(wave_at_detector)))[measurement>=0]
    updated_exit_wave = propagate_beam(wave_at_detector, (experiment_params[0],experiment_params[1],-experiment_params[2]),propagator=propagator)
    return updated_exit_wave