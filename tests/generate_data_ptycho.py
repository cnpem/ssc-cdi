from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import hsv_to_rgb, LogNorm
import scipy
import json
import sscCdi
import time
#print(f'sscCdi version: {sscCdi.__version__}')

#Generate synthetic data

def create_2d_gaussian(N, M, sigma_x, sigma_y, mu_x=None, mu_y=None):
    """
    Creates a 2D Gaussian distribution matrix.

    Parameters:
    - N: Number of rows.
    - M: Number of columns.
    - sigma_x: Standard deviation in the x direction.
    - sigma_y: Standard deviation in the y direction.
    - mu_x: Mean in the x direction. Defaults to the center of the matrix if not provided.
    - mu_y: Mean in the y direction. Defaults to the center of the matrix if not provided.

    Returns:
    A 2D array representing the Gaussian distribution.
    """
    if mu_x is None:
        mu_x = N / 2
    if mu_y is None:
        mu_y = M / 2

    x = np.linspace(0, N-1, N)
    y = np.linspace(0, M-1, M)
    x, y = np.meshgrid(x, y)

    gaussian = np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
    return gaussian


def create_2d_disk(N,M,sigma):
    x = np.linspace(0, N-1, N)
    y = np.linspace(0, M-1, M)
    x, y = np.meshgrid(x, y)
    disk = np.where((x-N/2)**2 + (y-M/2)**2 <= (sigma)**2, 1, 0)
    return disk


def apply_random_shifts_to_positions(pos_x, pos_y, range_x, range_y, mu=0, sigma_x=2, sigma_y=2, type='gaussian'):
    if type == 'gaussian':
        deltaX = np.random.normal(mu, sigma_x, pos_x.shape)
        deltaY = np.random.normal(mu, sigma_y, pos_y.shape) 
        
        # Apply a different delta to each X and Y
        new_positions_x = range_x[0] +  ((pos_x + deltaX)% range_x[1])
        new_positions_y = range_y[0] +  ((pos_y + deltaY)% range_y[1])
        
    elif type == 'random':
        deltaX = np.round(sigma_x * np.random.rand(*pos_x.shape))
        deltaY = np.round(sigma_y * np.random.rand(*pos_y.shape))
        
         # Apply a different delta to each X and Y
        new_positions_x = range_x[0] +  ((pos_x + deltaX)% range_x[1])
        new_positions_y = range_y[0] +  ((pos_y + deltaY)% range_y[1])
    elif type == "uniform":
        deltaX = np.round(np.random.uniform(-sigma_x,sigma_x,pos_x.shape))
        deltaY = np.round(np.random.uniform(-sigma_y,sigma_y,pos_y.shape))
        
        # Apply a different delta to each X and Y
        new_positions_x = pos_x + deltaX
        new_positions_y = pos_y + deltaY
   

    # Normalize to start from 0
    # new_positions_x -= np.min(new_positions_x)
    # new_positions_y -= np.min(new_positions_y)

    return new_positions_x, new_positions_y


def get_positions_array(frame_shape,range_x, range_y, nx, ny, random_positions=True, plot=True):
    y_pxls = np.linspace(range_y[0], range_y[1], ny)
    x_pxls = np.linspace(range_x[0], range_x[1], nx)
    
    pos_y,pos_x = np.meshgrid(y_pxls,x_pxls)
    
    # sigma_x = np.sqrt((frame_shape[0]/180.0)) #*2
    # sigma_y = np.sqrt((frame_shape[1]/180.0)) #*2 
    sigma_x = (range_x[1] - range_x[0])/(2*nx)
    sigma_y = (range_y[1] - range_y[0])/(2*ny)
    
    if random_positions == True:
       pos_x,pos_y = apply_random_shifts_to_positions(pos_x, pos_y, range_x, range_y,  sigma_x=sigma_x, sigma_y=sigma_y, type="uniform")
    
    positions = np.vstack((np.array(pos_x.flatten()) ,np.array(pos_y.flatten()))).T
    
    if plot: # Plot positions map
        figure, ax = plt.subplots(dpi=100)
        ax.plot(positions[:,0],positions[:,1],'.',label='Original',color='black')
        ax.set_title('Positions') 
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    return positions


def create_hsv_wheel_image(size=256):
    """Create an HSV color wheel image."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Create complex plane

    # Amplitude is the distance from the center
    Amplitude = np.abs(Z)
    
    # Phase is the angle (in radians) from the positive real axis
    Phase = np.angle(Z)

    # Create the complex image with varying amplitude and phase
    ComplexImg = np.exp(1j * Phase)
    
    # Normalize amplitude to the maximum distance (which is sqrt(2) for this range)
    ComplexImg *= Amplitude / np.max(Amplitude)

    # Convert the complex image to RGB using the provided function
    RGB_image = sscCdi.misc.convert_complex_to_RGB(ComplexImg, bias=0.001)

    # Mask out the region outside the unit circle (optional)
    mask = Amplitude <= 1
    RGB_image[~mask] = 1  # Set to white or any other background color

    return RGB_image


def interpolate_image(image, new_shape, interpolation_factor = 0.5):
    """
    Interpolates a single image to a new size new_shape using bicubic interpolation.
    
    Parameters:
    - image: numpy array of shape (H, W), the original image.
    - new_shape: int, the size of the output image (Hnew, Wnew).
    
    Returns:
    - interpolated_image: numpy array of shape new_shape, the resized image.
    """
    
    # define zoom factor
    zoom_factor_h = new_shape[0]/image.shape[0]
    zoom_factor_w = new_shape[1]/image.shape[1]
    
    # resize image to have new_shape 
    image_resized = scipy.ndimage.zoom(image, (zoom_factor_h, zoom_factor_w), order = 3)
    
    return image_resized

def generate_ptychography_dataset(obj, probe, positions, add_position_errors=False, plot=False):

    #TODO: add invalid points to diffraction pattern

    if obj.ndim != 2 or probe.ndim != 2:
        raise ValueError("obj and probe must be 2D arrays.")

    """ Create Probe """
    probe_size_y, probe_size_x = probe.shape

    if probe_size_y > obj.shape[0] or probe_size_x > obj.shape[1]:
        raise ValueError("Probe must be smaller than the object.")

    pos_x,pos_y = positions[:,0], positions[:,1] # (N,2) array, x and y positions

    #print('Model object: ',obj.shape)
    #print('Probe: ',probe.shape)
    #print('Positions: ',positions.shape)
    
    diff_patterns = np.empty((positions.shape[0],probe_size_y,probe_size_x),dtype=float)
    probe_story_mask = np.zeros_like(obj,dtype=float)
    for i, position in enumerate(positions):
    
        px,py = np.round(position) # round to nearest integer
        px = int(px)
        py = int(py)

        """ Exit wave-field """
        wavefront = obj[py:py+probe_size_y,px:px+probe_size_x]*probe
        probe_story_mask[py:py+probe_size_y,px:px+probe_size_x] += np.abs(probe)
    
        """ Prpagation and Measurement """
        diff_patterns[i] = np.absolute(np.fft.fftshift(np.fft.fft2(wavefront)))**2
    
    positions = np.hstack((np.array([pos_x]).T ,np.array([pos_y]).T)) # adjust positions format for proper input
    position_errors = None
    
    # fix positions 
    positions = np.roll(positions, shift=1, axis=1)

    probe_story_mask = np.where(probe_story_mask>0,1,0)

    if plot:
        fig, ax = plt.subplots(1, 5, dpi=150, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.5]})

        # Plot positions
        ax[0].plot(positions[:, 0], positions[:, 1], '.', label='Original', color='black')
        ax[0].set_title('Positions')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_aspect('equal')

        # Plot amplitude and phase for the object
        ax[1].set_title('Amplitude')
        amp_img = ax[1].imshow(np.abs(obj),cmap='viridis')
        # ax[1].imshow(probe_story_mask, alpha=0.3)
        fig.colorbar(amp_img, ax=ax[1], orientation='horizontal', fraction=0.046, pad=0.04)

        ax[2].set_title('Phase')
        phase_img = ax[2].imshow(np.angle(obj), cmap='viridis')
        ax[2].imshow(probe_story_mask, alpha=0.3)
        fig.colorbar(phase_img, ax=ax[2], orientation='horizontal', fraction=0.046, pad=0.04)

        # Plot amplitude and phase for the probe
        ax[3].set_title('Probe')
        probe_rgb_img = ax[3].imshow(sscCdi.misc.convert_complex_to_RGB(probe))

        # Create the HSV color wheel
        hsv_wheel = create_hsv_wheel_image(size=256)
        ax[4].imshow(hsv_wheel)
        ax[4].set_title('')

        # # Hide the axes for the color wheel plot
        ax[4].axis('off')

        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        # Parameters for placing the tick labels
        radius = 128  # Radius of the color wheel
        center = (128, 128)  # Center of the color wheel

        # Define the tick angles and corresponding labels
        tick_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        tick_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$']

        # Convert the angles to Cartesian coordinates for placing the labels
        for angle, label in zip(tick_angles, tick_labels):
            x = center[0] + (15+radius) * np.cos(angle)
            y = center[1] + (15+radius) * np.sin(angle)
            ax[4].text(x, y, label, fontsize=10, ha='center', va='center')

        # # Add a circular boundary to indicate the extent of the color wheel
        circle = plt.Circle(center, radius, color='black', fill=False, linestyle='-', linewidth=0.5)
        ax[4].add_patch(circle)

        plt.tight_layout()
        plt.show()

    return diff_patterns, positions, position_errors, probe_story_mask

def generate_ptychography(diffraction_pattern_size = 64, probe_size = 30, n_positions_axis = 10):
    #wavelength = sscCdi.misc.wavelength_meters_from_energy_keV(energy) # meters
    N = diffraction_pattern_size
    #probe_pixel_size = wavelength*distance/(N*detector_pixel_size)
    sigma = probe_size/2

    probe = create_2d_gaussian(N, N, sigma, sigma, mu_x=None, mu_y=None)
    disk = create_2d_disk(N, N, int(np.round(N/3.8))) # N/3
    probe = probe*disk
    probe[probe>0] = 1
    probe = probe/np.max(probe)
    probe_phase = (probe - np.min(probe)) * (2 * np.pi) / (np.max(probe) - np.min(probe)) - np.pi
    probe = probe*np.exp(1j*probe_phase*0)
    #print('probe shape', probe.shape)
    obj_amplitude =  getattr(data, 'camera')() 
    obj_phase = getattr(data, 'gravel')()
    obj_amplitude =  interpolate_image(obj_amplitude, (int(np.round(3*N)), int(np.round(3*N)))) 
    obj_phase =  interpolate_image(obj_phase, (int(np.round(3*N)), int(np.round(3*N))))          
    obj_phase = obj_phase/np.max(obj_phase)


    N2 = obj_phase.shape[0]
    pad = (int(1.5*N2)-N2)/2
    obj_amplitude = np.pad(obj_amplitude,((int(pad),int(pad)),(int(pad),int(pad))),mode='constant',constant_values=0)
    obj_phase = np.pad(obj_phase,((int(pad),int(pad)),(int(pad),int(pad))),mode='constant',constant_values=0)


    obj_amplitude = obj_amplitude/obj_amplitude.max()
    obj_phase_normalized = (obj_phase - np.min(obj_phase)) * (2*np.pi) / (np.max(obj_phase) - np.min(obj_phase)) - np.pi
    obj_phase_normalized = obj_phase_normalized/2
    obj = obj_amplitude*np.exp(1j*obj_phase_normalized)
    #print('obj shape', obj.shape)

    range_x = [pad//4, 3*N*1.1]     
    range_y = [pad//4, 3*N*1.1]  
    nx = n_positions_axis
    ny = n_positions_axis
    
    positions = np.round(get_positions_array(probe.shape, range_x, range_y, nx, ny, random_positions=True, plot=False))
    diff_patterns, positions, positions_errors, probe_story_mask = generate_ptychography_dataset(obj, probe, positions)

    return obj, probe, diff_patterns, positions, positions_errors, probe_story_mask