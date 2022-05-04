#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def masks_application(difpad, jason):

    center_row, center_col = jason["DifpadCenter"]

    if jason["DetectorExposure"][0]:  # still being tested
        print("Removing pixels above detector pile-up threshold")
        mask = np.zeros_like(difpad)
        difpad_region = np.zeros_like(difpad)
        half_size = 128 # 128 pixels halfsize mean the region has 256^2, i.e. the size of a single chip
        mask[center_row-half_size:center_row+half_size,center_col-half_size:center_col+half_size] = 1
        difpad_region = np.where(mask>0,difpad,0)        
        detector_pileup_count = 350000  # counts/sec; value according to Kalile
        detector_exposure_time = jason["DetectorExposure"][1]
        difpad_rescaled = difpad_region / detector_exposure_time # apply threshold
        difpad[difpad_rescaled > detector_pileup_count] = -1
    elif jason["CentralMask"][0]:  # circular central mask to block center of the difpad
        print("Applying circular mask to central pixels")
        radius = jason["CentralMask"][1] # pixels
        central_mask = create_circular_mask(center_col,center_row, radius, difpad.shape)
        difpad[central_mask > 0] = -1

    return difpad, jason

def create_circular_mask(center_row, center_col, radius, mask_shape):
    """Create a circular mask to block the center of the diffraction pattern

    Args:
        center_row (int): Center position in the vertical dimension
        center_col (int): Center position in the horizontal dimension
        radius (int): Radius of the circular mask in pixels
        mask_shape ([tuple]): [description]

    Returns:
        [2-dimensional ndarrya]: array containing 1s within the disk, 0 otherwise
    """
    print('Using manually set circular mask to the diffraction pattern...')
    """ All values in pixels """
    mask = np.zeros(mask_shape)
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)

    Xmesh, Ymesh = np.meshgrid(x_array, y_array)

    mask = np.where((Xmesh - center_col) ** 2 + (Ymesh - center_row) ** 2 <= radius ** 2, 1, 0)
    return mask

def fit_2d_lorentzian(dataset, fit_guess=(1, 1, 1, 1, 1, 1)):
    """ Fit of 2d lorentzian to a matrix

    Args:
        dataset : matrix to be fitted with a Lorentzian curve
        fit_guess: tuple with initial fit guesses. Defaults to (1, 1, 1, 1, 1, 1).

    Returns:
        lorentzian2d_fit : fitted surface
        params : best fit parameters
    """    
    from scipy.optimize import curve_fit

    x = np.arange(0, dataset.shape[0])
    y = np.arange(0, dataset.shape[1])
    X, Y = np.meshgrid(x, y)
    size_to_reshape = X.shape

    params, pcov = curve_fit(lorentzian2d, (X, Y), np.ravel(dataset), fit_guess)
    lorentzian2d_fit = lorentzian2d(np.array([X, Y]), params[0], params[1], params[2], params[3], params[4], params[5])
    lorentzian2d_fit = lorentzian2d_fit.reshape(size_to_reshape)

    return lorentzian2d_fit, params


def get_central_region(difpad, center_estimate, radius):
    """ Extract central region of a diffraction pattern

    Args:
        difpad : 2d diffraction pattern data
        center_estimate : the center of the image to be extracted
        radius : size of the squared region to be extracted

    Returns:
        region_around_center : extracted 2d region
    """    
    center_estimate = np.round(center_estimate)
    center_r, center_c = int(center_estimate[0]), int(center_estimate[1])
    region_around_center = difpad[center_r - radius:center_r + radius + 1, center_c - radius:center_c + radius + 1]
    return region_around_center

def refine_center_estimate(difpad, center_estimate, radius=20):
    """
    Finds a region of radius around center of mass estimate. Then fits a Lorentzian peak to this region.
    The position of the peak gives a displacement to correct the center of mass estimate

    Args:
        difpad : 2d diffraction pattern 
        center_estimate : initial estimate of the center
        radius : size of the squared region around the center to consider

    Returns:
        center : refined center position of the difpad
    """    

    region_around_center = get_central_region(difpad, center_estimate, int(radius))
    fit_guess = np.max(difpad), center_estimate[0], center_estimate[1], 5, 5, 0

    try:
        lorentzian2d_fit, fit_params = fit_2d_lorentzian(region_around_center, fit_guess=fit_guess)
        amplitude, centerx, centery, sigmax, sigmay, rotation = fit_params
        # print(f'Lorentzian center: ({centerx},{centery})')
        deltaX, deltaY = (region_around_center.shape[0] // 2 - round(centerx) + 1), ( 1 + region_around_center.shape[1] // 2 - round(centery))
    except:
        print('Fit failed')

    if 0:  # plot for debugging
        from matplotlib.colors import LogNorm
        figure, subplot = plt.subplots(1, 2)
        subplot[0].imshow(region_around_center, cmap='jet', norm=LogNorm())
        subplot[0].set_title('Central region preview')
        subplot[1].imshow(lorentzian2d_fit, cmap='jet')
        subplot[1].set_title('Lorentzian fit')

    center = (round(center_estimate[0]) - deltaX, round(center_estimate[1]) - deltaY)

    return center


def refine_center_estimate2(difpad, center_estimate, radius=20):
    """     Finds a region of radius around center of mass estimate. 
    The position of the max gives a displacement to correct the center of mass estimate

    Args:
        difpad : 2d diffraction pattern 
        center_estimate : initial estimate of the center
        radius : size of the squared region around the center to consider

    Returns:
        center : refined center position of the difpad
    """    
    from scipy.ndimage import center_of_mass

    region_around_center = get_central_region(difpad, center_estimate, int(radius))

    center_displaced = np.where(region_around_center == np.max(region_around_center))
    centerx, centery = center_displaced[0][0], center_displaced[1][0]

    deltaX, deltaY = (region_around_center.shape[0] // 2 - round(centerx)), ( region_around_center.shape[1] // 2 - round(centery))

    if 0:  # plot for debugging
        figure, subplot = plt.subplots(1, 2)
        subplot[0].imshow(region_around_center, cmap='jet', norm=LogNorm())
        subplot[0].set_title('Central region preview')
        region_around_center[centerx, centery] = 1e9
        subplot[1].imshow(region_around_center, cmap='jet', norm=LogNorm())

    center = (round(center_estimate[0]) - deltaX, round(center_estimate[1]) - deltaY)

    return center


def get_difpad_center(difpad, refine=True, fit=False, radius=20):
    """ Get central position of the difpad

    Args:
        difpad : diffraction pattern data
        refine (bool): Choose whether to refine the initial central position estimate. Defaults to True.
        fit (bool, optional): if true, refines using a lorentzian surface fit; else, gets the maximum. Defaults to False.
        radius (int, optional): size of the squared region around center used to refine the center estimate. Defaults to 20.

    Returns:
        center : diffraction pattern center
    """    
    from scipy.ndimage import center_of_mass
    center_estimate = center_of_mass(difpad)
    if refine:
        if fit:
            center = refine_center_estimate(difpad, center_estimate, radius=radius)
        else:
            center = refine_center_estimate2(difpad, center_estimate, radius=radius)
    else:
        center = (round(center_estimate[0]), round(center_estimate[1]))
    return center


    

