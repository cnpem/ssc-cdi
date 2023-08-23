import numpy as np
import os, sys, h5py, time
from scipy import ndimage
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from concurrent.futures import ProcessPoolExecutor
import tqdm

""" Sirius Scientific Computing Imports """
from sscIO import io
from sscPimega import pi540D, pi135D
from sscPimega import misc as miscPimega

""" sscCdi relative imports"""
from ..misc import read_hdf5

def restore_pimega(diffraction_pattern,geometry,detector):
    """ 
    Call PIMEGA 135D or 540D restoration given a certain geometry

    Args:
        diffraction_pattern (numpy.ndarray): volume containing diffraction pattern to be restored
        geometry (dic): PIMEGA geometry dictionary
        detector (string): detector model used

    Returns:
        (numpy.ndarray): restored diffractions patterns
    """

    if detector == '135D':
        return pi135D.backward135D(diffraction_pattern , geometry)
    elif detector == '540D':
        return pi540D.backward540D(diffraction_pattern, geometry)

def restore_CUDA(input_dict,geometry,hdf5_filepaths):
    """ 
    Accellerated version of the restoration algorithm in GPUs. 
    Calls CUDA codes that accompany the package. 

    Args:
        input_dict (dict): input diction
        geometry (dict): PIMEGA geometry dictionary
        hdf5_filepaths (list): list containing the full paths to all the desired hdf5s files to restorate

    Returns:
        (numpy.ndarray): volume of restored diffraction patterns
    """

    dic = {}
    dic['path']     = hdf5_filepaths
    dic['outpath']  = input_dict["temporary_output"]
    dic['order']    = "yx" 
    dic['rank']     = "ztyx" # order of axis
    dic['dataset']  = "entry/data/data"
    dic['ngpus']    = len(input_dict["GPUs"])
    dic['gpus']     = input_dict["GPUs"]
    dic['init']     = 0
    dic['final']    = -1 # -1 to use all DPs
    dic['saving']   = 1  # save or not
    dic['timing']   = 0  # print timers 
    dic['blocksize']= 10
    dic['geometry'] = geometry
    dic['roi']      = input_dict["detector_ROI_radius"] # 512
    dic['center']   = input_dict["DP_center"] # [1400,1400]
    dic['flat']     = read_hdf5(input_dict["flatfield"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
    dic['empty']    = np.zeros_like(dic['flat']) # OBSOLETE! empty is not used anymore;
    dic['daxpy']    = [0,np.zeros([3072,3072])] 

    file_number = 11 # which file idx from hdf5_filepaths to perform restoration

    restored_data_info = pi540D.ioSetM_Backward540D( dic )
    output = pi540D.ioGetM_Backward540D( dic, restored_data_info, file_number) 
    pi540D.ioCleanM_Backward540D( dic, restored_data_info ) # clean temporary files 
    return output

def restore_IO_SharedArray(input_dict, geometry, hdf5_path,method="IO"):
    """ 
    Read diffraction data using either sscIO or h5py and calls restoration algorithms.
    Includes preprocessing (flatfield correction, mask application, background subtraction) and post-processing (data binning, if wanted)

    Args:
        input_dict (_type_): _description_
        geometry (_type_): _description_
        hdf5_path (_type_): _description_
        method (str, optional): _description_. Defaults to "IO".

    Returns:
        (numpy.ndarray): volume of restored diffraction patterns 
    """

    if input_dict["detector"] == '540D':
        DP_shape = 3072
    elif input_dict["detector"] == '135D':
        DP_shape = 1536
    else:
        sys.error('Please selector correct detector type: 135D or 540D')

    if method == "IO":
        os.system(f"h5clear -s {hdf5_path}")
        raw_DPs, _ = io.read_volume(hdf5_path, 'numpy', use_MPI=True, nprocs=input_dict["CPUs"])
    elif method == "h5py":
        raw_DPs = read_hdf5(hdf5_path)
    
    binning = int(input_dict['binning'])

    if input_dict["detector_ROI_radius"] > 0:
        half_square_side = input_dict["detector_ROI_radius"]
        if half_square_side % 2 != 0: half_square_side = half_square_side -  1 # make it even
        DP_shape = 2*half_square_side

    if binning > 1: # if applying binning
        DP_shape = DP_shape // binning

    restoration_params = (input_dict,geometry)
    diffraction_patterns, _ = miscPimega.batch(raw_DPs, input_dict['CPUs'], [ DP_shape,DP_shape ], restoration_with_processing_and_binning,  restoration_params)
    
    return diffraction_patterns

def restoration_with_processing_and_binning(DP, args):
    input_dict, geometry = args
    flat, mask, subtraction_mask = read_masks(input_dict)
    DP = corrections_and_restoration(input_dict,DP,geometry, flat, mask, subtraction_mask)
    if input_dict["binning"] > 1: 
        DP = binning_G(DP,input_dict["binning"]) # binning strategy by G. Baraldi
    return DP

def read_masks(input_dict):
    """
    Reads hdf5 containing flatfield, mask and background.

    Args:
        input_dict (dict): dictionary of inputs containing:
            - input_dict["detector"]: detector model description: '135D' or '540D'
            - input_dict["flatfield"]: path to flatfield hdf5 file
            - input_dict["mask"]: path to mask of invalid pixels hdf5 file
            - input_dict["subtraction_path"]: path to background hdf5 file
    """

    if input_dict["detector"] == '135D':
        shape = (1536,1536)
    elif input_dict["detector"] == '540D':
        shape = (3072,3072)

    if input_dict["flatfield"] != "":
        flatfield = h5py.File(input_dict["flatfield"], 'r')['entry/data/data'][()]
    else:
        flatfield = np.ones(shape)

    if input_dict["mask"] != "":
        mask = h5py.File(input_dict["mask"], 'r')['entry/data/data'][()]
    else:
        mask = np.zeros(shape)

    if "subtraction_path" in input_dict and input_dict["subtraction_path"] != "":
        background = np.asarray(h5py.File(input_dict["subtraction_path"], 'r')['entry/data/data']).squeeze().astype(np.float32)
        background = background * np.squeeze(flatfield) # Apply flatfield
    else:
        background = np.zeros(shape)

    return flatfield, mask, background

def corrections_and_restoration(input_dict, DP,geometry, flat, mask, subtraction_mask):
    """
    Applies corrections to a diffraction pattern, restores it and finally crops it to a desired size.
    """
    
    cy, cx = input_dict['DP_center']

    DP = np.squeeze(DP)

    flat[np.isnan(flat)] = -1
    flat[flat == 0] = -1 # null points at flatfield are indication of bad points
    DP = DP * np.squeeze(flat) # apply flatfield
    DP[np.squeeze(flat)==-1] = -1 # null values in both the data and in the flat will be disconsidered
    
    DP = DP - np.squeeze(subtraction_mask) # apply subtraction mask; mask is null when no subtraction is wanted

    DP = DP.astype(np.float32) # convert to float
    
    DP[np.abs(np.squeeze(mask)) == 1] = -1 # apply mask
    
    DP = restore_pimega(DP, geometry,input_dict["detector"]) # restaurate

    if input_dict["keep_original_negative_values"] == False:
        DP[DP < 0] = -1 # all invalid values must be -1 by convention

    if input_dict["detector_ROI_radius"] < 0:
        hsize = min(min(cx,DP.shape[1]-cx),min(cy,DP.shape[0]-cy)) # get the biggest size possible such that the restored difpad is still squared
        if hsize % 2 != 0: 
            hsize = hsize -  1 # make it even
    else:
        hsize = input_dict["detector_ROI_radius"]

    if input_dict["detector_ROI_radius"] != 0:
        DP = DP[cy - hsize:cy + hsize, cx - hsize:cx + hsize] # select ROI from the center (cx,cy)

    return DP 


def binning_G(binning,DP):
    """
    Binning strategy of a 2D diffraction pattern implemented by Giovanni Baraldi
    """

    if binning % 2 != 0: # no binning
        sys.error(f'Please select an EVEN integer value for the binning parameters. Selected value: {binning}')
    else:
        while binning % 2 == 0 and binning > 0:
            avg = DP + np.roll(DP, -1, -1) + np.roll(DP, -1, -2) + np.roll(np.roll(DP, -1, -1), -1, -2)  # sum 4 neigboors at the top-left value

            div = 1 * (DP >= 0) + np.roll(1 * (DP >= 0), -1, -1) + np.roll(1 * (DP >= 0), -1, -2) + np.roll( np.roll(1 * (DP >= 0), -1, -1), -1, -2)  # Boolean array! Results in the n of valid points in the 2x2 neighborhood

            avg = avg + 4 - div  # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

            avgmask = (DP < 0) & ( div > 0)  # div > 0 means at least 1 neighbor is valid. DP < 0 means top-left values is invalid.

            DP[avgmask] = avg[avgmask] / div[ avgmask]  # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

            DP = DP[:, 0::2] + DP[:, 1::2]  # binning columns
            DP = DP[0::2] + DP[1::2]  # binning lines

            DP[DP < 0] = -1

            DP[div[0::2, 0::2] < 3] = -1  # why div < 3 ? Every neighborhood that had 1 or 2 invalid points is considered invalid?

            binning = binning // 2

        if binning > 1:
            print('Entering binning > 1 only')
            avg = -DP * 1.0 + binning ** 2 - 1
            div = DP * 0
            for j in range(0, binning):
                for i in range(0, binning):
                    avg += np.roll(np.roll(DP, j, -2), i, -1)
                    div += np.roll(np.roll(DP > 0, j, -2), i, -1)

            avgmask = (DP < 0) & (div > 0)
            DP[avgmask] = avg[avgmask] / div[avgmask]

            DPold = DP + 0
            DP = DP[0::binning, 0::binning] * 0
            for j in range(0, binning):
                for i in range(0, binning):
                    DP += DPold[j::binning, i::binning]

            DP[DP < 0] = -1

    return DP


def binning_G_parallel(DPs,binning, processes):
    """
    Calls binning function in parallel for certain number of processes
    """

    # def call_binning_parallel(DP):
    #     return binning_G(DP,binning) # binning strategy by G. Baraldi

    def binning_G2(binning,DP):
        return binning_G(DP,binning)

    from functools import partial
    call_binning_parallel = partial(binning_G, binning)

    n_frames = DPs.shape[0]

    binned_DPs = np.empty((DPs.shape[0],DPs.shape[1]//binning,DPs.shape[2]//binning))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(call_binning_parallel,[DPs[i,:,:] for i in range(n_frames)])
        for counter, result in enumerate(results):
            binned_DPs[counter,:,:] = result

    if binned_DPs.shape[1] % 2 != 0: # make shape even
        binned_DPs = binned_DPs[:,0:-1,:]
    if binned_DPs.shape[2] % 2 != 0:    
        binned_DPs = binned_DPs[:,:,0:-1]

    return binned_DPs


def get_DP_center_miqueles(dbeam, project):
    """
    Approach by Eduardo Miqueles to get center of the diffraction pattern
    """
    aDP = pi540D._worker_annotation_image ( np.clip( dbeam, 0, 100) )
    aDP = ndimage.gaussian_filter( aDP, sigma=0.95, order=0 )
    aDP = aDP/aDP.max()
    aDP = 1.0 * ( aDP > 0.98 )    
    u = np.array(range(3072))
    xx,yy = np.meshgrid(u,u)
    xc = ((aDP * xx).sum() / aDP.sum() ).astype(int)
    yc = ((aDP * yy).sum() / aDP.sum() ).astype(int)
    annotation = np.array([ [xc, yc] ])
    tracking = pi540D.annotation_points_standard ( annotation )
    tracking = pi540D.tracking540D_vec_standard ( project, tracking ) 
    xc = int( tracking[0][2] )
    yc = int( tracking[0][3] ) 
    return xc, yc


def get_DP_center(difpad, refine=True, fit=False, radius=20):
    """
    Get central position of the difpad

    Args:
        difpad : diffraction pattern data
        refine (bool): Choose whether to refine the initial central position estimate. Defaults to True.
        fit (bool, optional): if true, refines using a lorentzian surface fit; else, gets the maximum. Defaults to False.
        radius (int, optional): size of the squared region around center used to refine the center estimate. Defaults to 20.

    Returns:
        center : diffraction pattern center
    """    

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

        # params, pcov = curve_fit(lorentzian2d, (X, Y), np.ravel(dataset), fit_guess)
        # lorentzian2d_fit = lorentzian2d(np.array([X, Y]), params[0], params[1], params[2], params[3], params[4], params[5])
        lorentzian2d_fit = lorentzian2d_fit.reshape(size_to_reshape)

        return lorentzian2d_fit


    def get_central_region(difpad, center_estimate, radius):
        """ Extract central region of a diffraction pattern

        Args:
            difpad : 2d diffraction pattern data
            center_estimate : the center of the image to be extracteddata
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