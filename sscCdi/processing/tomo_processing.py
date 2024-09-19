# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################



import os, sys, time, ast, h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imsave
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import scipy

from ..misc import save_json_logfile_tomo, open_or_create_h5_dataset
from .unwrap import remove_phase_gradient, unwrap_in_parallel

####################### SORTING ###################################

def sort_sinogram_by_angle(object, angles,object_savepath='',angles_savepath=''):
    """
     
    Sorting script to reorder sinogram frames by angle, instead of acquisition order

    Args:
        object (ndarray): 3d sinogram to be sorted
        angles (ndarray): rotation angle for each frame of the sinogram
        object_savepath (str, optional): Path to hdf5 file in which new sinogram will be saved. Defaults to ''.
        angles_savepath (str, optional): Path to hdf5 file in which new angles will be saved. Defaults to ''.

    Returns:
        sorted_object
        sorted_angles

    """    

    start = time.time()
    sorted_angles = sort_angles(angles) # input colums with frame number and angle in rad
    sorted_object = reorder_slices_low_to_high_angle(object, sorted_angles)

    if angles_savepath != '':
        print('Saving sorted angles...')
        open_or_create_h5_dataset(angles_savepath,'entry','data',sorted_angles,create_group=True)
        print('Saved sorted angles at: ', angles_savepath)
    if object_savepath != '':
        print('Saving sorted sinogram...')
        open_or_create_h5_dataset(object_savepath,'entry','data',sorted_object,create_group=True)
        print('Saved sorted object at: ',object_savepath)

    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return sorted_object, sorted_angles

def remove_frames_from_sinogram(sinogram,angles,list_of_bad_frames,updated_object_filepath= '', updated_angles_filepath= ''):
    """ Remove wanted frames from volume of images and from the respective list of angle values

    Args:
        sinogram: 3d array of images. First index selects the frame.
        angles: array of corresponding rotation angle values to the sinogram frames. There must be 1 angle for each sinogram frame.
        list_of_bad_frames: list of values containing the frames to be removed
        ordered_object_filepath (str, optional): Path to hdf5 file in which new sinogram will be saved. Defaults to ''.
        ordered_angles_filepath (str, optional): Path to hdf5 file in which new angles will be saved. Defaults to ''.

    """    


    print('Original shape: ',sinogram.shape)

    new_object = np.delete(sinogram,list_of_bad_frames,axis=0)
    new_angles = np.delete(angles,list_of_bad_frames,axis=0)

    print('New shape: ',new_object.shape)

    if updated_object_filepath != '':
        open_or_create_h5_dataset(updated_object_filepath,'entry','data',new_angles,create_group=True)
    if updated_angles_filepath != '':
        open_or_create_h5_dataset(updated_angles_filepath,'entry','data',new_object,create_group=True)

    return new_object, new_angles

def sort_angles(angles):
    """ Sort angles array from smallest to highest angle

    Args:
        angles (array): angle array in time/acquisition order

    Returns:
        angles (array): angle array in sorted from smallest to highest angle
    """
    angles = np.asarray(angles)
    
    angles[:,0] = np.asarray([ i for i in range(angles.shape[0])]) # make sure numbering is from 0 to N

    sorted_angles = angles[angles[:,1].argsort(axis=0)]
    return sorted_angles 

def reorder_slices_low_to_high_angle(object, rois):
    """ 
    
    Reorder sinogram according to the sorted angles array

    Args:
        object (ndarray): sinogram to be sorted
        angles (array): angle array in sorted from smallest to highest angle

    Returns:
        sorted_object (array): sinogram sorted by angle

    """

    sorted_object = np.zeros_like(object)

    for k in range(object.shape[0]): # reorder slices from lowest to highest angle
        # print(f'New index: {k}. Old index: {int(rois[k,0])}')
        sorted_object[k] = object[int(rois[k,0])]

    return sorted_object

######################### CROP #################################################

def crop_volume(volume,top_crop,bottom_crop,left_crop,right_crop,cropped_savepath='',crop_mode=0):
    """ Crops images in a volume in the Y,X directions.

    Args:
        volume (ndarray): 3d array of shape (N,Y,X), N being the slice number
        top_crop (int): number of pixels on top
        bottom_crop (int): number of pixels on botto,
        left_crop (int): number of pixels on left
        right_crop (int): number of pixels on right
        cropped_savepath (str, optional): Path to hdf5 file in which new volume will be saved. Defaults to ''.
        crop_mode (int, optional): Crop mode == 0 means each crop will be like [:,top_crop:-bottom_crop,left_crop:-right_crop]. Mode ==1 means [:,top_crop:bottom_crop,left_crop:right_crop]. Defaults to 0.

    Returns:
        cropped volume 

    """    

    
    start = time.time()
    if crop_mode == 0:
        volume = volume[:,top_crop:-bottom_crop,left_crop:-right_crop] # Crop frame
    elif crop_mode == 1:
        volume = volume[:,top_crop:bottom_crop,left_crop:right_crop] # Crop frame
    else:
        sys.exit('Please select crop_mode 0 or 1')

    print(f"Cropped sinogram shape: {volume.shape}")
    if cropped_savepath != '':
        print('Saving cropped volume...')
        open_or_create_h5_dataset(cropped_savepath,'entry','data',volume,create_group=True)
        print('Saved cropped volume at: ',cropped_savepath)
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return volume

######################### UNWRAP #################################################

def unwrap_sinogram(sinogram,unwrapped_savepath=''):
    """ 
    Calls unwrapping algorithms in multiple processes for the sinogram frames
    """
    start = time.time()

    sinogram = unwrap_in_parallel(sinogram)
    if unwrapped_savepath != '':
        print('Saving unwrapped volume..')
        open_or_create_h5_dataset(unwrapped_savepath,'entry','data',sinogram,create_group=True)
        print('Saved unwrapped sinogram at: ',unwrapped_savepath)
 
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return sinogram

######################### EQUALIZATION #################################################

def equalize_sinogram(sinogram,invert=False,remove_phase_gradient=True,roi=[],cpus=1,iterations=1,min_max=(),remove_negative_values=False,remove_offset=False,savepath=''):
    start = time.time()

    equalized_sinogram = equalize_frames_parallel(sinogram,invert=invert,remove_phase_gradient=remove_phase_gradient,roi=roi,cpus=cpus,iterations=iterations,min_max=min_max,remove_negative_values=remove_negative_values,remove_offset=remove_offset)

    if savepath != '':
        print('Saving equalized sinogran...')
        open_or_create_h5_dataset(savepath,'entry','data',equalized_sinogram,create_group=True)
        print('Saved equalized object at: ',savepath)

    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return equalized_sinogram

def tomo_equalize3D(tomo,remove_outliers=3,threshold=20,local_offset=[],mask=[],savepath=''):
    """ Call equalization algorithms for tomogram frames

    Args:
        dic (dict): dictionary of inputs
        keys:
        "reconstruction_filepath"
        "tomo_remove_outliers"
        "tomo_threshold"
        "tomo_local_offset"
        "eq_reconstruction_filepath"

    """
    start = time.time()

    dic = {} 
    dic["tomo_remove_outliers"] = remove_outliers
    dic["tomo_threshold"] = threshold
    dic["tomo_local_offset"] = local_offset # [top,bottom,left,right, axis]
    dic["tomo_mask"] = mask

    equalized_tomogram = equalize_tomogram(
        tomo,np.mean(tomo),
        np.std(tomo),
        remove_outliers=dic["tomo_remove_outliers"],
        threshold=float(dic["tomo_threshold"]),
        bkg_window=dic["tomo_local_offset"]
    )

    if savepath != {}:
        open_or_create_h5_dataset(savepath,'entry','data',equalized_tomogram,create_group=True)
        print('Saved equalized object at: ',savepath)

    open_or_create_h5_dataset( savepath,'entry', 'data',  equalized_tomogram,create_group=True)
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return equalized_tomogram

def remove_outliers(data,sigma):
    """ Remove all values above/below +sigma/-sigma sigma values. 1 sigma = 1 standard deviation

    Args:
        data (array): sinogram slice
        sigma (int): number of sigmas to reject

    Returns:
        data (array): sinogram slice with filtered values
    
    """

    mean, std = np.mean(data), np.std(data)
    data = np.where(data > mean + sigma*std,0,data)
    data = np.where(data < mean - sigma*std,0,data)
    return data

def equalize_frame(dic,frame):
    """
    Performs a series of processing steps over a 2D array, namely:

    1. Removes a gradient (i.e. the phase ramp) for the image as a whole
    2. Makes any NaN values null
    3. Removes outlier values above/below a certain sigma
    4. Removes the global offset of the array, making the smallest value null
    5. Removes a local offset of the array, subtracting the mean value of a desired region from the entire array

    Args:
        dic (dict): dictionary of inputs
        keys:
        "equalize_invert" (boolean): Description of this key
        "equalize_remove_phase_gradient" (boolean): Description of this key
        "equalize_ROI" (type): Description of this key
        "equalize_remove_phase_gradient_iterations" (type): Description of this key
        "equalize_local_offset" (type): Description of this key
        "equalize_set_min_max" (type): Description of this key
        "equalize_non_negative" (type): Description of this key
        frame (array): 2D image/frame to be equalized

    Returns:
        array: equalized frame

    """

    if dic["equalize_invert"] == True:
        frame = -frame

    # Remove phase ramp
    if dic["equalize_remove_phase_gradient"] == True:
        if dic["equalize_ROI"] == []:
            mask = np.ones_like(frame,dtype=bool)
        else:
            mask = np.zeros_like(frame,dtype=bool)
            mask[dic["equalize_ROI"][0]:dic["equalize_ROI"][1],dic["equalize_ROI"][2]:dic["equalize_ROI"][3]] = True
        
        if "equalize_remove_phase_gradient_iterations" in dic:
            iterations = dic["equalize_remove_phase_gradient_iterations"]
        else:
            iterations = 5 
        frame = remove_phase_gradient(frame, mask,loop_count_limit=iterations)

    # Check for NaNs
    whereNaN = np.isnan(frame)
    if whereNaN.any():
        print("NaN values found in frame after removing gradient. Removing them!")
        frame = np.where(whereNaN,0,frame)

    # OBSOLETE: Remove outliers
    # if remove_outlier != 0:
    #     frame = remove_outliers(frame,remove_outlier)

    # OBSOLETE: Remove global offset
    # if remove_global_offset:
    #     frame -= np.min(frame)

    # Remove average offset from specific region
    if dic["equalize_local_offset"]:
        mean = np.mean(frame[dic["equalize_ROI"][0]:dic["equalize_ROI"][1],dic["equalize_ROI"][2]:dic["equalize_ROI"][3]])
        frame -= mean

    if dic["equalize_set_min_max"] != []:
        frame = np.where(frame<dic["equalize_set_min_max"][0],0,np.where(frame>dic["equalize_set_min_max"][1],0,frame))


    if dic["equalize_non_negative"]:    
        frame = np.where(frame<0,0,frame) # put remaining negative values to zero

    return frame

def equalize_frames_parallel(sinogram,invert=False,remove_phase_gradient=True,roi=[],cpus=1,iterations=1,min_max=(),remove_negative_values=False,remove_offset=False):
    """ 
    
    Calls function equalize_frame in parallel at multiple threads for each frameo of the sinogram

    Args:
        sinogram (array): sinogram to be equalized
        dic (dict): dictionary of inputs
        keys:
        "CPUs": number of CPUs

    Returns:
        equalized_sinogram: equalized sinogram

    """

    dic = {}
    dic["CPUs"] = cpus
    dic["equalize_invert"] = invert                           # invert phase shift signal from negative to positive
    dic["equalize_ROI"] = roi                   # region of interest of null region around the sample used for phase ramp and offset corrections
    dic["equalize_remove_phase_gradient"] = remove_phase_gradient            # if empty and equalize_ROI = [], will subtract best plane fit from whole image
    dic["equalize_remove_phase_gradient_iterations"] = iterations    # number of times the gradient fitting is performed
    dic["equalize_local_offset"] = remove_offset                     # remove offset of each frame from the mean of ROI 
    dic["equalize_set_min_max"]= min_max                         # [minimum,maximum] threshold values for whole volume
    dic["equalize_non_negative"] = remove_negative_values                    # turn any remaining negative values to zero


    minimum, maximum, mean, std = np.min(sinogram), np.max(sinogram), np.mean(sinogram), np.std(sinogram)

    # Remove NaNs
    whereNaN = np.isnan(sinogram)
    if whereNaN.any():
        print("NaN values found in unwrapped sinogram. Removing them!")
        sinogram = np.where(whereNaN,0,sinogram)

    # Call parallel equalization
    equalize_frame_partial = partial(equalize_frame, dic)
    print('Sinogram shape to equalize: ', sinogram.shape)

    n_frames = sinogram.shape[0]

    processes = dic["CPUs"]
    print(f'Using {processes} parallel processes')

    with ProcessPoolExecutor(max_workers=processes) as executor:
        equalized_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(equalize_frame_partial,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            # minimum, maximum, mean, std = np.min(result), np.max(result), np.mean(result), np.std(result)
            equalized_sinogram[counter,:,:] = result

    minimum1, maximum1, mean1, std1 = np.min(equalized_sinogram), np.max(equalized_sinogram), np.mean(equalized_sinogram), np.std(equalized_sinogram)
    print(f'Min \t Mean-3*sigma \t Mean \t Mean+3*sigma \t Max ')
    print(f'Old {minimum:.2f}, {mean-3*std:.2f}, {mean:.2f}, {mean+3*std:.2f},{maximum:.2f}')
    print(f'New: {minimum1:.2f}, {mean1-3*std1:.2f},{mean1:.2f}, {mean1+3*std1:.2f},{maximum1:.2f}')

    return equalized_sinogram

def equalize_scipy_optimization(mask,img,initial_guess=(1,1,1),method='Nelder-Mead',max_iter = 1,stop_criteria=(1e-5,1e-5,1e-2)):
    
    def equalization_cost_function(params,img,mask,x,y):
        a, b, c = params
        return np.linalg.norm(mask*(img- (a*x+b*y+c)))**2
    
    if img.shape != mask.shape:
        sys.exit('Image and Mask do not have the same shape')

    y, x = np.indices(img.shape)
    
    for i in range(0,max_iter):
    
        result = scipy.optimize.minimize(equalization_cost_function, initial_guess, args=(img,mask,x,y),method=method)

        success = result.success
        if success == False:
            print('Convergence failed. Scipy.minimize message: ',result.message)
        else:
            # print('Minization done')
            pass

        a,b,c = result.x
        plane = a*x+b*y+c
        img = img - plane
    
    return img, plane, (a,b,c)
    
def equalize_scipy_optimization_parallel(sinogram,mask,initial_guess=(0,0,0),method='Nelder-Mead',max_iter = 1,stop_criteria=(1e-5,1e-5,1e-2),processes=10):

    equalize_scipy_optimization_partial = partial(equalize_scipy_optimization, mask,initial_guess=initial_guess,method=method,max_iter = max_iter,stop_criteria=stop_criteria)
    n_frames = sinogram.shape[0]
    equalized_sinogram = np.empty_like(sinogram)
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(tqdm(executor.map(equalize_scipy_optimization_partial,sinogram),total=n_frames,desc='Equalizing frames'))
        for counter, result in tqdm(enumerate(results),desc="Populating result matrix"):
            equalized_sinogram[counter,:,:] = result[0]
            
    return equalized_sinogram


def equalize_parallel_gradient_descent(volume, iterations=50, mask=None, step_size=1e-6, initial_guess = (0,0,0)):
    
    init_a, init_b, init_c = initial_guess
    
    step_a = step_b = step_c = step_size
    
    equalized_volume = np.empty_like(volume)
    sizey, sizex = volume[0].shape
    
    Y, X = np.indices((sizey,sizex))
    Y = Y - sizey//2
    Y = Y/np.max(Y)
    X = X - sizex//2
    X = X/np.max(X)
    
    
    if mask is None:
        mask = np.ones_like(X)

    def calculate_gradients(data,X,Y,W,a,b,c):
        gradA = -2*np.sum(W*X*(W*data-(a*X+b*Y+c)))
        gradB = -2*np.sum(W*Y*(W*data-(a*X+b*Y+c)))
        gradC = -2*np.sum(W*(W*data-(a*X+b*Y+c)))
        return gradA, gradB, gradC

    def gradient_descent_iteration(a,b,c,gradA, gradB, gradC,step_a,step_b, step_c):
        a = a - step_a * gradA
        b = b - step_b * gradB
        c = c - step_c * gradC
        return a, b, c


    def process_slice(slice, X, Y, mask, iterations, step_a, step_b, step_c, init_a, init_b, init_c):
        a, b, c = init_a, init_b, init_c
        for j in range(iterations):
            gradA, gradB, gradC = calculate_gradients(slice, X, Y, mask, a, b, c)
            a, b, c = gradient_descent_iteration(a, b, c, gradA, gradB, gradC, step_a, step_b, step_c)
        return a, b, c


    fitted_coefficients = np.empty((volume.shape[0], 3))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [ executor.submit(process_slice, volume[i], X, Y, mask, iterations, step_a, step_b, step_c, init_a, init_b, init_c) for i in range(volume.shape[0])  ]
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing slices with {executor._max_workers} workers")):
            a,b,c = future.result()
            fitted_coefficients[i] = a,b,c
            equalized_volume[i] = volume[i] - (a*X+b*Y+c)

    return equalized_volume, fitted_coefficients


def equalize_tomogram(equalized_tomogram,mean,std,remove_outliers=0,threshold=0,bkg_window=[]):
    """ 
    
    Filters outliers in the tomogram

    Args:
        equalized_tomogram (array): 3D reconstructed volume from tomographic algorithm
        mean (float): mean value of 3D reconstruction
        std (float): standard deviation of 3D reconstruction
        remove_outliers (int, optional): if not zero, will remove outliers above/below a certain sigma, sigma being this variable. Defaults to 0.
        threshold (int, optional): value T to threshold the volume. All voxels with absolute value higher than T are set to zero. Defaults to 0.
        bkg_window (list, optional): List of type [top,bottom,left,right,direction] indication the the coordinates of a squared window over a certain direction. The mean value of the window will be subtracted from all voxels. Value below this mean are set to null. Defaults to [].

    Returns:
        equalized_tomogram (array): 3D equalized tomogram

    """
    
    if threshold != 0:
        equalized_tomogram = np.where( np.abs(equalized_tomogram) > threshold,0,equalized_tomogram)

    if remove_outliers != 0:
            equalized_tomogram = np.where( equalized_tomogram > mean+remove_outliers*std,0,equalized_tomogram)
            equalized_tomogram = np.where( equalized_tomogram < mean-remove_outliers*std,0,equalized_tomogram)

    if bkg_window !=[]:
        axis_direction = bkg_window[4] # last item of list indicates the direction of the slicing
        if axis_direction == 0:
            window = equalized_tomogram[:,bkg_window[0]:bkg_window[1],bkg_window[2]:bkg_window[3]]
        elif axis_direction == 1:
            window = equalized_tomogram[bkg_window[0]:bkg_window[1],:,bkg_window[2]:bkg_window[3]]
        elif axis_direction == 2:
            window = equalized_tomogram[bkg_window[0]:bkg_window[1],bkg_window[2]:bkg_window[3],:]

        offset = np.mean(window)
        equalized_tomogram = equalized_tomogram - offset
        equalized_tomogram = np.where(equalized_tomogram<0,0,equalized_tomogram)

    return equalized_tomogram

