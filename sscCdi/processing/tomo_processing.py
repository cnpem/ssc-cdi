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

####################### ALIGNMENT ###########################################

def adjust_rotation_axis(sinogram, angles,displacements=[0,10]):

    import sscRaft

    fig, ax = plt.subplots(1,2,dpi=150)
    ax[0].imshow(np.abs(sinogram.sum(0)))
    ax[1].imshow(np.angle(sinogram.sum(0)))
    fig.suptitle('Sum of projections')
    plt.show()
    
    slice_to_reconstruct = int(input("Choose a slice in the vertical direction to reconstruct (should be an integer):"))

    displacements = np.linspace(displacements[0],displacements[1],displacements[1]-displacements[0]+1,dtype=int)
    print("The horizontal position will be displaced for the following values: ",displacements)
    
    dic = {}
    dic["algorithm_dic"] = { # if FBP: filtered back-projection
    'algorithm': "FBP",
    'gpu': [0],
    'filter': 'lorentz', # 'gaussian','lorentz','cosine','rectangle'
    'angles':angles[:,1]*np.pi/180,
    'paganin regularization': 0, # 0 <= regularization <= 1; use for smoothening
    }

    biggest_side = np.max(sinogram[0].shape)
    tomos = np.empty((len(displacements),biggest_side,biggest_side))

    
    for i, dx in enumerate(displacements):
        shifted_sino = np.roll(sinogram[:,slice_to_reconstruct,:],shift=dx,axis=1)
        tomo = sscRaft.fbp(shifted_sino, dic["algorithm_dic"])
        tomos[i] = tomo


    return tomos,displacements

def apply_axis_adjustment(sinogram,displacements):

    user_value = int(input("Choose a displacement index to use (should be an integer):"))
    
    chosen_dx = displacements[user_value]
    sinogram_adjusted_axis = np.roll(sinogram,shift=chosen_dx,axis=2)
    
    return sinogram_adjusted_axis



def wiggle_sinogram_alignment(dic,object,angles,save=True):
    """ Calls alignment algorithms for aligning the object along the sinogram frames

    Args:
        dic (dict): dictionary of inputs
        keys:
        "ordered_angles_filepath"
        "wiggle_sinogram_selection"
        "bad_frames_before_wiggle"
        "project_angles_to_regular_grid": boolean
        "step_percentage"
        "wiggle_cmas_filepath"
        "wiggle_sinogram_filepath"
            
    Returns:
        dic (dict): updated dictionary of inputs 
        "n_of_used_angles" : if projected angles, set number of used angles
        "n_of_original_angles"
        "projected_angles_filepath": if projected angles, set filepath
        "wiggle_ctr_of_mas"

    """

    start = time.time()

    angles = angles*np.pi/180
    object = make_bad_frame_null(dic["bad_frames_before_wiggle"],object)

    if dic['project_angles_to_regular_grid']:
        object, _, _, projected_angles = angle_grid_organize(object, angles,percentage=dic["step_percentage"])
        dic['n_of_used_angles']     = projected_angles.shape 
        open_or_create_h5_dataset(dic["projected_angles_filepath"],'entry','data',projected_angles,create_group=True)

    dic['n_of_original_angles'] = angles.shape # save to output log

    minimum = np.min(object)
    object = object - minimum # make minimum equal to 0
    tomoP, wiggle_cmas = wiggle(dic, object)
    dic["wiggle_ctr_of_mas"] = wiggle_cmas
    tomoP = tomoP + minimum # recover original values prior to alignment

    if save:
        print('Saving wiggle aligned sinogram...')
        open_or_create_h5_dataset(dic["wiggle_cmas_filepath"],'entry','data',wiggle_cmas,create_group=True)
        open_or_create_h5_dataset(dic["wiggle_sinogram_filepath"],'entry','data',tomoP,create_group=True)
        print('Saved aligned object at: ',dic["wiggle_sinogram_filepath"])


    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic, tomoP, wiggle_cmas

def preview_angle_projection(dic,sinogram,angles):
    """ Simulates the projection of angles to regular grid

    Args:
        dic (dict): dictionary with necessary parameters
        keys:
        "ordered_angles_filepath"
        "wiggle_sinogram_selection"
        "step_percentage"
    """    

    print("Simulating projection of angles to regular grid...")
    angles = (np.pi/180.) * angles
    total_n_of_angles = angles.shape[0]
    
    _, selected_indices, n_of_padding_frames, projected_angles = angle_grid_organize(
                                                                    sinogram, 
                                                                    angles,percentage=dic["step_percentage"]
                                                                )
    n_of_negative_idxs = len([ i for i in selected_indices if i < 0])
    selected_positive_indices = [ i for i in selected_indices if i >= 0]
    complete_array = [i for i in range(total_n_of_angles)]

    # print('Selected indices: \n',selected_indices)
    print('Before+after frames added:',n_of_padding_frames)
    print('Intermediate null frames :',len([ i for i in selected_indices if i < -1]))
    print('                        + -----')
    print("Total null frames        :", n_of_negative_idxs)
    print("Frames being used        :", len(selected_positive_indices)," of ",len(complete_array))
    print('                        + -----')
    print('Projected Angles         :', projected_angles.shape[0])

def angle_grid_organize( original_frames, angles, percentage = 100 ):
    """ 
    Given non-regular steps between rotation angles, this function projects angles to regular grid and pad it to run from 0 to 180 degrees. 
    
    Args:
        original_frames (array): sinogram
        angles (array): array of angles with non-regular steps
        percentage (int, optional): Value from 0 to 100. 100 indicates the biggest gap between consecutive angles will be used as the step. 0 indicates the smallest gap will be used as step. Used intermediate values between 0 and 100 for gaps in between. Defaults to 100.

    Returns:
       projected_frames: modified sinogram, containing padded and null frames to account for new angles of the regular grid
       selected_indices: list of indexes indicating if frame at that positions was nulled or valid
       padding_frames_counter (int): number of frames padded before and after sinogram to account for angles from 0 to 180 degrees
       angles_array (array): projected angles array

    """
    
    angles_list = []
    padding_frames_counter = 0

    start_angle = angles[:,1].min()
    end_angle   = angles[:,1].max()

    neighbor_differences = angles[1::,1] - angles[0:-1,1]  # shift and subtract to get difference between neighbors
    
    maxdtheta = abs(neighbor_differences).max() 
    mindtheta = abs(neighbor_differences).min()

    divider = (percentage*maxdtheta - (percentage-100)*mindtheta)/100 # if 100, = max; if 0 = min; intermediary values between 0 and 100 results in values between min and max

    n_of_angles = int(np.ceil( np.pi/divider))
    
    dtheta = np.pi / (n_of_angles)
    projected_frames = np.zeros([n_of_angles,original_frames.shape[1],original_frames.shape[2]])

    previous_idx = -1
    previous_min_dif = neighbor_differences[0]
    selected_indices = np.zeros([n_of_angles], dtype=int)
    for k in range(n_of_angles):

        angle = -np.pi/2.0 + k*dtheta # start at -pi/2 and move in regular steps of dTheta
        angles_list.append(angle*180/np.pi)
        if angle > end_angle or angle < start_angle: # if current is before initial or final acquired angle, use a zeroed frame
            padding_frames_counter += 1
            selected_indices[k] = -1
            projected_frames[k,:,:] = np.zeros([original_frames.shape[1],original_frames.shape[2]])
        else:
            difference_array = abs(angle - angles[:,1]) 
            arg_min_dif = np.argmin( difference_array )
            min_diff = difference_array[arg_min_dif]
            selected_indices[k] = int( arg_min_dif)

            if selected_indices[k] == previous_idx: # evaluate if previous and last frames will be the same
                if previous_min_dif > min_diff: # if angle difference is smaller now than before, zero the previous frame and declare the current to be the projected one
                    projected_frames[k-1,:,:] = np.zeros([original_frames.shape[1],original_frames.shape[2]])
                    selected_indices[k-1] = -2
                    projected_frames[k,:,:] = original_frames[selected_indices[k],:]
                else: 
                    selected_indices[k] = -3
                    continue 
            else:
                projected_frames[k,:,:] = original_frames[selected_indices[k],:]

            previous_idx = selected_indices[k]
            previous_min_dif = min_diff
        
    angles_array = np.asarray(angles_list) - np.min(angles_list) # convert values to range 0 - 180
    
    return projected_frames, selected_indices, padding_frames_counter, angles_array 

def make_bad_frame_null(bad_list, sinogram):
    """
     Null frames of interest, listed in "bad_frames_before_wiggle" dic variable

    Args:
        bad_list (list): list of bad frames to be nulled
        sinogram (array): sinogram containing frames to be nulled

    Returns:
        siogram (array): updated sinogram, with nulled frames

    """
    for k in bad_list:
        sinogram[k,:,:] = 0
    return sinogram

def wiggle(dic, sinogram):
    """ 
    Calls wiggle algorithm in both direction and return the aligned sinogram as well as the center-of_mass coordinates for alignment after 3D tomographic recon by slices.

    Args:
        dic (dict): dictionary of inputs
        keys:
        "CPUs"
        "wiggle_reference_frame"
        sinogram (array): sinogram containing misaligned frames

    Returns:
        aligned_sinogram: updated sinogram after alignment
        wiggle_cmas: center of mass coordinates to be used for final alignment, after 3D tomographic reconstruction by slices
    
    """

    import sscRaft

    temp_tomogram, shift_vertical = sscRaft.get_wiggle( sinogram, "vertical", dic["CPUs"], dic["wiggle_reference_frame"] )
    print('\tFinished vertical wiggle. Starting horizontal wiggle...')
    aligned_sinogram, shift_horizontal, wiggle_cmas_temp = sscRaft.get_wiggle( temp_tomogram, "horizontal", dic["CPUs"], dic["wiggle_reference_frame"] )
    wiggle_cmas = [[],[]]
    wiggle_cmas[1], wiggle_cmas[0] =  wiggle_cmas_temp[:,1].tolist(), wiggle_cmas_temp[:,0].tolist()
    return aligned_sinogram, wiggle_cmas

def remove_null_frames_for_visualization(wiggle_sinogram):
    null_matrix = np.zeros_like(wiggle_sinogram[0])
    indices = []
    for i in range(wiggle_sinogram.shape[0]):
        if wiggle_sinogram[i].all() == null_matrix.all():
            indices.append(i)
    wiggle_sinogram = np.delete(wiggle_sinogram,indices,axis=0)
    return wiggle_sinogram

def vignetting(shape,alpha=36,m=36):

    N = 0.99 # controls thickness of null border
    x = np.linspace(-N,N,shape[1])
    y = np.linspace(-N,N,shape[0])
    X, Y = np.meshgrid(x,y)

    vignet = np.exp(-alpha*(X)**m)*np.exp(-alpha*(Y)**m)
    vignet[vignet<1e-10] = 0
    return vignet

def gaussian2D(shape,center,sigma):

    x = np.linspace(0,shape[1]-1,shape[1]) - shape[1]//2
    y = np.linspace(0,shape[0]-1,shape[0]) - shape[0]//2
    X, Y = np.meshgrid(x,y)

    # Evaluate the Gaussian function at each point in the domain
    Z = np.exp(-((X - center[1])**2  / (2 * sigma[1]**2))) * np.exp(-((Y - center[0])**2  / (2 * sigma[0]**2)))
    
    return Z

####################### TOMOGRAPHY ###########################################

def tomography(dic, sinogram,save=True):
    """ Calls tomographic algorithms from sscRaft

    Args:
        dic (dict): dictionary of inputs
        keys:
        "reconstruction_filepath"
        sinogram (array): sinogram

    Returns:
        reconstruction3D (array): 3D reconstructed volume via tomography

    """
    
    start = time.time()
    reconstruction3D = call_sscRaft(dic, sinogram,save)
    if save:
        np.save(dic["reconstruction_filepath"],reconstruction3D)
        open_or_create_h5_dataset(dic["reconstruction_filepath"],'recon','volume',reconstruction3D,create_group=True)
        print('Saved reconstructed object at: ',dic["wiggle_sinogram_filepath"])

    print(f'Time elapsed: Tomography: {time.time() - start} s' )
    return reconstruction3D

def automatic_regularization(sino, L=0.001):
    """
    Applies regularization according to: https://doi.org/10.1016/j.rinam.2019.100088

    Args:
        sino (array): aligned sinogram frame
        L (float): regulirization parameter 

    Returns:
        D (array): regularized frame
    
    """

    a = 1
    R = sino.shape[1]
    V = sino.shape[0]
    th = np.linspace(0, np.pi, V, endpoint=False)
    t  = np.linspace(-a, a, R)
    dt = (2*a)/float((R-1))
    wc = 1.0/(2*dt)
    w = np.linspace(-wc, wc, R)
    if 1: # two options
        h = np.abs(w) / (1 + 4 * np.pi * L * (w**2) )
    else:
        h = 1 / (1 + 4 * np.pi * L * (w**2) )
    G = np.fft.fftshift(np.transpose(np.kron(np.ones((V, 1)), h))).T
    B = np.fft.fft(sino, axis=1)
    D = np.fft.ifft(B * G, axis=1).real
    return D

def save_or_load_wiggle_ctr_mass(path,wiggle_cmass = [[],[]],save=True):
    """ 
    Save or load wiggle ctr of mass

    Args:
        path (str): path to ctr of mass
        wiggle_cmass (list, optional): . Defaults to [[],[]].
        save (bool, optional): boolean to save or not the ctr of mass. Defaults to True.

    Returns:
        int: flag if saved or ctr of mass
    
    """    
    
    if save:
        wiggle_cmass = np.asarray(wiggle_cmass)
        np.save(path, wiggle_cmass)
        return 0
    else:
        array = np.load(path)
        wiggle_cmas = [array[0,:],array[1,:]]
        return wiggle_cmas

def add_plot_suffix_to_file(path):
    """ Add plot suffix to file

    Args:
        path (str): string with the path to file

    Returns:
        str: path with plot suffix

    """    

    first_part = path.split(".")[0]
    second_part = path.split(".")[-1]
    return first_part + "_PLOT." + second_part

def get_and_save_downsampled_sinogram(sinogram,path,downsampling=4):
    """ 
    Get and save downsampled sinogram

    Args:
        sinogram (array): sinogram
        path (str): string with the path to save  downsample sinogram
        downsampling (int, optional): Defaults to 4.

    Returns:
        donwsampled_sinogram (array): downsampled sinogram

    """    

    downsampled_sinogram = sinogram[:,::downsampling,::downsampling]
    np.save(add_plot_suffix_to_file(path),downsampled_sinogram)
    return downsampled_sinogram

def check_sinogram_shape(sinogram):
    
    myshape = sinogram.shape

    if myshape[1]%2 != 0 or myshape[2]%2 != 0:

        if myshape[1]%2 != 0:
            sinogram = sinogram[:,0:-1,:]
        if myshape[2]%2 != 0:
            sinogram = sinogram[:,:,0:-1]

        print(f"As of now, filtering in sscRaft requires an array with even pixels. Your array shape is {myshape}. Adjusting shape to {sinogram.shape} \n")
            
    return sinogram

def call_sscRaft(dic, sinogram,save=True):
    """
    Performs tomography

    Args:
        dic (dict): dictionary of inputs
        keys:
        "ordered_angles_filepath"
        "using_wiggle": boolean
        "wiggle_cmas_filepath"
        "wiggle_ctr_of_mas"
        "project_angles_to_regular_grid"
        "algorithm_dic"
        keys:
        "angles"
        "nangles"
        "reconSize"
        "algorithm"
        "tomooffset"
        "is360"
        "automatic_regularization"
        sinogram (array): sinogram

    Returns:
        reconstruction3D (array): tomographic volume

    """

    import sscRaft

    sinogram = check_sinogram_shape(sinogram)

    angles_filepath = dic["ordered_angles_filepath"]
    
    if 'using_wiggle' not in dic:
        dic['using_wiggle'] = False
    if  'project_angles_to_regular_grid' not in dic:
        dic['project_angles_to_regular_grid'] = False

    if dic['using_wiggle']:
        wiggle_cmas_path  = dic["wiggle_cmas_filepath"]
        try:
            wiggle_cmas = dic["wiggle_ctr_of_mas"]
        except:
            wiggle_cmas = np.load(dic["wiggle_cmas_filepath"])

        if wiggle_cmas == [[],[]]: # if no ctr of mass, save file with it
            wiggle_cmas = save_or_load_wiggle_ctr_mass(wiggle_cmas_path,save=False)

    if dic['project_angles_to_regular_grid'] == True:
        angles_filepath = angles_filepath[:-3]+'_projected.h5'
        angles = h5py.File(angles_filepath,'r')['entry/data'][()]
        dic['algorithm_dic']['angles[rad]'] = angles*np.pi/180
    else:
        angles = h5py.File(angles_filepath,'r')['entry/data'][()]
        dic['algorithm_dic']['angles[rad]'] = angles[:,1]*np.pi/180 

    """ Automatic Regularization """
    if 'automatic_regularization' not in dic:  dic['automatic_regularization'] = 0
    if dic['automatic_regularization'] != 0:
        print('\tStarting automatic regularization frame by frame...')
        for k in range(sinogram.shape[1]):
            sinogram[:,k,:] = automatic_regularization( sinogram[:,k,:], dic['automatic_regularization'])

    """ Tomography """
    sinogram = np.swapaxes(sinogram, 0, 1) # exchange axis 0 and 1 
    dic['algorithm_dic']['nangles'] = sinogram.shape[1]
    dic['algorithm_dic']['reconSize'] = sinogram.shape[2]

    if dic["algorithm_dic"]['algorithm'] == "FBP": 
        print(f'Starting tomographic algorithm FBP algorithm')
        if 'tomooffset' not in dic["algorithm_dic"]: 
            dic["algorithm_dic"]['tomooffset'] = 0
        reconstruction3D = sscRaft.fbp(sinogram, dic["algorithm_dic"])
    elif dic["algorithm_dic"]['algorithm'] == "EM": # sinogram is what comes out of wiggle
        dic["algorithm_dic"]['is360'] = False
        print(f'Starting tomographic algorithm EM algorithm')
        reconstruction3D = sscRaft.em( sinogram, dic['algorithm_dic'] )
    else:
        sys.exit('Select a proper reconstruction method')
     
    if dic['using_wiggle']: 
        print("\tApplying wiggle center-of-mass correction to 3D reconstructed slices...")
        reconstruction3D = sscRaft.set_wiggle(reconstruction3D, 0, -np.array(wiggle_cmas[1]), -np.array(wiggle_cmas[0]), dic["CPUs"])

    print('\t Tomography done!')

    if save:
        print('Saving tomography logfile...')
        save_json_logfile_tomo(dic)

    return reconstruction3D


def gradient_y_wrapping_insensitive(sinogram,pixel_size):
    g = np.exp(1j*sinogram) 
    return np.angle(g * np.conj(np.roll(g,shift=1,axis=-1)))/pixel_size

def phase_derivative_hilbert_transform(sinogram,pixel_size=1):
    """
    Calculate phase derivative to use it with backprojection (no filter!) 
    Pixel size in meters
    """
    grad_sinogram = gradient_y_wrapping_insensitive(sinogram,pixel_size)
    hilbert = np.imag(scipy.signal.hilbert(grad_sinogram,axis=-1)/(2*np.pi))
    return hilbert

####################### PLOT ###########################################

def plot_histograms(recon3D, equalized_tomogram,bins=300):
    """ 
    Histogram plot funcion

    Args:
        recon3D (array): unequalized tomogram
        equalized_tomogram (array): equalized tomogram
        bins (int, optional): histogram bin number. Defaults to 300.
        
    """    
    recon_hist = recon3D.flatten()
    equalized_hist = equalized_tomogram.flatten()

    statistics_recon = (np.max(recon3D),np.min(recon3D),np.mean(recon3D),np.std(recon3D))
    print('Original data statistics: ',f'\n\tMax = {statistics_recon[0]:.2e}\n\t Min = {statistics_recon[1]:.2e}\n\t Mean = {statistics_recon[2]:.2e}\n\t StdDev = {statistics_recon[3]:.2e}')

    statistics_equalized = (np.max(equalized_tomogram),np.min(equalized_tomogram),np.mean(equalized_tomogram),np.std(equalized_tomogram))
    print('Thresholded data statistics: ',f'\n\tMax = {statistics_equalized[0]:.2e}\n\t Min = {statistics_equalized[1]:.2e}\n\t Mean = {statistics_equalized[2]:.2e}\n\t StdDev = {statistics_equalized[3]:.2e}')

    figure2, axs = plt.subplots(1,2,figsize=(13,2))

    axs[0].set_yscale('log'), axs[1].set_yscale('log')
    axs[0].set_title("Histogram: Original")
    axs[1].set_title("Histogram: Equalized")
    figure2.canvas.header_visible = False 
    figure2.tight_layout()


    axs[0].hist(recon_hist,bins=bins)
    axs[1].hist(np.where(equalized_hist==0,np.NaN,equalized_hist),bins=bins,color='green')

    plt.show()
