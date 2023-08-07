import os, sys, time, ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imsave
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import sscRaft, sscRadon

from ..misc import save_json_logfile, create_directory_if_doesnt_exist, save_json_logfile_tomo, open_or_create_h5_dataset
from ..processing.unwrap import remove_phase_gradient, unwrap_in_parallel, unwrap_in_sequence

def define_paths(dic):
    """ Defines all the path required for the remaining parts of the code; adds them to the dicitionary and creates necessary folders

    Args:
        dic (dict): dictionary of inputs  

    Returns:
        dic (dict): updated dictionary of inputs  
    """

    dic["output_folder"] = dic["sinogram_path"].rsplit('/',1)[0]
    dic["filename"]    = os.path.join(dic["sinogram_path"].rsplit('/',1)[1].split('.')[0]+'_'+dic["contrast_type"])
    dic["temp_folder"] = os.path.join(dic["output_folder"],'temp')
    dic["ordered_angles_filepath"]       = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles.npy')
    dic["projected_angles_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles_projected.npy')
    dic["ordered_object_filepath"]       = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_object.npy')
    dic["cropped_sinogram_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_cropped_sinogram.npy')
    dic["pre_aligned_sinogram_filepath"] = os.path.join(dic["temp_folder"],f'{dic["filename"]}_prealigned_sinogram.npy')
    dic["equalized_sinogram_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_equalized_sinogram.npy')
    dic["unwrapped_sinogram_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_unwrapped_sinogram.npy')
    dic["wiggle_sinogram_filepath"]      = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_sinogram.npy')
    dic["wiggle_cmas_filepath"]          = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_ctr_mass.npy')
    dic["reconstruction_filepath"]       = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo.npy')
    dic["eq_reconstruction_filepath"]    = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo_equalized.npy')

    create_directory_if_doesnt_exist(dic["output_folder"])
    create_directory_if_doesnt_exist(dic["temp_folder"])

    return dic

####################### SORTING ###################################

def tomo_sort(dic, object, angles):
    """Call sorting algorithm to reorder sinogram frames by angle, instead of acquisition order

    Args:
        dic (dict): dictionary of inputs  
        object (ndarray): read sinogram to be sorted
        angles (ndarray): rotation angle for each frame of the sinogram

    """
    start = time.time()
    sorted_angles = sort_angles(angles) # input colums with frame number and angle in rad
    sorted_object = reorder_slices_low_to_high_angle(object, sorted_angles)

    np.save(dic["ordered_angles_filepath"], sorted_angles)
    np.save(dic["ordered_angles_filepath"], sorted_object) 
    print(f'Time elapsed: {time.time() - start:.2f} s' )

def remove_frames_after_sorting(dic):

    sorted_object = np.load(dic["ordered_angles_filepath"])
    sorted_angles = np.load(dic["ordered_angles_filepath"])

    print('Original shape: ',sorted_object.shape)

    sorted_object = np.delete(sorted_object,dic["bad_frames_after_sorting"],axis=0)
    sorted_angles = np.delete(sorted_angles,dic["bad_frames_after_sorting"],axis=0)

    print('New shape: ',sorted_object.shape)

    np.save(dic["ordered_angles_filepath"], sorted_angles)
    np.save(dic["ordered_object_filepath"], sorted_object) 

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
    """ Reorder sinogram according to the sorted angles array

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

def tomo_crop(dic,object):
    """ Crops sinogram according to cropping parameters in dic

    Args:
        dic (dict): dictionary of inputs  

    """
    start = time.time()
    object = object[:,dic["top_crop"]:-dic["bottom_crop"],dic["left_crop"]:-dic["right_crop"]] # Crop frame
    print(f"Cropped sinogram shape: {object.shape}")
    np.save(dic["cropped_sinogram_filepath"],object) # save shaken and padded sorted sinogram
    print(f'Time elapsed: {time.time() - start:.2f} s' )

######################### UNWRAP #################################################

def tomo_unwrap(dic,object):
    """ Calls unwrapping algorithms in multiple processes for the sinogram frames

    Args:
        dic (dict): dictionary of inputs  

    """
    start = time.time()

    object = make_bad_frame_null(dic["bad_frames_before_unwrap"],object)

    object = unwrap_in_parallel(object)
    np.save(dic["unwrapped_sinogram_filepath"],object)  
    print(f'Time elapsed: {time.time() - start:.2f} s' )

######################### EQUALIZATION #################################################

def tomo_equalize(dic, sinogram):
    """ Calls equalization algorithms in multiple processes for sinogram frames

    Args:
        dic (dict): dictionary of inputs  

    """
    start = time.time()

    sinogram = make_bad_frame_null(dic["bad_frames_before_equalization"],sinogram)

    equalized_sinogram = equalize_frames_parallel(sinogram,dic)
    np.save(dic["equalized_sinogram_filepath"] ,equalized_sinogram)
    print(f'Time elapsed: {time.time() - start:.2f} s' )

def tomo_equalize3D(dic):
    """ Call equalization algorithms for tomogram frames

    Args:
        dic (dict): dictionary of inputs f 

    """
    start = time.time()
    reconstruction = np.load(dic["reconstruction_filepath"])
    equalized_tomogram = equalize_tomogram(reconstruction,np.mean(reconstruction),np.std(reconstruction),remove_outliers=dic["tomo_remove_outliers"],threshold=float(dic["tomo_threshold"]),bkg_window=dic["tomo_local_offset"])
    np.save(dic["eq_reconstruction_filepath"],equalized_tomogram)
    open_or_create_h5_dataset(dic["eq_reconstruction_filepath"].split('.npy')[0]+'.hdf5','recon','equalized_volume',equalized_tomogram,create_group=True)
    print(f'Time elapsed: {time.time() - start:.2f} s' )

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
    """ Performs a series of processing steps over a 2D array, namely:

        1) Removes a gradient (i.e. the phase ramp) for the image as a whole
        2) Makes any NaN values null
        3) Removes outlier values above/below a certain sigma
        4) Removes the global offset of the array, making the smallest value null
        5) Removes a local offset of the array, subtracting the mean value of a desired region from the entire array

    Args:
        remove_gradient (bool): select wheter to remove phase ramp over the whole image or not
        remove_global_offset (bool): removes global offset from the image, subtracting the minimum from all pixel values
        remove_outlier (int): integer value indicating the number of sigma. Values above/below sigma will be considered outliers and set as zero
        remove_avg_offset (list): coordinates of squared region (top,bottom,left, right). The mean value of such region in the frame will be subtracted from all pixels.
        frame (array): 2D image/frame to be equalized

    Returns:
        frame (array): equalized frame
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

def equalize_frames_parallel(sinogram,dic):
    """ Calls function equalize_frame in parallel at multiple threads for each frameo of the sinogram

    Returns:
        equalized_sinogram: equalized sinogram
    """

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
            minimum, maximum, mean, std = np.min(result), np.max(result), np.mean(result), np.std(result)
            equalized_sinogram[counter,:,:] = result

    minimum1, maximum1, mean1, std1 = np.min(equalized_sinogram), np.max(equalized_sinogram), np.mean(equalized_sinogram), np.std(equalized_sinogram)
    print(f'Min \t Mean-3*sigma \t Mean \t Mean+3*sigma \t Max ')
    print(f'Old {minimum:.2f}, {mean-3*std:.2f}, {mean:.2f}, {mean+3*std:.2f},{maximum:.2f}')
    print(f'New: {minimum1:.2f}, {mean1-3*std1:.2f},{mean1:.2f}, {mean1+3*std1:.2f},{maximum1:.2f}')

    return equalized_sinogram

def equalize_tomogram(equalized_tomogram,mean,std,remove_outliers=0,threshold=0,bkg_window=[]):
    """_summary_

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

def tomo_alignment(dic):
    """ Calls alignment algorithms for aligning the object along the sinogram frames

    Args:
        dic (dict): dictionary of inputs

    Returns:
        dic (dict): updated dictionary of inputs 
    """

    start = time.time()

    angles  = np.load(dic["ordered_angles_filepath"])*np.pi/180
    object = np.load(dic["wiggle_sinogram_selection"]) 

    object = make_bad_frame_null(dic["bad_frames_before_wiggle"],object)

    if dic['project_angles_to_regular_grid']:
        object, _, _, projected_angles = angle_grid_organize(object, angles,percentage=dic["step_percentage"])
        dic['n_of_used_angles']     = projected_angles.shape 
        np.save(dic["projected_angles_filepath"],projected_angles)

    dic['n_of_original_angles'] = angles.shape # save to output log

    tomoP, wiggle_cmas = wiggle(dic, object)
    dic["wiggle_ctr_of_mas"] = wiggle_cmas
    np.save(dic["wiggle_cmas_filepath"],wiggle_cmas)
    np.save(dic["wiggle_sinogram_filepath"],tomoP)

    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

def preview_angle_projection(dic):
    print("Simulating projection of angles to regular grid...")
    angles  = np.load(dic["ordered_angles_filepath"])
    angles = (np.pi/180.) * angles
    total_n_of_angles = angles.shape[0]
    
    _, selected_indices, n_of_padding_frames, projected_angles = angle_grid_organize(np.load(dic["wiggle_sinogram_selection"]), angles,percentage=dic["step_percentage"])
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
    """ Given non-regular steps between rotation angles, this function projects angles to regular grid and pad it to run from 0 to 180 degrees. 
    
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
    """ Null frames of interest, listed in "bad_frames_before_wiggle" dic variable

    Args:
        dic (dict): dictionary of inputs
        sinogram (array): sinogram containing frames to be nulled

    Returns:
        siogram (array): updated sinogram, with nulled frames
    """
    for k in bad_list:
        sinogram[k,:,:] = 0
    return sinogram

def wiggle(dic, sinogram):
    """ Calls sscRadon wiggle algorithm in both direction and return the aligned sinogram as well as the center-of_mass coordinates for alignment after 3D tomographic recon by slices.

    Args:
        dic (dict): dictionary of inputs
        sinogram (array): sinogram containing misaligned frames

    Returns:
        aligned_sinogram: updated sinogram after alignment
        wiggle_cmas: center of mass coordinates to be used for final alignment, after 3D tomographic reconstruction by slices
    """
    temp_tomogram, shift_vertical = sscRadon.radon.get_wiggle( sinogram, "vertical", dic["CPUs"], dic["wiggle_reference_frame"] )
    temp_tomogram, shift_vertical = sscRadon.radon.get_wiggle( temp_tomogram, "vertical", dic["CPUs"], dic["wiggle_reference_frame"] )
    print('\tFinished vertical wiggle. Starting horizontal wiggle...')
    aligned_sinogram, shift_horizontal, wiggle_cmas_temp = sscRadon.radon.get_wiggle( temp_tomogram, "horizontal", dic["CPUs"], dic["wiggle_reference_frame"] )
    wiggle_cmas = [[],[]]
    wiggle_cmas[1], wiggle_cmas[0] =  wiggle_cmas_temp[:,1].tolist(), wiggle_cmas_temp[:,0].tolist()
    return aligned_sinogram, wiggle_cmas

####################### TOMOGRAPHY ###########################################

def tomo_recon(dic, sinogram):
    """ Calls tomographic algorithms from sscRaft

    Args:
        dic (dict): dictionary of inputs

    Returns:
        reconstruction3D (array): 3D reconstructed volume via tomography

    """
    
    start = time.time()
    reconstruction3D = tomography(dic, sinogram)
    np.save(dic["reconstruction_filepath"],reconstruction3D)
    open_or_create_h5_dataset(dic["reconstruction_filepath"].split('.npy')[0]+'.hdf5','recon','volume',reconstruction3D,create_group=True)
    print(f'Time elapsed: Tomography: {time.time() - start} s' )
    return reconstruction3D

def automatic_regularization(sino, L=0.001):
    """ Applies regularization according to: https://doi.org/10.1016/j.rinam.2019.100088

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
    if save:
        wiggle_cmass = np.asarray(wiggle_cmass)
        np.save(path, wiggle_cmass)
        return 0
    else:
        array = np.load(path)
        wiggle_cmas = [array[0,:],array[1,:]]
        return wiggle_cmas

def add_plot_suffix_to_file(path):
    first_part = path.split(".")[0]
    second_part = path.split(".")[-1]
    return first_part + "_PLOT." + second_part

def get_and_save_downsampled_sinogram(sinogram,path,downsampling=4):
    downsampled_sinogram = sinogram[:,::downsampling,::downsampling]
    np.save(add_plot_suffix_to_file(path),downsampled_sinogram)
    return downsampled_sinogram

def tomography(dic, sinogram):
    """
    Args:
        dic (dict): dictionary of inputs
        use_regularly_spaced_angles (bool): boolean to select if angle steps are regular or not

    Returns:
        reconstruction3D (array): tomographic volume
    """

    angles_filepath          = dic["ordered_angles_filepath"]
    
    if dic['using_wiggle']:
        wiggle_cmas_path         = dic["wiggle_cmas_filepath"]
        try:
            wiggle_cmas = dic["wiggle_ctr_of_mas"]
        except:
            wiggle_cmas = np.load(dic["wiggle_cmas_filepath"])

        if wiggle_cmas == [[],[]]: # if no ctr of mass, save file with it
            wiggle_cmas = save_or_load_wiggle_ctr_mass(wiggle_cmas_path,save=False)

    if dic['project_angles_to_regular_grid'] == True:
        angles_filepath = angles_filepath[:-4]+'_projected.npy'
        dic['algorithm_dic']['angles'] = np.load(angles_filepath)*np.pi/180
    else:
        dic['algorithm_dic']['angles'] = np.load(angles_filepath)[:,1]*np.pi/180 

    """ Automatic Regularization """
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
        reconstruction3D = sscRadon.radon.set_wiggle(reconstruction3D, 0, -np.array(wiggle_cmas[1]), -np.array(wiggle_cmas[0]), dic["CPUs"])

    print('\t Tomography done!')

    print('Saving tomography logfile...')
    save_json_logfile_tomo(dic)
    print('\tSaved!')

    return reconstruction3D

####################### EXTRA ###########################################

def clean_pwcdi_dataset(input_path, inverted_frames, bad_frames,output_path):
    sinogram = np.load(input_path)

    for frame in inverted_frames:
        sinogram[frame] = sinogram[frame,::-1,::-1]
    for frame in bad_frames:
        sinogram[frame] = np.zeros_like(sinogram[0])

    np.save(output_path, sinogram)
    print("Saved data at: ",output_path)

def exponential_decay_at_border(x,alpha,cut=8):
    maximum = np.max(np.where(np.abs(x)<cut,0,1 - np.exp(alpha*x)))
    func = np.where(np.abs(x) < cut ,1 - np.exp(alpha*x),0)
    func = np.where(np.abs(x) < cut, 1 + func / np.max(np.abs(func)),0)
    return func

def pad_sinogram_frames(padding,sinogram):
    pad_row, pad_col = padding
    print("\tOld shape: ",sinogram.shape)
    sinogram = np.pad(sinogram,((0,0),(pad_row,pad_row),(pad_col,pad_col)),mode='constant')#,constant_values=((1,),(1,)))
    print("\tNew shape: ",sinogram.shape)
    return sinogram

def gradient_filter_and_pad(loadpath,savepath,background_region,filter_params, padding, preview, n_frame=0):

    print("Loading data from ",loadpath)
    data = np.load(loadpath)
    original_frame = data[n_frame]

    if background_region != ():
        figure, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(data[n_frame])
        print("Removing background from frames based on region ",background_region)
        top, bottom, left, right = background_region
        mask = np.zeros_like(data[0],dtype=bool) # mask indicating where to fit plane
        mask[top:bottom,left:right] = True
        get_best_plane_fit_inside_mask_partial = partial(get_best_plane_fit_inside_mask,mask)
        frames = [data[i] for i in range(data.shape[0])]
        """ Remove gradient from bkg """
        processes = min(os.cpu_count(),32)
        print(f'Using {processes} parallel processes')
        with ProcessPoolExecutor(max_workers=processes) as executor:
            results = list(tqdm(executor.map(get_best_plane_fit_inside_mask_partial,frames),total=data.shape[0]))
            for i, result in enumerate(results):
                data[i] = result - np.min(result)
        
        ax[1].imshow(data[n_frame])
        ax[0].set_title('Original')
        ax[1].set_title("Background removed")
        plt.show()                      
                          
    if filter_params != ():
        print("Filtering borders")
        cutoff, decay, null_size = filter_params
        
        frame = data[0]
        if 1: # exponential decay at border
            N = null_size
            x = np.linspace(-N,N,frame.shape[1])
            y = np.linspace(-N,N,frame.shape[0])
            func_x = np.where(x>=0,exponential_decay_at_border(x,decay,cut=cutoff) , exponential_decay_at_border(x,-decay,cut=cutoff))
            func_y = np.where(y>=0,exponential_decay_at_border(y,decay,cut=cutoff) , exponential_decay_at_border(y,-decay,cut=cutoff))
            meshY, meshX = np.meshgrid(func_x,func_y)
            border_attenuation_matrix = meshY*meshX
        elif 0: #. gaussian filter
            sigma = 2
            x = np.linspace(-5, 5,frame.shape[1])
            y = np.linspace(-5, 5,frame.shape[0])
            x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
            border_attenuation_matrix = gaus2d(x, y,sx=sigma,sy=sigma)

        """ Apply border filter """
        figure, ax = plt.subplots(1,4,figsize=(10,5))
        ax[0].imshow(data[n_frame])
        data[:] = data[:]*border_attenuation_matrix
        ax[3].imshow(data[n_frame])
        ax[2].imshow(border_attenuation_matrix)
        ax[1].plot(border_attenuation_matrix[border_attenuation_matrix.shape[0]//2,:])
        ax[1].set_aspect(100)
        ax[0].set_title('Before')
        ax[3].set_title("After filtering")
        plt.show()
        
    if padding != ():
        figure, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(data[n_frame])
        data = pad_sinogram_frames(padding,data)
        ax[1].imshow(data[n_frame])
        ax[0].set_title('Before')
        ax[1].set_title("After padding")
        plt.show()
        
    if savepath != '': 
        print("Saving data...")
        np.save(savepath,data) 
        print("Saved @ ",savepath)
        
    figure, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(original_frame)
    ax[1].imshow(data[n_frame])
    plt.show()
        
def flip_frames_of_interest(sinogram,frame_list):
    for i in frame_list:
        sinogram[i] = sinogram[i,::-1,::-1]
    return sinogram

def plot_histograms(recon3D, equalized_tomogram,bins=300):

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