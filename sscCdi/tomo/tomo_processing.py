import os, sys, time, ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imsave
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import sscRaft, sscRadon

from ..misc import save_json_logfile, create_directory_if_doesnt_exist, save_json_logfile_tomo
from ..processing.unwrap import remove_phase_gradient, unwrap_in_parallel, unwrap_in_sequence

def define_paths(dic):

    dic["output_folder"] = dic["sinogram_path"].rsplit('/',1)[0]
    dic["filename"]    = os.path.join(dic["contrast_type"]+'_'+dic["sinogram_path"].rsplit('/',1)[1].split('.')[0])
    dic["temp_folder"] = os.path.join(dic["output_folder"],'temp')
    dic["ordered_angles_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles.npy')
    dic["projected_angles_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_angles_projected.npy')
    dic["ordered_object_filepath"]     = os.path.join(dic["temp_folder"],f'{dic["filename"]}_ordered_object.npy')
    dic["cropped_sinogram_filepath"]   = os.path.join(dic["temp_folder"],f'{dic["filename"]}_cropped_sinogram.npy')
    dic["equalized_sinogram_filepath"] = os.path.join(dic["temp_folder"],f'{dic["filename"]}_equalized_sinogram.npy')
    dic["unwrapped_sinogram_filepath"] = os.path.join(dic["temp_folder"],f'{dic["filename"]}_unwrapped_sinogram.npy')
    dic["wiggle_sinogram_filepath"]    = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_sinogram.npy')
    dic["wiggle_cmas_filepath"]        = os.path.join(dic["temp_folder"],f'{dic["filename"]}_wiggle_ctr_mass.npy')
    dic["reconstruction_filepath"]     = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo.npy')
    dic["eq_reconstruction_filepath"]  = os.path.join(dic["output_folder"],f'{dic["filename"]}_tomo_equalized.npy')

    create_directory_if_doesnt_exist(dic["output_folder"])
    create_directory_if_doesnt_exist(dic["temp_folder"])

    return dic

####################### SORTING ###################################

def tomo_sort(dic, object, angles):
    start = time.time()
    sorted_angles = sort_angles(angles) # input colums with frame number and angle in rad
    object = reorder_slices_low_to_high_angle(object, sorted_angles)
    np.save(dic["ordered_angles_filepath"],angles)
    np.save(dic["ordered_object_filepath"], object) 
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

def sort_angles(angles):
    rois = np.asarray(angles)
    rois = rois[rois[:,1].argsort(axis=0)]
    return rois 

def reorder_slices_low_to_high_angle(object, rois):
    object_temporary = np.zeros_like(object)

    for k in range(object.shape[0]): # reorder slices from lowest to highest angle
            # print(f'New index: {k}. Old index: {int(rois[k,0])}')
            object_temporary[k,:,:] = object[int(rois[k,0]),:,:] 

    return object_temporary

######################### CROP #################################################

def tomo_crop(dic):
    start = time.time()
    object = np.load(dic["ordered_object_filepath"])
    object = object[:,dic["top_crop"]:-dic["bottom_crop"],dic["left_crop"]:-dic["right_crop"]] # Crop frame
    print(f"Cropped sinogram shape: {object.shape}")
    np.save(dic["cropped_sinogram_filepath"],object) # save shaken and padded sorted sinogram
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

######################### UNWRAP #################################################

def tomo_unwrap(dic):
    start = time.time()
    object = np.load(dic["cropped_sinogram_filepath"])  
    object = unwrap_in_parallel(object)
    np.save(dic["unwrapped_sinogram_filepath"],object)  
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

######################### EQUALIZATION #################################################

def tomo_equalize(dic):
    start = time.time()
    unwrapped_sinogram = np.load(dic["unwrapped_sinogram_filepath"] )
    equalized_sinogram = equalize_frames_parallel(unwrapped_sinogram,dic["equalize_invert"],dic["equalize_gradient"],dic["equalize_outliers"],dic["equalize_global_offset"], dic["equalize_local_offset"])
    np.save(dic["equalized_sinogram_filepath"] ,equalized_sinogram)
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

def tomo_equalize3D(dic):
    start = time.time()
    reconstruction = np.load(dic["reconstruction_filepath"])
    equalized_tomogram = equalize_tomogram(reconstruction,np.mean(reconstruction),np.std(reconstruction),remove_outliers=dic["tomo_remove_outliers"],threshold=float(dic["tomo_threshold"]),bkg_window=dic["tomo_local_offset"])
    np.save(dic["eq_reconstruction_filepath"],equalized_tomogram)
    imsave(dic["eq_reconstruction_filepath"][:-4] + '.tif',equalized_tomogram)
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

def remove_outliers(data,sigma):
    minimum, maximum, mean, std = np.min(data), np.max(data), np.mean(data), np.std(data)
    # print(f'Min \t Mean-{sigma}*sigma \t Mean \t Mean+{sigma}*sigma \t Max ')
    # print('Old',minimum, mean-sigma*std,mean, mean+sigma*std,maximum)
    data = np.where(data > mean + sigma*std,0,data)
    data = np.where(data < mean - sigma*std,0,data)
    minimum, maximum, mean, std = np.min(data), np.max(data), np.mean(data), np.std(data)
    # print('New',minimum, mean-sigma*std,mean, mean+sigma*std,maximum)
    return data

def equalize_frame(remove_gradient, remove_outlier, remove_global_offset, remove_avg_offset,frame):

    # Remove Gradient
    for i in range(0,remove_gradient):
        frame = remove_phase_gradient(frame,np.ones_like(frame,dtype=bool))

    # Check for NaNs
    whereNaN = np.isnan(frame)
    if whereNaN.any():
        print("NaN values found in frame after removing gradient. Removing them!")
        frame = np.where(whereNaN,0,frame)

    # Remove outliers
    for i in range(0,remove_outlier):
        frame = remove_outliers(frame,3)

    # Remove global offsset
    if remove_global_offset:
        frame -= np.min(frame)

    # Remove average offset from specific region
    if remove_avg_offset != []:
        frame -= np.mean(frame[remove_avg_offset[0]:remove_avg_offset[1],remove_avg_offset[2]:remove_avg_offset[3]])
        frame = np.where(frame<0,0,frame)

    return frame

def equalize_frames_parallel(sinogram,invert=False,remove_gradient=0, remove_outlier=0, remove_global_offset=0, remove_avg_offset=[0,slice(0,None),slice(0,None)]):

    minimum, maximum, mean, std = np.min(sinogram), np.max(sinogram), np.mean(sinogram), np.std(sinogram)

    
    # Invert sinogram
    if invert == True:
        sinogram = -sinogram

    # Remove NaNs
    whereNaN = np.isnan(sinogram)
    if whereNaN.any():
        print("NaN values found in unwrapped sinogram. Removing them!")
        sinogram = np.where(whereNaN,0,sinogram)

    # Call parallel equalization
    equalize_frame_partial = partial(equalize_frame, remove_gradient, remove_outlier, remove_global_offset, remove_avg_offset)
    print('Sinogram shape to unwrap: ', sinogram.shape)

    n_frames = sinogram.shape[0]

    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')


    with ProcessPoolExecutor(max_workers=processes) as executor:
        equalized_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(equalize_frame_partial,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            # if counter % 100 == 0: print('Populating results matrix...',counter)
            minimum, maximum, mean, std = np.min(result), np.max(result), np.mean(result), np.std(result)
            # print('New ',minimum, mean-3*std,mean, mean+3*std,maximum)
            equalized_sinogram[counter,:,:] = result

    minimum1, maximum1, mean1, std1 = np.min(equalized_sinogram), np.max(equalized_sinogram), np.mean(equalized_sinogram), np.std(equalized_sinogram)
    print(f'Min \t Mean-3*sigma \t Mean \t Mean+3*sigma \t Max ')
    print('Old ',minimum, mean-3*std,mean, mean+3*std,maximum)
    print('New ',minimum1, mean1-3*std1,mean1, mean1+3*std1,maximum1)

    return equalized_sinogram

def equalize_tomogram(recon,mean,std,remove_outliers=0,threshold=0,bkg_window=[],axis_direction=1,mask_slice=[]):
    
    if type(bkg_window) == type("a_string"):
        bkg_window = ast.literal_eval(bkg_window) # read string as list
        mask_slice = ast.literal_eval(mask_slice) # read string as list
    
    equalized_tomogram = recon

    if threshold != 0:
        equalized_tomogram = np.where( np.abs(equalized_tomogram) > threshold,0,equalized_tomogram)

    if remove_outliers != 0:
        for i in range(remove_outliers):
            equalized_tomogram = np.where( equalized_tomogram > mean+3*std,0,equalized_tomogram)
            equalized_tomogram = np.where( equalized_tomogram < mean-3*std,0,equalized_tomogram)

    if mask_slice != []:
        mask_matrix = np.zeros_like(recon)
        if axis_direction == 0:
            mask_matrix[:,mask_slice[0]:mask_slice[1],mask_slice[2]:mask_slice[3]] = 1
        elif axis_direction == 1:
            mask_matrix[mask_slice[0]:mask_slice[1],:,mask_slice[2]:mask_slice[3]] = 1
        elif axis_direction == 2:
            mask_matrix[mask_slice[0]:mask_slice[1],mask_slice[2]:mask_slice[3],:] = 1
        recon = recon*mask_matrix

    if bkg_window !=[]:

        if axis_direction == 0:
            window = recon[:,bkg_window[0]:bkg_window[1],bkg_window[2]:bkg_window[3]]
        elif axis_direction == 1:
            window = recon[bkg_window[0]:bkg_window[1],:,bkg_window[2]:bkg_window[3]]
        elif axis_direction == 2:
            window = recon[bkg_window[0]:bkg_window[1],bkg_window[2]:bkg_window[3],:]

        offset = np.mean(window)
        equalized_tomogram = equalized_tomogram - offset
        equalized_tomogram = np.where(equalized_tomogram<0,0,equalized_tomogram)

    return equalized_tomogram

####################### ALIGNMENT ###########################################

def tomo_alignment(dic):
    start = time.time()

    angles  = np.load(dic["ordered_angles_filepath"])
    object = np.load(dic["wiggle_sinogram_selection"]) 

    object = make_bad_frame_null(dic,object)
    object, _, _, projected_angles = angle_mesh_organize(object, angles,percentage=dic["step_percentage"])
    print(object.shape,projected_angles.shape)
    tomoP, _, _, wiggle_cmas = wiggle(dic, object)

    dic['n_of_original_angles'] = angles.shape # save to output log
    dic['n_of_used_angles']     = projected_angles.shape 
    dic["wiggle_ctr_of_mas"] = wiggle_cmas

    np.save(dic["wiggle_cmas_filepath"],wiggle_cmas)
    np.save(dic["projected_angles_filepath"],projected_angles)
    np.save(dic["wiggle_sinogram_filepath"],tomoP)
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return dic

def preview_angle_projection(dic):
    print("Simulating projection of angles to regular grid...")
    angles  = np.load(dic["ordered_angles_filepath"])
    angles = (np.pi/180.) * angles
    total_n_of_angles = angles.shape[0]
    
    _, selected_indices, n_of_padding_frames, projected_angles = angle_mesh_organize(np.load(dic["wiggle_sinogram_selection"]), angles,percentage=dic["step_percentage"])
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


def angle_mesh_organize( original_frames, angles, percentage = 100 ): 
    """ Project angles to regular mesh and pad it to run from 0 to 180
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
    idx = np.zeros([n_of_angles], dtype=int)
    for k in range(n_of_angles):

        angle = -np.pi/2.0 + k*dtheta # start at -pi/2 and move in regular steps of dTheta
        angles_list.append(angle*180/np.pi)
        if angle > end_angle or angle < start_angle: # if current is before initial or final acquired angle, use a zeroed frame
            padding_frames_counter += 1
            idx[k] = -1
            projected_frames[k,:,:] = np.zeros([original_frames.shape[1],original_frames.shape[2]])
        else:
            difference_array = abs(angle - angles[:,1]) 
            arg_min_dif = np.argmin( difference_array )
            min_diff = difference_array[arg_min_dif]
            idx[k] = int( arg_min_dif)

            if idx[k] == previous_idx: # evaluate if previous and last frames will be the same
                if previous_min_dif > min_diff: # if angle difference is smaller now than before, zero the previous frame and declare the current to be the projected one
                    projected_frames[k-1,:,:] = np.zeros([original_frames.shape[1],original_frames.shape[2]])
                    idx[k-1] = -2
                    projected_frames[k,:,:] = original_frames[idx[k],:]
                else: 
                    idx[k] = -3
                    continue 
            else:
                projected_frames[k,:,:] = original_frames[idx[k],:]

            previous_idx = idx[k]
            previous_min_dif = min_diff
        
    angles_array = np.asarray(angles_list) - np.min(angles_list) # convert values to range 0 - 180
    
    return projected_frames, idx, padding_frames_counter, angles_array 

def make_bad_frame_null(dic, object):
    for k in dic["bad_frames_before_wiggle"]:
        object[k,:,:] = 0
    return object

def wiggle(dic, object):
    temp_tomogram, shiftv = sscRadon.radon.get_wiggle( object, "vertical", dic["CPUs"], dic["wiggle_reference_frame"] )
    temp_tomogram, shiftv = sscRadon.radon.get_wiggle( temp_tomogram, "vertical", dic["CPUs"], dic["wiggle_reference_frame"] )
    print('\tFinished vertical wiggle. Starting horizontal wiggle...')
    tomoP, shifth, wiggle_cmas_temp = sscRadon.radon.get_wiggle( temp_tomogram, "horizontal", dic["CPUs"], dic["wiggle_reference_frame"] )
    wiggle_cmas = [[],[]]
    wiggle_cmas[1], wiggle_cmas[0] =  wiggle_cmas_temp[:,1].tolist(), wiggle_cmas_temp[:,0].tolist()
    return tomoP, shifth, shiftv, wiggle_cmas

####################### TOMOGRAPHY ###########################################

def tomo_recon(dic):
    start = time.time()
    reconstruction3D = tomography(dic,use_regularly_spaced_angles=True)
    np.save(dic["reconstruction_filepath"],reconstruction3D)
    imsave(dic["reconstruction_filepath"][:-4] + '.tif',reconstruction3D)
    print(f'Time elapsed: Tomography: {time.time() - start} s' )
    return dic

def regularization(sino, L):
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

def tomography(input_dict,use_regularly_spaced_angles=True):
    
    algorithm                = input_dict["tomo_algorithm"]
    angles_filepath          = input_dict["ordered_angles_filepath"]
    iterations               = input_dict["tomo_iterations"]
    GPUs                     = input_dict["GPUs"]
    do_regularization        = input_dict["tomo_regularization"]
    regularization_parameter = input_dict["tomo_regularization_param"]
    wiggle_cmas              = input_dict["wiggle_ctr_of_mas"]
    wiggle_cmas_path         = input_dict["wiggle_cmas_filepath"]

    if wiggle_cmas == [[],[]]:
        wiggle_cmas = save_or_load_wiggle_ctr_mass(wiggle_cmas_path,save=False)

    data = np.load(input_dict["wiggle_sinogram_filepath"])

    if use_regularly_spaced_angles == True:
        angles_filepath = angles_filepath[:-4]+'_projected.npy'

    angles = np.load(angles_filepath) # sorted angles?

    """ ######################## Regularization ################################ """
    if do_regularization == True and algorithm == "EEM": 
        print('\tBegin Regularization')
        for k in range(data.shape[1]):
            data[:,k,:] = regularization( data[:,k,:], regularization_parameter)

        print('\tRegularization Done')

    """ ######################## RECON ################################ """
    print('Starting tomographic algorithm: ',algorithm)
    if algorithm == "FBP": 
        reconstruction3D[:,i,:]= FBP( sino=sinogram,angs=angles,device=GPUs,csino=centersino1)
    elif algorithm == "EEM": #data é o que sai do wiggle! 
        data = np.swapaxes(data, 0, 1) #tem que trocar eixos 0,1 - por isso o swap.
        n_of_angles = data.shape[1]
        recsize = data.shape[2]
        iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
        dic = {'gpu': GPUs, 'blocksize':20, 'nangles': n_of_angles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
        reconstruction3D = sscRaft.emfs( data, dic )
    else:
        sys.exit('Select a proper reconstruction method')
     
    print("\tApplying wiggle center-of-mass correction to 3D recon slices...")
    reconstruction3D = sscRadon.radon.set_wiggle(reconstruction3D, 0, -np.array(wiggle_cmas[1]), -np.array(wiggle_cmas[0]), input_dict["CPUs"])
    print('\t\t Correction done!')

    print('\t Tomography done!')

    print('Saving tomography logfile...')
    save_json_logfile_tomo(input_dict)
    print('\tSaved!')

    return reconstruction3D

####################### EXTRA ###########################################

def plane(variables,u,v,a):
    Xmesh,Ymesh = variables
    return np.ravel(u*Xmesh+v*Ymesh+a)

def func(x,alpha,cut=8):
    maximum = np.max(np.where(np.abs(x)<cut,0,1 - np.exp(alpha*x)))
    # func += np.where(np.abs(x) < cut,maximum,1 - np.exp(alpha*x))
    func = np.where(np.abs(x) < cut ,1 - np.exp(alpha*x),0)
    func = np.where(np.abs(x) < cut, 1 + func / np.max(np.abs(func)),0)
    return func

def get_best_plane_fit_inside_mask(mask2,frame ):
    new   = np.zeros(frame.shape)
    row   = new.shape[0]
    col   = new.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))

    a = b = c = 1e9
    counter = 0
    while np.abs(a) > 1e-8 or np.abs(b) > 1e-8 or counter > 5:
        grad_removed, a,b,c = remove_phase_gradient(frame,mask2)
        plane_fit = plane((XX,YY),a,b,c).reshape(XX.shape)
        frame = frame - plane_fit
        counter += 1
    return frame

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
            func_x = np.where(x>=0,func(x,decay,cut=cutoff) , func(x,-decay,cut=cutoff))
            func_y = np.where(y>=0,func(y,decay,cut=cutoff) , func(y,-decay,cut=cutoff))
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
        
    # loadpath = 'data.npy'
    # savepath = 'data2.npy' # if "", data won't be saved
    # background_region = (500,650,850,1100) # Rectangular region of the background. Use () to skip gradient removal for the background
    # filter_params = (40,1,50) # Parameters for the filter: (cutoff, decay, null_size). Use () to skip filtering
    # padding = (10,20) # Number of pixels to add: (rows, columns). Use () to skip padding
    # frame_preview = 0 # select which frame of the sinogram to preview in the plots

    # gradient_filter_and_pad(loadpath,savepath,background_region,filter_params, padding,frame_preview)        

def flip_frames_of_interest(sinogram,frame_list):
    for i in frame_list:
        sinogram[i] = sinogram[i,::-1,::-1]
    return sinogram