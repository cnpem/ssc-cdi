import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from skimage.morphology import square, erosion, opening, convex_hull_image, dilation
from functools import partial
import ast

from ..misc import list_files_in_folder
from ..processing.unwrap import RemoveGrad

import sscRaft
from sscRadon import radon

####################### SORTING ###################################

def angle_mesh_organize( original_frames, angles, percentage = 100 ): 
    """ Project angles to regular mesh and pad it to run from 0 to 180

    Args:
        original_frames (_type_): _description_
        angles (_type_): _description_
        
    Returns:
        _type_: _description_
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

 

def angle_mesh_organize_old( original_frames, angles, percentage = 100 ): 
        """ Project angles to regular mesh and pad it to run from 0 to 180

        Args:
            original_frames (_type_): _description_
            angles (_type_): _description_
            use_max (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        angles_list = []

        start_angle = angles[:,1].min()
        end_angle   = angles[:,1].max()
        rangea = end_angle - start_angle
        neighbor_differences = np.roll(angles[:,1],1,0) - angles[:,1]
        neighbor_differences[-1] = neighbor_differences[-2]
        neighbor_differences[0] = neighbor_differences[1]
        maxdtheta = abs(neighbor_differences).max() 
        mindtheta = abs(neighbor_differences).min()

        divider = (percentage*maxdtheta - (percentage-100)*mindtheta)/100 # if 100, = max; if 0 = min; intermediary values between 0 and 100 results in values between min and max
        print(f'Chosen regular interval: {divider}')
        
        n_of_angles = int( (np.pi)/divider ) 
        dtheta = np.pi / (n_of_angles-1)
        projected_frames = np.zeros([n_of_angles,original_frames.shape[1],original_frames.shape[2]])
        idx = np.zeros([n_of_angles], dtype=np.int)
        for k in range(n_of_angles):
            angle = -np.pi/2.0 + k*dtheta
            if angle > end_angle or angle < start_angle:
                idx[k] = -1
                projected_frames[k,:,:] = np.zeros([original_frames.shape[1],original_frames.shape[2]])
            else:
                idx[k] = int( np.argmin( abs(angle - angles[:,1]) ) )
                projected_frames[k,:,:] = original_frames[idx[k],:]
            angles_list.append(angle*180/np.pi)
        first = np.argmin((idx < 0)) - 1
        angles_array = np.asarray(angles_list) - np.min(angles_list) # convert values to range 0 - 180
        return projected_frames, idx, first, angles_array 


def sort_frames_by_angle(ibira_path,foldernames):
    rois = []
    counter = -1 
    for folder in foldernames:

        print(f"Sorting data for {folder} folder")

        filepaths, filenames = list_files_in_folder(os.path.join(ibira_path, folder,'positions'), look_for_extension=".txt")

        print('\t # of files in folder:',len(filenames))
        for filepath in filepaths:
            roisname = filepath  
            if roisname == os.path.join(ibira_path,folder, 'positions', folder + '_Ry_positions.txt'): # ignore this file, to use only the positions file inside /positions/ folder
            # if roisname == os.path.join(ibira_path, folder) + '/Ry_positions.txt': # use for old file standard
                continue
            else:
                counter += 1 
                posfile = open(roisname)
                a = 0
                for line in posfile:
                    line = str(line)
                    if a < 1: # get value from first line of the file only
                        angle = line.split(':')[1].split('\t')[0]
                        rois.append([int(counter),float(angle)])
                        break    
                    a += 1

    
    rois = np.asarray(rois)
    rois = rois[rois[:,1].argsort(axis=0)]
    return rois 

def reorder_slices_low_to_high_angle(object, rois):
    object_temporary = np.zeros_like(object)

    for k in range(object.shape[0]): # reorder slices from lowest to highest angle
            # print(f'New index: {k}. Old index: {int(rois[k,0])}')
            object_temporary[k,:,:] = object[int(rois[k,0]),:,:] 

    return object_temporary

######################### EQUALIZATION #################################################

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
        frame = RemoveGrad(frame,np.ones_like(frame,dtype=bool))

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
    print(f'Min \t Mean-3*sigma \t Mean \t Mean+3*sigma \t Max ')
    print('Old ',minimum, mean-3*std,mean, mean+3*std,maximum)
    
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
            if counter % 100 == 0: print('Populating results matrix...',counter)
            minimum, maximum, mean, std = np.min(result), np.max(result), np.mean(result), np.std(result)
            # print('New ',minimum, mean-3*std,mean, mean+3*std,maximum)
            equalized_sinogram[counter,:,:] = result

    minimum, maximum, mean, std = np.min(equalized_sinogram), np.max(equalized_sinogram), np.mean(equalized_sinogram), np.std(equalized_sinogram)
    # print('New ',minimum, mean-3*std,mean, mean+3*std,maximum)

    return equalized_sinogram

######################### CONVEX HULL #################################################

def operator_T(u):
    d   = 1.0
    uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
    uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
    uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
    uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
    delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
    z = np.sqrt( delta )
    return z

def do_chull(sinogram,invert,tolerance,opening_param,erosion_param,chull_param,frame):
    img = sinogram[frame,:,:] 
    where = operator_T(img).real
    new = np.copy(img)
    if invert:
        new[ new > 0] = operator_T(new).real[ img > 0]
    else:
        new[ new < 0] = operator_T(new).real[ img < 0]

    mask = (np.abs( new - img) < tolerance) * 1.0
    mask2 = opening(mask, square(opening_param))
    mask3 = erosion(mask2, square(erosion_param))
    chull = dilation( convex_hull_image(mask3), square(chull_param) ) # EXPAND CASCA DA MASCARA
    img_masked = np.copy(img * chull)  #nova imagem apenas com o suporte
    # sinogram[frame,:,:] = img_masked
    return new,mask,mask2,mask3,chull,img_masked

def apply_chull_parallel(sinogram,invert=True,tolerance=1e-5,opening_param=10,erosion_param=30,chull_param=50):
    if sinogram.ndim == 2:
        sinogram = np.expand_dims(sinogram, axis=0) # add dummy dimension to get 3d array
    chull_sinogram = np.empty_like(sinogram)
    do_chull_partial = partial(do_chull,sinogram,invert,tolerance,opening_param,erosion_param,chull_param)
    frames = [f for f in range(sinogram.shape[0])]
    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(tqdm(executor.map(do_chull_partial,frames),total=sinogram.shape[0]))
        for counter, result in enumerate(results):
            new,mask,mask2,mask3,chull,img_masked = result
            chull_sinogram[counter,:,:] = img_masked
    return [new,mask,mask2,mask3,chull,img_masked,chull_sinogram]

####################### TOMOGRAPHY ###########################################3

def save_json_logfile(input_dict,output_folder):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        input_dict (dic): input_dict dictionary
    """    
    import json, os
    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    
    name = input_dict["folders_list"][0]

    name = dt_string + "_" + name+".json"

    filepath = os.path.join(output_folder,name)
    file = open(filepath,"w")
    file.write(json.dumps(input_dict,indent=3,sort_keys=True))
    file.close()

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
    data_selection           = input_dict["contrast_type"]
    angles_filepath          = input_dict["ordered_angles_filepath"]
    iterations               = input_dict["tomo_iterations"]
    GPUs                     = input_dict["GPUs"]
    CPUs                     = input_dict["CPUs"]
    do_regularization        = input_dict["tomo_regularization"]
    regularization_parameter = input_dict["tomo_regularization_param"]
    output_folder            = input_dict["output_folder"] # output should be the same folder where original sinogram is located
    wiggle_cmas              = input_dict["wiggle_ctr_of_mas"]
    wiggle_cmas_path         = input_dict["wiggle_ctr_mass_filepath"]

    if wiggle_cmas == [[],[]]:
        wiggle_cmas = save_or_load_wiggle_ctr_mass(wiggle_cmas_path,save=False)

    data = np.load(os.path.join(output_folder,f'{data_selection}_wiggle_sinogram.npy'))

    if use_regularly_spaced_angles == True:
        angles_filepath = angles_filepath[:-4]+'_projected.npy'

    angles = np.load(angles_filepath) # sorted angles?

    """ ######################## Regularization ################################ """

    if 0: # Paola's approach to correcting angles
        # Padded zeros for completion of missing wedge:  from (-70,70) - 140 degrees, to (-90,90) - 180 degrees
        angles = angles[:,1] # get the angles
        anglesmax, anglesmin = angles[-1],  angles[0]     # max and min angles
        angles = np.insert(angles, 0, -90)     # Insert the first angle as -90. Why I do that? Beacause I assume that the first angles is always zero, in order to correctly find the angle step size inside the EM algorithm fro all angles.
        data = np.pad(data,((1,0),(0,0),(0,0)),'constant') # Pad zeros corresponding to the extra -90 value
        angles = (angles + 90) # Transform the angles from (-90,90) to (0,180)

    if do_regularization == True and algorithm == "EEM": # If which_reconstruction == "EEM" MIQUELES
        print('\tBegin Regularization')
        for k in range(data.shape[1]):
            data[:,k,:] = regularization( data[:,k,:], regularization_parameter)

        print('\tRegularization Done')

    """ ######################## RECON ################################ """


    print('Starting tomographic algorithm: ',algorithm)
    if algorithm == "TEM" or algorithm == "EM":
        data = np.exp(-data)
    elif algorithm == "ART":
        flat = np.ones([1,data.shape[-2],data.shape[-2]],dtype=np.uint16)
        dark = np.zeros(flat.shape[1:],dtype=np.uint16)
        centersino1 = Centersino(frame0=data[0,:,:], frame1=data[-1,:,:], flat=flat[0], dark=dark, device=0) 

    if algorithm != "EEM": # for these
        
        rays, slices = data.shape[-1], data.shape[-2]
        reconstruction3D = np.zeros((rays,slices,rays))
        for i in range(slices):
            sinogram = data[:,i,:]
            if algorithm == "ART":
                reconstruction3D[:,i,:]= MaskedART( sino=sinogram,mask=flat,niter=iterations ,device=GPUs)
            elif algorithm == "FBP": 
                reconstruction3D[:,i,:]= FBP( sino=sinogram,angs=angles,device=GPUs,csino=centersino1)
            elif algorithm == "RegBackprojection":
                reconstruction3D[:,i,:]= Backprojection( sino=sinogram,device=GPUs)
            elif algorithm == "EM":
                reconstruction3D[:,i,:]= EM(sinogram, flat, iter=iterations, pad=2, device=GPUs, csino=0)
            elif algorithm == "SIRT":
                reconstruction3D[:,i,:]= SIRT_FST(sinogram, iter=iterations, zpad=2, step=1.0, csino=0, device=GPUs, art_alpha=0.2, reg_mu=0.2, param_alpha=0, supp_reg=0.2, img=None)
    elif algorithm == "EEM": #data é o que sai do wiggle! 
        data = np.swapaxes(data, 0, 1) #tem que trocar eixos 0,1 - por isso o swap.
        n_of_angles = data.shape[1]
        recsize = data.shape[2]
        iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
        dic = {'gpu': GPUs, 'blocksize':20, 'nangles': n_of_angles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
        reconstruction3D = sscRaft.emfs( data, dic )
    else:
        import sys
        sys.exit('Select a proper reconstruction method')
     
    print("\tApplying wiggle center-of-mass correction to 3D recon slices...")
    reconstruction3D = radon.set_wiggle(reconstruction3D, 0, -np.array(wiggle_cmas[1]), -np.array(wiggle_cmas[0]), input_dict["CPUs"])
    print('\t\t Correction done!')

    print('\t Tomography done!')

    print('Saving tomography logfile...')
    save_json_logfile(input_dict,output_folder)
    print('\tSaved!')

    return reconstruction3D



####################### EXTRA ###########################################3
"""
Created on Fri Nov 11 08:03:31 2022

@author: yuri
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os



def plane(variables,u,v,a):
    Xmesh,Ymesh = variables
    return np.ravel(u*Xmesh+v*Ymesh+a)

def RemoveGrad_3( img, mask ):
    xy = np.argwhere( mask > 0)
    n = len(xy)
    y = xy[:,0].reshape([n,1])
    x = xy[:,1].reshape([n,1])
    F = np.array([ img[y[k],x[k]] for k in range(n) ]).reshape([n,1])
    mat = np.zeros([3,3])
    vec = np.zeros([3,1])
    mat[0,0] = (x*x).sum()
    mat[0,1] = (x*y).sum()
    mat[0,2] = (x).sum()
    mat[1,0] = mat[0,1]
    mat[1,1] = (y*y).sum()
    mat[1,2] = (y).sum()
    mat[2,0] = mat[0,2]
    mat[2,1] = mat[1,2]
    mat[2,2] = n
    vec[0,0] = (x*F).sum()
    vec[1,0] = (y*F).sum()
    vec[2,0] = (F).sum()
    eye = np.eye(mat.shape[0])
    eps = 1e-3 # valor tirado do *
    if 1: # com regularização
        abc = np.dot( np.linalg.inv(mat + eps * eye), vec).flatten() 
    else: # sem regularização
        abc = np.dot( np.linalg.inv(mat), vec).flatten()
    a = abc[0]
    b = abc[1]
    c = abc[2]
    new   = np.zeros(img.shape)
    row   = new.shape[0]
    col   = new.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))
    new[y, x] = img[ y, x] - ( a*XX[y,x] + b*YY[y,x] + c )
    #for k in range(n):
    #    new[y[k], x[k]] = img[ y[k], x[k]] - ( a*x[k] + b*y[k] + c )
    return new, a,b,c

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
        grad_removed, a,b,c = RemoveGrad_3(frame,mask2)
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
        
    # from sscCdi.caterete.tomo_processing import gradient_filter_and_pad

    # loadpath = 'data.npy'
    # savepath = 'data2.npy' # if "", data won't be saved
    # background_region = (500,650,850,1100) # Rectangular region of the background. Use () to skip gradient removal for the background
    # filter_params = (40,1,50) # Parameters for the filter: (cutoff, decay, null_size). Use () to skip filtering
    # padding = (10,20) # Number of pixels to add: (rows, columns). Use () to skip padding
    # frame_preview = 0 # select which frame of the sinogram to preview in the plots

    # gradient_filter_and_pad(loadpath,savepath,background_region,filter_params, padding,frame_preview)        
