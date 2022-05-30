import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from skimage.morphology import square, erosion, opening, convex_hull_image, dilation
from functools import partial

from .misc import list_files_in_folder

from sscRaft import parallel




####################### SORTING ###################################

def angle_mesh_organize( mdata, angles, percentage = 100 ): 
        """ Project angles to regular mesh and pad it to run from 0 to 180

        Args:
            mdata (_type_): _description_
            angles (_type_): _description_
            use_max (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        angles_list = []

        starta = angles[:,1].min()
        enda   = angles[:,1].max()
        rangea = enda - starta
        forw = np.roll(angles[:,1],1,0) - angles[:,1]
        forw[-1] = forw[-2]
        forw[0] = forw[1]
        maxdth = abs(forw).max() 
        mindth = abs(forw).min()

        divider = (percentage*maxdth - (percentage-100)*mindth)/100 # if 100, = max; if 0 = min; intermediary values between 0 and 100 results in values between min and max
        print(f'Chosen regular interval: {divider}')
        
        nangles = int( (np.pi)/divider ) 
        dth = np.pi / (nangles-1)
        ndata = np.zeros([nangles,mdata.shape[1],mdata.shape[2]])
        idx = np.zeros([nangles], dtype=np.int)
        for k in range(nangles):
            angle = -np.pi/2.0 + k*dth
            if angle > enda or angle < starta:
                idx[k] = -1
                ndata[k,:,:] = np.zeros([mdata.shape[1],mdata.shape[2]])
            else:
                idx[k] = int( np.argmin( abs(angle - angles[:,1]) ) )
                ndata[k,:,:] = mdata[idx[k],:]
            angles_list.append(angle*180/np.pi)
        first = np.argmin((idx < 0)) - 1
        angles_array = np.asarray(angles_list) - np.min(angles_list) # convert values to range 0 - 180
        return ndata, idx, first, angles_array 

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

def save_json_logfile(jason,output_folder):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        jason (dic): jason dictionary
    """    
    import json, os
    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    
    name = jason["folders_list"][0]

    name = dt_string + "_" + name+".json"

    filepath = os.path.join(output_folder,name)
    file = open(filepath,"w")
    file.write(json.dumps(jason,indent=3,sort_keys=True))
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


def tomography(input_dict,use_regularly_spaced_angles=True):
    
    # input_dict = {} 
    # global_dict["tomo_algorithm"]
    # contrast_type
    # ordered_angles_filename
    # global_dict["tomo_iterations"]
    # global_dict["tomo_n_of_gpus"]
    # global_dict["tomo_regularization"]
    # global_dict["tomo_regularization_param"]
    # output_folder

    algorithm                = input_dict["tomo_algorithm"]
    data_selection           = input_dict["contrast_type"]
    angles_filepath          = input_dict["ordered_angles_filepath"]
    iterations               = input_dict["tomo_iterations"]
    GPUs                     = input_dict["tomo_n_of_gpus"]
    do_regularization        = input_dict["tomo_regularization"]
    regularization_parameter = input_dict["tomo_regularization_param"]
    output_folder            = input_dict["output_folder"] # output should be the same folder where original sinogram is located

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
        from sscOldRaft import Centersino, MaskedART, FBP, Backprojection, EM, SIRT_FST

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
        nangles = data.shape[1]
        recsize = data.shape[2]
        iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
        dic = {'gpu': GPUs, 'blocksize':20, 'nangles': nangles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
        reconstruction3D = parallel.emfs( data, dic )
    else:
        import sys
        sys.exit('Select a proper reconstruction method')
    print('\t Tomography done!')

    print('Saving tomography logfile...')
    save_json_logfile(input_dict,output_folder)
    print('\tSaved!')

    return reconstruction3D