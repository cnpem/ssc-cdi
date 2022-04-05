import sscCdi
import sscIO
from sscPimega import pi540D

from sys import argv
import os
from time import time
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

from operator import sub

from numpy.fft import fftshift as shift
from numpy.fft import ifftshift as ishift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from scipy import ndimage
from sscptycho_functions import *

# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# MODULES FOR THE RESTAURATION APPLICATION 
# (see main code below)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def split_angles(diffractionpattern, frames):

    difpads = np.asarray(np.array_split(diffractionpattern, int(diffractionpattern.shape[0]/frames), axis = 0))
    print('\tDifpads shape after split angles: ', difpads.shape)

    return difpads


def load_2d_data(jason, filepaths):

    
    fullpath = os.path.join(filepaths)
    h5f,_ = sscIO.io.read_volume(fullpath, 'numpy', use_MPI=True, nprocs=jason["Threads"])

    if 1:  # OPTIONAL: exclude first difpad to match with probe_positions_file list
        h5f = h5f[1:]  # TODO: why does this difference of 1 position happens? Fix it!

    return h5f


def Geometry(L):
    """ Detector geometry parameters for sscPimega restauration

    Args:
        L : sample-detector distance

    Returns:
        geo : geometry 
    """    

    project = pi540D.get_detector_dictionary( L, {'geo':'nonplanar','opt':True,'mode':'virtual'} ) 
    geo = pi540D.geometry540D( project )
    return geo

def Restaurate(img, geom):
    return pi540D.backward540D(img, geom)

def UnRestaurate(img, geom):
    return pi540D._worker_annotation_image(pi540D.forward540D(img, geom))

def _get_center(dbeam, project):
    aimg = pi540D._worker_annotation_image ( np.clip( dbeam, 0, 100) )
    aimg = ndimage.gaussian_filter( aimg, sigma=0.95, order=0 )
    aimg = aimg/aimg.max()
    aimg = 1.0 * ( aimg > 0.98 )    
    u = np.array(range(3072))
    xx,yy = np.meshgrid(u,u)
    xc = ((aimg * xx).sum() / aimg.sum() ).astype(int)
    yc = ((aimg * yy).sum() / aimg.sum() ).astype(int)
    annotation = np.array([ [xc, yc] ])
    tracking = pi540D.annotation_points_standard ( annotation )
    tracking = pi540D.tracking540D_vec_standard ( project, tracking ) 
    xc = int( tracking[0][2] )
    yc = int( tracking[0][3] ) 
    return xc, yc


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def restauration_cat_3d(args,preview  = False,save  = False, read = True):
    
    """_summary_

    Returns:
        _type_: _description_
    """    
    
    jason, ibira_datafolder, scans_string, _ = args

    diffractionpattern = 0 #dummy, necessary because of the first iteration
    count = -1
    time_difpads = 0

    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)

        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        if read:
            if jason['Projections'] != []:
                n = jason['Projections'] #n refers to the amount of angles (projections)
                difpads = 0
                for i in n:
                    if i == 0:
                        difpads = np.load(jason['SaveDifpadPath'] + filenames[0] + '.npy')
                    else:
                        difpad = np.load(jason['SaveDifpadPath'] + filenames[i] + '.npy')
                        difpads = np.concatenate((difpads, difpad), axis = 0)
                frames = difpad.shape[0]

            else:
                n = len(filenames)
                difpads = 0
                for i in range(n):
                    if i == 0:
                        difpads = np.load(jason['SaveDifpadPath'] + filenames[0] + '.npy')
                    else:
                        difpad = np.load(jason['SaveDifpadPath'] + filenames[i] + '.npy')
                        difpads = np.concatenate((difpads, difpad), axis = 0)
                frames = difpad.shape[0]

            
            # difpads = np.asarray(difpads)
            #difpads = np.load(jason['SaveDifpadPath'] + filenames + '.npy')
            print('\tdifpads.shape (after 1st for): ', difpads.shape)

        else: 
            difpads, time_difpads, _, jason, frames, bad_projections = pi540_restauration_cat_block(params,jason['SaveDifpadPath'],preview,save)

        
        if diffractionpattern == 0:
            diffractionpattern = difpads
        else:
            diffractionpattern = np.concatenate(difpads)
        
        print('\tdifpads.shape: ', difpads.shape)
    
    difpads = split_angles(diffractionpattern, frames) #returns 4D structure for difpads as: [angles, frames, rows, columns]

    difpads = masks_application(difpads, jason)

    return diffractionpattern, time_difpads, jason


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def pi540_restauration_cat_block(args, savepath = '', preview = False, save = False):
    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    scans_string        = args[5]
    
    difpads = []
    t0 = time()

    first_iteration = True
    bad_projections = []
    cont = -1

    if jason['Projections'] != []:
        filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)

    for measurement_file, measurement_filepath in zip(filenames, filepaths):
        cont += 1

        param = (jason,ibira_datafolder,measurement_file,acquisitions_folder,scans_string,measurement_filepath)
        difpad = load_2d_data(jason, measurement_filepath)
        #difpad, elapsedtime_one_difpad, jason = pi540_restauration_cat(param,savepath,preview,save, first_iteration)
        
        if first_iteration: 
            print('\ndifpad shape: ', difpad.shape)
            difpads = difpad
            difpads = np.asarray(difpads)
            frames = difpad.shape[0]
            first_iteration = False

        elif difpads.shape[0] != frames:
            bad_difpad = np.zeros((frames, difpads.shape[1], difpads.shape[2]))
            print('\nbad difpad shape: ', bad_difpad.shape)
            difpads = np.concatenate((difpads, bad_difpad), axis=0)
            bad_projections.append(cont)
            
        else:
            difpads = np.concatenate((difpads, difpad), axis=0)
    
    difpads, elapsedtime_one_difpad, jason = pi540_restauration_cat(difpads, param,savepath,preview,save, first_iteration) #calls restauration function
    print('difpads shape before restauration and binning of ', jason['Binning'], ':', difpads.shape)

    # if save:
    #     np.save(savepath + measurement_file, difpad)

    t1 = time()
    elapsedtime = t1-t0

    return difpads, elapsedtime, elapsedtime_one_difpad, jason, frames, bad_projections

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def pi540_restauration_cat(difpads, args, savepath = '', preview = False, save = False, first_iteration = True):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]
    measurement_filepath= args[5]

    if preview and first_iteration:  # preview only 
        difpad_number = 0 # selects which difpad to preview
        raw_difpads = h5py.File(measurement_filepath, 'r')['entry/data/data'][()][:, 0, :, :]
        sscCdi.caterete.misc.plotshow_cmap2(raw_difpads[difpad_number, :, :], title=f'Raw Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/03_difpad_raw.png')
        print('Raw diffraction pattern shape: ', raw_difpads.shape)

    t0 = time()

    print('Begin Restauration')
            
    if jason['OldRestauration'] == True: # OldRestauration is Giovanni's
        print('\nMeasurement file in pi540_restauration_cat: ', measurement_file)
        difpads, geometry, _, jason = get_restaurated_difpads(difpads, jason)

    else:
        print('Entering Miqueles Restauration.')
        dic = {}
        dic['susp'] = jason["ChipBorderRemoval"]  # parameter to ignore borders of the detector chip
        dic['roi'] = jason["DetectorROI"]  # radius of the diffraction pattern wrt to center. Changes according to the binning value!
        dic['binning'] = jason['Binning']
        dic['distance'] = jason['DetDistance'] * 1e+3
        dic['nproc'] = jason["Threads"]
        dic['data'] = ibira_datafolder + measurement_file
        dic['empty'] = jason['EmptyFrame']
        dic['flat'] = jason['FlatField']
        dic['order'] = 'only' #TODO: ask Miqueles what this 'order' is about! 
        dic['function'] = sscCdi.caterete.restauration.cat_preproc_ptycho_measurement

        difpads, elapsed_time, geometry = sscCdi.caterete.restauration.cat_preproc_ptycho_projections(dic)

        jason['RestauredPixelSize'] = geometry['pxlsize']*1e-6

        print('Diffraction pattern shape (post restauration):', difpads.shape)

    if preview: # save plots of restaured difpad and mean of all restaured difpads
        difpad_number = 0
        sscCdi.caterete.misc.plotshow_cmap2(difpads[difpad_number, :, :], title=f'Restaured Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/04_difpad_restaured.png')
        sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads, axis=0), title=f'Mean Restaured Diffraction Pattern #{difpad_number}', savepath=jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.png')

    print('Finished Restauration')

    t1 = time()
    elapsedtime = t1-t0

    if save:
        np.save(savepath + measurement_file, difpads)

    return difpads, elapsedtime, jason


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_restaurated_difpads(h5f, jason):
    print('\th5f shape in get_restaurated_difpads: ', h5f.shape)
    """Extracts the data from json and manipulate it according G restauration input format
        Then, call G restauration

    Args:
        jason (json file): json object
        path (list of dtrings): list of complete paths to all files
        name (list of strings): list of all file names

    Returns:
        3D array: restaured difpads
    """    

    z1 = float(jason["DetDistance"]) * 1000  # Here comes the distance Geometry(Z1):
    geometry = Geometry(z1)

    empty = np.asarray(h5py.File(jason['EmptyFrame'], 'r')['/entry/data/data']).squeeze().astype(np.float32)
    
    if 'OldFormat' not in jason:
        flat = h5py.File(jason["FlatField"], 'r')['entry/data/data'][()][0, :, :]
    else:
        flat = np.load(jason["FlatField"])

    flat = np.array(flat)
    flat[np.isnan(flat)] = -1
    flat[flat == 0] = 1

    if "OldFormat" in jason:
        if jason['Mask'] != 0:
            mask = np.load(jason['Mask'])
        else:
            mask = np.zeros_like(h5f[0])
    else:
        mask = h5py.File(jason["Mask"], 'r')['entry/data/data'][()][0, 0, :, :]
    # mask = np.flip(mask,0)

    if jason['DifpadCenter'] == []:
        proj  = pi540D.get_detector_dictionary(jason['DetDistance'], {'geo':'nonplanar','opt':True,'mode':'virtual'})
        centerx, centery = _get_center(h5f[0,:,:], proj)
        jason['DifpadCenter'] = (centerx, centery)
        cx, cy = get_difpad_center(h5f[0,:,:]) #TODO: under test! 
        print('Yuri Automatic Difpad Center :', cx, cy)
        print('sscPimega Automatic Difpad Center:',centerx, centery)
    else:
        centerx, centery = jason['DifpadCenter']
        print('Manual Difpad Center :',centerx, centery)

    hsize = jason['DetectorROI']   

    Binning = int(jason['Binning'])

    r_params = (Binning, empty, flat, centerx, centery, hsize, geometry, mask)

    if 1: # under test -> preview_full_difpad
        print('Restaurating single difpad to save preview difpad of 3072^2 shape')
        difpad_number = 0
        img = Restaurate(h5f[difpad_number,:,:].astype(np.float32), geometry) # restaurate
        np.save(jason[ 'PreviewFolder'] + '/03_difpad_raw_flipped_3072.npy',img)
        sscCdi.caterete.misc.plotshow_cmap2(img, title=f'Restaured Diffraction Pattern #{difpad_number}, pre-binning', savepath=jason['PreviewFolder'] + '/03_difpad_raw_flipped_3072.png')


    t0 = time()
    output, _ = pi540D.backward540D_nonplanar_batch(h5f, z1, jason['Threads'], [ hsize//2 , hsize//2 ], restauration_processing_binning,  r_params, 'only')
    t1 = time()

    elapsedtime = t1-t0

    return output, geometry, elapsedtime, jason

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def restauration_processing_binning(img, args):
    """Restaurate and process the binning on the diffraction patterns

    Args:
        img (array): image to be restaured and binned
    """    

    Binning, empty, flat, cx, cy, hsize, geometry, mask = args

    binning = Binning + 0
    img[empty > 1] = -1 # Apply empty 
    img = img * np.squeeze(flat) # Apply flatfield

    if 1: # if mask after restauration with 3072x3072 size
        img[mask ==1] = -1 # Apply Mask

    img = img.astype(np.float32) # convert to float
    img = Restaurate(img, geometry) # restaurate

    if 0: # if mask after restauration with 640x640 size
        img[mask ==1] = -1 # Apply Mask

    img[img < 0] = -1 # all invalid values must be -1 by convention

    # select ROI from the center (cx,cy)
    img = img[cy - hsize:cy + hsize, cx - hsize:cx + hsize] 

    # Binning
    while binning % 2 == 0 and binning > 0:
        avg = img + np.roll(img, -1, -1) + np.roll(img, -1, -2) + np.roll(np.roll(img, -1, -1), -1, -2)  # sum 4 neigboors at the top-left value

        div = 1 * (img >= 0) + np.roll(1 * (img >= 0), -1, -1) + np.roll(1 * (img >= 0), -1, -2) + np.roll( np.roll(1 * (img >= 0), -1, -1), -1, -2)  # Boolean array! Results in the n of valid points in the 2x2 neighborhood

        avg = avg + 4 - div  # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

        avgmask = (img < 0) & ( div > 0)  # div > 0 means at least 1 neighbor is valid. img < 0 means top-left values is invalid.

        img[avgmask] = avg[avgmask] / div[ avgmask]  # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

        img = img[:, 0::2] + img[:, 1::2]  # Binning columns
        img = img[0::2] + img[1::2]  # Binning lines

        img[img < 0] = -1

        img[div[0::2, 0::2] < 3] = -1  # why div < 3 ? Every neighborhood that had 1 or 2 invalid points is considered invalid?

        binning = binning // 2

    while binning % 3 == 0 and binning > 0:
        avg = np.roll(img, 1, -1) + np.roll(img, -1, -1) + np.roll(img, -1, -2) + np.roll(img, 1, -2) + np.roll( np.roll(img, 1, -2), 1, -1) + np.roll(np.roll(img, 1, -2), -1, -1) + np.roll(np.roll(img, -1, -2), 1, -1) + np.roll( np.roll(img, -1, -2), -1, -1)
        div = np.roll(img > 0, 1, -1) + np.roll(img > 0, -1, -1) + np.roll(img > 0, -1, -2) + np.roll(img > 0, 1, -2) + np.roll( np.roll(img > 0, 1, -2), 1, -1) + np.roll(np.roll(img > 0, 1, -2), -1, -1) + np.roll( np.roll(img > 0, -1, -2), 1, -1) + np.roll(np.roll(img > 0, -1, -2), -1, -1)

        avgmask = (img < 0) & (div > 0) / div[avgmask]

        img = img[:, 0::3] + img[:, 1::3] + img[:, 2::3]
        img = img[0::3] + img[1::3] + img[2::3]

        img[img < 0] = -1
        binning = binning // 3

    if binning > 1:
        print('Entering binning > 1 only')
        avg = -img * 1.0 + binning ** 2 - 1
        div = img * 0
        for j in range(0, binning):
            for i in range(0, binning):
                avg += np.roll(np.roll(img, j, -2), i, -1)
                div += np.roll(np.roll(img > 0, j, -2), i, -1)

        avgmask = (img < 0) & (div > 0)
        img[avgmask] = avg[avgmask] / div[avgmask]

        imgold = img + 0
        img = img[0::binning, 0::binning] * 0
        for j in range(0, binning):
            for i in range(0, binning):
                img += imgold[j::binning, i::binning]

        img[img < 0] = -1

    t1 = time()

    return img

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def restauration_cat_2d(args,preview = False,save = False,read = True):

    jason, ibira_datafolder, scans_string, _ = args
    time_difpads = 0

    filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, jason['Acquisition_Folders'][0],scans_string), look_for_extension=".hdf5")
    
    if jason['Projections'] != []:
        filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)

    params = (jason, ibira_datafolder, filenames[0], jason['Acquisition_Folders'][0], scans_string, filepaths[0])
    
    if read:
        difpads = np.load(jason['SaveDifpadPath'] + filenames[0] + '.npy')
    else:  
        difpad = load_2d_data(jason, filepaths[0]) 
        difpads, time_difpads, jason = pi540_restauration_cat(difpad, params,jason['SaveDifpadPath'],preview,save)
    
    difpads = np.expand_dims(difpads,axis=0)
    difpads = masks_application(difpads, jason)

    return difpads, time_difpads, jason

