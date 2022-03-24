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

def pi540_restauration_cat_block(args, savepath = '', preview = False, save = False):
    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    scans_string        = args[5]
    
    difpads = []
    t0 = time()

    for measurement_file, measurement_filepath in zip(filenames, filepaths):

        if preview:  # preview only 
            difpad_number = 0 # selects which difpad to preview
            raw_difpads = h5py.File(measurement_filepath, 'r')['entry/data/data'][()][:, 0, :, :]
            sscCdi.caterete.misc.plotshow_cmap2(raw_difpads[difpad_number, :, :], title=f'Raw Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/03_difpad_raw.png')
            print('Raw difpad shape: ', raw_difpads.shape)

        param = (jason,ibira_datafolder,measurement_file,acquisitions_folder,scans_string)
        difpad, elapsedtime_one_difpad = pi540_restauration_cat(param,savepath,preview,save)
        
        if difpads == [] or difpads[0].shape == difpad.shape:
            difpads.append(difpad)
        else:
            difpads.append(np.zeros(difpads[0].shape))
    
    difpads = np.asarray(difpads)
    print('difpads shape after restauration and binning of', jason['Binning'], ':', difpads.shape)
    
    # if save:
    #     np.save(savepath + measurement_file, difpad)

    t1 = time()
    elapsedtime = t1-t0

    return difpads, elapsedtime, elapsedtime_one_difpad

def pi540_restauration_cat(args, savepath = '', preview = False, save = False):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]

    t0 = time()
    print('Begin Restauration')
            
    if jason['OldRestauration'] == True: # OldRestauration is Giovanni's
        print(ibira_datafolder, measurement_file, acquisitions_folder)
        difpads, geometry, _ = get_restaurated_difpads_old_format(jason, os.path.join(ibira_datafolder, acquisitions_folder,scans_string), measurement_file)

        if 1:  # OPTIONAL: exclude first difpad to match with probe_positions_file list
            difpads = difpads[1:]  # TODO: why does this difference of 1 position happens? Fix it!

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
        sscCdi.caterete.misc.save_json_logfile(jason["LogfilePath"], jason) # save json again for new pixel size value

        print('Difraction pattern shape (post restauration):', difpads.shape)

        if preview: # save plots of restaured difpad and mean of all restaured difpads
            difpad_number = 0
            sscCdi.caterete.misc.plotshow_cmap2(difpads[difpad_number, :, :], title=f'Restaured Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/04_difpad_restaured.png')
            sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads, axis=0),  title=f'Mean Restaured Diffraction Pattern #{difpad_number}', savepath=jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.png')

        print('Finished Restauration')

    t1 = time()
    elapsedtime = t1-t0

    if save:
        np.save(savepath + measurement_file, difpads)

    return difpads, elapsedtime


def get_restaurated_difpads_old_format(jason, path, name):
    """Extracts the data from json and manipulate it according G restauration input format
        Then, call G restauration

    Args:
        jason (json file): json object
        path (list of dtrings): list of complete paths to all files
        name (list of strings): list of all file names

    Returns:
        3D array: restaured difpads
    """    

    fullpath = os.path.join(path, name)
    h5f,_ = sscIO.io.read_volume(fullpath, 'numpy', use_MPI=True, nprocs=jason["Threads"])

    z1 = float(jason["DetDistance"]) * 1000  # Here comes the distance Geometry(Z1):
    geometry = Geometry(z1)

    empty = np.asarray(h5py.File(jason['EmptyFrame'], 'r')['/entry/data/data']).squeeze().astype(np.float32)
    flat = h5py.File(jason["FlatField"], 'r')['entry/data/data'][()][0, 0, :, :]

    flat = np.array(flat)
    flat[np.isnan(flat)] = -1
    flat[flat == 0] = 1

    mask = np.load(jason['Mask'])
    # mask = h5py.File(jason["Mask"], 'r')['entry/data/data'][()][0, 0, :, :]
    mask = np.flip(mask,0)

    if jason['DifpadCenter'] == []:
        proj  = pi540D.get_detector_dictionary(jason['DetDistance'], {'geo':'nonplanar','opt':True,'mode':'virtual'})
        centerx, centery = _get_center(h5f[0,:,:], proj)
        jason['DifpadCenter'] = [centerx, centery]
        cx, cy = get_difpad_center(h5f[0,:,:])
        print('Yuri Automatic Difpad Center :', cx, cy)
        print('sscPimega Automatic Difpad Center:',centerx, centery)
    else:
        centerx, centery = jason['DifpadCenter']
        print('Manual Difpad Center :',centerx, centery)

    hsize = jason['DetectorROI']  # (2560/2) 

    Binning = int(jason['Binning'])

    r_params = (Binning, empty, flat, centerx, centery, hsize, geometry, mask)

    t0 = time()
    output, _ = pi540D.backward540D_nonplanar_batch(h5f, z1, jason['Threads'], [ hsize//2 , hsize//2 ], restauration_processing_binning,  r_params, 'only')
    t1 = time()

    elapsedtime = t1-t0

    return output, geometry, elapsedtime

def restauration_processing_binning(img, args):
    """Restaurate and process the binning on the diffraction patterns

    Args:
        img (array): image to be restaured and binned
    """    
    t0 = time()

    Binning, empty, flat, cx, cy, hsize, geometry, mask = args

    binning = Binning + 0
    img[empty > 1] = -1 # Apply empty 
    img = img * flat # Apply flatfield

    img = img.astype(np.float32) # convert to float
    img = Restaurate(img, geometry) # restaurate

    img[mask ==1] = -1 # Apply Mask

    img[img < 0] = -1 # all invalid values must be -1 by convention
    t2 = time()

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

def restauration_cat_3d(args,preview  = False,save  = False,read = False):
    
    jason, ibira_datafolder, scans_string, _ = args

    diffractionpattern = []
    count = -1
    time_difpads = 0

    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        if jason['Frames'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Frames'], filepaths,  filenames)
        
        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        if read:
            n = len(filenames)
            difpads = []
            for i in range(n):
                difpad = np.load(jason['SaveDifpadPath'] + filenames[i] + '.npy')
                difpads.append(difpad)
            difpads = np.asarray(difpads)
        else: 
            difpads, time_difpads, _ = pi540_restauration_cat_block(params,jason['SaveDifpadPath'],preview,save)

        difpads = masks_application(difpads, jason)

        diffractionpattern.append(difpads)

    return diffractionpattern, time_difpads

def restauration_cat_2d(args,preview = False,save = False,read = False):
    
    jason, ibira_datafolder, scans_string, _ = args
    time_difpads = 0

    _, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, jason['Acquisition_Folders'][0],scans_string), look_for_extension=".hdf5")
        
    params = (jason, ibira_datafolder, filenames[0], jason['Acquisition_Folders'][0], scans_string)
    
    if read:
        difpads = np.load(jason['SaveDifpadPath'] + filenames[0] + '.npy')
    else:   
        difpads, time_difpads = pi540_restauration_cat(params,jason['SaveDifpadPath'],preview,save)
    
    difpads = np.expand_dims(difpads,axis=0)
    print('difpadshape:',difpads.shape)
    difpads = masks_application(difpads, jason)

    return difpads, time_difpads