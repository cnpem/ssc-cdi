import h5py
from time import time
import os

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from sscCdi.caterete.ptycho_processing import *

from sscIO import io
from sscPimega import pi540D
from sscPimega import opt540D

############################ OLD RESTAURATION BY GIOVANNI #####################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# MODULES FOR THE RESTAURATION APPLICATION 
# (see main code below)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++



def pi540_restauration_cat_block(args):
    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    scans_string        = args[5]
    
    difpads = []

    for measurement_file, measurement_filepath in zip(filenames, filepaths):

        param = (jason,ibira_datafolder,measurement_file,acquisitions_folder,scans_string,measurement_filepath)
        difpad, jason = pi540_restauration_cat(param)
        
        if difpads == [] or difpads[0].shape == difpad.shape:
            difpads.append(difpad)
        else:
            difpads.append(np.zeros(difpads[0].shape))

    difpads = np.asarray(difpads)
    print('Difpads shape after restauration and binning of', jason['Binning'], ':', difpads.shape)

    return difpads

def pi540_restauration_cat(args):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]

    print('\nMeasurement file in pi540_restauration_cat: ', measurement_file)
    difpads = get_restaurated_difpads_old_format(jason, os.path.join(ibira_datafolder, acquisitions_folder,scans_string), measurement_file)
    print('Finished Restauration')

    return difpads, jason


def get_restaurated_difpads_old_format(jason, path, name):
    """Extracts the data from json and manipulate it according G restauration input format

    Args:
        jason (json file): json object
        path (list of dtrings): list of complete paths to all files
        name (list of strings): list of all file names

    Returns:
        3D array: restaured difpads
    """    

    fullpath = os.path.join(path, name)
    raw_difpads,_ = io.read_volume(fullpath, 'numpy', use_MPI=True, nprocs=jason["Threads"])

    z1 = float(jason["DetDistance"]) * 1000  # Here comes the distance Geometry(Z1):
    geometry = Geometry(z1)

    empty = np.asarray(h5py.File(jason['EmptyFrame'], 'r')['/entry/data/data']).squeeze().astype(np.float32)
    
    flat = np.load(jason["FlatField"])

    flat = np.array(flat)
    flat[np.isnan(flat)] = -1
    flat[flat == 0] = 1

    mask = np.load(jason['Mask'])

    centerx, centery = jason['DifpadCenter']
    print('Manual Difpad Center :',centerx, centery)

    hsize = jason['DetectorROI']  # HALF SIZE 

    Binning = int(jason['Binning'])

    r_params = (Binning, empty, flat, centerx, centery, hsize, geometry, mask, jason)

    output, _ = pi540D.backward540D_nonplanar_batch(raw_difpads, z1, jason['Threads'], [ hsize//2 , hsize//2 ], restauration_processing_binning,  r_params, 'only')

    return output

def restauration_processing_binning(img, args):

    flat, mask, geometry, cx, cy, hsize,  binning = args

    img = img * np.squeeze(flat) # Apply flatfield

    img = img.astype(np.float32) # convert to float
    
    img[np.abs(mask) == 1] = -1 # Apply Mask
    
    img = Restaurate(img, geometry) # restore

    img[img < 0] = -1 # all invalid values must be -1 by convention

    img = img[cy - hsize:cy + hsize, cx - hsize:cx + hsize]      # select ROI from the center (cx,cy)

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

    return img

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
    return opt540D._worker_annotation_image(pi540D.forward540D(img, geom))