#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sscResolution
import sscPtycho
import sscCdi

from time import time
import os
import json
import h5py
import pandas as pd
import numpy as np
import math
import uuid
import SharedArray as sa
import multiprocessing
import multiprocessing.sharedctypes
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.morphology import square, erosion, convex_hull_image
import skimage.filters
from scipy.ndimage import gaussian_filter
from numpy.fft import fftshift as shift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from .misc import create_directory_if_doesnt_exist

def cat_ptycho_3d(difpads,jason):
    sinogram = []
    probe = []
    background = [] 

    count = -1
    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

        print('\nFilenames: ', filenames)

        if count == 0: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
            object_shape, half_size, object_pixel_size, jason =sscCdi.caterete.ptycho_processing.set_object_shape(difpads[count],jason,filenames,filepaths,acquisitions_folder)
            jason["object_pixel"] = object_pixel_size

        args = (jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,len(filenames))
        sinogram3d ,probe3d, background3d = sscCdi.caterete.ptycho_processing.ptycho3d_batch(difpads[count], args) # Main ptycho iteration over ALL frames in threads

        sinogram.append(sinogram3d)
        probe.append(probe3d)
        background.append(background3d)
    
    return sinogram,probe,background,jason


def cat_ptycho_serial(jason):

    sinogram_list   = []
    probe_list      = []
    background_list = []
    counter = 0
    first_iteration = True
    first_of_folder = True
    time_elasped_restauration = 0
    time_elasped_ptycho = 0

    for acquisitions_folder in jason['Acquisition_Folders']:  
        print('Acquisiton folder: ',acquisitions_folder)
        filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

        for measurement_file, measurement_filepath in zip(filenames, filepaths):   
            print('File: ',measurement_file)
            args1 = (jason,acquisitions_folder,measurement_file,measurement_filepath,len(filenames))
            t_start = time()
            difpads, _ , jason = sscCdi.caterete.ptycho_restauration.restauration_cat_2d(args1,first_run=first_iteration) # Restauration of 2D Projection (difpads - real, is a ndarray of size (1,:,:,:))
            time_elasped_restauration += time() - t_start
            
            if first_iteration: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                object_shape, half_size, object_pixel_size, jason = sscCdi.caterete.ptycho_processing.set_object_shape(difpads,jason, [measurement_file], [measurement_filepath], acquisitions_folder)
                jason["object_pixel"] = object_pixel_size
                first_iteration = False

            object_dummy     = np.zeros((1,object_shape[1],object_shape[0]),dtype = complex) # build 3D Sinogram
            probe_dummy      = np.zeros((1,1,difpads.shape[-2],difpads.shape[-1]),dtype = complex)
            background_dummy = np.zeros((1,difpads.shape[-2],difpads.shape[-1]), dtype=np.float32)
            
            args2 = (jason,[measurement_file], [measurement_filepath], acquisitions_folder,half_size,object_shape,len([measurement_file]),object_dummy,probe_dummy,background_dummy)

            t_start2 = time()
            object2d, probe2d, background2d = sscCdi.caterete.ptycho_processing.ptycho_main(difpads, args2, 0, 1,jason['GPUs'])   # Main ptycho iteration on ALL frames in threads
            time_elasped_ptycho += time() - t_start2

            if first_of_folder:
                object = object2d
                probe  = probe2d
                background    = background2d
                first_of_folder = False
            else:
                object = np.concatenate((object,object2d), axis = 0)
                probe  = np.concatenate((probe,probe2d),  axis = 0)
                background = np.concatenate((background,background2d), axis = 0)
            counter +=1

        first_of_folder = True

        sinogram_list.append(object)
        probe_list.append(probe)
        background_list.append(background)

    return sinogram_list, probe_list, background_list, time_elasped_restauration, time_elasped_ptycho, jason

def define_paths(jason):
    if 'PreviewGCC' not in jason: jason['PreviewGCC'] = [False,""] # flag to save previews of interest only to GCC, not to the beamline user
    
    #=========== Set Parameters and Folders =====================
    print('Proposal path: ',jason['ProposalPath'] )
    print('Acquisition folder: ',jason["Acquisition_Folders"][0])
 
    if jason["PreviewGCC"][0] == True: # path convention for GCC users
        if 'LogfilePath' not in jason: jason['LogfilePath'] = ''
        jason["PreviewGCC"][1]  = os.path.join(jason["PreviewGCC"][1],jason["Acquisition_Folders"][0])
        jason["PreviewFolder"]  = os.path.join(jason["PreviewGCC"][1])
        jason["SaveDifpadPath"] = os.path.join(jason["PreviewGCC"][1])
        jason["ReconsPath"]     = os.path.join(jason["PreviewGCC"][1])
    else:
        beamline_outputs_path = os.path.join(jason['ProposalPath'] .rsplit('/',3)[0], 'proc','recons',jason["Acquisition_Folders"][0]) # standard folder chosen by CAT for their outputs
        print("Output path:",     beamline_outputs_path)
        jason["LogfilePath"]    = beamline_outputs_path
        jason["PreviewFolder"]  = beamline_outputs_path
        jason["SaveDifpadPath"] = beamline_outputs_path
        jason["ReconsPath"]     = beamline_outputs_path


    if jason['InitialObj'] in jason and jason['InitialObj']   != "": jason['InitialObj']   = os.path.join(jason['ReconsPath'], jason['InitialObj']) # append initialObj filename to path
    if jason['InitialObj'] in jason and jason['InitialProbe'] != "": jason['InitialProbe'] = os.path.join(jason['ReconsPath'], jason['InitialProbe'])
    if jason['InitialObj'] in jason and jason['InitialBkg']   != "": jason['InitialBkg']   = os.path.join(jason['ReconsPath'], jason['InitialBkg'])

    jason['scans_string'] = 'scans'
    jason['positions_string']  = 'positions'

    images_folder    = os.path.join(jason["Acquisition_Folders"][0],'images')

    input_dict = json.load(open(os.path.join(jason['ProposalPath'] ,jason["Acquisition_Folders"][0],'mdata.json')))
    jason["Energy"] = input_dict['/entry/beamline/experiment']["energy"]
    jason["DetDistance"] = input_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
    jason["RestauredPixelSize"] = input_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
    jason["DetectorExposure"][1] = input_dict['/entry/beamline/detector']['pimega']["exposure time"]
    jason["EmptyFrame"] = os.path.join(jason['ProposalPath'] ,images_folder,'empty.hdf5')
    jason["FlatField"]  = os.path.join(jason['ProposalPath'] ,images_folder,'flat.hdf5')
    jason["Mask"]       = os.path.join(jason['ProposalPath'] ,images_folder,'mask.hdf5')
    return jason


def get_files_of_interest(jason,acquistion_folder=''):

    if acquistion_folder != '':
            filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(jason['ProposalPath'] , acquistion_folder,jason['scans_string'] ), look_for_extension=".hdf5")
    else:
        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(jason['ProposalPath'] , jason["Acquisition_Folders"][0],jason['scans_string'] ), look_for_extension=".hdf5")

    if jason['Projections'] != []:
        filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths, filenames)

    return filepaths, filenames


def match_cropped_frame_dimension(sinogram,frame):
    """ Match the new incoming frame to the same squared shape of the sinogram. Sinogram should have shape (M,N,N)!

    Args:
        sinogram : sinogram of shape (M,N,N)
        frame : frame of shape (A,B)

    Returns:
        frame : frame of shape (N,N)
    """

    if sinogram.shape != frame.shape:
        print('Frame shape do not match the sinogram. Applying correction')
        print(f'\t Sinogram shape: {sinogram.shape}. Frame shape: {frame.shape}')

    if sinogram.shape[1] < frame.shape[0]:
        frame = frame[0:sinogram.shape[1],:]
    elif sinogram.shape[1] > frame.shape[0]:
        frame = np.concatenate((frame,frame[-1,:]),axis=0) # appended a repeated last line

    if sinogram.shape[2] < frame.shape[1]:
        frame = frame[:,0:sinogram.shape[2]]
    elif sinogram.shape[2] > frame.shape[1]:
        frame = np.concatenate((frame,frame[:,-1]),axis=1) # appended a repeated last line

    if sinogram.shape != frame.shape:
        print(f'\t Corrected drame shape: {frame.shape}')

    return frame


def make_1st_frame_squared(frame):
    """ Crops frame of dimension (A,B) to (A,A) or (B,B), depending if A or B is smaller

    Args:
        frame: 2D frame

    Returns:
        frame: cropped frame with smalelr dimension
    """
    if frame.shape[0] != frame.shape[1]:
        smallest_shape = min(frame.shape[0],frame.shape[1])
        frame = frame[0:smallest_shape,0:smallest_shape]
    return frame


def crop_sinogram(sinogram, jason): 

    cropped_sinogram = sinogram
    if jason['AutoCrop'] == True: # automatically crop borders with noise
        print('Auto cropping frames...')
        
        if 1: # Miqueles approach using scan positions
            frame = 0
            ibira_datafolder = jason["ProposalPath"]
            for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon
                
                filepaths, filenames = sscCdi.caterete.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

                for measurement_file, measurement_filepath in zip(filenames, filepaths):

                    if sinogram.shape[0] == len(filenames): print("SHAPES MATCH!")

                    probe_positions_file = os.path.join(acquisitions_folder, jason['positions_string'], measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
                    probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)

                    cropped_frame = autocrop_using_scan_positions(sinogram[frame,:,:],jason,probe_positions) # crop
                    if frame == 0: 
                        cropped_frame =  make_1st_frame_squared(cropped_frame)
                        cropped_sinogram = np.empty((sinogram.shape[0],cropped_frame.shape[0],cropped_frame.shape[1]),dtype=complex)
                    
                    cropped_frame = match_cropped_frame_dimension(cropped_sinogram,cropped_frame)
                    print("SHAPES:",len(filenames),sinogram.shape, cropped_frame.shape)
                    cropped_sinogram[frame,:,:] = cropped_frame
                    frame += 1
       
        if 0: # Miqueles approach using T operator
            for frame in range(sinogram.shape[0]):
                cropped_sinogram[frame,:,:] = autocrop_miqueles_operatorT(sinogram[frame,:,:])
            
        if 0: # Yuri approach using local entropy
            for frame in range(sinogram.shape[0]):
                min_crop_value = []
                best_crop = auto_crop_noise_borders(sinogram[frame,:,:])
                min_crop_value.append(best_crop)
            min_crop = min(min_crop_value)
            cropped_sinogram = sinogram[:, min_crop:-min_crop-1, min_crop:-min_crop-1]

        if cropped_sinogram.shape[1] % 2 != 0:  # object array must have even number of pixels to avoid bug during the phase unwrapping later on
            cropped_sinogram = cropped_sinogram[:,0:-1, :]
        if cropped_sinogram.shape[2] % 2 != 0:
            cropped_sinogram = cropped_sinogram[:,:, 0:-1]
        print('\t Done!')
        
    return cropped_sinogram


def autocrop_using_scan_positions(image,jason,probe_positions):

    probe_positions = 1e-6 * probe_positions / jason['object_pixel']     #scanning positions @ image domain

    n         = image.shape[0]
    x         = (n//2 - probe_positions [:,0]).astype(int)
    y         = (n//2 - probe_positions [:,1]).astype(int)
    pinholesize = 0 #tirado do bolso! 
    xmin      = x.min() - pinholesize//2
    xmax      = x.max() + pinholesize//2
    ymin      = y.min() - pinholesize//2
    ymax      = y.max() + pinholesize//2
    new = image[xmin:xmax, ymin:ymax] 
    new = new + abs(new.min())

    return new


def autocrop_miqueles_operatorT(image):

    def _operator_T(u):
        d   = 1.0
        uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
        uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
        uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
        uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
        delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
        return np.sqrt( delta )
    
    def removeBorders(img, n):
        r,c = img.shape
        img[0:n,:] = 0
        img[r-n:r,:] = 0
        img[:,0:n] = 0
        img[:,c-n:c] = 0
        return img

    img = np.angle(image) ### um certo frame, que vem direto de ptycho
    img_gradient = skimage.filters.scharr(img)
    img_gradient = skimage.util.img_as_ubyte(img_gradient / img_gradient.max())
    img_gradient = gaussian_filter(img_gradient, sigma=10)
    where = _operator_T(img_gradient).real
    new = np.copy(img_gradient)
    new[ new > 0] = _operator_T(new).real[ img_gradient > 0]
    tol = 1e-6
    mask = ( np.abs( new - img_gradient) < tol ) * 1.0
    mask = erosion(mask, square(5))
    mask = removeBorders(mask, 100)
    chull = convex_hull_image(mask)
    image[ chull == 0 ] = 0 #cropando
    return image


def apply_phase_unwrap(sinogram, jason):

    if jason['Phaseunwrap'][2] != [] and jason['Phaseunwrap'][3] != []:
        print('Manual cropping of the data')
        """ Fine manual crop of the reconstruction for a proper phase unwrap
        jason['Phaseunwrap'][2] = [upper_crop,lower_crop]
        jason['Phaseunwrap'][3] = [left_crop,right_crop] """
        sinogram = sinogram[:,jason['Phaseunwrap'][2][0]: -jason['Phaseunwrap'][2][1], jason['Phaseunwrap'][3][0]: -jason['Phaseunwrap'][3][1]]
    
    print('Cropped object shape:', sinogram.shape)

    print('Phase unwrapping the cropped image')
    n_iterations = jason['Phaseunwrap'][1]  # number of iterations to remove gradient from unwrapped image.
    
    phase = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))
    absol = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))

    for frame in range(sinogram.shape[0]):
        original_object = sinogram[frame,:,:]  # create copy of object
        absol[frame,:,:] = np.abs(sinogram[frame,:,:])
        # phase[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.angle(sscPtycho.RemovePhaseGrad(sinogram[frame,:,:])))#, n_iterations, non_negativity=0, remove_gradient=0)
        phase[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.angle(sinogram[frame,:,:]))

        if 1:  # plot original and cropped object phase and save!
            figure, subplot = plt.subplots(1, 2,dpi=300,figsize=(5,5))
            subplot[0].imshow(-np.angle(original_object),cmap='gray')
            subplot[1].imshow(phase[frame,:,:],cmap='gray')
            subplot[0].set_title('Original')
            subplot[1].set_title('Cropped and Unwrapped')
            figure.savefig(jason['PreviewFolder'] + f'/autocrop_and_unwrap_{frame}.png')

    return phase,absol


def calculate_FRC(sinogram, jason):

    if sinogram.shape[1]%2!=0:
        sinogram = sinogram[:,0:-1,:]
    if sinogram.shape[2]%2!=0:
        sinogram = sinogram[:,:,0:-1]

    object_pixel_size = jason["object_pixel"] 

    frame = 0 # selects first frame of the sinogram to calculate resolution

    if jason['FRC'] == True:
        print('Estimating resolution via Fourier Ring Correlation')
        resolution = resolution_frc(sinogram[frame,:,:], object_pixel_size, plot=True,plot_output_folder=os.path.join(jason["PreviewFolder"]+'/'),savepath=jason["PreviewFolder"])
        try:
            print('\tResolution for frame ' + str(frame) + ':', resolution['halfbit_resolution'])
            jason["halfbit_resolution"] = resolution['halfbit_resolution']
        except:
            print('Could not calculate halfbit FRC resolution')
        try:
            print('\tResolution for frame ' + str(frame) + ':', resolution['sigma_resolution'])
            jason["3sigma_resolution"] = resolution['sigma_resolution']
        except:
            print('Could not calculate 3sigma FRC resolution')

    return jason


def plotshow(imgs, file, subplot_title=[], legend=[], cmap='jet', nlines=1, bLog=False, interpolation='bilinear'):  # legend = plot titles
    """ Show plot in a specific format 

    Args:
        imgs ([type]): [description]
        file ([type]): [description]
        subplot_title (list, optional): [description]. Defaults to [].
        legend (list, optional): [description]. Defaults to [].
        cmap (str, optional): [description]. Defaults to 'jet'.
        nlines (int, optional): [description]. Defaults to 1.
        bLog (bool, optional): [description]. Defaults to False.
        interpolation (str, optional): [description]. Defaults to 'bilinear'.
    """    
    num = len(imgs)

    for j in range(num):
        if type(cmap) == str:
            colormap = cmap
        elif len(cmap) == len(imgs):
            colormap = cmap[j]
        else:
            colormap = cmap[j // (len(imgs) // nlines)]

        sb = plt.subplot(nlines, (num + nlines - 1) // nlines, j + 1)
        if type(imgs[j][0, 0]) == np.complex64 or type(imgs[j][0, 0]) == np.complex128:
            sb.imshow(sscPtycho.CMakeRGB(imgs[j]), cmap='hsv', interpolation=interpolation)
        elif bLog:
            sb.imshow(np.log(1 + np.maximum(imgs[j], -0.1)) / np.log(10), cmap=colormap, interpolation=interpolation)
        else:
            sb.imshow(imgs[j], cmap=colormap, interpolation=interpolation)

        if len(legend) > j:
            sb.set_title(legend[j])

        sb.set_yticks([])
        sb.set_xticks([])
        sb.set_aspect('equal')
        if subplot_title != []:
            sb.set_title(subplot_title[j])

    plt.savefig(file + '.png', format='png', dpi=300)
    plt.show()
    plt.clf()
    plt.close()



def read_probe_positions(probe_positions_filepath, measurement):
    print('Reading probe positions (probe_positions)...')
    probe_positions = []
    positions_file = open(probe_positions_filepath)

    line_counter = 0
    for line in positions_file:
        line = str(line)
        if line_counter >= 1:  # skip first line, which is the header
            T = -3E-3  # why did Giovanni rotated by this amount? not using this correction seems to result in an error in the number of positions
            pxl = float(line.split()[1])
            pyl = float(line.split()[0])
            px = pxl * np.cos(T) - np.sin(T) * pyl
            py = pxl * np.sin(T) + np.cos(T) * pyl
            probe_positions.append([px, py, 1, 1])
        line_counter += 1

    probe_positions = np.asarray(probe_positions)

    pshape = pd.read_csv(probe_positions_filepath,sep=' ').shape  # why read pshape from file? can it be different from probe_positions.shape+1?

    os.system(f"h5clear -s {measurement}")
    with h5py.File(measurement, 'r') as file:
        mshape = file['entry/data/data'].shape

    if pshape[0] == mshape[0]:  # check if number of recorded beam positions in txt matches the positions saved to the hdf
        print('\tSuccess in reading positions file:' + probe_positions_filepath)
        print("\tShape probe_positions:", probe_positions.shape, pshape, mshape)
    else:
        print("\tError in probe_positions shape. {0} is different from diffraction pattern shape {1}".format(probe_positions.shape, mshape,pshape))
        print('\t\t Setting object as null array with correct shape.')
        probe_positions = np.zeros((mshape[0]-1, 4))
        print('\t\tNew probe positions shape',probe_positions.shape)
    return probe_positions


def create_squared_mask(start_row, start_column, height, width, mask_shape):
    """ Create squared mask. Start position is the top-left corner. All values in pixels!

    Args:
        start_row ([type]): [description]
        start_column ([type]): [description]
        height ([type]): [description]
        width ([type]): [description]
        mask_shape ([type]): [description]

    Returns:
        [type]: [description]
    """
    mask = np.zeros(mask_shape)
    mask[start_row:start_row + height, start_column:start_column + width] = 1
    return mask


def set_initial_parameters(jason, difpads, probe_positions, radius, center_x, center_y, object_size, dx):
    """Defines the structure to get started with the reconstruction

    Args:
        jason (json file): json with the input parameter
        difpads (3D array)): measured diffraction patterns
        probe_positions (array): probe positions in x and y directions
        radius (int): probe support radius in pixels
        center_x (int): probe support center at x coordinate
        center_y (int): probe support center at y coordinate
        maxroi (int): total object size plus padding
        dx (int): pixel size

    Returns:
        initial data for reconstruction
    """    
    half_size = difpads.shape[-1] // 2

    if jason['f1'] == -1:  # Manually choose wether to find Fresnel number automatically or not
        jason['f1'] = setfresnel(dx, pixel=jason['RestauredPixelSize'], energy=jason['Energy'], z=jason['DetDistance'])
        jason['f1'] = -jason['f1']
    print('\tF1 value:', jason['f1'])

    # Compute probe: initial guess:
    probe = set_initial_probe(difpads, jason)

    # Adicionar modulos incoerentes
    probe = set_modes(probe, jason)

    # Object initial guess:
    obj = set_initial_obj(jason, object_size, probe, difpads)

    # Mask of 1 and 0:
    sigmask = set_sigmask(difpads)

    # Background: better not use any for now.
    if 0:
        background = set_background(difpads, jason)
    else:
        background = np.ones(difpads[0].shape) # dummy

    # Compute probe support:
    probesupp = probe_support(probe, half_size, radius, center_x, center_y)

    probe_positionsi = probe_positions + 0  # what's the purpose of declaring probe_positionsi?

    # Set data for Ptycho algorithms:
    datapack = set_datapack(obj, probe, probe_positions, difpads, background, probesupp)

    return datapack, probe_positionsi, sigmask


def set_object_pixel_size(jason,half_size):
    c = 299792458             # Speed of Light [m/s]
    planck = 4.135667662E-18  # Plank constant [keV*s]
    wavelength = planck * c / jason['Energy'] # meters
    jason["wavelength"] = wavelength
    
    # Convert pixel size:
    dx = wavelength * jason['DetDistance'] / ( jason['Binning'] * jason['RestauredPixelSize'] * half_size * 2)

    return dx, jason


def setfresnel(dx=1, pixel=55.55E-6, energy=3.8E3, z=1):
    """Calculate Fresnel number

    Args:
        dx (int, optional): effective pixel size. Defaults to 1.
        pixel (float, optional): pixel size. Defaults to 55.55E-6.
        energy (float, optional): beam energy. Defaults to 3.8E3.
        z (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """    
    print('Setting Fresnel number automatically...')
    c = 299792458  # Velocity of Light [m/s]
    plank = 4.135667662E-15  # Plank constant [ev*s]
    const = (plank * c)
    wave = const / (energy * 1000)  # [m]  waveleght

    magn = pixel / dx  # m
    F1 = ((dx * dx) * magn) / (wave * z)

    print('\tFresnel number (F1) - F1:', F1)
    print('\tMagnification:', magn)
    print('\tEffective Pixel size:', dx)

    return F1


def set_initial_probe(difpads, jason):
    """Create initial guess for the probe

    Args:
        difpads (array): measured diffraction patterns
        jason (json): file with inputs

    Returns:
        probe (array)
    """    
    print('Setting initial probe...')
    # Compute probe: initial guess:
    if jason['InitialProbe'] == "":
        # Initial guess for none probe:
        probe = np.average(difpads, 0)[None]
        ft = shift(fft2(shift(probe)))
        probe = np.sqrt(shift(ifft2(shift(ft))))
    else:
        # Load probe:
        probe = np.load(jason['InitialProbe'])[0][0]

    print("\tProbe shape:", probe.shape)
    return probe


def set_modes(probe, jason):
    """ Set number of probe modes

    Args:
        probe : probe matrix
        jason : jason input dictionary

    Returns:
        probe : probe with different modes
    """    
    print('Setting modes...')
    mode = probe.shape[0]
    print('\tNumber of modes:', mode)
    # Adicionar modulos incoerentes
    if jason['Modes'] > mode:
        add = jason['Modes'] - mode
        probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
        for i in range(add):
            probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

    print("\tProbe shape ({0},{1}) with {2} incoherent modes".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

    return probe


def set_gpus(jason):
    """ Function to set all 4 GPUs if json input value is negative

    Args:
        jason : json input dictionary
    """    
    print('Setting GPUs...')
    if jason['GPUs'][0] < 0:
        jason['GPUs'] = [0, 1, 2, 3]
    sscPtycho.SetDevices(jason['GPUs'])


def set_initial_obj(jason, object_shape, probe, difpads):
    """Create initial guess for the object

    Args:
        jason (json file): file with inputs
        half_size (int): half size of the object
        maxroi (int): size of the padding object
        probe (array): probe
        difpads (array): measured data

    Returns:
        obj (array)
    """    
    # Object initial guess:
    if jason['InitialObj'] == "":
        print('Setting initial guess for Object...')
        obj = np.random.rand(object_shape[0], object_shape[1]) * (np.sqrt(np.average(difpads) / np.average(abs(np.fft.fft2(probe)) ** 2)))
    else:
        obj = np.load(jason['InitialObj'])[0]

    return obj


def set_sigmask(difpads):
    """Create a mask for invalid pixels

    Args:
        difpads (array): measured diffraction patterns

    Returns:
        sigmask (array): 2D-array, same shape of a diffraction pattern, maps the invalid pixels
        0 for negative values, intensity measured elsewhere
    """    
    # Mask of 1 and 0:
    sigmask = np.ones(difpads[0].shape)
    sigmask[difpads[0] < 0] = 0

    return sigmask


def set_background(difpads, jason):
    """Creates background mask: constant signal on detector, for instance, the illumination of the room

    Args:
        difpads (array): measured intensities 
        jason (json file): json file with the inputs for reconstruction

    Returns:
        background (array): mask for take into account the background
    """    
    print('Setting background...')
    # Background: better not use any for now.
    if jason['InitialBkg'] == "":
        print('\tUsing no background!')
        background = np.zeros(difpads[0].shape)
    else:
        try:
            background = np.maximum(abs(np.load(jason['ReconPath'] + jason['InitialBkg'])), 1)
        except:
            background = np.ones(difpads[0].shape)

    return background


def probe_support(probe, half_size, radius, center_x, center_y):
    """Create a support for probe

    Args:
        probe (array): initial guess for the probe
        half_size (int): half difraction pattern size
        radius (): probe support radius
        center_x (int): probe support center in x
        center_y (int): probe support center in y

    Returns:
        probesupp (array): probe support
    """    
    print('Setting probe support...')
    # Compute probe support:
    ar = np.arange(-half_size, half_size)
    xx, yy = np.meshgrid(ar, ar)
    probesupp = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2  # offset of 30 chosen by hand?
    probesupp = np.asarray([probesupp for k in range(probe.shape[0])])

    # No support:
    # probesupp = probesupp*0 + 1

    return probesupp


def Prop(img, f1): # Probe propagation
    """ Frunction for free space propagation of the probe in the Fraunhoffer regime

    See paper `Memory and CPU efficient computation of the Fresnel free-space propagator in Fourier optics simulations <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28750&id=420820>`_. Are terms missing after convolution?
    
    Args:
        img (array): probe
        f1 (float): Fresnel number

    Returns:
        [type]: [description]
    """    
    hs = img.shape[-1] // 2
    ar = np.arange(-hs, hs) / float(2 * hs)
    xx, yy = np.meshgrid(ar, ar)
    g = np.exp(-1j * np.pi / f1 * (xx ** 2 + yy ** 2))
    return np.fft.ifft2(np.fft.fft2(img) * np.fft.fftshift(g))


def set_datapack(obj, probe, probe_positions, difpads, background, probesupp):
    """Create a dictionary to store the data needed for reconstruction

    Args:
        obj (array): guess for ibject
        probe (array): guess for probe
        probe_positions (array): position in x and y directions
        difpads (array): intensities (diffraction patterns) measured
        background (array): background
        probesupp (array): probe support

    Returns:
        datapack (dictionary)
    """    
    print('Creating datapack...')
    # Set data for Ptycho algorithms:
    datapack = {}
    datapack['obj'] = obj
    datapack['probe'] = probe
    datapack['rois'] = probe_positions
    datapack['difpads'] = difpads
    datapack['bkg'] = background
    datapack['probesupp'] = probesupp

    return datapack


def save_variable(variable, predefined_name, savename=""):
    """ Function to save reconstruction object, probe and/or background. 
    
    This function presents some redundancy. Should be improved!

    Args:
        variable : variable to be saved (e.g. sinogram, probe reconstruction and/or background)
        predefined_name: predefined name for saving the output variable
        savename (str, optional): Name to be used instead of predefined_name. Defaults to "".
    """    
    variable = np.asarray(variable, dtype=object)

    # for i in range(variable.shape[0]):
    #     print('shapes', variable[i].shape)
    for i in range(variable.shape[0]):  # loop to circumvent problem with nan values
        if math.isnan(variable[i][:, :].imag.sum()):
            variable[i][:, :] = np.zeros(variable[i][:, :].shape)

    variable = np.asarray(variable, dtype=np.complex64)

    if savename != "":
        np.save(savename, variable)
    else:
        np.save(predefined_name, variable)


def resolution_frc(data, pixel, plot=False,plot_output_folder="./outputs/preview",savepath='./outputs/reconstruction'):
    """     
    Fourier Ring Correlation for 2D images:
    The routine inputs are besides the two images for correlation
    Resolution threshold curves desired: "half" for halfbit, "sigma" for 3sigma, "both" for them both
    Pixelsize of the object

    # Output is a dictionary with the resolution values in the object pixelsize unit, and the FRC, frequency and threshold arrays
    # resolution['halfbit'] :  resolution values in the object pixelsize unit
    # resolution['curve']   :  FRC array
    # resolution['freq']    :  frequency array
    # resolution['sthresh'] :  threshold array
    # resolution['hthresh'] :  threshold array

    Args:
        data : image to calculate resolution
        pixel : effective pixel size of the object
        plot_output_folder (str, optional): _description_. Defaults to "./outputs/preview".
        savepath : folder to save FRC dictionary with outputs. Defaults to './outputs/reconstruction'.

    Returns:
        resolution: FRC output dictionary
    """    
    
    print('Calculating resolution by Fourier Ring Correlation...')

    sizex = data.shape[-1]
    sizey = data.shape[-2]

    data = data.reshape((1,sizey,sizex)) # reshape so that fourier_ring_correlation interprets data correctly

    # For this case we will use the odd/odd even/even divisions of one image in a dataset
    data1 = data[:,0:sizey:2, 0:sizex:2]  # even
    data2 = data[:,1:sizey:2, 1:sizex:2]  # odd
    
    resolution = sscResolution.fourier_ring_correlation(data1,data2,pixel)
    if plot:
        sscResolution.get_fourier_correlation_fancy_plot(resolution, plot_output_folder, plot=True)

    if savepath != '':
        export_json(resolution,savepath+'/frc_outputs.txt')

    return resolution


def export_json(params,output_path):
    """ Exports a dictionary to a json file

    Args:
        params : dictionary
        output_path : path to output file
    """    
    import json, numpy
    export = {}
    for key in params:
        export[key] = params[key]
        if isinstance(params[key], numpy.ndarray):
            export[key] = export[key].tolist()
    json.dumps(export)

    out_file = open(output_path, "w")
    json.dump(export,out_file)
    return 0


def auto_crop_noise_borders(complex_array):
    """ Crop noisy borders of the reconstructed object using a local entropy map of the phase

    Args:
        complex_array : reconstructed object

    Returns:
        cropped_array : object without noisy borders
    """    
    import skimage.filters
    from skimage.morphology import disk

    img = np.angle(complex_array)  # get phase to perform cropping analysis

    img_gradient = skimage.filters.scharr(img)
    img_gradient = skimage.util.img_as_ubyte(img_gradient / img_gradient.max())
    local_entropy_map = skimage.filters.rank.entropy(img_gradient, disk(5)) # disk gives size of the region used to calculate local entropy

    smallest_img_dimension = 200
    max_crop = img.shape[0] // 2 - smallest_img_dimension // 2  # smallest image after cropping will have 2*100 pixels in each direction

    crop_sizes = range(1, max_crop, 10)

    mean_list = []
    for c in (crop_sizes):
        """
        mean is a good metric since we expect it to decrease as high entropy border
        gets cropped, and increase again as the low entropy smooth background gets cropped.
        it may become an issue if the sample is displaced from the center, though
        """
        mean = (local_entropy_map[c:-c, c:-c].ravel()).mean()
        mean_list.append(mean)

    best_crop = crop_sizes[np.where(mean_list == min(mean_list))[0][0]]

    # cropped_array = complex_array[best_crop:-best_crop, best_crop:-best_crop]  # crop original complex image

    # if 0:  # debug / see results
    #     figure, subplot = plt.subplots(1, 3, figsize=(10, 10), dpi=200)
    #     subplot[0].imshow(img)
    #     subplot[1].imshow(local_entropy_map)
    #     subplot[2].imshow(np.angle(cropped_array))
    #     subplot[0].set_title('Original')
    #     subplot[1].set_title('Local entropy')

    #     subplot[2].set_title('Cropped')

    #     figure, subplot = plt.subplots()
    #     subplot.plot(crop_sizes, mean_list)
    #     subplot.set_xlabel('Crop size')
    #     subplot.set_ylabel('Mean')
    #     subplot.grid()

    # if cropped_array.shape[0] % 2 != 0:  # object array must have even number of pixels to avoid bug during the phase unwrapping later on
    #     cropped_array = cropped_array[0:-1, :]
    # if cropped_array.shape[1] % 2 != 0:
    #     cropped_array = cropped_array[:, 0:-1]

    return best_crop


def create_output_directories(jason):
    if jason["PreviewGCC"][0] == True:
        try:
            create_directory_if_doesnt_exist(jason["PreviewGCC"][1])
        except:
            print('ERROR: COULD NOT CREATE OUTPUT DIRECTORY')
    if jason["LogfilePath"] != "":
        create_directory_if_doesnt_exist(jason["LogfilePath"])
    if jason["PreviewFolder"] != "":
        create_directory_if_doesnt_exist(jason["PreviewFolder"])
    if jason["ReconsPath"] != "":
        create_directory_if_doesnt_exist(jason["ReconsPath"])
    if jason["ReconsPath"] != "":
        create_directory_if_doesnt_exist(jason["ReconsPath"])
    if jason["SaveDifpadPath"] != "":
        create_directory_if_doesnt_exist(jason["SaveDifpadPath"])


def convert_probe_positions(dx, probe_positions, offset_topleft = 20):
    """Set probe positions considering maxroi and effective pixel size

    Args:
        difpads (3D array): measured diffraction patterns
        jason (json file): file with the setted parameters and directories for reconstruction
        probe_positions (array): each element is an 2-array with x and y probe positions
        offset_topleft (int, optional): [description]. Defaults to 20.

    Returns:
        object pixel size (float), maximum roi (int), probe positions (array)
    """    

    # Subtract the probe positions minimum to start at 0
    probe_positions[:, 0] -= np.min(probe_positions[:, 0])
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    offset_bottomright = offset_topleft #define padding width
    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / dx + offset_topleft #shift probe positions to account for the padding
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / dx + offset_topleft #shift probe positions to account for the padding

    return probe_positions, offset_bottomright


def set_object_shape(difpads,jason,filenames,filepaths,acquisitions_folder,offset_topleft = 20):

    ibira_datafolder    = jason['ProposalPath']
    positions_string    = jason['positions_string']

    # Pego a PRIMEIRA medida de posicao, supondo que ela nao tem erro
    measurement_file = filenames[0]
    measurement_filepath = filepaths[0]
    
    # Compute half size of diffraction patterns:
    half_size = difpads.shape[-1] // 2

    # Compute/convert pixel size:
    dx, jason = set_object_pixel_size(jason,half_size)

    probe_positions_file = os.path.join(acquisitions_folder, positions_string, measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
    probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)
    probe_positions, offset_bottomright = convert_probe_positions(dx, probe_positions, offset_topleft = offset_topleft)

    maxroi        = int(np.max(probe_positions)) + offset_bottomright
    object_shape  = 2 * half_size + maxroi
    print('Object shape:',object_shape)

    return (object_shape,object_shape), half_size, dx, jason


def ptycho_main(difpads, args, _start_, _end_,gpu):
    t0 = time()

    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    acquisitions_folder = args[3]
    half_size           = args[4]
    object_shape        = args[5]
    sinogram            = args[7]
    probe3d             = args[8]
    backg3d             = args[9]

    ibira_datafolder  = jason['ProposalPath']
    positions_string  = jason['positions_string']

    for i in range(_end_ - _start_):
        
        measurement_file     = filenames[_start_ + i]
        measurement_filepath = filepaths[_start_ + i]
        
        if i == 0:
            current_frame = str(0).zfill(4)  # start at 0. this variable will name the output preview images of the object and probe
        else:
            current_frame = str(int(current_frame) + 1).zfill(4)  # increment one
        
        frame = int(current_frame)

        probe_positions_file = os.path.join(acquisitions_folder, positions_string, measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
        probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)
        probe_positions, _ = convert_probe_positions(jason["object_pixel"], probe_positions)

        run_ptycho = np.any(probe_positions)  # check if probe_positions == null matrix. If so, won't run current iteration. #TODO: output is null when #difpads != #positions. How to solve this?

        if i == 0: t1 = time()

        if run_ptycho == True:
                
            if i == 0: t2 = time()

            if jason["PreviewGCC"][0] and i == 0: # save plots of processed difpad and mean of all processed difpads
                difpad_number = 0
                sscCdi.caterete.misc.plotshow_cmap2(difpads[frame,difpad_number, :, :], title=f'Restaured + Processed Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/05_difpad_processed.png')
                sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads[frame], axis=0),    title=f"Mean of all difpads: {measurement_filepath.split('/')[-1]}", savepath=jason[ "PreviewFolder"] + '/05_difpad_processed_mean.png')

            probe_support_radius, probe_support_center_x, probe_support_center_y = jason["ProbeSupport"]

            print(f'Object shape: {object_shape}. Detector half-size: {half_size}')

            datapack, _, sigmask = set_initial_parameters(jason,difpads[frame],probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,object_shape,jason["object_pixel"])

            if i == 0: t3 = time()

            run_algorithms = True
            loop_counter = 1
            while run_algorithms:  # run Ptycho:
                try:
                    algorithm = jason['Algorithm' + str(loop_counter)]
                except:
                    run_algorithms = False

                if run_algorithms:
                    if algorithm['Name'] == 'GL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                    probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'],
                                                    epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                    probef1=jason['f1'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'positioncorrection':
                        datapack['bkg'] = None
                        datapack = sscPtycho.PosCorrection(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                               probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], 
                                                               epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                               probef1=jason['f1'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'RAAR':
                        datapack = sscPtycho.RAAR(iter=algorithm['Iterations'], beta=algorithm['Beta'],
                                                      probecycles=algorithm['ProbeCycles'], batch=algorithm['Batch'],
                                                      epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'],
                                                      sigmask=sigmask, probef1=jason['f1'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'GLL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                    probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'],
                                                    epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                    probef1=jason['f1'], data=datapack,params={'device':gpu})

                    loop_counter += 1
                    RF = datapack['error']

            print('Original object shape:', datapack['obj'].shape)

            if i == 0: t4 = time()

            sinogram[frame, :, :] = datapack['obj']  # build 3D Sinogram
            probe3d[frame, :, :, :]  = datapack['probe']
            backg3d[frame, :, :]  = datapack['bkg']

        else:
            print('CAUTION! Zeroing frame:',frame,' for error in position file.')
            sinogram[frame, :, :]   = np.zeros((object_shape[0],object_shape[1])) # build 3D Sinogram
            probe3d[frame, :, :, :] = np.zeros((1,difpads.shape[-2],difpads.shape[-1]))
            backg3d[frame, :, :]    = np.zeros((difpads.shape[-2],difpads.shape[-1]))

        if i == 0: t5 = time()

    print(f'\nElapsed time for reconstruction of 1st frame: {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes')
    print(f'Total time iteration: {t5 - t0:.2f} seconds = {(t5 - t0) / 60:.2f} minutes')

    return sinogram, probe3d, backg3d


def _worker_batch_frames_(params, idx_start, idx_end, gpu):
    
    output_object       = params[0]
    output_probe        = params[1]
    output_backg        = params[2]
    difpads             = params[3]
    jason               = params[4] 
    filenames           = params[5]
    filepaths           = params[6]
    acquisitions_folder = params[7]
    half_size           = params[8]
    object_shape        = params[9]
    total_frames        = params[10]   

    _start_ = idx_start
    _end_   = idx_end

    args = (jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames,output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:])

    output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:] = ptycho_main( difpads[_start_:_end_,:,:,:], args, _start_, _end_,gpu)
    

def _build_batch_of_frames_(params):

    total_frames = params[9]
    threads      = len(params[4]['GPUs']) 
    
    b = int( np.ceil( total_frames/threads )  ) 
    
    processes = []
    for k in range( threads ):
        begin_ = k*b
        end_   = min( (k+1)*b, total_frames )
        gpu = [k]

        p = multiprocessing.Process(target=_worker_batch_frames_, args=(params, begin_, end_, gpu))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    

def ptycho3d_batch( difpads, params):
    
    name         = str( uuid.uuid4())
    name1        = str( uuid.uuid4())
    name2        = str( uuid.uuid4())


    jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames = params

    try:
        sa.delete(name)
    except:
        pass
    try:
        sa.delete(name1)
    except:
        pass
    try:
        sa.delete(name2)
    except:
        pass
            
    output_object = sa.create(name,[total_frames, object_shape[0], object_shape[1]], dtype=np.complex64)
    output_probe  = sa.create(name1,[total_frames, 1, difpads.shape[-2],difpads.shape[-1]], dtype=np.complex64)
    output_backg  = sa.create(name2,[total_frames, difpads.shape[-2],difpads.shape[-1]], dtype=np.float32)

    _params_ = ( output_object, output_probe, output_backg, difpads, jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames)
    
    _build_batch_of_frames_ ( _params_ )

    sa.delete(name)
    sa.delete(name1)
    sa.delete(name2)

    return output_object,output_probe,output_backg


def masks_application(difpad, jason):

    center_row, center_col = jason["DifpadCenter"]

    if jason["DetectorExposure"][0]: 
        print("Removing pixels above detector pile-up threshold")
        mask = np.zeros_like(difpad)
        difpad_region = np.zeros_like(difpad)
        half_size = 128 # 128 pixels halfsize mean the region has 256^2, i.e. the size of a single chip
        mask[center_row-half_size:center_row+half_size,center_col-half_size:center_col+half_size] = 1
        difpad_region = np.where(mask>0,difpad,0)        
        detector_pileup_count = 350000  # counts/sec; value according to Kalile
        detector_exposure_time = jason["DetectorExposure"][1]
        difpad_rescaled = difpad_region / detector_exposure_time # apply threshold
        difpad[difpad_rescaled > detector_pileup_count] = -1
    elif jason["CentralMask"][0]:  # circular central mask to block center of the difpad
        radius = jason["CentralMask"][1] # pixels
        central_mask = create_circular_mask(center_row,center_col, radius, difpad.shape)
        difpad[central_mask > 0] = -1

    return difpad


def create_circular_mask(center_row, center_col, radius, mask_shape):
    """Create a circular mask to block the center of the diffraction pattern

    Args:
        center_row (int): Center position in the vertical dimension
        center_col (int): Center position in the horizontal dimension
        radius (int): Radius of the circular mask in pixels
        mask_shape ([tuple]): [description]

    Returns:
        [2-dimensional ndarrya]: array containing 1s within the disk, 0 otherwise
    """
    """ All values in pixels """
    mask = np.zeros(mask_shape)
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)

    Xmesh, Ymesh = np.meshgrid(x_array, y_array)

    mask = np.where((Xmesh - center_col) ** 2 + (Ymesh - center_row) ** 2 <= radius ** 2, 1, 0)
    return mask


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


def get_difpad_center(difpad, refine=True, fit=False, radius=20):
    """ Get central position of the difpad

    Args:
        difpad : diffraction pattern data
        refine (bool): Choose whether to refine the initial central position estimate. Defaults to True.
        fit (bool, optional): if true, refines using a lorentzian surface fit; else, gets the maximum. Defaults to False.
        radius (int, optional): size of the squared region around center used to refine the center estimate. Defaults to 20.

    Returns:
        center : diffraction pattern center
    """    
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


def preview_ptycho(jason, phase, absol, probe, frame = 0):
    if jason['Preview']:  # Preview Reconstruction:
        ''' Plot scan points
        plt.figure()
        plt.scatter(probe_positionsi[:, 0], probe_positionsi[:, 1])
        plt.scatter(datapack['rois'][:, 0, 0], datapack['rois'][:, 0, 1])
        plt.savefig(jason['PreviewFolder'] + '/scatter_2d.png', format='png', dpi=300)
        plt.clf()
        plt.close()
        '''

        plotshow([abs(Prop(p, jason['f1'])) for p in probe[frame]] + [p for p in probe[frame]], file=jason['PreviewFolder'] + '/probe_'  + str(frame), nlines=2)
        plotshow([phase[frame], absol[frame]], subplot_title=['Phase', 'Magnitude'],            file=jason['PreviewFolder'] + '/object_' + str(frame), nlines=1, cmap='gray')
        