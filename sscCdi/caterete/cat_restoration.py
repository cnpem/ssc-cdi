import numpy as np
import os, sys, h5py, time
from scipy import ndimage
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

""" Sirius Scientific Computing Imports """
import sscCdi
from sscIO import io
from sscPimega import pi540D
from sscPimega import misc as miscPimega

""" sscCdi relative imports"""
from .ptycho_processing import get_difpad_center, Geometry
from ..misc import read_hdf5

def Restorate(DP, geom):
    return pi540D.backward540D(DP, geom)

def restoration_cuda_parallel(jason):
    
    ibira_datafolder, scans_string  = jason['proposal_path'],jason['scans_string']

    count = -1
    for acquisitions_folder in jason['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if jason['projections'] != []:
            filepaths, filenames = sscCdi.misc.misc.select_specific_angles(jason['projections'], filepaths,  filenames)
            print('\nMeasurement file in restoration_cat_3d: ', filenames)
        
        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        # Restorate
        distance = jason["DetDistance"]*1000 # distance in milimeters
        geometry = Geometry(distance)
        params   = {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual' ,'susp': jason["suspect_border_pixels"]}
        project  = pi540D.dictionary540D(distance, params )
        geometry = pi540D.geometry540D( project )
        datapath = '/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/data/ptycho3d/glass21/'
        dic = {}
        dic['path']     = sorted(glob.glob( datapath + '/scans/*.hdf5') )
        dic['outpath']  = "/home/ABTLUS/eduardo.miqueles/test/"
        dic['order']    = "yx" 
        dic['rank']     = "ztyx" # order of axis
        dic['dataset']  = "entry/data/data"
        dic['nGPUs']    = len(jason["GPUs"])
        dic['GPUs']     = jason["GPUs"]
        dic['init']     = 0
        dic['final']    = -1 # -1 to use all DPs
        dic['saving']   = 1 # save or not
        dic['timing']   = 0 # print timers 
        dic['blocksize']= 10
        dic['geometry'] = geometry
        dic['roi']      = jason["detector_ROI_radius"]#512
        dic['center']   = jason["DP_center"] #[1400,140
        dic['flat']     = read_hdf5(jason["FlatField"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
        dic['empty']    = read_hdf5(jason['EmptyFrame']).squeeze().astype(np.float32) # numpy.zeros([3072,3072]) 
        
        restored_data_info = pi540D.ioSetM_Backward540D( dic )

    return dic, restored_data_info


def restoration_cat_3d(jason):
    
    ibira_datafolder, scans_string  = jason['proposal_path'],jason['scans_string']

    diffractionpattern = []
    count = -1
    time_difpads = 0

    for acquisitions_folder in jason['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if jason['projections'] != []:
            filepaths, filenames = sscCdi.misc.misc.select_specific_angles(jason['projections'], filepaths,  filenames)
            print('\nMeasurement file in restoration_cat_3d: ', filenames)
        
        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        difpads, time_difpads, _, jason = pi540_restoration_cat_block(params,jason['SaveDifpadPath'],preview,save)

        diffractionpattern.append(difpads)

    return diffractionpattern, time_difpads, jason


def pi540_restoration_cat_block(args, savepath = '', preview = False, save = False):
    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    scans_string        = args[5]
    
    difpads = []
    t0 = time.time()

    first_iteration = True
    for measurement_file, measurement_filepath in zip(filenames, filepaths):

        param = (jason,ibira_datafolder,measurement_file,acquisitions_folder,scans_string,measurement_filepath)
        difpad, elapsedtime_one_difpad, jason = pi540_restoration_cat(param,savepath,preview,save, first_iteration)
        
        if difpads == [] or difpads[0].shape == difpad.shape:
            difpads.append(difpad)
        else:
            difpads.append(np.zeros(difpads[0].shape))

        if first_iteration == True: first_iteration == False

    difpads = np.asarray(difpads)
    print('difpads shape after restoration and binning of', jason['binning'], ':', difpads.shape)
    
    # if save:
    #     np.save(savepath + measurement_file, difpad)

    t1 = time.time()
    elapsedtime = t1-t0

    return difpads, elapsedtime, elapsedtime_one_difpad, jason


def restoration_cat_2d(args,first_run=True):
    jason, acquisition_folder, filename, filepath, geometry = args[0] , args[1], args[2], args[3], args[5]
    ibira_datafolder, scans_string  = jason['proposal_path'],jason['scans_string']

    time_difpads = 0

    params = (jason, ibira_datafolder, filename, acquisition_folder, scans_string, filepath, geometry)
    
    difpads, time_difpads, jason = pi540_restoration_cat(params,jason['SaveDifpadPath'],preview,save,first_iteration=first_run)

    difpads = np.expand_dims(difpads,axis=0)

    return difpads, time_difpads, jason



def _get_center(dbeam, project):
    aDP = pi540D._worker_annotation_image ( np.clip( dbeam, 0, 100) )
    aDP = ndimage.gaussian_filter( aDP, sigma=0.95, order=0 )
    aDP = aDP/aDP.max()
    aDP = 1.0 * ( aDP > 0.98 )    
    u = np.array(range(3072))
    xx,yy = np.meshgrid(u,u)
    xc = ((aDP * xx).sum() / aDP.sum() ).astype(int)
    yc = ((aDP * yy).sum() / aDP.sum() ).astype(int)
    annotation = np.array([ [xc, yc] ])
    tracking = pi540D.annotation_points_standard ( annotation )
    tracking = pi540D.tracking540D_vec_standard ( project, tracking ) 
    xc = int( tracking[0][2] )
    yc = int( tracking[0][3] ) 
    return xc, yc


def pi540_restoration_cat(args, savepath = '', preview = False, save = False, first_iteration = True):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]
    geometry            = args[6]

    t0 = time.time()
    print('Begin restoration')
            
    print('\nMeasurement file in pi540_restoration_cat: ', measurement_file)

    difpads, geometry, _, jason = get_restaurated_difpads_old_format(jason, geometry,os.path.join(ibira_datafolder, acquisitions_folder,scans_string), measurement_file,first_iteration=first_iteration,preview=preview)

    print('Finished restoration')

    t1 = time.time()

    if save:
        np.save( os.path.join(savepath, measurement_file), difpads)

    return difpads, t1-t0, jason


def get_restaurated_difpads_old_format(jason, geometry, path, name,first_iteration,preview):
    """Extracts the data from json and manipulate it according G restoration input format
        Then, call G restoration

    Args:
        jason (json file): json object
        path (list of dtrings): list of complete paths to all files
        name (list of strings): list of all file names

    Returns:
        3D array: restaured difpads
    """    

    fullpath = os.path.join(path, name)
    os.system(f"h5clear -s {fullpath}")
    raw_difpads,_ = io.read_volume(fullpath, 'numpy', use_MPI=True, nprocs=jason["CPUs"])

    if first_iteration:  # preview only 
        print('Raw diffraction pattern shape: ', raw_difpads.shape)
        difpad_number = 0 # selects which difpad to preview
        mean_raw_difpads = np.mean(raw_difpads, axis=0)
        np.save(jason[ 'output_path'] + '/02_difpad_raw_mean.npy',mean_raw_difpads)

    empty = read_hdf5(jason['EmptyFrame']).squeeze().astype(np.float32)

    flat = read_hdf5(jason["FlatField"])[()][0, 0, :, :]

    flat = np.array(flat)
    flat[np.isnan(flat)] = -1
    flat[flat == 0] = -1 # null points at flatfield are indication of bad points

    print('Loading Mask from: ',jason['Mask'])
    mask = read_hdf5(jason["Mask"])[()][0, 0, :, :]
        
    if jason['DP_center'] == []: 
        if 1: # Miqueles's approach for getting diffraction pattern center
            proj  = pi540D.dictionary540D(jason['DetDistance'], {'geo':'nonplanar','opt':True,'mode':'virtual'})
            centerx, centery = _get_center(raw_difpads[0,:,:], proj)
        else: # Yuri's approach for getting diffraction pattern center
            centerx, centery = get_difpad_center(raw_difpads[0,:,:]) #TODO: under test! 
        jason['DP_center'] = (centerx, centery)
        print('Automatic Diffraction Pattern Center :', centery, centerx)
    else:
        centery, centerx = jason['DP_center']
        print('Manual Diffraction Pattern Center :',centerx, centery)

    binning = int(jason['binning'])

    apply_crop = True
    
    if binning != 1:
        apply_binning = True
    else:
        apply_binning = False

    half_square_side = 1536

    if apply_crop:
        L = 3072 # PIMEGA540D size
        if jason["detector_ROI_radius"] == 0:
            half_square_side = min(min(centerx,L-centerx),min(centery,L-centery)) # get the biggest size possible such that the restored difpad is still squared
        else:
            half_square_side = jason["detector_ROI_radius"]
        
        if half_square_side % 2 != 0: 
            half_square_side = half_square_side -  1 # make it even
        DP_shape = 2*half_square_side
    else:
        DP_shape = 3072
        half_square_side = DP_shape // 2

    if apply_binning:
        DP_shape = DP_shape // binning

    r_params = (binning, empty, flat, centerx, centery, half_square_side, geometry, mask, jason, apply_crop, apply_binning, np.zeros_like(raw_difpads[0]),False)

    if first_iteration: # difpad used in jupyter to find center position!
    
        print('Restaurating single difpad to save preview difpad of 3072^2 shape')
        difpad_number = 0
        DP0 = Restorate(raw_difpads[difpad_number,:,:].astype(np.float32), geometry) # restaurate
        DP = corrections_and_restoration(raw_difpads[difpad_number,:,:],empty,flat,np.zeros_like(flat),mask,geometry,jason,apply_crop,centerx,centery,DP_shape,False)
        np.save(jason[ 'output_path'] + '/03_difpad_restaured_flipped.npy',DP0)
        np.save(jason[ 'output_path'] + '/03_difpad_restaured_flipped_masked.npy',DP)

    t0 = time.time()

    output, _ = miscPimega.batch(raw_difpads, jason['CPUs'], [ DP_shape,DP_shape ], restoration_processing_binning,  r_params)
    
    t1 = time.time()

    elapsedtime = t1-t0

    return output, geometry, elapsedtime, jason


def restoration_processing_binning(DP, args):

    binning, empty, flat, cx, cy, hsize, geometry, mask,jason, apply_crop, apply_binning, subtraction_mask, keep_original_negatives = args

    img = corrections_and_restoration(DP,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives)

    img = G_binning(img,apply_binning,binning,mask) # binning strategy by G. Baraldi
    
    return img

def corrections_and_restoration(DP,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives):
    
    DP[empty > 1] = -1 # apply empty 
    
    DP = DP * np.squeeze(flat) # apply flatfield
    DP[flat==-1] = -1 # null values in both the data and in the flat will be disconsidered
    
    DP = DP - subtraction_mask # apply subtraction mask; mask is null when no subtraction is wanted

    DP = DP.astype(np.float32) # convert to float
    
    DP[np.abs(mask) ==1] = -1 # apply Mask
    
    DP = Restorate(DP, geometry) # restaurate

    if keep_original_negatives == False:
        DP[DP < 0] = -1 # all invalid values must be -1 by convention

    if hsize == 0:
        hsize = min(min(cx,DP.shape[1]-cx),min(cy,DP.shape[0]-cy)) # get the biggest size possible such that the restored difpad is still squared
        if hsize % 2 != 0: 
            hsize = hsize -  1 # make it even

    if apply_crop:
        DP = DP[cy - hsize:cy + hsize, cx - hsize:cx + hsize] # select ROI from the center (cx,cy)


    return DP 


def G_binning(DP,apply_binning,binning,mask):

    if apply_binning == False: 
        pass
    else:
        if binning % 2 != 0:
            sys.exit(f"binning = {binning}. Please select an EVEN value for the binning parameters.")
        while binning % 2 == 0 and binning > 0:
            avg = DP + np.roll(DP, -1, -1) + np.roll(DP, -1, -2) + np.roll(np.roll(DP, -1, -1), -1, -2)  # sum 4 neigboors at the top-left value

            div = 1 * (DP >= 0) + np.roll(1 * (DP >= 0), -1, -1) + np.roll(1 * (DP >= 0), -1, -2) + np.roll( np.roll(1 * (DP >= 0), -1, -1), -1, -2)  # Boolean array! Results in the n of valid points in the 2x2 neighborhood

            avg = avg + 4 - div  # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

            avgmask = (DP < 0) & ( div > 0)  # div > 0 means at least 1 neighbor is valid. DP < 0 means top-left values is invalid.

            DP[avgmask] = avg[avgmask] / div[ avgmask]  # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

            DP = DP[:, 0::2] + DP[:, 1::2]  # binning columns
            DP = DP[0::2] + DP[1::2]  # binning lines

            DP[DP < 0] = -1

            DP[div[0::2, 0::2] < 3] = -1  # why div < 3 ? Every neighborhood that had 1 or 2 invalid points is considered invalid?

            binning = binning // 2

        if binning > 1:
            print('Entering binning > 1 only')
            avg = -DP * 1.0 + binning ** 2 - 1
            div = DP * 0
            for j in range(0, binning):
                for i in range(0, binning):
                    avg += np.roll(np.roll(DP, j, -2), i, -1)
                    div += np.roll(np.roll(DP > 0, j, -2), i, -1)

            avgmask = (DP < 0) & (div > 0)
            DP[avgmask] = avg[avgmask] / div[avgmask]

            DPold = DP + 0
            DP = DP[0::binning, 0::binning] * 0
            for j in range(0, binning):
                for i in range(0, binning):
                    DP += DPold[j::binning, i::binning]

            DP[DP < 0] = -1

    return DP