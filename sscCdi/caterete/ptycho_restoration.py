import numpy
import numpy as np
import h5py
from scipy import ndimage, signal
from time import time
import os

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from scipy import ndimage
from .ptycho_processing import get_difpad_center, masks_application

from sscIO import io
import sscCdi
from sscPimega import pi540D
from sscPimega import opt540D
from sscPimega import miscPimega

############################ OLD restoration BY GIOVANNI #####################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# MODULES FOR THE restoration APPLICATION 
# (see main code below)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def Geometry(L,susp=3,scale=0.98,fill=False):
    """ Detector geometry parameters for sscPimega restoration

    Args:
        L : sample-detector distancef
    Returns:
        geo : geometry 
    """    

    project = pi540D.dictionary540D( L, {'geo':'nonplanar','opt':True,'mode':'virtual', 'fill': fill, 'scale': scale, 'susp': susp } ) 
    geo = pi540D.geometry540D( project )
    return geo

def Restaurate(DP, geom):
    return pi540D.backward540D(DP, geom)

def UnRestaurate(DP, geom):
    return opt540D._worker_annotation_image(pi540D.forward540D(DP, geom))


def pi540_restoration_cat_block(args, savepath = '', preview = False, save = False):
    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    scans_string        = args[5]
    
    difpads = []
    t0 = time()

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
    print('difpads shape after restoration and binning of', jason['Binning'], ':', difpads.shape)
    
    # if save:
    #     np.save(savepath + measurement_file, difpad)

    t1 = time()
    elapsedtime = t1-t0

    return difpads, elapsedtime, elapsedtime_one_difpad, jason

def pi540_restoration_cat(args, savepath = '', preview = False, save = False, first_iteration = True):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]
    geometry            = args[6]

    t0 = time()
    print('Begin restoration')
            
    if jason['OldRestauration'] == True: # OldRestauration is Giovanni's
        print('\nMeasurement file in pi540_restoration_cat: ', measurement_file)

        difpads, geometry, _, jason = get_restaurated_difpads_old_format(jason, geometry,os.path.join(ibira_datafolder, acquisitions_folder,scans_string), measurement_file,first_iteration=first_iteration,preview=preview)

        if 1:  # OPTIONAL: exclude first difpad to match with probe_positions_file list
            difpads = difpads[1:]  # TODO: why does this difference of 1 position happens? Fix it!

    else:
        print('Entering Miqueles restoration.')
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
        dic['function'] = sscCdi.caterete.ptycho_restoration.cat_preproc_ptycho_measurement

        difpads, elapsed_time, geometry = sscCdi.caterete.ptycho_restoration.cat_preproc_ptycho_projections(dic)

        jason['RestauredPixelSize'] = geometry['pxlsize']*1e-6

        print('Diffraction pattern shape (post restoration):', difpads.shape)

    if preview: # save plots of restaured difpad and mean of all restaured difpads
        difpad_number = 0
        np.save(jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.npy',np.mean(difpads, axis=0))
        np.save(jason[ 'PreviewFolder'] + '/04_difpad_restaured.npy',difpads[difpad_number, :, :])
        sscCdi.caterete.misc.plotshow_cmap2(difpads[difpad_number, :, :], title=f'Restaured Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/04_difpad_restaured.png')
        sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads, axis=0), title=f'Mean Restaured Diffraction Pattern #{difpad_number}', savepath=jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.png')

    print('Finished restoration')

    t1 = time()
    elapsedtime = t1-t0

    if save:
        np.save( os.path.join(savepath, measurement_file), difpads)

    return difpads, elapsedtime, jason


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
    raw_difpads,_ = io.read_volume(fullpath, 'numpy', use_MPI=True, nprocs=jason["Threads"])

    if first_iteration:  # preview only 
        print('Raw diffraction pattern shape: ', raw_difpads.shape)
        difpad_number = 0 # selects which difpad to preview
        mean_raw_difpads = np.mean(raw_difpads, axis=0)
        np.save(jason[ 'PreviewFolder'] + '/02_difpad_raw_mean.npy',mean_raw_difpads)
    if preview and first_iteration:
        sscCdi.caterete.misc.plotshow_cmap2(raw_difpads[difpad_number, :, :], title=f'Raw Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/02_difpad_raw.png')
        sscCdi.caterete.misc.plotshow_cmap2(mean_raw_difpads, title=f'Raw Diffraction Patterns mean', savepath=jason['PreviewFolder'] + '/02_difpad_raw_mean.png')

    os.system(f"h5clear -s {jason['EmptyFrame']}")
    empty = np.asarray(h5py.File(jason['EmptyFrame'], 'r')['/entry/data/data']).squeeze().astype(np.float32)

    if 'OldFormat' not in jason:
        flat = h5py.File(jason["FlatField"], 'r')['entry/data/data'][()][0, 0, :, :]
    else:
        flat = np.load(jason["FlatField"])

    flat = np.array(flat)
    flat[np.isnan(flat)] = -1
    flat[flat == 0] = -1 # null points at flatfield are indication of bad points

    print('Loading Mask from: ',jason['Mask'])
    if 'OldFormat' in jason:
        if jason['Mask'] != 0:
            mask = np.load(jason['Mask'])
        else:
            mask = np.zeros_like(raw_difpads[0])
    else:
        mask = h5py.File(jason["Mask"], 'r')['entry/data/data'][()][0, 0, :, :]
        # mask = np.flip(mask,0)
        
    if preview and first_iteration:
        sscCdi.caterete.misc.plotshow_cmap2(empty, title=f'Empty', savepath=jason['PreviewFolder'] + '/01_empty.png')
        sscCdi.caterete.misc.plotshow_cmap2(flat,  title=f'Flat',  savepath=jason['PreviewFolder'] + '/01_flat.png')
        sscCdi.caterete.misc.plotshow_cmap2(mask,  title=f'Mask',  savepath=jason['PreviewFolder'] + '/01_mask.png')

    if jason['DifpadCenter'] == []:
        proj  = pi540D.dictionary540D(jason['DetDistance'], {'geo':'nonplanar','opt':True,'mode':'virtual'})
        centerx, centery = _get_center(raw_difpads[0,:,:], proj)
        jason['DifpadCenter'] = (centerx, centery)
        cx, cy = get_difpad_center(raw_difpads[0,:,:]) #TODO: under test! 
        print('Yuri Automatic Difpad Center :', cx, cy)
        print('sscPimega Automatic Difpad Center:',centerx, centery)
    else:
        centery, centerx = jason['DifpadCenter']
        print('Manual Difpad Center :',centerx, centery)


    Binning = int(jason['Binning'])

    apply_crop = True
    
    if Binning != 1:
        apply_binning = True
    else:
        apply_binning = False

    half_square_side = 1536

    if apply_crop:
        L = 3072 # PIMEGA540D size
        if jason["DetectorROI"] == 0:
            half_square_side = min(min(centerx,L-centerx),min(centery,L-centery)) # get the biggest size possible such that the restored difpad is still squared
        else:
            half_square_side = jason["DetectorROI"]
        
        if half_square_side % 2 != 0: 
            half_square_side = half_square_side -  1 # make it even
        DP_shape = 2*half_square_side
    else:
        DP_shape = 3072
        half_square_side = DP_shape // 2

    if apply_binning:
        DP_shape = DP_shape // Binning

    r_params = (Binning, empty, flat, centerx, centery, half_square_side, geometry, mask, jason, apply_crop, apply_binning, np.zeros_like(raw_difpads[0]),False)

    if first_iteration: # difpad used in jupyter to find center position!
    
        print('Restaurating single difpad to save preview difpad of 3072^2 shape')
        difpad_number = 0
        DP0 = Restaurate(raw_difpads[difpad_number,:,:].astype(np.float32), geometry) # restaurate
        DP = corrections_and_restoration(raw_difpads[difpad_number,:,:],empty,flat,np.zeros_like(flat),mask,geometry,jason,apply_crop,centerx,centery,DP_shape,False)
        np.save(jason[ 'PreviewFolder'] + '/03_difpad_restaured_flipped.npy',DP0)
        np.save(jason[ 'PreviewFolder'] + '/03_difpad_restaured_flipped_masked.npy',DP)
        sscCdi.caterete.misc.plotshow_cmap2(DP, title=f'Restaured Diffraction Pattern #{difpad_number}, pre-binning', savepath=jason['PreviewFolder'] + '/03_difpad_restaured_flipped.png')

    t0 = time()
    use_GPU = True
    if use_GPU == True: 
        output = restoration_processing_binning_GPU(raw_difpads, jason, r_params)
    else:  
        output, _ = miscPimega.batch(raw_difpads, jason['Threads'], [ DP_shape,DP_shape ], restoration_processing_binning,  r_params)
    
    t1 = time()

    elapsedtime = t1-t0

    return output, geometry, elapsedtime, jason

def restoration_processing_binning(DP, args):

    Binning, empty, flat, cx, cy, hsize, geometry, mask,jason, apply_crop, apply_binning, subtraction_mask, keep_original_negatives = args

    img = corrections_and_restoration(DP,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives)

    img = G_binning(img,apply_binning,Binning,mask) # binning strategy by G. Baraldi
    
    return img


def restoration_processing_binning_GPU(raw_difpads, jason, args):

    Binning, empty, flat, cx, cy, hsize, geometry, mask,jason, apply_crop, apply_binning, subtraction_mask, keep_original_negatives = args

    geometry["gpu"] = jason["GPUs"][0] # fix to use multiple

    blockSize = 10
    nblocks = raw_difpads.shape[0] // blockSize
    for k in range( nblocks ):
        
        DP_block = raw_difpads[k * nblocks: min( (k+1)*nblocks, raw_difpads.shape[0] ), :, :]

        DP_block = corrections_and_restoration_block(DP_block,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives) # fix inputs

        for i, DP in enumerate(DP_block): # to be optimized!
            DP_block[i] = G_binning(DP,apply_binning,Binning,mask) # binning strategy by G. Baraldi

    return DP_block

def Restaurate_GPU(DP,geom):
    return pi540D.backward540D(DP, geom)

def corrections_and_restoration_block(DP,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives):
    DP[:,empty > 1] = -1 # Apply empty 
    DP = DP * np.squeeze(flat) # Apply flatfield
    DP = DP - subtraction_mask # apply subtraction mask; mask is null when no subtraction is wanted

    DP = DP.astype(np.float32) # convert to float
    
    DP[:,np.abs(mask) ==1] = -1 # Apply Mask
    
    DP = Restaurate_GPU(DP, geometry) # restaurate

    DP[DP < 0] = -1 # all invalid values must be -1 by convention

    if keep_original_negatives == False:
        DP[DP < 0] = -1 # all invalid values must be -1 by convention

    if hsize == 0:
        hsize = min(min(cx,DP.shape[1]-cx),min(cy,DP.shape[0]-cy)) # get the biggest size possible such that the restored difpad is still squared
        if hsize % 2 != 0: 
            hsize = hsize -  1 # make it even

    if apply_crop:
        DP = DP[:,cy - hsize:cy + hsize, cx - hsize:cx + hsize] # select ROI from the center (cx,cy)


    return DP 

def corrections_and_restoration(DP,empty,flat,subtraction_mask,mask,geometry,jason,apply_crop,cx,cy,hsize,keep_original_negatives):
    DP[empty > 1] = -1 # Apply empty 
    DP = DP * np.squeeze(flat) # Apply flatfield
    DP = DP - subtraction_mask # apply subtraction mask; mask is null when no subtraction is wanted

    DP = DP.astype(np.float32) # convert to float
    
    DP[np.abs(mask) ==1] = -1 # Apply Mask
    
    DP = Restaurate(DP, geometry) # restaurate

    DP[DP < 0] = -1 # all invalid values must be -1 by convention

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
        # print("SKIP BINNING")
        pass
    else:
        # Binning
        while binning % 2 == 0 and binning > 0:
            avg = DP + np.roll(DP, -1, -1) + np.roll(DP, -1, -2) + np.roll(np.roll(DP, -1, -1), -1, -2)  # sum 4 neigboors at the top-left value

            div = 1 * (DP >= 0) + np.roll(1 * (DP >= 0), -1, -1) + np.roll(1 * (DP >= 0), -1, -2) + np.roll( np.roll(1 * (DP >= 0), -1, -1), -1, -2)  # Boolean array! Results in the n of valid points in the 2x2 neighborhood

            avg = avg + 4 - div  # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

            avgmask = (DP < 0) & ( div > 0)  # div > 0 means at least 1 neighbor is valid. DP < 0 means top-left values is invalid.

            DP[avgmask] = avg[avgmask] / div[ avgmask]  # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

            DP = DP[:, 0::2] + DP[:, 1::2]  # Binning columns
            DP = DP[0::2] + DP[1::2]  # Binning lines

            DP[DP < 0] = -1

            DP[div[0::2, 0::2] < 3] = -1  # why div < 3 ? Every neighborhood that had 1 or 2 invalid points is considered invalid?

            binning = binning // 2

        while binning % 3 == 0 and binning > 0:
            avg = np.roll(DP, 1, -1) + np.roll(DP, -1, -1) + np.roll(DP, -1, -2) + np.roll(DP, 1, -2) + np.roll( np.roll(DP, 1, -2), 1, -1) + np.roll(np.roll(DP, 1, -2), -1, -1) + np.roll(np.roll(DP, -1, -2), 1, -1) + np.roll( np.roll(DP, -1, -2), -1, -1)
            div = np.roll(DP > 0, 1, -1) + np.roll(DP > 0, -1, -1) + np.roll(DP > 0, -1, -2) + np.roll(DP > 0, 1, -2) + np.roll( np.roll(DP > 0, 1, -2), 1, -1) + np.roll(np.roll(DP > 0, 1, -2), -1, -1) + np.roll( np.roll(DP > 0, -1, -2), 1, -1) + np.roll(np.roll(DP > 0, -1, -2), -1, -1)

            avgmask = (DP < 0) & (div > 0) / div[avgmask]

            DP = DP[:, 0::3] + DP[:, 1::3] + DP[:, 2::3]
            DP = DP[0::3] + DP[1::3] + DP[2::3]

            DP[DP < 0] = -1
            binning = binning // 3

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

def restoration_cat_3d(args):
    
    ibira_datafolder, scans_string  = jason['ProposalPath'],jason['scans_string']
    preview,save, read = jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads']

    diffractionpattern = []
    count = -1
    time_difpads = 0

    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)
            print('\nMeasurement file in restoration_cat_3d: ', filenames)
        
        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        if read:
            n = len(filenames)
            difpads = []
            for i in range(n):
                difpad = np.load( os.path.join(jason['SaveDifpadPath'], filenames[i] + '.npy'))
                difpads.append(difpad)
            difpads = np.asarray(difpads)
        else: 
            difpads, time_difpads, _, jason = pi540_restoration_cat_block(params,jason['SaveDifpadPath'],preview,save)


        diffractionpattern.append(difpads)

    return diffractionpattern, time_difpads, jason


def restoration_cat_2d(args,first_run=True):

    jason, acquisition_folder, filename, filepath, geometry = args[0] , args[1], args[2], args[3], args[5]
    ibira_datafolder, scans_string  = jason['ProposalPath'],jason['scans_string']
    preview,save, read = jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads']

    time_difpads = 0

    params = (jason, ibira_datafolder, filename, acquisition_folder, scans_string, filepath, geometry)
    
    if read:
        difpads = np.load( os.path.join(jason['SaveDifpadPath'],filename + '.npy'))
    else:   
        difpads, time_difpads, jason = pi540_restoration_cat(params,jason['SaveDifpadPath'],preview,save,first_iteration=first_run)

    difpads = np.expand_dims(difpads,axis=0)

    return difpads, time_difpads, jason

def inpaint_lonely_neighbors(DP):
    valids_mask = np.where(DP > 0 , 1 , -1)
    sum_of_mask_neighbors = np.roll(valids_mask,1,0) + np.roll(valids_mask,-1,0) + np.roll(valids_mask,1,1) + np.roll(valids_mask,-1,1) + np.roll(np.roll(valids_mask,1,0),1,1) + np.roll(np.roll(valids_mask,1,0),-1,1) + np.roll(np.roll(valids_mask,-1,0),1,1) + np.roll(np.roll(valids_mask,-1,0),-1,1)
    sum_of_neighbors = np.roll(DP,1,0) + np.roll(DP,-1,0) + np.roll(DP,1,1) + np.roll(DP,-1,1) + np.roll(np.roll(DP,1,0),1,1) + np.roll(np.roll(DP,1,0),-1,1) + np.roll(np.roll(DP,-1,0),1,1) + np.roll(np.roll(DP,-1,0),-1,1)
    pixels_to_paint = np.where(sum_of_mask_neighbors == 8,True,False)
    DP_corrected = DP.copy()
    DP_corrected[pixels_to_paint] = sum_of_neighbors[pixels_to_paint] / 8
    return DP_corrected

################# MIQUELES restoration ###################################

def cat_preproc_ptycho_measurement( data, args ):
    """ Miqueles function for new restoration approach. Passed as an argument (part of dictionary) to cat_preproc_ptycho_projections 

    Args:
        data (_type_): _description_
        args (_type_): _description_
    Returns:
        backroi : restaured diffraction pattern with binning
    """    

    def _get_center(dbeam, project):
        aDP = pi540D._worker_annotation_image ( numpy.clip( dbeam, 0, 100) )
        aDP = ndimage.gaussian_filter( aDP, sigma=0.95, order=0 )
        aDP = aDP/aDP.max()
        aDP = 1.0 * ( aDP > 0.98 )    
        u = numpy.array(range(3072))
        xx,yy = numpy.meshgrid(u,u)
        xc = ((aDP * xx).sum() / aDP.sum() ).astype(int)
        yc = ((aDP * yy).sum() / aDP.sum() ).astype(int)
        annotation = numpy.array([ [xc, yc] ])
        tracking = pi540D.annotation_points_standard ( annotation )
        tracking = pi540D.tracking540D_vec_standard ( project, tracking ) 
        xc = int( tracking[0][2] )
        yc = int( tracking[0][3] ) 
        return xc, yc
    def _operator_T(u):
        d   = 1.0
        uxx = (numpy.roll(u,1,1) - 2 * u + numpy.roll(u,-1,1) ) / (d**2)
        uyy = (numpy.roll(u,1,0) - 2 * u + numpy.roll(u,-1,0) ) / (d**2)
        uyx = (numpy.roll(numpy.roll(u,1,1),1,1) - numpy.roll(numpy.roll(u,1,1),-1,0) - numpy.roll(numpy.roll(u,1,0),-1,1) + numpy.roll(numpy.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
        uxy = (numpy.roll(numpy.roll(u,1,1),1,1) - numpy.roll(numpy.roll(u,-1,1),1,0) - numpy.roll(numpy.roll(u,-1,0),1,1) + numpy.roll(numpy.roll(u,-1,1),-1,0)   )/ (2 * d**2)
        delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
        z = numpy.sqrt( delta )
        return z
    def _get_roi( DP, roi, center,binning):
        xc, yc = center
        X = DP[yc-roi:yc+roi:binning,xc-roi:xc+roi:binning]
        return X
    def set_binning(data, binning):
        if binning>0:
            # Define kernel for convolution                                         
            kernel = numpy.ones([binning,binning]) 
            # Perform 2D convolution with input data and kernel 
            X = signal.convolve2d(data, kernel, mode='same')/kernel.sum()
        else:
            X = data
        return X

    ######
    dic   = args[0]
    geo   = args[1]
    proj  = args[2]
    empty = args[3]
    flat  = args[4]
    xc, yc = _get_center(data, proj)

    #operation on the detector domain:
    flat[flat == 0] = 1 
    flat[numpy.isnan(flat)] = 1
    data = data*flat # Flatfield application. Convention with DET group is a product between data and flat!
    
    if 0:
        data[empty > 0] = _operator_T(data).real[ empty > 0] # remove bad datapoints with Miquele's operator
    else:
        data[empty > 0] = -1

    back  = pi540D.backward540D ( data , geo)
    where = (back == -1)
    back = set_binning ( back, dic['binning'] )
    back[where] = -1
    
    backroi  = _get_roi( back, dic['roi'], [xc, yc],dic['binning'])
    where = _get_roi( where, dic['roi'], [xc, yc],dic['binning'])
    backroi[where] = -1

    return backroi
##############
def cat_preproc_ptycho_projections( dic ):
    """ Miqueles' function call to new restoration approach

    Args:
        dic (_type_): _description_

    Returns:
        output : restaured diffraction patterns
        elapsed_time : restoration time
    """    
    #-----------------------
    #read data using ssc-io:
    empty     = h5py.File(dic['empty'], 'r')['entry/data/data/'][0,0,:,:]
    measure,_ = io.read_volume(dic['data'],'numpy', use_MPI=True, nprocs=32)
    flat      = numpy.load(dic['flat']) # flatfield file needs to be a numpy!
    #------------------------------------
    # computing ssc-pimega 540D geometry:
    xdet     = pi540D.get_project_values_geometry()
    project  = pi540D.dictionary540D( xdet, dic['distance'] )
    project['s'] = [dic['susp'],dic['susp']]
    geometry = pi540D.geometry540D( project )
    #-------------------------------------------------------------------------------
    # applying the input dic['function'] to the ptychographic sequence measured (2d)
    # -> function must include restoration + all preprocessing details
    params = (dic, geometry, project, empty, flat )
    start = time()
    output,_ = pi540D.backward540D_batch( measure[1:], dic['distance'], dic['nproc'], [2*dic['roi']//dic['binning'], 2*dic['roi']//dic['binning']], dic['function'], params, dic['order'] )
    elapsed = time() - start
    return output, elapsed,geometry
#################
