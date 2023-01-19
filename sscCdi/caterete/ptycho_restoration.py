import numpy
import h5py
from scipy import ndimage, signal
from time import time
import os

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from scipy import ndimage
from sscCdi.caterete.ptycho_processing import *

from sscIO import io
import sscCdi
from sscPimega import pi540D
from sscPimega import opt540D

############################ OLD RESTAURATION BY GIOVANNI #####################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# MODULES FOR THE RESTAURATION APPLICATION 
# (see main code below)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def Geometry(L):
    """ Detector geometry parameters for sscPimega restauration

    Args:
        L : sample-detector distancef
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
    for measurement_file, measurement_filepath in zip(filenames, filepaths):

        param = (jason,ibira_datafolder,measurement_file,acquisitions_folder,scans_string,measurement_filepath)
        difpad, elapsedtime_one_difpad, jason = pi540_restauration_cat(param,savepath,preview,save, first_iteration)
        
        if difpads == [] or difpads[0].shape == difpad.shape:
            difpads.append(difpad)
        else:
            difpads.append(np.zeros(difpads[0].shape))

        if first_iteration == True: first_iteration == False

    difpads = np.asarray(difpads)
    print('difpads shape after restauration and binning of', jason['Binning'], ':', difpads.shape)
    
    # if save:
    #     np.save(savepath + measurement_file, difpad)

    t1 = time()
    elapsedtime = t1-t0

    return difpads, elapsedtime, elapsedtime_one_difpad, jason

def pi540_restauration_cat(args, savepath = '', preview = False, save = False, first_iteration = True):
    
    jason               = args[0]
    ibira_datafolder    = args[1]
    measurement_file    = args[2]
    acquisitions_folder = args[3]
    scans_string        = args[4]
    measurement_filepath= args[5]

    t0 = time()
    print('Begin Restauration')
            
    if jason['OldRestauration'] == True: # OldRestauration is Giovanni's
        print('\nMeasurement file in pi540_restauration_cat: ', measurement_file)
        difpads, geometry, _, jason = get_restaurated_difpads_old_format(jason, os.path.join(ibira_datafolder, acquisitions_folder,scans_string), measurement_file,first_iteration=first_iteration,preview=preview)

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
        dic['function'] = sscCdi.caterete.ptycho_restoration.cat_preproc_ptycho_measurement

        difpads, elapsed_time, geometry = sscCdi.caterete.ptycho_restoration.cat_preproc_ptycho_projections(dic)

        jason['RestauredPixelSize'] = geometry['pxlsize']*1e-6

        print('Diffraction pattern shape (post restauration):', difpads.shape)

    if preview: # save plots of restaured difpad and mean of all restaured difpads
        difpad_number = 0
        np.save(jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.npy',np.mean(difpads, axis=0))
        np.save(jason[ 'PreviewFolder'] + '/04_difpad_restaured.npy',difpads[difpad_number, :, :])
        sscCdi.caterete.misc.plotshow_cmap2(difpads[difpad_number, :, :], title=f'Restaured Diffraction Pattern #{difpad_number}', savepath=jason['PreviewFolder'] + '/04_difpad_restaured.png')
        sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads, axis=0), title=f'Mean Restaured Diffraction Pattern #{difpad_number}', savepath=jason[ 'PreviewFolder'] + '/04_difpad_restaured_mean.png')

    print('Finished Restauration')

    t1 = time()
    elapsedtime = t1-t0

    if save:
        np.save( os.path.join(savepath, measurement_file), difpads)

    return difpads, elapsedtime, jason


def get_restaurated_difpads_old_format(jason, path, name,first_iteration,preview):
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

    z1 = float(jason["DetDistance"]) * 1000  # Here comes the distance Geometry(Z1):
    geometry = Geometry(z1)

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
        proj  = pi540D.get_detector_dictionary(jason['DetDistance'], {'geo':'nonplanar','opt':True,'mode':'virtual'})
        centerx, centery = _get_center(raw_difpads[0,:,:], proj)
        jason['DifpadCenter'] = (centerx, centery)
        cx, cy = sscCdi.ptycho_processing.get_difpad_center(raw_difpads[0,:,:]) #TODO: under test! 
        print('Yuri Automatic Difpad Center :', cx, cy)
        print('sscPimega Automatic Difpad Center:',centerx, centery)
    else:
        centery, centerx = jason['DifpadCenter']
        print('Manual Difpad Center :',centerx, centery)


    Binning = int(jason['Binning'])

    apply_crop = True

    if apply_crop:
        cropsize = jason['DetectorROI']   
    else:
        cropsize = 2*1536

    if Binning != 1:
        hsize = cropsize
        apply_binning = True
    else: 
        hsize = 2*2560
        apply_binning = False

    r_params = (Binning, empty, flat, centerx, centery, cropsize, geometry, mask, jason, apply_crop, apply_binning, np.ones_like(raw_difpads[0]))

    if first_iteration: # difpad used in jupyter to find center position!
        print('Restaurating single difpad to save preview difpad of 3072^2 shape')
        difpad_number = 0
        img = Restaurate(raw_difpads[difpad_number,:,:].astype(np.float32), geometry) # restaurate
        np.save(jason[ 'PreviewFolder'] + '/03_difpad_restaured_flipped.npy',img)
        sscCdi.caterete.misc.plotshow_cmap2(img, title=f'Restaured Diffraction Pattern #{difpad_number}, pre-binning', savepath=jason['PreviewFolder'] + '/03_difpad_restaured_flipped.png')

    t0 = time()
    output, _ = pi540D.backward540D_nonplanar_batch(raw_difpads, z1, jason['Threads'], [ hsize//2 , hsize//2 ], restauration_processing_binning,  r_params, 'only')
    t1 = time()

    elapsedtime = t1-t0

    return output, geometry, elapsedtime, jason

def restauration_processing_binning(img, args):
    """Restaurate and process the binning on the diffraction patterns

    Args:
        img (array): image to be restaured and binned
    """    

    Binning, empty, flat, cx, cy, hsize, geometry, mask,jason, apply_crop, apply_binning, subtraction_mask = args

    binning = Binning + 0
    img[empty > 1] = -1 # Apply empty 
    img = img * np.squeeze(flat) # Apply flatfield
    img = img - subtraction_mask # apply subtraction mask; mask is null when no subtraction is wanted

    img = img.astype(np.float32) # convert to float
    
    unbinned_mask = True
    if unbinned_mask: # if mask before restauration with 3072x3072 size
        img[np.abs(mask) ==1] = -1 # Apply Mask
    
    img = Restaurate(img, geometry) # restaurate

    img[img < 0] = -1 # all invalid values must be -1 by convention

    img = sscCdi.ptycho_processing.masks_application(img,jason)

    if apply_crop:
        # select ROI from the center (cx,cy)
        img = img[cy - hsize:cy + hsize, cx - hsize:cx + hsize] 

    if apply_binning == False:
        print("SKIP BINNING")
    else:
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

        if unbinned_mask == False: # if mask after restauration with 640x640 size
            img[np.abs(mask) ==1] = -1 # Apply Mask

        t1 = time()

    return img

def restauration_cat_3d(args):
    
    ibira_datafolder, scans_string  = jason['ProposalPath'],jason['scans_string']
    preview,save, read = jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads']

    diffractionpattern = []
    count = -1
    time_difpads = 0

    for acquisitions_folder in jason['Acquisition_Folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restauration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder,scans_string), look_for_extension=".hdf5")
        
        if jason['Projections'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Projections'], filepaths,  filenames)
            print('\nMeasurement file in restauration_cat_3d: ', filenames)
        
        params = (jason, filenames, filepaths, ibira_datafolder, acquisitions_folder, scans_string)

        if read:
            n = len(filenames)
            difpads = []
            for i in range(n):
                difpad = np.load( os.path.join(jason['SaveDifpadPath'], filenames[i] + '.npy'))
                difpads.append(difpad)
            difpads = np.asarray(difpads)
        else: 
            difpads, time_difpads, _, jason = pi540_restauration_cat_block(params,jason['SaveDifpadPath'],preview,save)


        diffractionpattern.append(difpads)

    return diffractionpattern, time_difpads, jason


def restauration_cat_2d(args,first_run=True):

    jason, acquisition_folder, filename, filepath = args[0] , args[1], args[2], args[3]
    ibira_datafolder, scans_string  = jason['ProposalPath'],jason['scans_string']
    preview,save, read = jason['PreviewGCC'][0],jason['SaveDifpads'],jason['ReadRestauredDifpads']

    time_difpads = 0

    params = (jason, ibira_datafolder, filename, acquisition_folder, scans_string, filepath)
    
    if read:
        difpads = np.load( os.path.join(jason['SaveDifpadPath'],filename + '.npy'))
    else:   
        difpads, time_difpads, jason = pi540_restauration_cat(params,jason['SaveDifpadPath'],preview,save,first_iteration=first_run)

    difpads = np.expand_dims(difpads,axis=0)

    return difpads, time_difpads, jason


################# MIQUELES RESTAURATION ###################################

def cat_preproc_ptycho_measurement( data, args ):
    """ Miqueles function for new restauration approach. Passed as an argument (part of dictionary) to cat_preproc_ptycho_projections 

    Args:
        data (_type_): _description_
        args (_type_): _description_
    Returns:
        backroi : restaured diffraction pattern with binning
    """    

    def _get_center(dbeam, project):
        aimg = pi540D._worker_annotation_image ( numpy.clip( dbeam, 0, 100) )
        aimg = ndimage.gaussian_filter( aimg, sigma=0.95, order=0 )
        aimg = aimg/aimg.max()
        aimg = 1.0 * ( aimg > 0.98 )    
        u = numpy.array(range(3072))
        xx,yy = numpy.meshgrid(u,u)
        xc = ((aimg * xx).sum() / aimg.sum() ).astype(int)
        yc = ((aimg * yy).sum() / aimg.sum() ).astype(int)
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
    def _get_roi( img, roi, center,binning):
        xc, yc = center
        X = img[yc-roi:yc+roi:binning,xc-roi:xc+roi:binning]
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
    """ Miqueles' function call to new restauration approach

    Args:
        dic (_type_): _description_

    Returns:
        output : restaured diffraction patterns
        elapsed_time : restauration time
    """    
    #-----------------------
    #read data using ssc-io:
    empty     = h5py.File(dic['empty'], 'r')['entry/data/data/'][0,0,:,:]
    measure,_ = io.read_volume(dic['data'],'numpy', use_MPI=True, nprocs=32)
    flat      = numpy.load(dic['flat']) # flatfield file needs to be a numpy!
    #------------------------------------
    # computing ssc-pimega 540D geometry:
    xdet     = pi540D.get_project_values_geometry()
    project  = pi540D.get_detector_dictionary( xdet, dic['distance'] )
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
