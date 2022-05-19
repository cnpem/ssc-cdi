import sscResolution
import sscPtycho
import sscCdi

import os
from time import time
import h5py
import pandas as pd
import json
import numpy as np
import math
import uuid
import SharedArray as sa
import multiprocessing
import multiprocessing.sharedctypes

import matplotlib.pyplot as plt

from numpy.fft import fftshift as shift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2


# +++++++++++++++++++++++++++++++++++++++++++++++++
#
#
#
# MODULES FOR THE FINAL APPLICATION 
# (see main code below)
#
#
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

def crop_sinogram(sinogram, jason): 

    min_crop_value = []

    if jason['AutoCrop'] == True: # automatically crop borders with noise
        for frame in range(sinogram.shape[0]):
            best_crop = auto_crop_noise_borders(sinogram[frame,:,:])
            min_crop_value.append(best_crop)

        min_crop = min(min_crop_value)
        sinogram = sinogram[:, min_crop:-min_crop-1, min_crop:-min_crop-1]

        if sinogram.shape[1] % 2 != 0:  # object array must have even number of pixels to avoid bug during the phase unwrapping later on
            sinogram = sinogram[:,0:-1, :]
        if sinogram.shape[2] % 2 != 0:
            sinogram = sinogram[:,:, 0:-1]

    return sinogram

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
        #TODO: insert non_negativity and remove_gradient optionals in the json input? We do not understand why they are needed yet!
    
    phase = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))
    absol = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))

    for frame in range(sinogram.shape[0]):
        original_object = sinogram[frame,:,:]  # create copy of object
        absol[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.abs(sscPtycho.RemovePhaseGrad(sinogram[frame,:,:])), n_iterations, non_negativity=0, remove_gradient=0)
        phase[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.angle(sscPtycho.RemovePhaseGrad(sinogram[frame,:,:])), n_iterations, non_negativity=0, remove_gradient=0)
        # sinogram[frame,:,:] = absolute * np.exp(-1j * angle)

        if 1:  # plot original and cropped object phase and save!
            figure, subplot = plt.subplots(1, 2,dpi=300,figsize=(5,5))
            subplot[0].imshow(-np.angle(original_object),cmap='gray')
            subplot[1].imshow(phase[frame,:,:],cmap='gray')
            subplot[0].set_title('Original')
            subplot[1].set_title('Cropped and Unwrapped')
            figure.savefig(jason['PreviewFolder'] + f'/autocrop_and_unwrap_{frame}.png')

    return phase,absol


def calculate_FRC(sinogram, jason):

    object_pixel_size = jason["object_pixel"] 

    frame = 0 # selects first frame of the sinogram to calculate resolution

    if jason['FRC'] == True:
        print('Estimating resolution via Fourier Ring Correlation')
        resolution = resolution_frc(sinogram[frame,:,:], object_pixel_size, plot=True,plot_output_folder=jason["PreviewFolder"],savepath=jason["PreviewFolder"])
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
    """Read probe positions from .txt data file

    Args:
        probe_positions_filepath (string): path to file storing the probe positions
        measurement (string): path to measurement folder

    Returns:
        probe_positions: array, each item is an array with [x position, y position, 1, 1]
    """    
    print('Reading probe positions (probe_positions)...')
    probe_positions = []
    positions_file = open(probe_positions_filepath)

    line_counter = 0
    for line in positions_file:
        line = str(line)
        if line_counter > 1:  # skip first line, which is the header
            T = -3E-3  # why rotate by this amount?
            pxl = float(line.split()[1])
            pyl = float(line.split()[0])
            px = pxl * np.cos(T) - np.sin(T) * pyl
            py = pxl * np.sin(T) + np.cos(T) * pyl
            probe_positions.append([px, py, 1, 1])
        line_counter += 1

    probe_positions = np.asarray(probe_positions)

    pshape = pd.read_csv(probe_positions_filepath,sep=' ').shape  # why read pshape from file? can it be different from probe_positions.shape+1?

    with h5py.File(measurement, 'r') as file:
        mshape = file['entry/data/data'].shape

    if pshape[0] == mshape[0]:  # check if number of recorded beam positions in txt matches the positions saved to the hdf
        print('\tSuccess in read positions file:' + probe_positions_filepath)
        print("\tShape probe_positions:", probe_positions.shape, pshape, mshape)
    else:
        print("\tError in probe_positions shape. {0} is different from diffraction pattern shape {1}".format(probe_positions.shape, mshape))
        print('\npshape: ', pshape)
        print('\t\t Setting object as null array with correct shape.')
        # probe_positions = np.zeros([1,1,1,1])
        probe_positions = np.zeros((mshape[0], 4))
        print('teste',probe_positions.shape)
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
    hsize = difpads.shape[-1] // 2

    if jason['f1'] == -1:  # Manually choose wether to find Fresnel number automatically or not
        jason['f1'] = setfresnel(dx, pixel=jason['RestauredPixelSize'], energy=jason['Energy'], z=jason['DetDistance'])
        jason['f1'] = -jason['f1']
    print('\tF1 value:', jason['f1'])

    # Compute probe: initial guess:
    probe = set_initial_probe(difpads, jason)

    # Adicionar modulos incoerentes
    probe = set_modes(probe, jason)

    # GPUs selection:
    # set_gpus(jason)

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
    probesupp = probe_support(probe, hsize, radius, center_x, center_y)

    probe_positionsi = probe_positions + 0  # what's the purpose of declaring probe_positionsi?

    # Set data for Ptycho algorithms:
    datapack = set_datapack(obj, probe, probe_positions, difpads, background, probesupp)

    return datapack, probe_positionsi, sigmask


def set_object_pixel_size(jason,hsize):
    c = 299792458  # Velocity of Light [m/s]
    planck = 4.135667662E-18  # Plank constant [keV*s]
    wavelength = planck * c / jason['Energy'] # meters
    jason["wavelength"] = wavelength
    # Compute/convert pixel size:
    dx = wavelength * jason['DetDistance'] / ( jason['Binning'] * jason['RestauredPixelSize'] * hsize * 2)

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
        probe = np.load(jason['InitialProbe'])[0]

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
        hsize (int): half size of the object
        maxroi (int): size of the padding object
        probe (array): probe
        difpads (array): measured data

    Returns:
        obj (array)
    """    
    print('Setting initial guess for Object...')
    # Object initial guess:
    if jason['InitialObj'] == "":
        obj = np.random.rand(object_shape[0], object_shape[1]) * (
            np.sqrt(np.average(difpads) / np.average(abs(np.fft.fft2(probe)) ** 2)))
        # obj = np.random.rand(2048,2048) * (np.sqrt(np.average(difpads)/np.average(abs(np.fft.fft2(probe))**2)))
    else:
        obj = np.load(jason['InitialObj'])

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

def probe_support(probe, hsize, radius, center_x, center_y):
    """Create a support for probe

    Args:
        probe (array): initial guess for the probe
        hsize (int): half difraction pattern size
        radius (): probe support radius
        center_x (int): probe support center in x
        center_y (int): probe support center in y

    Returns:
        probesupp (array): probe support
    """    
    print('Setting probe support...')
    # Compute probe support:
    ar = np.arange(-hsize, hsize)
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
    print(f'Saving variable {predefined_name}...')
    print(len(variable))
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

    print('\t', savename, variable.shape)

def save_variable2(variable, predefined_name, savename=""):
    """ Function to save reconstruction object, probe and/or background. 
    
    This function presents some redundancy. Should be improved!

    Args:
        variable : variable to be saved (e.g. sinogram, probe reconstruction and/or background)
        predefined_name: predefined name for saving the output variable
        savename (str, optional): Name to be used instead of predefined_name. Defaults to "".
    """    
    print(f'Saving variable {predefined_name}...')
    print(len(variable))
    variable = np.asarray(variable, dtype=object)
    # for i in range(variable.shape[0]):
    #     print('shapes', variable[i].shape)
    for i in range(variable.shape[0]):  # loop to circumvent problem with nan values
        if math.isnan(variable[i][:, :].sum()):
            variable[i][:, :] = np.zeros(variable[i][:, :].shape)

    variable = np.asarray(variable, dtype=np.float32)

    if savename != "":
        np.save(savename, variable)
    else:
        np.save(predefined_name, variable)

    print('\t', savename, variable.shape)


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

def create_directory_if_doesnt_exist(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

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
    if jason["ObjPath"] != "":
        create_directory_if_doesnt_exist(jason["ObjPath"])
    if jason["ProbePath"] != "":
        create_directory_if_doesnt_exist(jason["ProbePath"])
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

def set_object_shape(difpads,args,offset_topleft = 20):

    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    ibira_datafolder    = args[3]
    acquisitions_folder = args[4]
    positions_string    = args[6]

    # Pego a PRIMEIRA medida de posicao, supondo que ela nao tem erro
    measurement_file = filenames[0]
    measurement_filepath = filepaths[0]
    
    # Compute half size of diffraction patterns:
    hsize = difpads.shape[-1] // 2

    # Compute/convert pixel size:
    dx, jason = set_object_pixel_size(jason,hsize)

    probe_positions_file = os.path.join(acquisitions_folder, positions_string, measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
    probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)
    probe_positions, offset_bottomright = convert_probe_positions(dx, probe_positions, offset_topleft = 20)

    if 0: #TODO: test to compute object of rectangular size
        maxroiy       = int(np.max(probe_positions[:, 0])) + offset_bottomright
        maxroix       = int(np.max(probe_positions[:, 1])) + offset_bottomright
        object_shapey = 2 * hsize + maxroiy
        object_shapex = 2 * hsize + maxroix

    maxroi        = int(np.max(probe_positions)) + offset_bottomright
    object_shape  = 2 * hsize + maxroi
    print('Object shape:',object_shape,object_shape)
    # print(f'\tmaxroi: {np.max(probe_positions)}, int(maxroi):{maxroi}')

    return object_shape,object_shape, maxroi, hsize, dx, jason

def ptycho_main(difpads, sinogram, probe3d, backg3d, args, _start_, _end_, gpu):
    t0 = time()
    
    jason               = args[0][0]
    filenames           = args[0][1]
    filepaths           = args[0][2]
    ibira_datafolder    = args[0][3]
    acquisitions_folder = args[0][4]
    scans_string        = args[0][5]
    positions_string    = args[0][6]
    maxroi              = args[1]
    hsize               = args[2]
    object_shape        = args[3]

    for i in range(_end_ - _start_):
    # for measurement_file, measurement_filepath in zip(filenames, filepaths):  # loop through each hdf5, one for each sample angle
        
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

            print(f'Object shape: {object_shape}. Detector half-size: {hsize}')

            datapack, _, sigmask = set_initial_parameters(jason,difpads[frame],probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,object_shape,jason["object_pixel"])

            param = {'device':gpu}

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
                                                    probef1=jason['f1'], data=datapack,params=param)

                    elif algorithm['Name'] == 'positioncorrection':
                        datapack['bkg'] = None
                        datapack = sscPtycho.PosCorrection(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                               probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], 
                                                               epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                               probef1=jason['f1'], data=datapack,params=param)

                    elif algorithm['Name'] == 'RAAR':
                        datapack = sscPtycho.RAAR(iter=algorithm['Iterations'], beta=algorithm['Beta'],
                                                      probecycles=algorithm['ProbeCycles'], batch=algorithm['Batch'],
                                                      epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'],
                                                      sigmask=sigmask, probef1=jason['f1'], data=datapack,params=param)

                    elif algorithm['Name'] == 'GLL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                    probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'],
                                                    epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                    probef1=jason['f1'], data=datapack,params=param)

                    loop_counter += 1
                    RF = datapack['error']

            print('Original object shape:', datapack['obj'].shape)

            if i == 0: t4 = time()

            sinogram[frame, :, :] = datapack['obj']  # build 3D Sinogram
            probe3d[frame, :, :]  = datapack['probe']
            backg3d[frame, :, :]  = datapack['bkg']

        else:
            print('CAUTION! Zeroing frame:',frame,' for error in position file.')
            sinogram[frame, :, :]   = np.zeros((object_shape[0],object_shape[1])) # build 3D Sinogram
            probe3d[frame, :, :, :] = np.zeros((1,difpads.shape[-2],difpads.shape[-1]))
            backg3d[frame, :, :]    = np.zeros((difpads.shape[-2],difpads.shape[-1]))

        if i == 0: t5 = time()

    # print(f'\nElapsed time for reconstruction of 1st frame: {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes')
    # print(f'Total time iteration: {t5 - t0:.2f} seconds = {(t5 - t0) / 60:.2f} minutes')

    return sinogram, probe3d, backg3d

def _worker_batch_frames_(params, idx_start, idx_end, gpu):
    
    output_object = params[0]
    output_probe  = params[1]
    output_backg  = params[2]
    difpads       = params[3]
    args          = params[5]
    
    _start_ = idx_start
    _end_   = idx_end

    output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:] = ptycho_main( difpads[_start_:_end_,:,:,:], output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:], args, _start_, _end_, gpu)
    
def _build_batch_of_frames_(params):

    total_frames = params[5][4]
    threads      = params[4]
    
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
    

def ptycho3d_batch( difpads, threads, args):
    
    name         = str( uuid.uuid4())
    name1        = str( uuid.uuid4())
    name2        = str( uuid.uuid4())

    object_shape = args[3]
    total_frames = args[4]

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

    _params_ = ( output_object, output_probe, output_backg, difpads, threads, args)
    
    _build_batch_of_frames_ ( _params_ )

    sa.delete(name)
    sa.delete(name1)
    sa.delete(name2)

    return output_object,output_probe,output_backg

