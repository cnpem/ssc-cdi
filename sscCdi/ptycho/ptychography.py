

import numpy as np
import cupy as cp
import sys, os, h5py
import sscPtycho
from ..misc import estimate_memory_usage, add_to_hdf5_group, concatenate_array_to_h5_dataset, wavelength_from_energy

def call_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):

    if 'algorithms' in input_dict:
        obj, probe, error, positions = call_GCC_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)
    else:
        obj, probe, error, positions = call_GB_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)

    return obj, probe, error, positions

def call_GCC_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):
    """ Ptychography algorithms in Python by GCC

    Args:
        input_dict (_type_): _description_
        DPs (_type_): _description_
        positions (_type_): _description_
        initial_obj (_type_, optional): _description_. Defaults to None.
        initial_probe (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if initial_probe == None:
        probe = set_initial_probe(input_dict, (DPs.shape[1], DPs.shape[2]), np.average(DPs, 0)[None] ) # probe initial guess
    if initial_obj == None:
        obj = set_initial_object(input_dict,DPs,probe[0]) # object initial guess
        obj = np.expand_dims(obj,axis=0)

    positions = positions.astype(np.int32)
    positions = np.roll(positions,shift=1,axis=1) # adjusting to the same standard as GB ptychography
    
    error = np.empty((0,))

    inputs = {}
    for counter in range(1,1+len(input_dict['algorithms'].keys())):

        inputs['iterations'] = input_dict['algorithms'][str(counter)]['iterations'] 
        inputs["n_of_modes"] = input_dict['incoherent_modes']
        inputs['object_pixel'] = input_dict['object_pixel']
        inputs['wavelength'] = input_dict['wavelength']
        inputs['distance'] = input_dict['detector_distance']
        inputs["position_rotation"] = input_dict["position_rotation"]
        inputs["object_padding"] = input_dict["object_padding"]
        inputs['regularization_object'] = input_dict['algorithms'][str(counter)]['regularization_object'] 
        inputs['regularization_probe']  = input_dict['algorithms'][str(counter)]['regularization_probe'] 
        inputs['step_object']= input_dict['algorithms'][str(counter)]['step_object'] 
        inputs['step_probe'] = input_dict['algorithms'][str(counter)]['step_probe'] 
        # POSITION CORRECTION. TO BE DONE.
        inputs['position_correction_beta'] = 0 # if 0, does not apply position correction
        inputs['beta'] = 1 # position correction beta value
        inputs['epsilon'] = 0.001 # small value to add to probe/object update denominator
        # inputs['centralize_probe'] = False # not implemented 


        if input_dict["algorithms"][str(counter)]['name'] == 'ePIE_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of ePIE algorithm...")
            inputs['use_mPIE'] = False # friction and momentum counter only relevant if this is True
            inputs['friction_object'] = input_dict['algorithms'][str(counter)]['mPIE_friction_obj'] 
            inputs['friction_probe'] = input_dict['algorithms'][str(counter)]['mPIE_friction_probe'] 
            inputs['momentum_counter'] = input_dict['algorithms'][str(counter)]['mPIE_momentum_counter'] 
            obj, probe, algo_error = PIE_multiprobe_loop(DPs, positions,obj[0],probe[0], inputs)

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            obj, probe, algo_error = RAAR_multiprobe_parallel(DPs, positions,obj[0],probe[0],inputs,probe_support=None,processes=96)
            obj = np.expand_dims(obj,axis=0) # obj coming with one dimensions less. needs to be fixed
        else:
            sys.exit('Please select a proper algorithm! Selected: ', inputs["algorithm"])

        error = np.concatenate((error,algo_error),axis=0)


    return obj, probe, error, None


def call_GB_ptychography(input_dict,DPs, probe_positions, initial_obj=None, initial_probe=None):
    """ Call Ptychography CUDA codes developed by Giovanni Baraldi

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "hdf5_output": output file path/name
                "GPUs": number of GPUs
                "CPUs": number of CPUs
                "fresnel_number": Fresnel number

        DPs (numpy array): array of diffraction patterns of shape (N,Y,X)
        probe_positions (numpy array): array of probe positions of shape (N,2)
        initial_obj (numpy array, optional): initial object array of shape (Y,X). Defaults to np.ones(1), which then uses input from input_dict
        initial_probe (numpy array, optional): _description_. Defaults to np.ones(1), which then uses input from input_dict.

    Returns:
        object: object of shape (Y,X)
        probe: probe of shape (1,M,DY,DX)
        error: array of ptychography errors along iterations
    """


    datapack, sigmask = set_initial_parameters_for_GB_algorithms(input_dict,DPs,probe_positions)
    
    # if initial_obj!=np.ones(1):
    if initial_obj is not None:
        datapack["obj"] = initial_obj
    
    # if initial_probe!=np.ones(1):
    if initial_probe is not None:
        datapack["probe"] = initial_probe

    concatenate_array_to_h5_dataset(input_dict["hdf5_output"],'recon','initial_object',datapack["obj"],concatenate=False)
    concatenate_array_to_h5_dataset(input_dict["hdf5_output"],'recon','initial_probe',datapack["probe"],concatenate=False)
    concatenate_array_to_h5_dataset(input_dict["hdf5_output"],'recon','probe_support',datapack["probesupp"],concatenate=False)

    print(f'Starting ptychography... using {len(input_dict["GPUs"])} GPUs {input_dict["GPUs"]} and {input_dict["CPUs"]} CPUs')
    run_algorithms = True
    loop_counter = 1
    error = np.empty((0,))

    corrected_positions = None

    while run_algorithms:  # run Ptycho:
        try:
            algorithm = input_dict['Algorithm' + str(loop_counter)]
            algo_name = algorithm["Name"]
            n_of_iterations = algorithm['Iterations']
            print(f"\tCalling {n_of_iterations} iterations of {algo_name} algorithm...")
        except:
            run_algorithms = False

        if run_algorithms:
            if algorithm['Name'] == 'GL':
                datapack = sscPtycho.GL(iter      = algorithm['Iterations'], 
                                        objbeta   = algorithm['ObjBeta'],
                                        probebeta = algorithm['ProbeBeta'],
                                        batch     = algorithm['Batch'],
                                        epsilon   = algorithm['Epsilon'],
                                        tvmu      = algorithm['TV'],
                                        sigmask   = sigmask,
                                        data      = datapack,
                                        params    = {'device':input_dict["GPUs"]},
                                        probef1=input_dict['fresnel_number'])

            elif algorithm['Name'] == 'positioncorrection':
                datapack['bkg'] = None
                datapack = sscPtycho.PosCorrection(iter       = algorithm['Iterations'],
                                                    objbeta   = algorithm['ObjBeta'],
                                                    probebeta = algorithm['ProbeBeta'], 
                                                    batch     = algorithm['Batch'],
                                                    epsilon   = algorithm['Epsilon'], 
                                                    tvmu      = algorithm['TV'], 
                                                    sigmask   = sigmask,
                                                    data      = datapack,
                                                    params    = {'device':input_dict["GPUs"]},
                                                    probef1=input_dict['fresnel_number'])
                corrected_positions = datapack['rois']

            elif algorithm['Name'] == 'RAAR':
                datapack = sscPtycho.RAAR(iter         = algorithm['Iterations'],
                                           beta        = algorithm['Beta'],
                                           probecycles = algorithm['ProbeCycles'],
                                           batch       = algorithm['Batch'],
                                           epsilon     = algorithm['Epsilon'], 
                                           tvmu        = algorithm['TV'],
                                           sigmask     = sigmask,
                                           data        = datapack,
                                           params      = {'device':input_dict["GPUs"]}, 
                                           probef1=input_dict['fresnel_number']) 

            loop_counter += 1
            error = np.concatenate((error,datapack["error"]),axis=0)

    datapack['obj'] = datapack['obj'].astype(np.complex64)
    datapack['probe'] = datapack['probe'].astype(np.complex64)

    return datapack['obj'], datapack['probe'], error, corrected_positions


def set_initial_parameters_for_GB_algorithms(input_dict, DPs, probe_positions):
    """ Adjust probe initial data to be accepted by Giovanni's algorithm

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
        DPs (numpy array): array of diffraction patterns of shape (N,Y,X)
        probe_positions (numpy array): array of probe positions of shape (N,2)

    Returns:
        datapack (dic): dictionary containing the inputs divided by keys
        sigmask (array): mask of invalid pixels in diffraction data
    """

    def set_datapack(obj, probe, probe_positions, DPs, background, probesupp):
        """Create a dictionary to store the data needed for reconstruction

        Args:
            obj (array): guess for ibject
            probe (array): guess for probe
            probe_positions (array): position in x and y directions
            DPs (array): intensities (diffraction patterns) measured
            background (array): background
            probesupp (array): probe support

        Returns:
            datapack (dictionary)
        """    
        print('Creating datapack...') # Set data for Ptycho algorithms
        datapack = {}
        datapack['obj'] = obj
        datapack['probe'] = probe
        datapack['rois'] = probe_positions
        datapack['difpads'] = DPs
        datapack['bkg'] = background
        datapack['probesupp'] = probesupp

        return datapack

    def set_sigmask(DPs):
        """Create a mask for invalid pixels

        Args:
            DPs (array): measured diffraction patterns

        Returns:
            sigmask (array): 2D-array, same shape of a diffraction pattern, maps the invalid pixels. 0 for negative values
        """    
        sigmask = np.ones(DPs[0].shape)
        sigmask[DPs[0] < 0] = 0
        return sigmask

    def get_probe_support(input_dict,probe_shape):
        """ Create mask containing probe support region

        Args:
            input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
                keys:
                    "probe_support": probe support
            probe_shape (array): probe size 

        Returns:
            probesupp: mask containing probe support
        """

        
        print('Setting probe support...')

        probe = np.zeros(probe_shape)
        
        if input_dict["probe_support"][0] == "circular":
            radius, center_x, center_y = input_dict["probe_support"][1], input_dict["probe_support"][2], input_dict["probe_support"][3]

            half_size = probe_shape[-1]//2

            ar = np.arange(-half_size, half_size)
            xx, yy = np.meshgrid(ar, ar)
            support = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2

            probe[:] = support # all frames and all modes with same support

        elif input_dict["probe_support"][0] == "cross":
            cross_width_y, border, center_square_side = input_dict['probe_support'][1],input_dict['probe_support'][2],input_dict['probe_support'][3]
            probe[:] = create_cross_mask((probe_shape[1],probe_shape[2]),cross_width_y, border, center_square_side)
        else: 
            sys.exit('Please select the correct probe support: circular or cross')

        return probe

    def append_ones(probe_positions):
        """ Adjust shape and column order of positions array to be accepted by Giovanni's code

        Args:
            probe_positions (array): initial positions array in (PY,PX) shape

        Returns:
            probe_positions2 (array): rearranged probe positions array
        """
        zeros = np.zeros((probe_positions.shape[0],1))
        probe_positions = np.concatenate((probe_positions,zeros),axis=1)
        probe_positions = np.concatenate((probe_positions,zeros),axis=1) # concatenate columns to use Giovanni's ptychography code
        probe_positions2 = np.ones_like(probe_positions)
        probe_positions2[:,0] = probe_positions[:,1] # change x and y column order
        probe_positions2[:,1] = probe_positions[:,0]
        return probe_positions2
    

    half_size = DPs.shape[-1] // 2

    print('Fresnel number:', input_dict['fresnel_number'])

    probe_positions = append_ones(probe_positions)

    probe = set_initial_probe(input_dict, (DPs.shape[1], DPs.shape[2]), np.average(DPs, 0)[None] ) # probe initial guess
    
    probe_support = get_probe_support(input_dict, probe.shape)
    
    obj = set_initial_object(input_dict,DPs,probe) # object initial guess

    sigmask = set_sigmask(DPs) # mask for invalid pixels
    background = np.ones(DPs[0].shape) # dummy array 

    print(f"Diffraction Patterns: {DPs.shape}\nInitial Object: {obj.shape}\nInitial Probe: {probe.shape}\nProbe Support: {probe_support.shape}\nProbe Positions: {probe_positions.shape}")
    
    datapack = set_datapack(obj, probe, probe_positions, DPs, background, probe_support)     # Set data for Ptycho algorithms:

    print(f"Total datapack size: {estimate_memory_usage(datapack['obj'],datapack['probe'],datapack['rois'],datapack['difpads'],datapack['bkg'],datapack['probesupp'])[3]:.2f} GBs")

    return datapack, sigmask


def set_initial_probe(input_dict,DP_shape,DPs_avg):
    """ Get initial probe with multiple modes, with format required by Giovanni's code

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "incoherent_modes": incoherent nodes
                "initial_probe": initial probe
                "output_path": path of output files
                
        DP_shape (tuple): shape of single diffraction pattern
        DPs_avg (array): average of all diffraction data

    Returns:
        probe: initial probe array 
    """

    def set_modes(probe, input_dict):
        mode = probe.shape[0]

        if input_dict['incoherent_modes'] > mode:
            add = input_dict['incoherent_modes'] - mode
            probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
            for i in range(add):
                probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

        print("\tProbe shape ({0},{1}) with {2} incoherent mode(s)".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

        return probe

    print('Creating initial probe...')

    if isinstance(input_dict['initial_probe'],list): # if no path to file given
        
        type = input_dict['initial_probe'][0]

        if type == 'circular':
            probe = create_circular_mask(DP_shape,input_dict['initial_probe'][1])
            probe = probe + 1j*probe
        elif type == 'cross':
            probe = create_cross_mask(DP_shape,input_dict['initial_probe'][1],input_dict['initial_probe'][2],input_dict['initial_probe'][3])
        elif type == 'constant':
            probe = np.ones(DP_shape)
        elif type == 'random':
            probe = np.random.rand(*DP_shape)
        elif type == 'inverse':
            ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(DPs_avg)))
            probe = np.sqrt(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft))))
        else:
            sys.exit("Please select an appropriate type for probe initial guess: circular, squared, rectangular, cross, constant, random")

    elif isinstance(input_dict['initial_probe'],str):
        if os.path.splitext(input_dict['initial_probe'])[1] == '.hdf5' or os.path.splitext(input_dict['initial_probe'])[1] == '.h5':
            probe = h5py.File(input_dict['initial_probe'],'r')['recon/probe'][()]
        elif os.path.splitext(input_dict['initial_probe'])[1] == '.npy':
            probe = np.load(input_dict['initial_probe']) # load guess from file
        probe = probe[0]
    elif isinstance(input_dict['initial_probe'],int):
        probe = np.load(os.path.join(input_dict["output_path"],input_dict["output_path"].rsplit('/',2)[1]+"_probe.npy"))
    else:
        sys.exit("Please select an appropriate path or type for probe initial guess: circular, squared, cross, constant")

    probe = probe.astype(np.complex64)

    if probe.shape[0] <= 1:
        probe = set_modes(probe, input_dict) # add incoherent modes 

    return probe


def set_initial_object(input_dict,DPs, probe):
    """ Get initial object from file at input dictionary or define a constant or random matrix for it

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "initial_obj": initial object
                "object_shape": object shape
                "output_path": path for output files
        DPs (array): diffraction data used for calculating normalization factor
        probe (array): probe used for calculating normalization factor

    Returns:
        _type_: _description_
    """

    print('Creating initial object...')

    if isinstance(input_dict['initial_obj'],list):
        type = input_dict['initial_obj'][0]
        if type == 'constant':
            obj = np.ones(input_dict["object_shape"])
        elif type == 'random':
            normalization_factor = np.sqrt(np.average(DPs) / np.average(abs(np.fft.fft2(probe))**2))
            obj = np.random.rand(*input_dict["object_shape"]) * normalization_factor
        elif type == 'complex_random':
            obj =  1 * (np.random.rand(*input_dict["object_shape"]) + 1j*np.random.rand(*input_dict["object_shape"]))
        elif type == 'initialize':
            pass #TODO: implement method from https://doi.org/10.1364/OE.465397
    elif isinstance(input_dict['initial_obj'],str): 
        if os.path.splitext(input_dict['initial_obj'])[1] == '.hdf5' or os.path.splitext(input_dict['initial_obj'])[1] == '.h5':
            obj = h5py.File(input_dict['initial_obj'],'r')['recon/object'][0] # select first frame of object
        elif os.path.splitext(input_dict['initial_obj'])[1] == '.npy':
            obj = np.load(input_dict['initial_obj'])
        obj = np.squeeze(obj)
    elif isinstance(input_dict['initial_obj'],int):
        obj = np.load(os.path.join(input_dict["output_path"],input_dict["output_path"].rsplit('/',2)[1]+"_object.npy"))
    else:
        sys.exit("Please select an appropriate path or type for object initial guess: autocorrelation, constant, random")

    return obj.astype(np.complex64)


def create_circular_mask(mask_shape, radius):
    """" Create circular mask 

    Args:
        mask_shape (tuple): Y,X shape of the mask
        radius (int): radius of the mask in pixels

    Returns:
        mask (array): circular mask of 1s and 0s
    """


    """ All values in pixels """
    center_row, center_col = mask_shape
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    return np.where((Xmesh - center_col//2) ** 2 + (Ymesh - center_row//2) ** 2 <= radius ** 2, 1, 0)


def create_cross_mask(mask_shape, cross_width_y=15, border=3, center_square_side = 10, cross_width_x=0):
    """ Create cross mask
    Args:
        mask_shape (tuple): y and x sizes of the mask
        cross_width_y (int, optional): _description_. Defaults to 15.
        border (int, optional): Distance from edge of cross mask to the domain border. Defaults to 3.
        center_square_side (int, optional): _description_. Defaults to 10.
        cross_width_x (int, optional): _description_. Defaults to 0.

    Returns:
        mask (array): cross mask
    """
    
    if cross_width_x == 0: cross_width_x = cross_width_y
    
    """ All values in pixels """
    
    # center
    center_row, center_col = mask_shape[0]//2, mask_shape[1]//2
    mask = np.zeros(mask_shape)
    mask[center_row-cross_width_y//2:center_row+cross_width_y//2,:] = 1
    mask[:,center_col-cross_width_x//2:center_col+cross_width_x//2] = 1
    
    # null border
    mask[0:border,:]  = 0 
    mask[:,0:border]  = 0 
    mask[-border::,:] = 0
    mask[:,-border::] = 0

    # center square
    mask[center_row-center_square_side:center_row+center_square_side,center_col-center_square_side:center_col+center_square_side] = 1
    
    return mask 

def set_object_pixel_size(input_dict,DP_size):
    """ Get size of object pixel given energy, distance and detector pixel size

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "energy": beamline energy
                "wavelength": 
                "binning": 
                "restored_pixel_size": restored pixel size
        DP_size (int): lateral size of detector array

    Returns:
        input_dict: update input dictionary containing size of object pixel
    """

    wavelength = wavelength_from_energy(input_dict["energy"])
    input_dict["wavelength"] = wavelength
    
    object_pixel_size = wavelength * input_dict['detector_distance'] / (input_dict["binning"]*input_dict['restored_pixel_size'] * DP_size)
    input_dict["object_pixel"] = object_pixel_size # in meters

    print(f"\tObject pixel size = {object_pixel_size*1e9:.2f} nm")
    PA_thickness = 4*object_pixel_size**2/(0.61*wavelength)
    print(f"\tLimit thickness for resolution of 1 pixel: {PA_thickness*1e6:.3f} microns")
    return input_dict


def set_object_shape(input_dict, DP_shape, probe_positions):
    """ Determines shape (Y,X) of object matrix from size of probe and its positions.

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "object_padding":
                "object_shape": object size/shape
        DP_shape (tuple): shape of the diffraction patterns array
        probe_positions (numpy array): array os probe positiions in pixels

    Returns:
        input_dict (dict)): updated input dictionary containing object_shape information
    """

    offset_bottomright = input_dict["object_padding"]
    DP_size_y, DP_size_x = DP_shape[1:]

    maximum_probe_coordinate_x = int(np.max(probe_positions[:,1]))
    object_shape_x  = DP_size_x + maximum_probe_coordinate_x + offset_bottomright

    maximum_probe_coordinate_y = int(np.max(probe_positions[:,0]))
    object_shape_y  = DP_size_y + maximum_probe_coordinate_y + offset_bottomright

    input_dict["object_shape"] = (object_shape_y, object_shape_x)

    return input_dict


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import time
import h5py, os
import random
import tqdm
import pyfftw
import scipy

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
import threading

from skimage.registration import phase_cross_correlation
from numpy.fft import fft2, fftshift, ifftshift, ifft2

random.seed(0)

""" RAAR + probe decomposition   """

def RAAR_multiprobe_cupy(diffraction_patterns,positions,obj,probe,inputs, probe_support = None):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    # Numpy to Cupy
    diffraction_patterns = cp.array(diffraction_patterns)
    positions = cp.array(positions)
    obj = cp.array(obj)
    probe = cp.array(probe)

    if probe_support is None:
        probe_support = cp.ones_like(probe)
    else:
        probe_support = cp.array(probe_support)

    obj_matrix = cp.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = cp.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = cp.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        wavefronts[index] = probe_modes*obj_box
        
    error = []
    for iteration in range(0,iterations):
        """
        RAAR update function:
        psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
        psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi
        psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
        """

        t1 = time.time()
        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
            
            psi_after_Pr = probe_modes*obj_box
            
            psi_after_reflection_Rspace = 2*psi_after_Pr-wavefronts[index]
            psi_after_projection_Fspace, _ = update_exit_wave_multiprobe_cupy(psi_after_reflection_Rspace,diffraction_patterns[index]) # Projection in Fourier space
            wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 
        t2 =time.time()
        print(t2-t1)

        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR_cupy(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = calculate_recon_error_Fspace_cupy(diffraction_patterns,wavefronts,(dx,wavelength,distance)).get()
        if iteration%10==0:
            print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0].get(), probe_modes.get(), error

def projection_Rspace_multiprobe_RAAR_cupy(wavefronts,obj,probes,positions,epsilon):
    probes = RAAR_multiprobe_update_probe_cupy(wavefronts, obj, probes.shape,positions, epsilon=epsilon) 
    obj   = RAAR_multiprobe_update_object_cupy(wavefronts, probes, obj.shape, positions,epsilon=epsilon)
    return probes, obj

def RAAR_multiprobe_update_object_cupy(wavefronts, probe, object_shape, positions,epsilon):

    modes,m,n = probe.shape
    k,l = object_shape

    probe_sum  = cp.zeros((k,l),dtype=complex)
    wave_sum   = cp.zeros((k,l),dtype=complex)
    probe_intensity  = cp.abs(probe)**2
    probe_conj = cp.conj(probe)

    for mode in range(modes):
        for index, (posx, posy) in enumerate((positions)):
            probe_sum[posy:posy + m , posx:posx+n] += probe_intensity[mode]
            wave_sum[posy:posy + m , posx:posx+n]  += probe_conj[mode]*wavefronts[index,mode] 

    obj = wave_sum/(probe_sum + epsilon)

    return obj

def RAAR_multiprobe_update_probe_cupy(wavefronts, obj, probe_shape,positions, epsilon=0.01):
    
    l,m,n = probe_shape

    object_sum = cp.zeros((m,n),dtype=complex)
    wave_sum = cp.zeros((l,m,n),dtype=complex)
    
    obj_intensity = cp.abs(obj)**2
    obj_conj = cp.conj(obj)
    
    for index, (posx, posy) in enumerate(positions):
        object_sum += obj_intensity[posy:posy + m , posx:posx+n] 
        for mode in range(l):
            wave_sum[mode] += obj_conj[posy:posy + m , posx:posx+n]*wavefronts[index,mode]

    probes = wave_sum/(object_sum + epsilon) # epsilon to avoid division by zero. 

    return probes


""" parallel RAAR + probe decompositon """

def RAAR_multiprobe_parallel(diffraction_patterns,positions,obj,probe,inputs, probe_support = None, processes=32):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    if probe_support is None:
        probe_support = np.ones_like(probe)
    else:
        probe_support = np.array(probe_support)

    obj_matrix = np.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = np.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = np.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        
    error = []
    for iteration in range(0,iterations):

        # t1 = time.time()
        wavefronts = update_wavefronts_parallel(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta,processes)
        # wavefronts = update_wavefronts_parallel2(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta)
        # t2 = time.time()
        # print(t2-t1)

        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = 0 # calculate_recon_error_Fspace(diffraction_patterns,wavefronts,(dx,wavelength,distance))
        print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0], probe_modes, error

def update_wavefronts_parallel2(obj_matrix, probe_modes, wavefronts0, diffraction_patterns0,positions,beta):

    global wavefronts,projected_wavefronts,diffraction_patterns

    wavefronts,diffraction_patterns = wavefronts0, diffraction_patterns0

    shapey,shapex = probe_modes.shape[1], probe_modes.shape[2]

    projected_wavefronts = np.empty_like(wavefronts)

    for index, (posx, posy) in enumerate(positions):
        projected_wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]

    indexes = [i for i in range(wavefronts.shape[0])]
    
    processes = []
    for index in indexes:
        # process = multiprocessing.Process(target=update_wavefront2,args=(index, beta))
        process = threading.Thread(target=update_wavefront2,args=(index, beta))
        process.start()
        processes.append(process)

    for p in processes: # wait for processes to finish
        p.join()

    return wavefronts

def update_wavefront2(index,beta):
    """
    RAAR update function:
    psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
    psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
    psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
    """
    global wavefronts,projected_wavefronts,diffraction_patterns

    
    psi_after_reflection_Rspace = 2*projected_wavefronts[index]-wavefronts[index]
    psi_after_projection_Fspace, _ = update_exit_wave_multiprobe(psi_after_reflection_Rspace,diffraction_patterns[index]) # Projection in Fourier space
    wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*projected_wavefronts[index] 

def update_wavefronts_parallel(obj_matrix, probe_modes, wavefronts, diffraction_patterns,positions,beta, processes):
    t0 = time.time()

    shapey,shapex = probe_modes.shape[1], probe_modes.shape[2]

    # update_wavefront_partial = partial(update_wavefront,beta)

    projected_wavefronts = np.empty_like(wavefronts)
    for index, (posx, posy) in enumerate(positions):
        projected_wavefronts[index] = probe_modes*obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = executor.map(update_wavefront,wavefronts, projected_wavefronts, diffraction_patterns)
        for counter, result in enumerate(results):
            wavefronts[counter,:,:] = result

    return wavefronts

def update_wavefront(wavefront,psi_after_Pr,data,beta=1):
    """
    RAAR update function:
    psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
    psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
    psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
    """
    
    wavefront = cp.asarray(wavefront)
    psi_after_Pr = cp.asarray(psi_after_Pr)
    data = cp.asarray(data)

    psi_after_reflection_Rspace = 2*psi_after_Pr-wavefront
    psi_after_projection_Fspace, _ = update_exit_wave_multiprobe_cupy(psi_after_reflection_Rspace.copy(),data) # Projection in Fourier space
    wavefront = beta*(wavefront + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 


    return wavefront.get()

def RAAR_multiprobe(diffraction_patterns,positions,obj,probe,inputs, probe_support = None):
    iterations = inputs['iterations']
    beta       = inputs['beta']
    epsilon    = inputs['epsilon']
    dx         = inputs['object_pixel']
    wavelength = inputs['wavelength']
    distance   = inputs['distance']
    n_of_modes = inputs["n_of_modes"]

    if probe_support is None:
        probe_support = np.ones_like(probe)
    else:
        probe_support = np.array(probe_support)

    obj_matrix = np.ones((n_of_modes,obj.shape[0],obj.shape[1]),dtype=complex) 
    obj_matrix[:] = obj # create matrix of repeated object to facilitate slice-wise product with probe modes
    
    shapey,shapex = probe.shape
    wavefronts = np.ones((len(positions),n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex) # wavefronts contain the wavefront for each probe mode, and for all probe positions
    
    probe_modes = np.ones((n_of_modes,probe.shape[0],probe.shape[1]),dtype=complex)
    probe_modes[:] = probe
    
    for index, (posx, posy) in enumerate(positions):
        obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
        wavefronts[index] = probe_modes*obj_box
        
    error = []
    for iteration in range(0,iterations):
        """
        RAAR update function:
        psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
        psi' = beta*(Pf*Rr + I)*psi + (1-2*beta)*Pr*psi 
        psi' = beta*(Pf*Rr*psi + psi) + (1-2*beta)*Pr*psi (eq 1)
        """

        for index, (posx, posy) in enumerate(positions):
            
            obj_box = obj_matrix[:,posy:posy + shapey , posx:posx+ shapex]
            psi_after_Pr = probe_modes*obj_box

            psi_after_reflection_Rspace = 2*psi_after_Pr-wavefronts[index]
            psi_after_projection_Fspace, _ = update_exit_wave_multiprobe(psi_after_reflection_Rspace.copy(),diffraction_patterns[index]) # Projection in Fourier space

            wavefronts[index] = beta*(wavefronts[index] + psi_after_projection_Fspace) + (1-2*beta)*psi_after_Pr 

        probe_modes, single_obj_box = projection_Rspace_multiprobe_RAAR(wavefronts,obj_matrix[0],probe_modes,positions,epsilon) # Update Object and Probe! Projection in Real space (consistency condition)
        obj_matrix[:] = single_obj_box # update all obj slices to be the same;

        probe_modes = probe_modes[:]*probe_support

        iteration_error = calculate_recon_error_Fspace(diffraction_patterns,wavefronts,(dx,wavelength,distance)).get()
        if iteration%10==0:
            print(f'\tIteration {iteration}/{iterations} \tError: {iteration_error:.2e}')
        error.append(iteration_error) 
        
    return obj_matrix[0], probe_modes, error

def update_exit_wave_multiprobe(wavefront_modes,measurement):
    wavefront_modes = propagate_farfield_multiprobe(wavefront_modes)
    wavefront_modes_at_detector = Fspace_update_multiprobe(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_farfield_multiprobe(wavefront_modes_at_detector,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_farfield_multiprobe_pyfftw(wavefront_modes,backpropagate=False):
    
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): 
            wavefront_modes[m] = np.fft.fftshift(pyfftw_FT2D(mode))
    else:
        for m in range(wavefront_modes.shape[0]):
            wavefront_modes[m] = pyfftw_FT2D(np.fft.ifftshift(wavefront_modes[m]),inverse=True)
    return wavefront_modes
    
def pyfftw_FT2D(data,inverse=False):
    data_aligned = pyfftw.byte_align(data,dtype = 'complex128')
    # result_aligned = pyfftw.empty_aligned(data_aligned.shape,dtype = 'complex128')
    if inverse == False:
        # fft_object = pyfftw.FFTW(data_aligned, result_aligned, axes=(0,1),threads=30)
        fft = pyfftw.interfaces.numpy_fft.fft(data_aligned)
    else:    
        # fft_object = pyfftw.FFTW(data_aligned, result_aligned, axes=(0,1),direction='FFTW_BACKWARD',threads=30)
        fft = pyfftw.interfaces.numpy_fft.ifft(data_aligned)
    # return fft_object()
    return fft

def propagate_farfield_multiprobe(wavefront_modes,backpropagate=False):
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): 
            # wavefront_modes[m] = np.fft.fftshift(np.fft.fft2(mode)) # NUMPY
            wavefront_modes[m] = scipy.fft.fftshift(scipy.fft.fft2(mode,workers=-1)) #SCIPY
    else:
        for m in range(wavefront_modes.shape[0]):
            # wavefront_modes[m] = np.fft.ifft2(np.fft.ifftshift(wavefront_modes[m]))
            wavefront_modes[m] = scipy.fft.ifft2(scipy.fft.ifftshift(wavefront_modes[m]),workers=-1)
    return wavefront_modes

def Fspace_update_multiprobe(wavefront_modes,measurement,epsilon=0.001):
    
    total_wave_intensity = np.zeros_like(wavefront_modes[0])

    for mode in wavefront_modes:
        total_wave_intensity += np.abs(mode)**2
    total_wave_intensity = np.sqrt(total_wave_intensity)
    
    updated_wavefront_modes = wavefront_modes
    for m, mode in enumerate(wavefront_modes): 
        updated_wavefront_modes[m][measurement>=0] = np.sqrt(measurement[measurement>=0])*mode[measurement>=0]/(total_wave_intensity[measurement>=0]+epsilon)
    
    return updated_wavefront_modes

def projection_Rspace_multiprobe_RAAR(wavefronts,obj,probes,positions,epsilon):
    probes = RAAR_multiprobe_update_probe(wavefronts, obj, probes.shape,positions, epsilon=epsilon) 
    obj   = RAAR_multiprobe_update_object(wavefronts, probes, obj.shape, positions,epsilon=epsilon)
    return probes, obj

def RAAR_multiprobe_update_object(wavefronts, probe, object_shape, positions,epsilon):

    modes,m,n = probe.shape
    k,l = object_shape

    probe_sum  = np.zeros((k,l),dtype=complex)
    wave_sum   = np.zeros((k,l),dtype=complex)
    probe_intensity  = np.abs(probe)**2
    probe_conj = np.conj(probe)

    for mode in range(modes):
        for index, (posx, posy) in enumerate((positions)):
            probe_sum[posy:posy + m , posx:posx+n] += probe_intensity[mode]
            wave_sum[posy:posy + m , posx:posx+n]  += probe_conj[mode]*wavefronts[index,mode] 

    obj = wave_sum/(probe_sum + epsilon)

    return obj

def RAAR_multiprobe_update_probe(wavefronts, obj, probe_shape,positions, epsilon=0.01):
    
    l,m,n = probe_shape

    object_sum = np.zeros((m,n),dtype=complex)
    wave_sum = np.zeros((l,m,n),dtype=complex)
    
    obj_intensity = np.abs(obj)**2
    obj_conj = np.conj(obj)
    
    for index, (posx, posy) in enumerate(positions):
        object_sum += obj_intensity[posy:posy + m , posx:posx+n] 
        for mode in range(l):
            wave_sum[mode] += obj_conj[posy:posy + m , posx:posx+n]*wavefronts[index,mode]

    probes = wave_sum/(object_sum + epsilon) # epsilon to avoid division by zero. 

    return probes


"""  mPIE + probe decomposition   """

def PIE_multiprobe_loop(diffraction_patterns, positions, object_guess, probe_guess, inputs):

    r_o = inputs["regularization_object"]
    r_p = inputs["regularization_probe"]
    s_o = inputs["step_object"]
    s_p = inputs["step_probe"]
    f_o = inputs["friction_object"]
    f_p = inputs["friction_probe"]
    m_counter_limit = inputs["momentum_counter"]
    n_of_modes = inputs["n_of_modes"]
    iterations = inputs["iterations"]
    experiment_params =  (inputs['object_pixel'], inputs['wavelength'],inputs['distance'])

    object_guess = cp.array(object_guess) # convert from numpy to cupy
    probe_guess  = cp.array(probe_guess)
    positions    = cp.array(positions)
    diffraction_patterns = cp.array(diffraction_patterns)

    obj = cp.ones((n_of_modes,object_guess.shape[0],object_guess.shape[1]),dtype=complex)
    obj[:] = object_guess # object matrix repeats for each slice; each slice will operate with a different probe mode

    offset = probe_guess.shape

    if inputs["n_of_modes"] > 1:
        probe_modes = cp.empty((inputs["n_of_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[0] = probe_guess # first mode is guess
        for mode in range(1,inputs["n_of_modes"]): # remaining modes are random
            probe_modes[mode] = cp.random.rand(*probe_guess.shape)
    elif inputs["n_of_modes"] == 1:
        probe_modes = cp.empty((inputs["n_of_modes"],probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)
        probe_modes[:] = probe_guess
    else:
        sys.exit('Please select the correct amount of modes: ',inputs["n_of_modes"])

    wavefronts = cp.empty((len(diffraction_patterns),probe_guess.shape[0],probe_guess.shape[1]),dtype=complex)

    probe_velocity = cp.zeros_like(probe_modes,dtype=complex)
    obj_velocity   = cp.zeros_like(obj,dtype=complex)
    
    momentum_counter = 0
    error_list = []
    for i in range(iterations):
        
        temporary_obj, temporary_probe = obj.copy(), probe_modes.copy()
        
        for j in cp.random.permutation(len(diffraction_patterns)):  
            py, px = positions[:,1][j],  positions[:,0][j]

            obj_box = obj[:,py:py+offset[0],px:px+offset[1]]

            """ Wavefront at object exit plane """
            wavefront_modes = obj_box*probe_modes

            wavefronts[j] = wavefront_modes[0] # save mode 0 wavefront to calculate recon error
 
            """ Propagate + Update + Backpropagate """
            updated_wavefront_modes, _ = update_exit_wave_multiprobe_cupy(wavefront_modes.copy(),diffraction_patterns[j]) #copy so it doesn't work as a pointer!
            
            obj[:,py:py+offset[0],px:px+offset[1]] , probe_modes = PIE_update_func_multiprobe(obj_box[0],probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p)

            if inputs["use_mPIE"] == True: # momentum addition                                                                                      
                momentum_counter,obj_velocity,probe_velocity,temporary_obj,temporary_probe,obj,probe_modes = momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,temporary_obj,temporary_probe,obj,probe_modes,f_o,f_p,m_counter_limit,momentum_type="")

        iteration_error = calculate_recon_error_Fspace_cupy(diffraction_patterns,wavefronts,experiment_params).get()
        if i%10==0:
            print(f'\tIteration {i}/{iterations} \tError: {iteration_error:.2e}')
        error_list.append(iteration_error) # error in fourier space 

    return obj.get(), probe_modes.get(), error_list

def PIE_update_func_multiprobe(obj,probe_modes,wavefront_modes,updated_wavefront_modes,s_o,s_p,r_o,r_p):

    """ 
    s: step constant
    r: regularization constant
    """
    
    def get_denominator_p(obj,reg_p):
        power = cp.abs(obj)**2
        denominator = (1-reg_p)*power+ reg_p*cp.max(power)
        return denominator  

    def get_denominator_o(probe_modes,reg_o):
        
        total_probe_power = cp.zeros_like(cp.abs(probe_modes[0]))
        for mode in probe_modes:
            total_probe_power += cp.abs(mode)**2    
            
        denominator = (1-reg_o)*total_probe_power + reg_o*cp.max(total_probe_power)
        
        return denominator  

    # r_o,r_p,s_o,s_p,_,_,_ = mPIE_params

    # Pre-calculating to avoid repeated operations
    denominator_object = get_denominator_o(probe_modes,r_o)
    probe_modes_conj = probe_modes.conj()
    Delta_wavefront_modes =  updated_wavefront_modes - wavefront_modes

    obj = obj + s_o * cp.sum(probe_modes_conj*Delta_wavefront_modes,axis=0) / denominator_object # object update

    obj_conj = obj.conj()
    denominator_probe  = get_denominator_p(obj,r_p)
    for m in range(probe_modes.shape[0]): # P_(i+1) = P_(i) + s_p * DeltaP_(i)
        probe_modes[m] = probe_modes[m] + s_p * obj_conj*Delta_wavefront_modes[m] / denominator_probe # probe update


    return obj, probe_modes

def momentum_addition_multiprobe(momentum_counter,probe_velocity,obj_velocity,O_aux,P_aux,obj,probe,friction_object,friction_probe,m_counter_limit,momentum_type=""):
    

    momentum_counter += 1    
    if momentum_counter == m_counter_limit : 

        probe_velocity = friction_probe*probe_velocity + (probe - P_aux) # equation 19 in the paper
        obj_velocity   = friction_object*obj_velocity  + (obj - O_aux)  

        if momentum_type == "Nesterov": # equation 21
            obj = obj + friction_object*obj_velocity
            probe = probe + friction_object*probe_velocity 
        else: # equation 20     
            obj = O_aux + obj_velocity
            probe = P_aux + probe_velocity 

        O_aux = obj
        P_aux = probe            
        momentum_counter = 0
    
    return momentum_counter,obj_velocity,probe_velocity,O_aux,P_aux,obj,probe

""" GENERAL """

def update_exit_wave_multiprobe_cupy(wavefront_modes,measurement):
    wavefront_modes = propagate_farfield_multiprobe_cupy(wavefront_modes)
    wavefront_modes_at_detector = Fspace_update_multiprobe_cupy(wavefront_modes,measurement)
    updated_wavefront_modes = propagate_farfield_multiprobe_cupy(wavefront_modes_at_detector,backpropagate=True)
    return updated_wavefront_modes, wavefront_modes_at_detector

def propagate_farfield_multiprobe_cupy(wavefront_modes,backpropagate=False):
    if backpropagate == False:
        for m, mode in enumerate(wavefront_modes): #TODO: worth propagating in parallel?
            wavefront_modes[m] = cp.fft.fftshift(cp.fft.fft2(mode))
    else:
        for m in range(wavefront_modes.shape[0]):
            wavefront_modes[m] = cp.fft.ifft2(cp.fft.ifftshift(wavefront_modes[m]))
    return wavefront_modes

def Fspace_update_multiprobe_cupy(wavefront_modes,measurement,epsilon=0.001):
    
    total_wave_intensity = cp.zeros_like(wavefront_modes[0])

    for mode in wavefront_modes:
        total_wave_intensity += cp.abs(mode)**2
    total_wave_intensity = cp.sqrt(total_wave_intensity)
    
    updated_wavefront_modes = wavefront_modes
    for m, mode in enumerate(wavefront_modes): #TODO: worth updating in parallel?
        updated_wavefront_modes[m][measurement>=0] = cp.sqrt(measurement[measurement>=0])*mode[measurement>=0]/(total_wave_intensity[measurement>=0]+epsilon)
    
    return updated_wavefront_modes

def calculate_recon_error_Fspace_cupy(diffractions_patterns,wavefronts,experiment_params):

    error_numerator = 0
    error_denominator = 0
    for DP, wave in zip(diffractions_patterns,wavefronts):
        wave_at_detector = propagate_beam_cupy(wave, experiment_params,propagator='fourier')
        intensity = cp.abs(wave_at_detector)**2
        
        error_numerator += cp.sum(cp.abs(DP-intensity))
        error_denominator += cp.sum(cp.abs(DP))

    return error_numerator/error_denominator 

def calculate_recon_error_Fspace(diffractions_patterns,wavefronts,experiment_params):

    error_numerator = 0
    error_denominator = 0
    for DP, wave in zip(diffractions_patterns,wavefronts):
        wave_at_detector = propagate_beam_cupy(wave, experiment_params,propagator='fourier')
        intensity = np.abs(wave_at_detector)**2
        
        error_numerator += np.sum(np.abs(DP-intensity))
        error_denominator += np.sum(np.abs(DP))

    return error_numerator/error_denominator 

def propagate_beam_cupy(wavefront, experiment_params,propagator='fourier'):
    

    """ Propagate a wavefront using fresnel ou fourier propagator

    Args:
        wavefront : the wavefront to propagate
        dx : pixel spacing of the wavefront input
        wavelength : wavelength of the illumination
        distance : distance to propagate
        propagator (str, optional): 'fresenel' or 'fourier'. Defaults to 'fresnel'.

    Returns:
        output: propagated wavefront
    """    
    
    dx, wavelength,distance = experiment_params 
    
    if propagator == 'fourier':
        if distance > 0:
            output = cp.fft.fftshift(cp.fft.fft2(wavefront))
        else:
            output = cp.fft.ifft2(cp.fft.ifftshift(wavefront))
    
    elif propagator == 'fresnel':
    
        ysize, xsize = wavefront.shape
        x_array = cp.linspace(-xsize/2,xsize/2-1,xsize)
        y_array = cp.linspace(-ysize/2,ysize/2-1,ysize)

        fx = x_array/(xsize)
        fy = y_array/(ysize)

        FX,FY = cp.meshgrid(fx,fy)
        # Calculate approx phase distribution for each plane wave component
        w = FX**2 + FY**2 
        # Compute FFT
        F = fftshift(fft2(fftshift(wavefront)))
        # multiply by phase-shift and inverse transform 
        a = cp.exp(-1j*cp.pi*( distance*wavelength/dx**2)*w)
        output = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(F*a)))
    return output


""" DEV: Position Correction """

def update_beta(positions1,positions2, beta):
    
    k = np.corrcoef(positions1,positions2)[0,1]

    if np.isnan(k).any():
        print('Skipping')
    else:
        threshold1 = +0.3
        threshold2 = -0.3
        
        if k > threshold1:
            beta = beta*1.1 # increase by 10%
        elif k < threshold2:
            beta = beta*0.9 #reduce by 10%
        else:
            pass # keep same value
        
    return beta

def get_illuminated_mask(probe,probe_threshold):
    probe = np.abs(probe)
    mask = np.where(probe > np.max(probe)*probe_threshold, 1, 0)
    return mask

def position_correction(i, obj,previous_obj,probe,position_x,position_y, betas, probe_threshold=0.5, upsampling=100):

    beta_x,beta_y = betas

    illumination_mask = get_illuminated_mask(probe,probe_threshold)

    obj = obj*illumination_mask
    previous_obj = previous_obj*illumination_mask

    relative_shift, error, diffphase = phase_cross_correlation(obj, previous_obj, upsample_factor=upsampling)

    # if 0 :
    #     threshold = 5
    #     if np.abs(beta_y*relative_shift[0]) > threshold or np.abs(beta_x*relative_shift[1]) > threshold:
    #         new_position = np.array([position_x,position_y])
    #     else:
    #         new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    #         # new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # else:
    
    # new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # new_position = np.array([position_x + beta_x*relative_shift[0], position_y + beta_y*relative_shift[1]])

    if i == 0:
        print(position_x, beta_x*relative_shift[1],'\t',position_y,beta_y*relative_shift[0],relative_shift)

    return new_position, relative_shift, illumination_mask

def position_correction2(i,updated_wave,measurement,obj,probe,px,py,offset,betas,experiment_params):
    """ Position correct of the gradient of intensities """ 
    
    beta_x, beta_y = betas
    
    
    # Calculate intensity difference
    updated_intensity_at_detector = np.abs(updated_wave)**2
    intensity_diff = (updated_intensity_at_detector-measurement).flatten()
    
    # Calculate wavefront gradient
    obj_dy = np.roll(obj,1,axis=0)
    obj_dx = np.roll(obj,1,axis=1)
    
    obj_box     = obj[py:py+offset[0],px:px+offset[1]]
    obj_dy_box  = obj_dy[py:py+offset[0],px:px+offset[1]]
    obj_dx_box  = obj_dx[py:py+offset[0],px:px+offset[1]]
    
    wave_at_detector    = propagate_beam(obj_box*probe,    experiment_params,propagator='fourier')
    wave_at_detector_dy = propagate_beam(obj_dy_box*probe, experiment_params,propagator='fourier')
    wave_at_detector_dx = propagate_beam(obj_dx_box*probe, experiment_params,propagator='fourier')

    obj_pxl = experiment_params[0]
    wavefront_gradient_x = (wave_at_detector-wave_at_detector_dx)/obj_pxl
    wavefront_gradient_y = (wave_at_detector-wave_at_detector_dy)/obj_pxl
   
    # Calculate intensity gradient
    intensity_gradient_x = 2*np.real(wavefront_gradient_x*np.conj(wave_at_detector))
    intensity_gradient_y = 2*np.real(wavefront_gradient_y*np.conj(wave_at_detector))
    
    
    # Solve linear system
    A_matrix = np.column_stack((intensity_gradient_x.flatten(),intensity_gradient_y.flatten()))
    A_transpose = np.transpose(A_matrix)
    relative_shift = np.linalg.pinv(A_transpose@A_matrix)@A_transpose@intensity_diff

    # Update positions
    # new_positions = np.array([px - beta_x*relative_shift[0], py - beta_y*relative_shift[1]])
    new_positions = np.array([py - beta_y*relative_shift[1],px - beta_x*relative_shift[0]])
    
    if i == 0:
        print(px, beta_x*relative_shift[1],'\t',py,beta_y*relative_shift[0],relative_shift)
    
    return new_positions

def plot_positions_and_errors(data_folder,dataname,offset,PIE_positions=[],positions_story=[]):
    
    import os, json
    
    metadata = json.load(open(os.path.join(data_folder,dataname,'mdata.json')))
    distance = metadata['/entry/beamline/experiment']['distance']*1e-3
    energy = metadata['/entry/beamline/experiment']['energy']
    pixel_size = metadata['/entry/beamline/detector']['pimega']['pixel size']*1e-6
    wavelength, wavevector = calculate_wavelength(energy)
    
    diffraction_patterns = np.load(os.path.join(data_folder,dataname,f"0000_{dataname}_001.hdf5.npy"))

    n_pixels = diffraction_patterns.shape[1]
    obj_pixel_size = wavelength*distance/(n_pixels*pixel_size)
    
    _,_,measured = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}",obj_pixel_size,offset,0)
    _,_,true = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}_without_error",obj_pixel_size,offset,0)
    
    colors = np.linspace(0,positions.shape[0]-1,positions.shape[0])
    fig, ax = plt.subplots(dpi=150)
    ax.legend(["True" , "Measured", "Corrected", "Path"],loc=(1.05,0.84))    
    ax.scatter(measured[:,1],measured[:,0],marker='o',c='red')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    if positions_story != []:
        for i in range(PIE_positions.shape[0]):
            y = positions_story[:,i,1]
            x = positions_story[:,i,0]
            ax.scatter(y,x,color='blue',s=2,marker=',',alpha=0.2)
    if PIE_positions != []:
        ax.scatter(PIE_positions[:,1],PIE_positions[:,0],marker='x',color='blue')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    ax.scatter(true[:,1],true[:,0],marker='*',color='green')#,c=colors,cmap='jet')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid()

