import numpy as np
import sys, os, h5py
import random

import sscPtycho
from ..misc import estimate_memory_usage, concatenate_array_to_h5_dataset, wavelength_from_energy
from .pie import PIE_multiprobe_loop
from .raar import RAAR_multiprobe_cupy

random.seed(0)

def call_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):

    if 'algorithms' in input_dict:
        obj, probe, error, positions = call_GCC_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)
    else:
        obj, probe, error, positions = call_GB_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)

    return obj, probe, error, positions

def call_GCC_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):
    """ Ptychography algorithms in Python by GCC

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
            inputs['friction_object'] = input_dict['algorithms'][str(counter)]['mPIE_friction_obj'] 
            inputs['friction_probe'] = input_dict['algorithms'][str(counter)]['mPIE_friction_probe'] 
            inputs['momentum_counter'] = input_dict['algorithms'][str(counter)]['mPIE_momentum_counter'] 
            obj, probe, algo_error = PIE_multiprobe_loop(DPs, positions,obj[0],probe[0], inputs)

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            obj, probe, algo_error = RAAR_multiprobe_cupy(DPs,positions,obj,probe,inputs, probe_support = None)
            obj = np.expand_dims(obj,axis=0) # obj coming with one dimensions less. needs to be fixed
        else:
            sys.exit('Please select a proper algorithm! Selected: ', inputs["algorithm"])

        error = np.concatenate((error,algo_error),axis=0)


    return obj, probe, error, positions

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

    complex_obj = obj.astype(np.complex64)


    return complex_obj

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





