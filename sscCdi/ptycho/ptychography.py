import numpy as np
import sys, os, h5py
import random

from ..cditypes import GL, PosCorrection, PIE, RAAR

from ..misc import estimate_memory_usage, concatenate_array_to_h5_dataset, wavelength_meters_from_energy_keV
from ..processing.propagation import fresnel_propagator_cone_beam
from .pie import PIE_multiprobe_loop
from .raar import RAAR_multiprobe_cupy

from .. import log_event

random.seed(0)

@log_event
def call_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):
    """ Call Ptychography algorithms. Options are:

        - RAAR_PYTHON: Relaxed Averaged Alternating Reflections. Single GPU, Python implementation using CuPy
        - ePIE_PYTHON: Extended Ptychographic Iterative Engine. Single GPU, Python implementation using CuPy
        - RAAR_CUDA:   Relaxed Averaged Alternating Reflections. Multi GPU, CUDA implementation
        - AP_CUDA:     Alternate Projections. Multi GPU, CUDA implementation
        - AP_PC_CUDA:  Alternate Projections with Position Correction via Annealing method. Multi GPU, CUDA implementation
        - ePIE_CUDA:   Extended Ptychographic Iterative Engine. Single GPU, CUDA implementation

    Args:
        DPs (ndarray): diffraction data with shape (N,Y,X). N is the number of diffraction patterns.
        positions (array): positions array with shape (N,2) with (x,y) position pairs in each line.
        initial_obj (ndarray, optional): Initial guess for object. Shape to be determined from DPs and positions. If None, will use the input in "input_dict" to determine the initial object. Defaults to None.
        initial_probe (ndarray, optional): Initial guess for probe of shape (M,Y,X), where M is the number of probe modes. If None, will use the input in "input_dict" to determine the initial probe. Defaults to None.
        input_dict (dict): dictionary of input required for Ptychography. Example below are:
           
            input_dict = {
                'CPUs': 32,  # number of cpus to use for parallel execution    
                
                'GPUs': [0], # list of numbers (e.g. [0,1,2]) containg the number of the GPU
                
                'position_rotation': 0, # angle in radians. Rotation angle between detector and probe transverse coordinates
                
                'object_padding': 50, # pixels. Number of pixels to add around the object matrix
                
                'incoherent_modes': 0, # int. Number of incoherent modes to use
                
                'probe_support': [ "circular", 300,0,0 ], # support to be applied to the probe matrix after probe update. Options are:
                                                          # - ["circular",radius_pxls,center_y, center_x]; (0,0) is the center of the image
                                                          # - [ "cross", cross_width, border_padding, center_width ]; all values in pixels

                'distance_sample_focus': 0, # float. Distance in meters between sample and focus or pinhole. This distance is used to propagated the probe prior to application of the probe support. 
                
                "initial_obj": ["random"], # 2d array. Initial guess for the object. Options are:
                                           # - path to .npy, 
                                           # - path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                                           # - ["random"], random matrix with values between
                                           # - ["constant"], constant matrix of 1s

                "initial_probe": ["inverse"], # 2d array. Initial guess for the probe. Options are:
                                              # - path to .npy, 
                                              # - path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                                              # - ["random"], random matrix with values between
                                              # - ["constant"], constant matrix of 1s
                                              # - ["inverse"], matrix of the Inverse Fourier Transform of the mean of DPs.
                                              # - ["circular",radius,distance], circular mask with a pixel of "radius". If a distance (in meters) is given, it propagated the round probe using the ASM method.

                'Algorithm1': {'Batch': 64,
                                'Beta': 0.995,
                                'Epsilon': 0.01,
                                'Iterations': 70,
                                'Name': 'RAAR',
                                'ProbeCycles': 4,
                                'TV': 0},

                'Algorithm2': {'Batch': 64,
                                'Epsilon': 0.01,
                                'Iterations': 50,
                                'Name': 'GL',
                                'ObjBeta': 0.97,
                                'ProbeBeta': 0.95,
                                'TV': 0.0001},

                'Algorithm2': {'Batch': 64,
                                'Epsilon': 0.01,
                                'Iterations': 50,
                                'Name': 'positioncorrection',
                                'ObjBeta': 0.97,
                                'ProbeBeta': 0.95,
                                'TV': 0.0001},
                    
                'Algorithm3': { 'Name': 'PIE',
                                'Iterations': 100,
                                'step_obj': 0.5,    # step size for object update
                                'step_probe': 1,    # step size for probe update
                                'reg_obj': 0.25,    # regularization for object update
                                'reg_probe': 0.5,   # regularization for probe update
                                'Batch':1}
            }

    Returns:
        obj: object matrix 
        probe: probe matrix
        error: error metric along iterations
        positions: final positions of the scan (which may be corrected if AP_PC_CUDA is used)
    """    

    check_shape_of_inputs(DPs,positions,initial_probe) # check if dimensions are correct; exit program otherwise

    print(f'Pixel size = {input_dict["detector_pixel_size"]*1e6:.2f} um')
    
    print(f'Energy = {input_dict["energy"]} keV')
    
    if "wavenlegnth" not in input_dict:
        input_dict["wavelength"] = wavelength_meters_from_energy_keV(input_dict['energy'])
        print(f"Wavelength = {input_dict['wavelength']*1e9:.3f} nm")
    
    if "object_pixel" not in input_dict:
        input_dict["object_pixel"] = calculate_object_pixel_size(input_dict['wavelength'],input_dict['detector_distance'], input_dict['detector_pixel_size'],DPs.shape[1],binning=input_dict["binning"]) # in meters
        print(f"Object pixel = {input_dict['object_pixel']*1e9:.2f} nm")
    
    if "object_shape" not in input_dict:
        input_dict["object_shape"] = set_object_shape(input_dict["object_padding"], DPs.shape, positions)
        print(f"Object shape: {input_dict['object_shape']}")
    
    create_output_h5_file(input_dict)

    if 'Algorithm1' not in input_dict:
        obj, probe, error, positions = call_python_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)
    else:
        obj, probe, error, positions = call_CUDA_ptychography(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe)

    return obj, probe, error, positions

def check_shape_of_inputs(DPs,positions,initial_probe):

    if DPs.shape[0] != positions.shape[0]:
        raise ValueError(f'There is a problem with input data!\nThere are {DPs.shape[0]} diffractiom patterns and {positions.shape[0]} positions. These values should be the same.')    

    if initial_probe is not None:
        if DPs[0].shape[1] != initial_probe.shape[1] or DPs[0].shape[2] != initial_probe.shape[2]:
            raise ValueError(f'There is a problem with your input data!\nThe dimensions of input_probe and diffraction pattern differ in the X,Y directions: {DPs.shape} vs {initial_probe.shape}')

def create_output_h5_file(input_dict):

    with  h5py.File(input_dict["hdf5_output"], "w") as h5file:

        h5file.create_group("recon")
        h5file.create_group("metadata")

        h5file["metadata"].create_dataset('energy_keV',data=input_dict['energy']) 
        h5file["metadata"].create_dataset('wavelength_meters',data=input_dict['wavelength']) 
        h5file["metadata"].create_dataset('detector_distance_meters',data=input_dict['detector_distance']) 
        h5file["metadata"].create_dataset('distance_sample_focus',data=input_dict['distance_sample_focus']) 
        h5file["metadata"].create_dataset('detector_pixel_microns',data=input_dict['energy']) 
        h5file["metadata"].create_dataset('object_pixel_meters',data=input_dict['object_pixel']) 
        h5file["metadata"].create_dataset('cpus',data=input_dict['CPUs']) 
        h5file["metadata"].create_dataset('binning',data=input_dict['binning']) 
        h5file["metadata"].create_dataset('position_rotation_rad',data=input_dict['position_rotation']) 
        h5file["metadata"].create_dataset('object_padding_pixels',data=input_dict['object_padding'])
        h5file["metadata"].create_dataset('incoherent_modes',data=input_dict['incoherent_modes'])
        h5file["metadata"].create_dataset('fresnel_regime',data=input_dict['fresnel_regime']) 

        # lists, tuples, arrays
        h5file["metadata"].create_dataset('gpus',data=input_dict['GPUs']) 
        h5file["metadata"].create_dataset('object_shape',data=list(input_dict['object_shape']))
        
        h5file.create_group(f'metadata/probe_support')
        for key in input_dict['probe_support']: # save input probe
            h5file[f'metadata/probe_support'].create_dataset(key,data=input_dict['probe_support'][key])

        h5file.create_group(f'metadata/initial_obj')
        for key in input_dict['initial_obj']: # save input probe
            h5file[f'metadata/initial_obj'].create_dataset(key,data=input_dict['initial_obj'][key])

        h5file.create_group(f'metadata/initial_probe')
        for key in input_dict['initial_probe']: # save input probe
            h5file[f'metadata/initial_probe'].create_dataset(key,data=input_dict['initial_probe'][key])
        
        for key in input_dict['algorithms']: # save algorithms used
            h5file.create_group(f'metadata/algorithms/{key}')
            for subkey in input_dict['algorithms'][key]:
                h5file[f'metadata/algorithms/{key}'].create_dataset(subkey,data=input_dict['algorithms'][key][subkey])

    h5file.close()

def call_python_ptychography(input_dict,DPs, positions, initial_obj=None, initial_probe=None):
    """ 
    Wrapper for ptychography algorithms in Python by GCC.
    """

    if initial_probe == None:
        probe = set_initial_probe(input_dict, DPs ) # probe initial guess
    if initial_obj == None:
        obj = set_initial_object(input_dict,DPs,probe[0]) # object initial guess
        obj = np.expand_dims(obj,axis=0)

    if 'probe_support' in input_dict:
        input_dict["probe_support"] = get_probe_support(input_dict,probe.shape)
    else:
        input_dict["probe_support"] = np.ones_like(DPs[0])
        

    positions = positions.astype(np.int32)
    positions = np.roll(positions,shift=1,axis=1) # adjusting to the same standard as GB ptychography
    
    error = np.empty((0,1))
    
    inputs = input_dict
    for counter in range(1,1+len(input_dict['algorithms'].keys())):

        inputs['iterations'] = input_dict['algorithms'][str(counter)]['iterations'] 
        inputs['distance'] = input_dict["detector_distance"]
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
            inputs['use_mPIE'] = input_dict['algorithms'][str(counter)]['use_mPIE'] 
            obj, probe, algo_error = PIE_multiprobe_loop(DPs, positions,obj[0],probe[0], inputs)

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            obj, probe, algo_error = RAAR_multiprobe_cupy(DPs,positions,obj[0],probe[0],inputs)
            obj = np.expand_dims(obj,axis=0) # obj coming with one dimensions less. needs to be fixed
        else:
            sys.exit('Please select a proper algorithm! Selected: ', inputs["algorithm"])

        error = np.concatenate((error,algo_error),axis=0)

    return obj, probe, error, None

def call_CUDA_ptychography(input_dict,DPs, probe_positions, initial_obj=None, initial_probe=None):
    """ Call Ptychography CUDA codes
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
    loop_counter = 1
    error = np.empty((0,))

    corrected_positions = None

    while True:  # run Ptycho:
        try:
            algorithm: dict = input_dict['Algorithm' + str(loop_counter)]
            algo_name = algorithm["Name"]
            n_of_iterations = algorithm['Iterations']
            print(f"\tCalling {n_of_iterations} iterations of {algo_name} algorithm...")
        except:
            break

        if algorithm['Name'] == 'GL':
            datapack = GL(iter      = algorithm['Iterations'],
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
            datapack = PosCorrection(iter       = algorithm['Iterations'],
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
            datapack = RAAR(iter         = algorithm['Iterations'],
                                       beta        = algorithm['Beta'],
                                       probecycles = algorithm['ProbeCycles'],
                                       batch       = algorithm['Batch'],
                                       epsilon     = algorithm['Epsilon'],
                                       tvmu        = algorithm['TV'],
                                       sigmask     = sigmask,
                                       data        = datapack,
                                       params      = {'device':input_dict["GPUs"]},
                                       probef1=input_dict['fresnel_number'])

        elif algorithm['Name'] == 'PIE':
            datapack = PIE(iterations = algorithm['Iterations'],
                                     step_obj = algorithm['step_obj'],
                                     step_probe = algorithm['step_probe'],
                                     reg_obj = algorithm['reg_obj'],
                                     reg_probe = algorithm['reg_probe'],
                                     rois = datapack['rois'],
                                     difpads = datapack['difpads'],
                                     obj = datapack['obj'],
                                     probe = datapack['probe'],
                                     params = { 'device': input_dict["GPUs"] })

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
    
    if input_dict["distance_sample_focus"] == 0:
        input_dict['fresnel_number'] = 0
    else:
        input_dict['fresnel_number'] = input_dict["detector_pixel_size"]**2/(input_dict["wavelength"]*input_dict["distance_sample_focus"])
    

    print(f'Distance between sample and focus: {input_dict["distance_sample_focus"]*1e3}mm')
    print(f'Fresnel number: {input_dict["fresnel_number"]}')

    probe_positions = append_ones(probe_positions)

    probe = set_initial_probe(input_dict, DPs) # probe initial guess.
    
    probe_support = get_probe_support(input_dict, probe.shape)
    
    obj = set_initial_object(input_dict,DPs,probe) # object initial guess

    sigmask = set_sigmask(DPs) # mask for invalid pixels
    background = np.ones(DPs[0].shape) # dummy array 

    print(f"Diffraction Patterns: {DPs.shape}\nInitial Object: {obj.shape}\nInitial Probe: {probe.shape}\nProbe Support: {probe_support.shape}\nProbe Positions: {probe_positions.shape}")
    
    datapack = set_datapack(obj, probe, probe_positions, DPs, background, probe_support)     # Set data for Ptycho algorithms:

    print(f"Total datapack size: {estimate_memory_usage(datapack['obj'],datapack['probe'],datapack['rois'],datapack['difpads'],datapack['bkg'],datapack['probesupp'])[3]:.2f} GBs")

    return datapack, sigmask

def set_initial_probe(input_dict,DPs):
    print('Creating initial probe...')

    DP_shape = (DPs.shape[1], DPs.shape[2])

    def set_modes(probe, input_dict):
        mode = probe.shape[0]

        if input_dict['incoherent_modes'] > mode:
            add = input_dict['incoherent_modes'] - mode
            probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
            for i in range(add):
                probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

        print("Probe shape ({0},{1}) with {2} incoherent mode(s)".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

        return probe


    type_of_initial_guess = detect_variable_type_of_guess(input_dict['initial_probe']["probe"])

    if type_of_initial_guess == 'standard':
        
        if input_dict['initial_probe']['probe'] == 'circular':
            probe = create_circular_mask(DP_shape,input_dict['initial_probe']["radius"])
            probe = probe*np(1j*probe)
            if input_dict['initial_probe'][2] != 0: # propagate probe 
                probe = fresnel_propagator_cone_beam(probe, input_dict['wavelength'],input_dict['object_pixel'], input_dict["distance_sample_focus"])
            else:
                pass
        elif input_dict['initial_probe']['probe'] == 'cross':
            cross_width_y, border, center_square_side = input_dict['initial_probe']["cross_width"],input_dict['initial_probe']["border_padding"],input_dict['initial_probe'][center_width]
            probe = create_cross_mask(DP_shape,cross_width_y, border, center_square_side)
        elif input_dict['initial_probe']['probe'] == 'constant':
            probe = np.ones(DP_shape)
        elif input_dict['initial_probe']['probe'] == 'random':
            probe = np.random.rand(*DP_shape)
        elif input_dict['initial_probe']['probe'] == 'inverse' or input_dict['initial_probe']['probe'] == 'ift':
            DPs_avg =  np.average(DPs, 0)[None] 
            ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(DPs_avg)))
            probe = np.sqrt(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft))))
        else:
            sys.exit("Please select an appropriate type for probe initial guess: circular, squared, rectangular, cross, constant, random")

    elif type_of_initial_guess == 'path':
        path = input_dict['initial_probe']["probe"]        
        if os.path.splitext(path)[1] == '.hdf5' or os.path.splitext(path)[1] == '.h5':
            probe = h5py.File(path,'r')['recon/probe'][()]
        elif os.path.splitext(path)[1] == '.npy':
            probe = np.load(path) # load guess from file
    
    elif type_of_initial_guess == 'array':
        probe = input_dict['initial_probe']['probe']
    
    else:
        raise ValueError(f"Select an appropriate initial guess for the probe:{input_dict['initial_probe']}")

    probe = probe.astype(np.complex64)

    if probe.shape[0] <= 1:
        probe = set_modes(probe, input_dict) # add incoherent modes

    return probe

def detect_variable_type_of_guess(variable):
    if isinstance(variable, str):
        if os.path.isfile(variable):
            return "path"
        else:
            return "standard"
    elif isinstance(variable, np.ndarray):
        return "array"
    else:
        raise ValueError("Your input for the initial guess is wrong.")

def set_initial_object(input_dict,DPs, probe):
    print('Creating initial object...')

    type_of_initial_guess = detect_variable_type_of_guess(input_dict['initial_obj']["obj"])

    if type_of_initial_guess == 'standard':
        if input_dict['initial_obj']['obj'] == 'constant':
            obj = np.ones(input_dict["object_shape"])
        elif input_dict['initial_obj']['obj'] == 'random':
            normalization_factor = np.sqrt(np.average(DPs) / np.average(abs(np.fft.fft2(probe))**2))
            obj = np.random.rand(*input_dict["object_shape"]) * normalization_factor
        elif input_dict['initial_obj']['obj'] == 'complex_random':
            obj =  1 * (np.random.rand(*input_dict["object_shape"]) + 1j*np.random.rand(*input_dict["object_shape"]))
        elif input_dict['initial_obj']['obj'] == 'initialize':
            pass #TODO: implement method from https://doi.org/10.1364/OE.465397
    elif type_of_initial_guess == 'path':
        if os.path.splitext(input_dict['initial_obj']['obj'])[1] == '.hdf5' or os.path.splitext(input_dict['initial_obj']['obj'])[1] == '.h5':
            obj = h5py.File(input_dict['initial_obj']['obj'],'r')['recon/object'][0] # select first frame of object
        elif os.path.splitext(input_dict['initial_obj']['obj'])[1] == '.npy':
            obj = np.load(input_dict['initial_obj']['obj'])
        obj = np.squeeze(obj)
    elif type_of_initial_guess == 'array':
        obj = input_dict['initial_obj']['obj']
    else:
        raise ValueError(f"Select an appropriate initial guess for the object:{input_dict['initial_obj']}")

    complex_obj = obj.astype(np.complex64)


    return complex_obj

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
    
    if input_dict["probe_support"]["type"] == "circular":
        radius, center_x, center_y = input_dict["probe_support"]["radius"], input_dict["probe_support"]["center_y"], input_dict["probe_support"]["center_x"]

        half_size = probe_shape[-1]//2

        ar = np.arange(-half_size, half_size)
        xx, yy = np.meshgrid(ar, ar)
        support = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2

        probe[:] = support # all frames and all modes with same support

    elif input_dict["probe_support"]["type"] == "cross":
        cross_width_y, border, center_square_side = input_dict['probe_support']["cross_width"],input_dict['probe_support']["border_padding"],input_dict['probe_support']['center_width']
        probe[:] = create_cross_mask((probe_shape[1],probe_shape[2]),cross_width_y, border, center_square_side)

    elif input_dict["probe_support"]["type"] == "array":
        probe = input_dict["probe_support"]["type"]["data"]

    else: 
        raise ValueError(f"Select an appropriate probe support:{input_dict['probe_support']}")

    return probe

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

def calculate_object_pixel_size(wavelength,detector_distance, detector_pixel_size,n_of_pixels,binning=1):
    return wavelength * detector_distance / (binning*detector_pixel_size * n_of_pixels)

def set_object_pixel_size(input_dict,DP_size):
    """ Get size of object pixel given the energy, distance and detector pixel size

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "energy": beamline energy in keV
                "detetor_distance": distance in meters
                "binning": binning factor, if diffraction patterns were binned
                "detector_pixel_size": detector pixel size in meters
        DP_size (int): lateral size of detector array in pixels

    Returns:
        input_dict: update input dictionary containing size of object pixel and wavelength
    """
    
    object_pixel_size = calculate_object_pixel_size(input_dict['wavelength'],input_dict['detector_distance'], input_dict['detector_pixel_size'],DP_size,binning=input_dict["binning"])
    input_dict["object_pixel"] = object_pixel_size # in meters
    print(f"\tObject pixel size = {object_pixel_size*1e9:.2f} nm")

    PA_thickness = 4*object_pixel_size**2/(0.61*input_dict['wavelength'])
    print(f"\tLimit thickness for resolution of 1 pixel: {PA_thickness*1e6:.3f} microns")
    return input_dict

def set_object_shape(object_padding, DP_shape, probe_positions):
    """ Determines shape (Y,X) of object matrix from size of probe and its positions.

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys:
                "object_padding": number of pixels to pad in the border of the object array.
                "object_shape": object size/shape
        DP_shape (tuple): shape of the diffraction patterns array
        probe_positions (numpy array): array os probe positiions in pixels

    Returns:
        input_dict (dict)): updated input dictionary containing object_shape information
    """

    offset_bottomright = object_padding
    DP_size_y, DP_size_x = DP_shape[1:]

    maximum_probe_coordinate_x = int(np.max(probe_positions[:,1]))
    object_shape_x  = DP_size_x + maximum_probe_coordinate_x + offset_bottomright

    maximum_probe_coordinate_y = int(np.max(probe_positions[:,0]))
    object_shape_y  = DP_size_y + maximum_probe_coordinate_y + offset_bottomright

    return (object_shape_y, object_shape_x)





