

import numpy as np
import sys, os
import sscPtycho
from ..misc import estimate_memory_usage, add_to_hdf5_group, concatenate_array_to_h5_dataset

def call_GB_ptychography(input_dict,DPs, probe_positions, initial_obj=np.ones(1), initial_probe=np.ones(1)):
    """ Call Ptychography CUDA codes developed by Giovanni Baraldi

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
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
    
    if initial_obj!=np.ones(1):
        datapack["obj"] = initial_obj
    
    if initial_probe!=np.ones(1):
        datapack["probe"] = initial_obj

    concatenate_array_to_h5_dataset(input_dict["hdf5_output"],'recon','initial_object',datapack["obj"],concatenate=False)
    concatenate_array_to_h5_dataset(input_dict["hdf5_output"],'recon','initial_probe',datapack["probe"],concatenate=False)

    print(f'Starting ptychography... using {len(input_dict["GPUs"])} GPUs {input_dict["GPUs"]} and {input_dict["CPUs"]} CPUs')
    run_algorithms = True
    loop_counter = 1
    error = np.empty((0,))
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

    return datapack['obj'], datapack['probe'], error


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

    def probe_support(n_of_probes, half_size, radius, center_x, center_y):
        """ Create mask containing probe support region

        Args:
            n_of_probes (_type_): number of probes
            half_size (_type_): half the size of one dimension of the probe array
            radius (_type_): radius of the support region
            center_x (_type_): center coordinate of support ball in x direction
            center_y (_type_): center coordinate of support ball in y direction

        Returns:
            probesupp: mask containing probe support
        """
        print('Setting probe support...')
        ar = np.arange(-half_size, half_size)
        xx, yy = np.meshgrid(ar, ar)
        probesupp = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2 
        probesupp = np.asarray([probesupp for k in range(n_of_probes)])
        return probesupp

    def append_ones(probe_positions):
        """ Adjust shape and column order of positions array to be accepted by Giovanni's  code

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
    
    obj = set_initial_object(input_dict,DPs,probe) # object initial guess

    sigmask = set_sigmask(DPs) # mask for invalid pixels

    background = np.ones(DPs[0].shape) # dummy array 

    probe_support_radius, probe_support_center_x, probe_support_center_y = input_dict["probe_support"]
    probesupp = probe_support(probe.shape[0], half_size, probe_support_radius, probe_support_center_x, probe_support_center_y)  

    print(f"\nDiffraction Patterns: {DPs.shape}\nInitial Object: {obj.shape}\nInitial Probe: {probe.shape}\nProbe Support: {probesupp.shape}\nProbe Positions: {probe_positions.shape}\n")
    
    datapack = set_datapack(obj, probe, probe_positions, DPs, background, probesupp)     # Set data for Ptycho algorithms:

    print(f"Total datapack size: {estimate_memory_usage(datapack['obj'],datapack['probe'],datapack['rois'],datapack['difpads'],datapack['bkg'],datapack['probesupp'])[3]:.2f} GBs")

    return datapack, sigmask


def set_initial_probe(input_dict,DP_shape,DPs_avg):
    """ Get initial probe with multiple modes, with format required by Giovanni's code

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
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
            probe = create_cross_mask(DP_shape,input_dict["DP_center"],input_dict['initial_probe'][1],input_dict['initial_probe'][2])
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
        probe = np.load(input_dict['initial_probe'])[0] # load guess from file
        probe = np.squeeze(probe)
        probe = probe.reshape((1,*probe.shape))
    elif isinstance(input_dict['initial_probe'],int):
        probe = np.load(os.path.join(input_dict["output_path"],input_dict["output_path"].rsplit('/',2)[1]+"_probe.npy"))
    else:
        sys.exit("Please select an appropriate path or type for probe initial guess: circular, squared, cross, constant")

    probe = probe.astype(np.complex64)
    probe = np.expand_dims(probe,axis=0)

    probe = set_modes(probe, input_dict) # add incoherent modes 

    return probe


def set_initial_object(input_dict,DPs, probe):
    """ Get initial object from file at input dictionary or define a constant or random matrix for it

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
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
        elif type == 'initialize':
            pass #TODO: implement method from https://doi.org/10.1364/OE.465397
    elif isinstance(input_dict['initial_obj'],str): 
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


def create_cross_mask(mask_shape,center, length_y, length_x=0):
    if length_x == 0: length_x = length_y
    """ All values in pixels """
    center_row, center_col = center
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    mask = np.zeros(*mask_shape)
    mask[center_row-length_y//2:center_row+length_y//2,:] = 1
    mask[:,center_col-length_x//2:center_col+length_x//2] = 1
    return mask 