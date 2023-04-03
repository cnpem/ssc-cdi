

import numpy as np
import sys, os
import sscPtycho
from ..processing.propagation import calculate_fresnel_number
from ..misc import  create_circular_mask, create_rectangular_mask, create_cross_mask, estimate_memory_usage, plot_error

def call_G_ptychography(input_dict,DPs, probe_positions, initial_obj=np.ones(1), initial_probe=np.ones(1)):

    if initial_obj != np.ones(1):
        input_dict["initial_obj"] = initial_obj
    if initial_probe != np.ones(1):
        input_dict["initial_probe"] = initial_probe

    probe_support_radius, probe_support_center_x, probe_support_center_y = input_dict["probe_support"]

    datapack, _, sigmask = set_initial_parameters_for_G_algos(input_dict,DPs,probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,input_dict["object_shape"],input_dict["object_pixel"])

    print('\nStarting ptychography...')
    run_algorithms = True
    loop_counter = 1
    error = np.empty((0,))
    while run_algorithms:  # run Ptycho:
        try:
            algorithm = input_dict['Algorithm' + str(loop_counter)]
            algo_name = algorithm["Name"]
            n_of_iterations = algorithm['Iterations']
            print(f"Calling {n_of_iterations} iterations of {algo_name} algorithm...")
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

    np.save(os.path.join(input_dict["output_path"],f"error.npy"),error)
    plot_error(error,path=os.path.join(input_dict["output_path"],f"error.png"),log=True)

    return datapack['obj'], datapack['probe']



def set_initial_parameters_for_G_algos(input_dict, DPs, probe_positions, radius, center_x, center_y, object_size, dx):

    def set_sigmask(DPs):
        """Create a mask for invalid pixels

        Args:
            DPs (array): measured diffraction patterns

        Returns:
            sigmask (array): 2D-array, same shape of a diffraction pattern, maps the invalid pixels
            0 for negative values, intensity measured elsewhere
        """    
        # mask of 1 and 0:
        sigmask = np.ones(DPs[0].shape)
        sigmask[DPs[0] < 0] = 0

        return sigmask

    def probe_support(probe, half_size, radius, center_x, center_y):
        print('Setting probe support...')
        ar = np.arange(-half_size, half_size)
        xx, yy = np.meshgrid(ar, ar)
        probesupp = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2 
        probesupp = np.asarray([probesupp for k in range(probe.shape[0])])
        return probesupp

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

    def append_zeros(probe_positions):
        zeros = np.zeros((probe_positions.shape[0],1))
        probe_positions = np.concatenate((probe_positions,zeros),axis=1)
        probe_positions = np.concatenate((probe_positions,zeros),axis=1) # concatenate columns to use Giovanni's ptychography code
        probe_positions2 = np.zeros_like(probe_positions)
        probe_positions2[:,0] = probe_positions[:,1] # change x and y column order
        probe_positions2[:,1] = probe_positions[:,0]
        return probe_positions2
    
    half_size = DPs.shape[-1] // 2

    if input_dict['fresnel_number'] == -1:  # Manually choose wether to find Fresnel number automatically or not
        input_dict['fresnel_number'] = calculate_fresnel_number(dx, pixel=input_dict['restored_pixel_size'], energy=input_dict['energy'], z=input_dict['detector_distance'])
    print('Fresnel number:', input_dict['fresnel_number'])

    probe_positions = append_zeros(probe_positions)

    probe = set_initial_probe(input_dict, (DPs.shape[1], DPs.shape[2]) ) # Compute probe: initial guess:
    
    obj = set_initial_object(input_dict) # Object initial guess:

    sigmask = set_sigmask(DPs)  # mask of 1 and 0:

    background = np.ones(DPs[0].shape) # dummy

    probesupp = probe_support(probe, half_size, radius, center_x, center_y)  # Compute probe support:

    print(f"\n\tDiffraction Patterns: {DPs.shape}\n\tInitial Object: {obj.shape}\n\tInitial Probe: {probe.shape}\n\tProbe Support: {probesupp.shape}\n\tProbe Positions: {probe_positions.shape}\n")
    
    datapack = set_datapack(obj, probe, probe_positions, DPs, background, probesupp)     # Set data for Ptycho algorithms:

    print(f"Total datapack size: {estimate_memory_usage(datapack['obj'],datapack['probe'],datapack['rois'],datapack['difpads'],datapack['bkg'],datapack['probesupp'])[3]:.2f} GBs")

    return datapack, probe_positions, sigmask


def set_initial_probe(input_dict,DP_shape):

    def set_modes(probe, input_dict):

        mode = probe.shape[0]

        if input_dict['incoherent_modes'] > mode:
            add = input_dict['incoherent_modes'] - mode
            probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
            for i in range(add):
                probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

        print("\tProbe shape ({0},{1}) with {2} incoherent modes".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

        return probe

    print('Creating initial probe...')

    if isinstance(input_dict['initial_probe'],list): # if no path to file given
        
        type = input_dict['initial_probe'][0]

        if type == 'circular':
            probe = create_circular_mask(input_dict["DP_center"],input_dict['initial_probe'][1],DP_shape)
        elif type == 'squared':
            probe = create_rectangular_mask(DP_shape,input_dict["DP_center"],input_dict['initial_probe'][1])
        elif type == 'rectangular':
            probe = create_rectangular_mask(DP_shape,input_dict["DP_center"],input_dict['initial_probe'][1],input_dict['initial_probe'][2])
        elif type == 'cross':
            probe = create_cross_mask(DP_shape,input_dict["DP_center"],input_dict['initial_probe'][1],input_dict['initial_probe'][2])
        elif type == 'constant':
            probe = np.ones(DP_shape)
        elif type == 'random':
            probe = np.random.rand(*DP_shape)
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

    probe = np.expand_dims(probe,axis=0)

    probe = set_modes(probe, input_dict) # add incoherent modes 

    return probe


def set_initial_object(input_dict):

        print('Creating initial object...')

        if isinstance(input_dict['initial_obj'],list):
            type = input_dict['initial_obj'][0]
            if type == 'constant':
                obj = np.ones(input_dict["object_shape"])
            elif type == 'random':
                obj = np.random.rand(*input_dict["object_shape"])
            elif type == 'initialize':
                pass #TODO: implement method from https://doi.org/10.1364/OE.465397
        elif isinstance(input_dict['initial_obj'],str): 
            obj = np.load(input_dict['initial_obj'])
            obj = np.squeeze(obj)
        elif isinstance(input_dict['initial_obj'],int):
            obj = np.load(os.path.join(input_dict["output_path"],input_dict["output_path"].rsplit('/',2)[1]+"_object.npy"))
        else:
            sys.exit("Please select an appropriate path or type for object initial guess: autocorrelation, constant, random")

        return obj



