

import numpy as np
import sys
import sscPtycho
from ..processing.propagation import calculate_fresnel_number
from ..misc import  create_circular_mask, create_rectangular_mask, create_cross_mask

def call_G_ptychography(input_dict,DPs, probe_positions, initial_obj=np.ones(1), initial_probe=np.ones(1)):

    if initial_obj != np.ones(1):
        input_dict["initial_obj"] = initial_obj
    if initial_probe != np.ones(1):
        input_dict["initial_probe"] = initial_probe

    probe_support_radius, probe_support_center_x, probe_support_center_y = input_dict["probe_support"]

    datapack, _, sigmask = set_initial_parameters_for_G_algos(input_dict,DPs,probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,input_dict["object_shape"],input_dict["object_pixel"])

    run_algorithms = True
    loop_counter = 1
    while run_algorithms:  # run Ptycho:
        try:
            algorithm = input_dict['Algorithm' + str(loop_counter)]
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
                                        probefresnel_number=input_dict['fresnel_number'])

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
                                                    probefresnel_number=input_dict['fresnel_number'])

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
                                           probefresnel_number=input_dict['fresnel_number']) 

            loop_counter += 1
            
            RF = datapack['error']

    return datapack['obj'], datapack['probe']



def set_initial_parameters_for_G_algos(input_dict, difpads, probe_positions, radius, center_x, center_y, object_size, dx):

    def set_sigmask(difpads):
        """Create a mask for invalid pixels

        Args:
            difpads (array): measured diffraction patterns

        Returns:
            sigmask (array): 2D-array, same shape of a diffraction pattern, maps the invalid pixels
            0 for negative values, intensity measured elsewhere
        """    
        # mask of 1 and 0:
        sigmask = np.ones(difpads[0].shape)
        sigmask[difpads[0] < 0] = 0

        return sigmask


    def probe_support(probe, half_size, radius, center_x, center_y):
        print('Setting probe support...')
        ar = np.arange(-half_size, half_size)
        xx, yy = np.meshgrid(ar, ar)
        probesupp = (xx + center_x) ** 2 + (yy + center_y) ** 2 < radius ** 2  # offset of 30 chosen by hand?
        probesupp = np.asarray([probesupp for k in range(probe.shape[0])])
        return probesupp

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

    half_size = difpads.shape[-1] // 2

    if input_dict['fresnel_number'] == -1:  # Manually choose wether to find Fresnel number automatically or not
        input_dict['fresnel_number'] = calculate_fresnel_number(dx, pixel=input_dict['restored_pixel_size'], energy=input_dict['energy'], z=input_dict['detector_distance'])
        input_dict['fresnel_number'] = -input_dict['fresnel_number']
    print('\tF1 value:', input_dict['fresnel_number'])

    # Compute probe: initial guess:
    probe = set_initial_probe(difpads, input_dict)

    # Object initial guess:
    obj = set_initial_object(input_dict, object_size, probe, difpads)

    # mask of 1 and 0:
    sigmask = set_sigmask(difpads)

    background = np.ones(difpads[0].shape) # dummy

    # Compute probe support:
    probesupp = probe_support(probe, half_size, radius, center_x, center_y)

    probe_positionsi = probe_positions + 0  # what's the purpose of declaring probe_positionsi?

    # Set data for Ptycho algorithms:
    datapack = set_datapack(obj, probe, probe_positions, difpads, background, probesupp)

    return datapack, probe_positionsi, sigmask


def set_initial_probe(input_dict,DP_shape):

    def set_modes(probe, input_dict):

        mode = probe.shape[0]
        print('\tNumber of modes:', mode)
        # Adicionar modulos incoerentes
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
            probe = np.ones(*DP_shape)
        elif type == 'random':
            probe = np.random.rand(*DP_shape)
        else:
            sys.error("Please select an appropriate type for probe initial guess: circular, squared, rectangular, cross, constant, random")

    elif isinstance(input_dict['initial_probe'],str):
        probe = np.load(input_dict['initial_probe'])[0][0] # load guess from file
        probe = probe.reshape((1,1,*probe.shape))
    elif isinstance(input_dict['initial_probe'],np.ndarray):
        pass
    else:
        sys.error("Please select an appropriate path or type for probe initial guess: circular, squared, cross, constant")

    print("\tProbe shape:", probe.shape)

    if input_dict['incoherent_modes'] > 1:
        print(f"\tSetting initial incoherent modes: {input_dict['incoherent_modes']} modes")
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
            obj = np.load(input_dict['initial_obj'])[0]
        elif isinstance(input_dict['initial_obj'],np.ndarray):
            pass
        else:
            sys.error("Please select an appropriate path or type for object initial guess: autocorrelation, constant, random")

        return obj



