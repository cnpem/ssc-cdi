# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import numpy as np
import sys, os, h5py
import random

from ..cditypes import AP, PIE, RAAR

from ..misc import estimate_memory_usage, wavelength_meters_from_energy_keV, calculate_object_pixel_size
from ..processing.propagation import fresnel_propagator_cone_beam
from .pie import PIE_python
from .raar import RAAR_python
from .ML import ML_cupy

from .ptycho_plots import plot_ptycho_scan_points, plot_probe_modes, get_extent_from_pixel_size, plot_iteration_error, plot_amplitude_and_phase, get_plot_extent_from_positions, plot_probe_support,plot_ptycho_corrected_scan_points,plot_object_spectrum

random.seed(0)

def call_ptychography(input_dict, DPs, positions, initial_obj=None, initial_probe=None,plot=True):
    """
    Call Ptychography algorithms. Options are:

    - `RAAR_python`: Relaxed Averaged Alternating Reflections. Single GPU, Python implementation using CuPy
    - `ePIE_python`: Extended Ptychographic Iterative Engine. Single GPU, Python implementation using CuPy
    - `RAAR`: Relaxed Averaged Alternating Reflections. Multi GPU, CUDA implementation
    - `AP`: Alternate Projections. Multi GPU, CUDA implementation
    - `rPIE`: regularized Ptychographic Iterative Engine. Single GPU, CUDA implementation

    Args:
        DPs (ndarray): Diffraction data with shape (N, Y, X). N is the number of diffraction patterns.
        positions (ndarray): Positions array with shape (N, 2) with (y, x) position pairs in each line.
        initial_obj (ndarray, optional): Initial guess for object. Shape to be determined from DPs and positions. 
            If None, will use the input in "input_dict" to determine the initial object. Defaults to None.
        initial_probe (ndarray, optional): Initial guess for probe of shape (M, Y, X), where M is the number of probe modes. 
            If None, will use the input in "input_dict" to determine the initial probe. Defaults to None.
        input_dict (dict): Dictionary of inputs required for Ptychography.
            
    Returns:
        obj: object matrix
        probe: probe matrix
        corrected_positions: None, except if AP_PC_CUDA is used
        input_dict: updated input dictionary
        error: Error metric along iterations

    Example of input_dict::

        input_dict = {
            "hdf5_output": './output.h5',  # Path to hdf5 file to contain all outputs
            'CPUs': 32,  # Number of CPUs to use in parallel execution
            'GPUs': [0],  # List of GPU indices (e.g. [0,1,2])
            "fresnel_regime": False,  # Only available for Python engines
            'energy': 6,  # Energy in keV
            'detector_distance': 13,  # Distance in meters
            'distance_sample_focus': 0,  # Distance in meters between sample and focus or pinhole
            'detector_pixel_size': 55e-6,  # Detector pixel size in meters
            'binning': 1,  # Binning factor. Must be an even number. If 1, no binning occurs.
            'position_rotation': 0,  # Rotation angle in radians to correct for misalignment
            'positions_unit': 'pixels',  # Units of positions. Options are 'pixels' or 'meters'
            'object_padding': 0,  # Number of pixels to add around the object matrix
            'incoherent_modes': 1,  # Number of incoherent modes to use

            'probe_support': {"type": "circular",  "radius": 1000,  "center_y": 0, "center_x": 0} , # support to be applied to the probe matrix after probe update. Options are:
                                                                                                    # - {"type": "circular",  "radius": 300,  "center_y": 0, "center_x": 0} (0,0) is the center of the image
                                                                                                    # - {"type": "cross",  "center_width": 300,  "cross_width": 0, "border_padding": 0} 
                                                                                                    # - {"type": "array",  "data": myArray}


            "initial_obj": {"obj": 'random'},     # 2d array. Initial guess for the object. Options are:
                                                # - {"obj": my2darray}, numpy array 
                                                # - {"obj": 'path/to/numpyFile.npy'}, path to .npy, 
                                                # - {"obj": 'path/to/hdf5File.h5'}, path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                                                # - {"obj": 'random'}, random matrix with values between 0 and 1
                                                # - {"obj": 'constant'}, constant matrix of 1s

            'initial_probe': { "probe": 'fzp', # creates initial guess based on modelled FZP
                            'beam_type': 'disc',  # 'disc' or 'gaussian'                 
                            'distance_sample_fzpf': 2.9e-3, # distance between sample and fzp focus        
                            'fzp_diameter': 50e-6,               
                            'fzp_outer_zone_width': 50e-9,     
                            'beamstopper_diameter': 20e-6,  # beamstopper placed before fzp. if 0, no beamstopper used      
                            'probe_diameter': 50e-6, # if not included, will use the same diameter s the fzp
                            'probe_normalize':False},  # normalizes fzp probe at end        
                            # - {"probe": my2darray}, numpy array 
                            # - {"probe": 'path/to/numpyFile.npy'}, path to .npy, 
                            # - {"probe": 'path/to/hdf5File.h5'}, path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                            # - {"probe": 'random'}, random matrix with values between 0 and 1
                            # - {"probe": 'constant'}, constant matrix of 1s
                            # - {"probe": 'inverse'}, matrix of the Inverse Fourier Transform of the mean of DPs.
                            # - {"probe": 'circular', "radius": 100, "distance":0},  circular mask with a pixel of "radius". If a distance (in meters) is given, it propagated the round probe using the ASM method 

            'algorithms': {  # Algorithms to be used
                '1': {
                    'name': 'RAAR_python',
                    'iterations': 50,
                    'regularization_object': 0.01,
                    'regularization_probe': 0.01,
                    'step_object': 1.0,
                    'step_probe': 1.0,
                },
                '2': {
                    'name': 'ePIE_python',
                    'iterations': 20,
                    'regularization_object': 0.25,
                    'regularization_probe': 0.5,
                    'step_object': 0.5,
                    'step_probe': 1,
                    'use_mPIE': False,
                    'mPIE_friction_obj': 0.9,
                    'mPIE_friction_probe': 0.99,
                    'mPIE_momentum_counter': 10,
                },
                '3': {
                    'name':'RAAR',
                    'iterations': 100,
                    'beta': 0.9,
                    'step_object': 1.0,
                    'step_probe': 1.0,
                    'regularization_object': 0.01,
                    'regularization_probe': 0.01,
                    'momentum_obj': 0.0,
                    'momentum_probe': 0.0,
                    'batch': 64
                },
                '4': {
                    'name':'AP',
                    'iterations': 50,
                    'step_object': 1.0,
                    'step_probe': 1.0,
                    'regularization_object': 0.01,
                    'regularization_probe': 0.01,
                    'momentum_obj': 0.5,
                    'momentum_probe': 0.5,
                    'batch': 64,
                },
                '5': {
                    'name':'PIE',
                    'iterations': 50,
                    'step_object': 1.0,
                    'step_probe': 1.0,
                    'regularization_object': 0.5,
                    'regularization_probe': 0.5,
                    'momentum_obj': 0.5,
                    'momentum_probe': 0.5,
                    'batch': 64,
                }
                }
            }
    """

    check_shape_of_inputs(DPs,positions,initial_probe) # check if dimensions are correct; exit program otherwise

    DPs, initial_obj, initial_probe = check_dtypes(DPs,initial_obj,initial_probe) # check if dtypes are correct; exit program otherwise

    input_dict = check_and_set_defaults(input_dict)

    if check_consecutive_keys(input_dict['algorithms']) == False:
        raise ValueError('Keys in algorithms dictionary should be consecutive integers starting from 1. For example: {1: {...}, 2: {...}, 3: {...}}')


    if input_dict['n_of_positions_to_remove'] > 0:
        positions,DPs = remove_positions_randomly(positions,DPs, input_dict['n_of_positions_to_remove'])

    if input_dict['binning']>1:
        DPs = bin_volume(DPs, input_dict['binning']) # binning of diffraction patterns
        print(f'Detector pixel size downsampled from {input_dict["detector_pixel_size"]*1e6:.2f} um to {input_dict["detector_pixel_size"]*1e6*input_dict["binning"]:.2f} um')
        input_dict["detector_pixel_size"] = input_dict["detector_pixel_size"]*input_dict['binning']

    print(f'Data shape: {DPs.shape}')

    estimated_size_for_all_DPs = estimate_memory_usage(DPs)[3]
    print(f"Estimated size for {DPs.shape[0]} DPs of type {DPs.dtype}: {estimated_size_for_all_DPs:.2f} GBs")
    print(f'Detector pixel size = {input_dict["detector_pixel_size"]*1e6:.2f} um')
    
    print(f'Energy = {input_dict["energy"]} keV')
    
    if "wavenlegnth" not in input_dict:
        input_dict["wavelength"] = wavelength_meters_from_energy_keV(input_dict['energy'])
        print(f"Wavelength = {input_dict['wavelength']*1e9:.3f} nm")
    
    if "object_pixel" not in input_dict:
        input_dict["object_pixel"] = calculate_object_pixel_size(input_dict['wavelength'],input_dict['detector_distance'], input_dict['detector_pixel_size'],DPs.shape[1]) # meters
        print(f"Object pixel = {input_dict['object_pixel']*1e9:.2f} nm")
    
    if 'positions_unit' not in input_dict:
        pass
    elif input_dict['positions_unit'] == 'meters' or input_dict['positions_unit'] == 'm':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1)
    elif input_dict['positions_unit'] == 'millimeters' or input_dict['positions_unit'] == 'mm':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1e-3)
    elif input_dict['positions_unit'] == 'microns' or input_dict['positions_unit'] == 'micrometers' or input_dict['positions_unit'] == 'um':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1e-6)
    elif input_dict['positions_unit'] == 'pixels':
        pass

    if "object_shape" not in input_dict:
        input_dict["object_shape"] = set_object_shape(input_dict["object_padding"], DPs.shape, positions)
        print(f"Object shape: {input_dict['object_shape']}")
    
    if plot: plot_ptycho_scan_points(positions,pixel_size=input_dict["object_pixel"])

    if input_dict['hdf5_output'] is not None:
        print('Creating output hdf5 file...')
        if os.path.exists(os.path.dirname(input_dict['hdf5_output'])) == False:
            print('Folder does not exist. Creating it...')
            os.makedirs(os.path.dirname(input_dict['hdf5_output']))

        create_output_h5_file(input_dict)

    obj, probe, error, corrected_positions, initial_obj, initial_probe = call_ptychography_engines(input_dict,DPs, positions, initial_obj=initial_obj, initial_probe=initial_probe,plot=plot)

    if plot == True and corrected_positions is not None:
        plot_ptycho_corrected_scan_points(positions,corrected_positions)

    if plot: 
        print('Plotting final object and probe...')
        plot_amplitude_and_phase(obj, positions=positions+probe.shape[-1]//2,extent=get_plot_extent_from_positions(positions))
        plot_object_spectrum(obj,cmap='gray')
        plot_probe_modes(probe,extent=get_extent_from_pixel_size(probe[0].shape,input_dict["object_pixel"]))

        if input_dict["distance_sample_focus"] != 0:
            print(f'Plotting probe at focus... Propagating it to source by {-input_dict["distance_sample_focus"]*1e3:.3f} mm')
            propagated_probe = np.empty_like(probe)
            for i, probe_mode in enumerate(probe):
                propagated_probe[i] = fresnel_propagator_cone_beam(probe_mode,input_dict["wavelength"],input_dict["detector_pixel_size"],-input_dict["distance_sample_focus"]) 
            plot_probe_modes(propagated_probe,extent=get_extent_from_pixel_size(probe[0].shape,input_dict["object_pixel"]))

        plot_iteration_error(error)

    if input_dict['hdf5_output'] is not None:
        print('Saving output hdf5 file at: ', input_dict['hdf5_output'])
        save_recon_output_h5_file(input_dict, obj, probe, positions,corrected_positions, error,initial_probe,initial_obj)

    return obj, probe, corrected_positions, input_dict, error

def call_ptychography_engines(input_dict,DPs, positions, initial_obj=None, initial_probe=None,plot=True):
    
    if initial_probe is None:
        initial_probe = set_initial_probe(input_dict, DPs, input_dict['incoherent_modes']) # probe initial guess
    probe = initial_probe.copy()

    if initial_obj is None:
        initial_obj = set_initial_object(input_dict,DPs,probe[0],input_dict["object_shape"]) # object initial guess
    obj = initial_obj.copy()

    print('Plotting initial guesses...')
    if plot: plot_probe_modes(probe,extent=get_extent_from_pixel_size(probe[0].shape,input_dict["object_pixel"]))
    if plot: plot_amplitude_and_phase(obj, positions=positions+probe.shape[-1]//2,extent=get_plot_extent_from_positions(positions))

    probe_positions = positions.astype(np.int32)
    probe_positions = np.roll(probe_positions,shift=1,axis=1) # change from (Y,X) to (X,Y) for the algorithms

    if np.any(probe_positions < 0):
        raise ValueError(f"Positions array cannot have negative values. Min = {probe_positions.min()}")  

    if 'probe_support' in input_dict:
        input_dict["probe_support_array"] = get_probe_support(input_dict,probe.shape)
    else:
        input_dict["probe_support_array"] = np.ones_like(DPs[0])

    if plot: plot_probe_support(input_dict["probe_support_array"][0],extent=get_extent_from_pixel_size(probe[0].shape,input_dict["object_pixel"]))

    if input_dict["distance_sample_focus"] == 0:
        input_dict['fresnel_number'] = 0
    else:   
        input_dict['fresnel_number'] = input_dict["object_pixel"]**2/(input_dict["wavelength"]*input_dict["distance_sample_focus"])

    print(f'Distance between sample and focus: {input_dict["distance_sample_focus"]*1e3}mm. Corresponding Fresnel number: {input_dict["fresnel_number"]}')

    print(f"Total datapack size: {estimate_memory_usage(obj,probe,probe_positions,DPs,input_dict['probe_support_array'])[3]:.2f} GBs")

    print(f'Starting ptychography... using {len(input_dict["GPUs"])} GPUs {input_dict["GPUs"]} and {input_dict["CPUs"]} CPUs')

    corrected_positions = None
    error_rfactor = []
    error_nmse = []
    error_llk = []
    for counter in range(1,1+len(input_dict['algorithms'].keys())):

        check_for_nans(obj,probe,DPs) # check if there are NaNs in the input data; exit program otherwise

        algo_inputs = {**input_dict, **{ k: v for k,v in input_dict['algorithms'][str(counter)].items() }  }

        if input_dict["algorithms"][str(counter)]['name'] == 'ePIE_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of rPIE algorithm...")
            

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
                        
            algo_inputs['friction_object'] = input_dict['algorithms'][str(counter)]['mPIE_friction_obj'] 
            algo_inputs['friction_probe'] = input_dict['algorithms'][str(counter)]['mPIE_friction_probe'] 
            algo_inputs['momentum_counter'] = input_dict['algorithms'][str(counter)]['mPIE_momentum_counter'] 
            algo_inputs['use_mPIE'] = input_dict['algorithms'][str(counter)]['use_mPIE'] 
            
            obj, probe, algo_error,probe_positions = PIE_python(DPs, probe_positions,obj,probe[0], algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
            
            algo_inputs['epsilon'] = 0.001 # small value to add to probe/object update denominator
            
            obj, probe, algo_error = RAAR_python(DPs,probe_positions,obj,probe[0],algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'ML_python':
            obj, new_probe, algo_error = ML_cupy(DPs,positions,obj,probe[0],algo_inputs) #TODO: expand to deal with multiple probe modes
            probe[0] = new_probe
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'AP': # former GL
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of Alternate Projections CUDA algorithm...")
            DPs, obj, probe = check_dtypes(DPs,obj,probe) # check if dtypes are correct; exit program otherwise

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])

            obj, probe, algo_error, probe_positions = AP(iterations=algo_inputs['iterations'],
                                                        objbeta=algo_inputs['momentum_obj'],
                                                        probebeta=algo_inputs['momentum_probe'],
                                                        batch=algo_inputs['batch'],
                                                        step_obj=algo_inputs['step_object'],
                                                        step_probe=algo_inputs['step_probe'],
                                                        reg_obj=algo_inputs['regularization_object'],
                                                        reg_probe=algo_inputs['regularization_probe'],
                                                        difpads=DPs,
                                                        obj=obj,
                                                        rois=probe_positions,
                                                        probe=probe,
                                                        probesupp = algo_inputs['probe_support_array'],
                                                        params={'device': input_dict["GPUs"]},
                                                        poscorr_iter=algo_inputs["position_correction"],
                                                        wavelength_m=input_dict["wavelength"],
                                                        pixelsize_m=input_dict["object_pixel"],
                                                        distance_m=input_dict["distance_sample_focus"])

            error_rfactor.append(algo_error)
            error_nmse.append(np.full_like(algo_error, np.nan))
            error_llk.append(np.full_like(algo_error, np.nan))

            if algo_inputs["position_correction"] > 0:
                corrected_positions = probe_positions      

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            DPs, obj, probe = check_dtypes(DPs,obj,probe) # check if dtypes are correct; exit program otherwise

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
            obj, probe, algo_error, probe_positions  = RAAR(iterations=algo_inputs['iterations'],
                                                            probebeta=algo_inputs['momentum_probe'],
                                                            objbeta=algo_inputs['momentum_obj'],
                                                            beta=algo_inputs['beta'],
                                                            batch=algo_inputs['batch'],
                                                            step_obj=algo_inputs['step_object'],
                                                            step_probe=algo_inputs['step_probe'],
                                                            reg_obj=algo_inputs['regularization_object'],
                                                            reg_probe=algo_inputs['regularization_probe'],
                                                            rois=probe_positions,
                                                            difpads=DPs,
                                                            obj=obj,
                                                            probe=probe,
                                                            probesupp = algo_inputs['probe_support_array'],
                                                            params={'device': input_dict["GPUs"]},
                                                            poscorr_iter=algo_inputs["position_correction"],
                                                            wavelength_m=input_dict["wavelength"],
                                                            pixelsize_m=input_dict["object_pixel"],
                                                            distance_m=input_dict["distance_sample_focus"])

            error_rfactor.append(algo_error)
            error_nmse.append(np.full_like(algo_error, np.nan))
            error_llk.append(np.full_like(algo_error, np.nan))

            if algo_inputs["position_correction"] > 0:
                corrected_positions = probe_positions      

        elif input_dict["algorithms"][str(counter)]['name'] == 'PIE':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of rPIE algorithm...")
            DPs, obj, probe = check_dtypes(DPs,obj,probe) # check if dtypes are correct; exit program otherwise

            if len(input_dict["GPUs"]) > 1:
                print(f"WARNING: PIE algorithm is not implemented for multi-GPU. Using single GPU {input_dict['GPUs'][0:1]} (batch size = 1) instead.")

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
            obj, probe, algo_error, probe_positions = PIE(iterations=algo_inputs['iterations'],
                                                        step_obj=algo_inputs['step_object'],
                                                        step_probe=algo_inputs['step_probe'],
                                                        reg_obj=algo_inputs['regularization_object'],
                                                        reg_probe=algo_inputs['regularization_probe'],
                                                        poscorr_iter=algo_inputs["position_correction"],
                                                        rois=probe_positions,
                                                        difpads=DPs,
                                                        obj=obj,
                                                        probe=probe,
                                                        wavelength_m=input_dict["wavelength"],
                                                        pixelsize_m=input_dict["object_pixel"],
                                                        distance_m=input_dict["distance_sample_focus"],
                                                        params={'device': input_dict["GPUs"][0:1]})


            error_rfactor.append(algo_error)
            error_nmse.append(np.full_like(algo_error, np.nan))
            error_llk.append(np.full_like(algo_error, np.nan))

            if algo_inputs["position_correction"] > 0:
                corrected_positions = probe_positions                                        
            
        else:
            sys.exit('Please select a proper algorithm! Selected: ', input_dict["algorithms"][str(counter)]['name'])

        if counter != len(input_dict['algorithms'].keys()) and plot == True:
            plot_amplitude_and_phase(obj, positions=positions+probe.shape[-1]//2,extent=get_plot_extent_from_positions(positions))
            

    error_rfactor =  np.concatenate(error_rfactor).ravel()
    error_nmse = np.concatenate(error_nmse).ravel()
    error_llk = np.concatenate(error_llk).ravel()
    error = np.column_stack((error_rfactor,error_nmse,error_llk)) # must be (iterartions,3) array shape
    return obj, probe, error, corrected_positions, initial_obj, initial_probe

def check_for_nans(*arrays):
    """
    Check for NaN values in an arbitrary number of NumPy arrays and raise a ValueError if any are found.

    Parameters:
    arrays (any number of np.array): A variable number of NumPy arrays to check for NaNs.

    Raises:
    ValueError: If any NaN values are found in the arrays.
    """
    for idx, array in enumerate(arrays):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input {idx+1} is not a NumPy array.")

        # Check if there are any NaNs in the array
        if np.isnan(array).any():
            raise ValueError(f"Array {idx+1} contains NaN values.")

def check_shape_of_inputs(DPs,positions,initial_probe):

    if DPs.shape[0] != positions.shape[0]:
        raise ValueError(f'There is a problem with input data!\nThere are {DPs.shape[0]} diffractiom patterns and {positions.shape[0]} positions. These values should be the same.')    

    if initial_probe is not None: # mudei como conversado com Mauro
        if DPs[0].shape[0] != initial_probe.shape[1] or DPs[0].shape[1] != initial_probe.shape[2]:
            raise ValueError(f'There is a problem with your input data!\nThe dimensions of input_probe and diffraction pattern differ in the X,Y directions: {DPs.shape} vs {initial_probe.shape}')

def remove_positions_randomly(arr1, arr2, R):

    """
    Reduce the number of positions randomly by R
    """

    # Get the number of columns (N) from the first array
    N = arr1.shape[0]

    # Ensure R is less than N
    if R >= N:
        raise ValueError(f"R should be less than N ({N})")

    # Randomly sample (N - R) indices to keep
    indices_to_keep = np.random.choice(N, N - R, replace=False)

    # Sort the indices to maintain the original order
    indices_to_keep = np.sort(indices_to_keep)

    # Slice both arrays using the sampled indices
    reduced_arr1 = arr1[indices_to_keep, :]
    reduced_arr2 = arr2[indices_to_keep, :, :]

    return reduced_arr1, reduced_arr2

def check_and_set_defaults(input_dict):
    # Define the default values
    default_values = {
        'CPUs': 32,
        'GPUs': [0],
        'fresnel_regime': False,
        'energy': 10,  # keV
        'detector_distance': 10,  # meters
        'distance_sample_focus': 0,
        'detector_pixel_size': 55e-6, # meters
        'binning': 1,
        'position_rotation': 0,
        'object_padding': 0,
        'incoherent_modes': 1,
        'n_of_positions_to_remove':0,
        'probe_support': {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0},
        'initial_obj': {"obj": 'random'},
        'initial_probe': {"probe": 'inverse'},
        'algorithms': {'1': {'name':'RAAR',
                            'batch': 64,
                            'iterations': 30, 
                            'beta': 0.9,
                            'step_object': 1.0,
                            'step_probe': 1.0,   
                            'regularization_object': 0.01,
                            'regularization_probe': 0.01,
                            'momentum_obj': 0.0,
                            'momentum_probe': 0.0,
                            'position_correction': 0,
                            },  } 
    }
    
    # Loop over the default values and check if they exist in input_dict
    for key, default_value in default_values.items():
        if key not in input_dict:
            input_dict[key] = default_value
            print(f"WARNING: key '{key}' was missing in the input dictionary. Set to default value: {default_value}")

    return input_dict

def check_dtypes(DPs,initial_obj,initial_probe):
    if DPs.dtype != np.float32:
        print('WARNING: Diffraction patterns dtype is not np.float32. Converting to np.float32...')
        DPs = DPs.astype(np.float32)

    if initial_obj is not None:
        if initial_obj.dtype != np.complex64:
            print('WARNING: Initial object dtype is not np.complex64. Converting to np.complex64...')
            initial_obj = initial_obj.astype(np.complex64)

    if initial_probe is not None:
        if initial_probe.dtype != np.complex64:
            print('WARNING: Initial probe dtype is not np.complex64. Converting to np.complex64...')
            initial_probe = initial_probe.astype(np.complex64)

    return DPs, initial_obj, initial_probe

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
                if subkey == 'initial_obj':
                   h5file.create_group(f'metadata/algorithms/{key}/{subkey}')
                   h5file[f'metadata/algorithms/{key}/{subkey}'].create_dataset(subkey,data=input_dict['algorithms'][key][subkey]['obj'])
                elif subkey == 'initial_probe':
                    h5file.create_group(f'metadata/algorithms/{key}/{subkey}')
                    h5file[f'metadata/algorithms/{key}/{subkey}'].create_dataset(subkey,data=input_dict['algorithms'][key][subkey]["probe"])
                else:
                    h5file[f'metadata/algorithms/{key}'].create_dataset(subkey,data=input_dict['algorithms'][key][subkey])

    h5file.close()

def convert_probe_positions_to_pixels(pixel_size, probe_positions,factor=1):
    """
    Subtratcs minimum of position in each direction, converts from microns to pixels and then apply desired offset 
    """

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = factor * probe_positions[:, 0] / pixel_size  # convert from microns to pixels
    probe_positions[:, 1] = factor * probe_positions[:, 1] / pixel_size 
    
    return probe_positions

def check_consecutive_keys(algorithms):
    keys = list(map(int, algorithms.keys()))
    keys.sort()
    return keys == list(range(1, len(keys) + 1))

def bin_volume(volume, downsampling_factor):
    """ Downsample a 3D volume (N,Y,X) in the Y, X directions by averaging over a specified downsampling factor.

    Args:
        volume (ndarray): 3D numpy array of shape (N,Y,X)
        downsampling_factor (int): downsampling_factor

    Raises:
        ValueError: error in case Y and X dimensions are not divisible by the downsampling factor

    Returns:
        downsampled_volume: 3D numpy array of shape (N, Y//downsampling_factor, X//downsampling_factor)

    """

    print('Binning data...')

    def suggest_crop_dimensions(Y, X, downsampling_factor):
        # Calculate the largest dimensions divisible by the downsampling factor
        new_Y = Y - (Y % downsampling_factor)
        new_X = X - (X % downsampling_factor)
        
        # Return the new suggested dimensions
        return new_Y, new_X

    N, Y, X = volume.shape

    # Ensure that Y and X are divisible by the downsampling factor
    if Y % downsampling_factor != 0 or X % downsampling_factor != 0:
        new_Y, new_X = suggest_crop_dimensions(Y, X, downsampling_factor)
        print("WARNING: Issue when binning. Y and X dimensions must be divisible by the downsampling factor. Cropping volume from ({Y},{X}) to ({new_Y},{new_X})")
        volume = volume[:, :new_Y, :new_X]

    def numpy_downsampling(volume, downsampling_factor):
        N, Y, X = volume.shape
        new_shape = (N, Y // downsampling_factor, downsampling_factor, X // downsampling_factor, downsampling_factor)
        downsampled_volume = volume.reshape(new_shape).mean(axis=(2, 4))
        return downsampled_volume
    
    binned_volume = numpy_downsampling(volume, downsampling_factor)

    print('Binned data to new shape: ', binned_volume.shape)

    return binned_volume

def save_recon_output_h5_file(input_dict, obj, probe, positions,corrected_positions, error,initial_probe,initial_obj):

    with  h5py.File(input_dict["hdf5_output"], "a") as h5file:

        h5file["recon"].create_dataset('object',data=obj) 
        h5file["recon"].create_dataset('probe',data=probe) 
        h5file["recon"].create_dataset('positions',data=positions)
        h5file["recon"].create_dataset('initial_probe',data=initial_probe)
        h5file["recon"].create_dataset('initial_obj',data=initial_obj) 
        h5file["recon"].create_dataset('error',data=error) 
        if corrected_positions is not None:
            h5file["recon"].create_dataset('corrected_positions',data=corrected_positions) 

    h5file.close()

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

    return probe_positions

def set_initial_probe(input_dict, DPs, incoherent_modes):
    print('Creating initial probe of type: ',input_dict['initial_probe']["probe"])

    DP_shape = (DPs.shape[1], DPs.shape[2])

    def set_modes(probe, input_dict):
        mode = probe.shape[0]

        if incoherent_modes > mode:
            add = incoherent_modes - mode
            probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
            for i in range(add):
                probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

        print("Probe shape ({0},{1}) with {2} incoherent mode(s)".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

        return probe

    type_of_initial_guess = detect_variable_type_of_guess(input_dict['initial_probe']["probe"])

    if type_of_initial_guess == 'standard':
        
        if input_dict['initial_probe']['probe'] == 'circular':
            probe = create_circular_mask(DP_shape,input_dict['initial_probe']["radius"])
            probe = probe*np.exp(1j*probe)
            if input_dict['initial_probe']['distance']!= 0: # propagate probe 
                probe = fresnel_propagator_cone_beam(probe, input_dict['wavelength'],input_dict['object_pixel'], input_dict['initial_probe']['distance'])
        elif input_dict['initial_probe']['probe'] == 'cross':
            cross_width_y, border, center_square_side = input_dict['initial_probe']["cross_width"],input_dict['initial_probe']["border_padding"],input_dict['initial_probe']['center_width']
            probe = create_cross_mask(DP_shape,cross_width_y, border, center_square_side)
        elif input_dict['initial_probe']['probe'] == 'constant':
            probe = np.ones(DP_shape)
        elif input_dict['initial_probe']['probe'] == 'random':
            probe = np.random.rand(*DP_shape)
        elif input_dict['initial_probe']['probe'] == 'inverse' or input_dict['initial_probe']['probe'] == 'ift':
            DPs_avg =  np.average(DPs, 0)[None] 
            ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(DPs_avg)))
            probe = np.sqrt(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft))))
        elif input_dict['initial_probe']['probe'] == 'fzp':
            wavelength = input_dict['wavelength']
            pixel_size_object = input_dict['object_pixel']
            beam_type = input_dict['initial_probe'].get('beam_type','disc')
            distance_sample_fzpf = input_dict['initial_probe']['distance_sample_fzpf']
            fzp_diameter = input_dict['initial_probe']['fzp_diameter']
            fzp_outer_zone_width = input_dict['initial_probe']['fzp_outer_zone_width']
            beamstopper_diameter = input_dict['initial_probe']['beamstopper_diameter']
            probe_diameter = input_dict['initial_probe'].get('probe_diameter',fzp_diameter)  
            probe_normalize = input_dict['initial_probe'].get('probe_normalize',False)
            probe = probe_model_fzp(wavelength = wavelength,
                                    grid_shape = 2*input_dict["detector_ROI_radius"],
                                    pixel_size_object = pixel_size_object , 
                                    beam_type =  beam_type,
                                    distance_sample_fzpf = distance_sample_fzpf,
                                    fzp_diameter = fzp_diameter,
                                    fzp_outer_zone_width = fzp_outer_zone_width,
                                    beamstopper_diameter = beamstopper_diameter,
                                    probe_diameter = probe_diameter,
                                    probe_normalize = probe_normalize )
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

    if probe.ndim == 2:
        probe = np.expand_dims(probe, axis=0)
        print(f'A 2D probe array was given. Adding a third dimension to the probe array: {probe.shape}')

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

def set_initial_object(input_dict,DPs, probe, obj_shape):
    print('Creating initial object of type: ', input_dict['initial_obj']["obj"])

    type_of_initial_guess = detect_variable_type_of_guess(input_dict['initial_obj']["obj"])

    if type_of_initial_guess == 'standard':
        if input_dict['initial_obj']['obj'] == 'constant':
            obj = np.ones(obj_shape)
        elif input_dict['initial_obj']['obj'] == 'random':
            normalization_factor = np.sqrt(np.average(DPs) / np.average(abs(np.fft.fft2(probe))**2))
            obj = np.random.rand(*obj_shape) * normalization_factor
        elif input_dict['initial_obj']['obj'] == 'complex_random':
            obj =  1 * (np.random.rand(*obj_shape) + 1j*np.random.rand(*obj_shape))
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

    print('Object shape:', complex_obj.shape, 'with dtype:', complex_obj.dtype)

    return complex_obj

def get_probe_support(input_dict,probe_shape):
    """ 
    Create mask containing probe support region

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
            keys: "probe_support": probe support
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

def set_object_pixel_size(wavelength, detector_distance, detector_pixel_size, DP_size):
    """
    Calculate and display the object pixel size

    This function computes the object pixel size based on the provided wavelength, detector distance,
    detector pixel size, diffraction pattern (DP) size, and optional binning factor. It also calculates
    the limit thickness for a resolution of 1 pixel and prints these values.

    Args:
        wavelength (float): Wavelength of the light used in meters.
        detector_distance (float): Distance from the sample to the detector in meters.
        detector_pixel_size (float): Size of a pixel on the detector in meters.
        DP_size (int): Size of the diffraction pattern (number of pixels).
        binning (int, optional): Binning factor. Must be an even number. If 1, no binning occurs. Defaults to 1.

    Returns:
        float: Calculated object pixel size in meters.

    """
    
    object_pixel_size = calculate_object_pixel_size(wavelength, detector_distance, detector_pixel_size, DP_size)
    print(f"\tObject pixel size = {object_pixel_size*1e9:.2f} nm")

    PA_thickness = 4 * object_pixel_size ** 2 / (0.61 * wavelength)
    print(f"\tLimit thickness for resolution of 1 pixel: {PA_thickness*1e6:.3f} microns")
    return object_pixel_size

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

def probe_model_fzp(wavelength, 
                    grid_shape,
                    pixel_size_object, 
                    beam_type,
                    distance_sample_fzpf,
                    fzp_diameter,
                    fzp_outer_zone_width,
                    beamstopper_diameter,
                    probe_diameter,
                    probe_normalize,
                    upsample = 10):

    """
    Args:
        wavelength (float): Wavelength of the probe in meters.
        grid_shape (int or list of int): Shape of the grid, either an int (for a square grid) or a list [int, int] for a rectangular grid.
        pixel_size_object (float): Size of a pixel in the object plane in meters.
        beam_type (str): Type of the beam, either 'gaussian' or 'disc'.
        distance_sample_fzpf (float): Distance between the sample and the focus of the FZP in meters.
        fzp_diameter (float): Diameter of the FZP in meters.
        fzp_outer_zone_width (float): Width of the outermost zone of the FZP in meters.
        beamstopper_diameter (float): Diameter of the beamstopper in meters.
        probe_diameter (float): Diameter of the probe in meters.
        probe_normalize (bool): Whether to normalize the probe.
        upsample (int, optional): Upsampling factor for the grid. Default is 10.

    Returns:
        initial_probe (numpy.ndarray): The initial probe after applying the FZP and beamstopper.
    """
    
    # FZP focus
    fzp_f = fzp_diameter*fzp_outer_zone_width/wavelength      

    # handle grid_shape being int or [int,int] or something else
    if isinstance(grid_shape, int):
        grid_shape = [grid_shape,grid_shape]
    elif isinstance(grid_shape, (list, list)) is False or len(grid_shape) != 2:
        raise ValueError("grid_shape must be an int or a list/tuple of two integers.")
        
    # upsample grid 
    grid_shape[0] = int(grid_shape[0]*upsample)
    grid_shape[1] = int(grid_shape[1]*upsample)

    # define a common sampling grid 
    x = np.linspace(-grid_shape[0]//2,(grid_shape[0]+1)//2,grid_shape[0])*pixel_size_object
    y = np.linspace(-grid_shape[1]//2,(grid_shape[1]+1)//2,grid_shape[0])*pixel_size_object
    x, y = np.meshgrid(x, y)
    
    # distances: z1 = d(fzp, fzp_f) and z2 = d(fzp_f, obj)   
    z1 = 0                             # -fzp_f 
    z2 = fzp_f + distance_sample_fzpf  # distance_sample_fzpf 
    

    if beam_type=="gaussian": 
        # define a gaussian wavefront
        sigma = (probe_diameter/2) / (2*np.sqrt(2*np.log(2))) # full width at half maximum
        w = np.exp(-(x**2 + y**2)/(2*sigma**2))
    elif beam_type=="disc":
        # define a disc wavefront 
        w = (x**2 + y**2 <= (probe_diameter/2)**2).astype(float) 
    else:  
        raise ValueError("Invalid beam_type. Must be either 'gaussian' or 'disc'.")

    ## beamstopper transfer function
    if beamstopper_diameter>0:
        # define the beamstopper mask 
        beamstopper = 1.0-(x**2 + y**2 <= (beamstopper_diameter/2)**2).astype(float)

        # apply beamstopper mask to w
        w = w*beamstopper

    # generate the fzp transfer function
    r2 =  x**2 + y**2
    # transfer_fzp = np.exp(1j*np.pi*r2/(wavelength*fzp_f))
    transfer_fzp = np.exp(-1j*np.pi*r2/(wavelength*fzp_f))
    
    
    # compute the initial probe 
    w = fresnel_propagator_cone_beam(w*transfer_fzp,wavelength,pixel_size_object,z2,z1)


    # crop grid 
    start_y = (grid_shape[0] - int(grid_shape[0]/upsample))//2
    start_x = (grid_shape[1] - int(grid_shape[1]/upsample))//2
    end_y = start_y + int(grid_shape[0]/upsample)
    end_x = start_x + int(grid_shape[1]/upsample) 
    w = w[start_y:end_y,start_x:end_x]

    # normalize if needed
    if probe_normalize is True:
        w = w*np.sqrt(grid_shape[0]*grid_shape[1]/np.sum(np.abs(w)**2,axis=(0,1)));

    return np.expand_dims(w,axis=0)
