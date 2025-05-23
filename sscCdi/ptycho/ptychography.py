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

from concurrent.futures import ProcessPoolExecutor

from ..cditypes import AP, PIE, RAAR

from ..misc import estimate_memory_usage, wavelength_meters_from_energy_keV, calculate_object_pixel_size, get_datetime
from ..processing.propagation import fresnel_propagator_cone_beam
from .pie import PIE_python
from .raar import RAAR_python
from .ML import ML_cupy

from .ptycho_plots import (plot_ptycho_scan_points, plot_probe_modes, 
                           get_extent_from_pixel_size, plot_iteration_error, 
                           plot_amplitude_and_phase, get_plot_extent_from_positions, 
                           plot_probe_support,plot_ptycho_corrected_scan_points,plot_object_spectrum)

random.seed(0)

def call_ptychography(input_dict, DPs, positions, initial_obj=None, initial_probe=None,plot=True):
    """This function creates the object and probe, converts the positions in SI units to pixel then  
    calls the engines utilized to perform the Ptychography reconstruction [1]. 

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        DPs (ndarray): Measured diffraction patterns organize in dimensions: (DPs index/number, DPs y-size, DPs x-size).
        positions (ndarray): Measured positions in metric units (m, mm, um) or pixel, organized in (y,x) list.  
        initial_obj (ndarray): Initial approximation for the object, see the dictionaty for the options. [default: None].
        initial_probe (ndarray, optional): Initial approximation for the probe, see the dictionaty for the options. [default: None].
        plot (bool, optional): Option to plot initial object and probe, input positions, object during all called engines and final object, probe, positions. [default: True].

    Returns: 
        obj (ndarray): Reconstructed complex object (amplitude and phase).
        probe (ndarray): Reconstructed complex probe (amplitude and phase) for all the incoherent modes organized in (mode index, probe).
        corrected_positions (ndarray): Final positions. Same as input without the position correction and the optimized positions with position correction.
        input_dict (dict):  Dictionary with the utilized parameters. 
        error (ndarray): Reconstruction errors (r_factor: r-factor or residue, mse: normalized mean square, llk: log-likehood ) for all the iterations, organized in (iteration, r-factor, mse, llk).

    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]

            #. Circular probe support: ``input_dict['probe_support'] = {\"type\": \"circular\", \"radius\": float, \"center_y\": int, \"center_x\": int}``.
            #. Cross probe support: ``input_dict['probe_support'] = {\"type\": \"cross\", \"center_width\": int, \"cross_width\": int, \"border_padding\": int }``.
            #. Numpy array probe support: ``input_dict['probe_support'] = {"type": "array", "data": myArray}``.
        
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        
        * ``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters,
                'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.   ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        
        * ``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
               'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
               'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
               'position_correction': int (0: no correction, N: performs correction every N iterations)}
            
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
               'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
               'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
               'momentum_obj': float , momentum_probe': float, 
               'position_correction': (0: no correction, N: performs correction every N iterations)}
            
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
               'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
               'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
               'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
               'momentum_obj': float , momentum_probe': float, 
               'position_correction': (0: no correction, N: performs correction every N iterations)}
            
            #. ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
               'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            
            #. ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            
            #. ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            
            #. ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            
            #. ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
     
    """  

    input_dict["datetime"] = get_datetime()

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

    if "wavelength" not in input_dict:
        input_dict["wavelength"] = wavelength_meters_from_energy_keV(input_dict['energy'])
        print(f"Wavelength = {input_dict['wavelength']*1e9:.3f} nm")

    if "object_pixel" not in input_dict:
        input_dict["object_pixel"] = calculate_object_pixel_size(input_dict['wavelength'],
                                                                 input_dict['detector_distance'],
                                                                 input_dict['detector_pixel_size'],
                                                                 DPs.shape[1]) # meters

        print(f"Object pixel = {input_dict['object_pixel']*1e9:.2f} nm")

    if input_dict['positions_unit'] is None:
        print("WARNING: assuming positions are in pixels. If not, please set 'positions_unit' in the input dictionary.")
        if plot: plot_ptycho_scan_points(positions,pixel_size=None)
    elif input_dict['positions_unit'] == 'meters' or input_dict['positions_unit'] == 'm':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1)
        if plot: plot_ptycho_scan_points(positions,pixel_size=input_dict["object_pixel"])
    elif input_dict['positions_unit'] == 'millimeters' or input_dict['positions_unit'] == 'mm':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1e-3)
        if plot: plot_ptycho_scan_points(positions,pixel_size=input_dict["object_pixel"])
    elif input_dict['positions_unit'] == 'microns' or input_dict['positions_unit'] == 'micrometers' or input_dict['positions_unit'] == 'um':
        positions = convert_probe_positions_to_pixels(input_dict["object_pixel"], positions,factor=1e-6)
        if plot: plot_ptycho_scan_points(positions,pixel_size=input_dict["object_pixel"])
    elif input_dict['positions_unit'] == 'pixels':
        if plot: plot_ptycho_scan_points(positions,pixel_size=None)
        pass


    if "object_shape" not in input_dict:
        input_dict["object_shape"] = set_object_shape(input_dict["object_padding"], DPs.shape, positions)
        print(f"Object shape: {input_dict['object_shape']}")


    if input_dict['hdf5_output'] is not None:
        print('Creating output hdf5 file...')
        if os.path.exists(os.path.dirname(input_dict['hdf5_output'])) == False:
            print('Folder does not exist. Creating it...')
            os.makedirs(os.path.dirname(input_dict['hdf5_output']))

    # call engines
    obj, probe, error, corrected_positions, initial_obj, initial_probe = call_ptychography_engines(input_dict,
                                                                                                   DPs,
                                                                                                   positions,
                                                                                                   initial_obj=initial_obj,
                                                                                                   initial_probe=initial_probe,
                                                                                                   plot=plot)
    # print(positions)
    # print(corrected_positions)
    # input_dict['positions'] = corrected_positions

    if plot is True and corrected_positions is not None:
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
        print('Saving output hdf5 file... ')
        create_parent_folder(input_dict["hdf5_output"]) # create parent folder to output file if it does not exist
        if input_dict['save_restored_data'] == True:
            save_h5_output(input_dict,obj, probe, positions, error,initial_obj,initial_probe,corrected_positions,restored_data=DPs)
        else:
            save_h5_output(input_dict,obj, probe, positions, error,initial_obj,initial_probe,corrected_positions,restored_data=None)

    return obj, probe, corrected_positions, input_dict, error

def call_ptychography_engines(input_dict, DPs, positions, initial_obj=None, initial_probe=None, plot=True):

    """This function calls all ptychography engines
    
    :meta private:

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        DPs (ndarray): Measured diffraction patterns organize in (DPs index/number, DPs y-size, DPs x-size).
        positions (ndarray_): Measured positions in metric units (m, mm, um) or pixel, organized in (y,x) list.  
        initial_obj (ndarray): Initial approximation for the object, see the dictionaty for the options. [default: None].
        initial_probe (ndarray, optional): Initial approximation for the probe, see the dictionaty for the options. [default: None].
        plot (bool, optional): Option to plot initial object and probe, input positions, object during all called engines and final object, probe, positions. [default: True].
    Returns:
        ndarray: Reconstructed complex object (amplitude and phase).
        ndarray: Reconstructed complex probe (amplitude and phase) for all the incoherent modes organized in (mode index, probe).
        ndarray: Reconstruction errors (r_factor: r-factor or residue, mse: normalized mean square, llk: log-likehood ) for all the iterations, organized in (iteration, r-factor, mse, llk).
        ndarray: Final positions. Same as input without the position correction and the optimized positions with position correction.
        ndarray: Initial complex object (amplitude and phase).
        ndarray: Initial complex probe (amplitude and phase) for all the incoherent modes organized in (mode index, probe).
    
    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.   ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
         

    """
    # define initial guess probe
    if initial_probe is None:
        initial_probe = set_initial_probe(input_dict, DPs, input_dict['incoherent_modes'])
    probe = initial_probe.copy()

    # define initial guess object
    if initial_obj is None:
        initial_obj = set_initial_object(input_dict,DPs,probe[0],input_dict["object_shape"])
    obj = initial_obj.copy()

    # copy probe positions
    # probe_positions = positions

    # copy probe positions and type cast from float to int
    # notice that simply np.round()'ing the positions could cause some roi to exceed boundaries.
    # check sanitize_rois() function in cditypes.py to see if that makes sense indeed.
    probe_positions = positions.astype(np.int32)


    # change from (Y,X) to (X,Y) for the algorithms
    probe_positions = np.roll(probe_positions, shift=1, axis=1)

    print('Plotting initial guesses...')
    if plot:
        plot_probe_modes(probe,extent=get_extent_from_pixel_size(probe[0].shape,input_dict["object_pixel"]))
    if plot:
        plot_amplitude_and_phase(obj, positions=positions+probe.shape[-1]//2, extent=get_plot_extent_from_positions(positions))

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

        if input_dict["algorithms"][str(counter)]['name'] == 'rPIE_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of rPIE algorithm...")

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])

            algo_inputs['friction_object'] = input_dict['algorithms'][str(counter)]['momentum_obj']
            algo_inputs['friction_probe'] = input_dict['algorithms'][str(counter)]['momentum_probe']
            algo_inputs['momentum_counter'] = input_dict['algorithms'][str(counter)]['mPIE_momentum_counter']

            obj, probe, algo_error, probe_positions = PIE_python(DPs, probe_positions,obj,probe[0], algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR_python':

            """ RAAR update function:     psi' = [ beta*(Pf*Rr + I) + (1-2*beta)*Pr ]*psi
            beta = 0 -> AP
            beta = 1 -> DM
            """

            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])

            obj, probe, algo_error = RAAR_python(DPs, probe_positions, obj,probe[0], algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'DM_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])

            algo_inputs['beta'] = input_dict['algorithms'][str(counter)]['beta'] = 1 # beta = 1 for DM

            obj, probe, algo_error = RAAR_python(DPs, probe_positions, obj, probe[0], algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'AP_python':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])

            algo_inputs['beta'] = input_dict['algorithms'][str(counter)]['beta'] = 0 # beta = 0 for AP

            obj, probe, algo_error = RAAR_python(DPs, probe_positions, obj, probe[0], algo_inputs)
            error_rfactor.append(algo_error[:,0])
            error_nmse.append(algo_error[:,1])
            error_llk.append(algo_error[:,2])

        elif input_dict["algorithms"][str(counter)]['name'] == 'ML_python':
            obj, new_probe, algo_error = ML_cupy(DPs,positions, obj, probe[0], algo_inputs) #TODO: expand to deal with multiple probe modes
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

            obj, probe, algo_error_rfactor, algo_error_llk, algo_error_mse, probe_positions = AP(iterations=algo_inputs['iterations'],
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


            # error_nmse.append(np.full_like(algo_error, np.nan))
            error_rfactor.append(algo_error_rfactor)
            error_nmse.append(algo_error_mse)
            error_llk.append(algo_error_llk)

            # if algo_inputs["position_correction"] > 0:
            # corrected_positions = probe_positions


        elif input_dict["algorithms"][str(counter)]['name'] == 'RAAR':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of RAAR algorithm...")
            DPs, obj, probe = check_dtypes(DPs,obj,probe) # check if dtypes are correct; exit program otherwise

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
            obj, probe, algo_error_rfactor, algo_error_llk, algo_error_mse, probe_positions  = RAAR(iterations=algo_inputs['iterations'],
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

            error_rfactor.append(algo_error_rfactor)
            # error_nmse.append(np.full_like(algo_error, np.nan))
            error_nmse.append(algo_error_mse)
            error_llk.append(algo_error_llk)

            #if algo_inputs["position_correction"] > 0:
            # corrected_positions = probe_positions

        elif input_dict["algorithms"][str(counter)]['name'] == 'PIE':
            print(f"Calling {input_dict['algorithms'][str(counter)]['iterations'] } iterations of rPIE algorithm...")
            DPs, obj, probe = check_dtypes(DPs,obj,probe) # check if dtypes are correct; exit program otherwise

            if len(input_dict["GPUs"]) > 1:
                print(f"WARNING: PIE algorithm is not implemented for multi-GPU. Using single GPU {input_dict['GPUs'][0:1]} (batch size = 1) instead.")

            if 'initial_probe' in input_dict["algorithms"][str(counter)]:
                probe = set_initial_probe(input_dict["algorithms"][str(counter)], DPs, input_dict['incoherent_modes'])
            if 'initial_obj' in input_dict["algorithms"][str(counter)]:
                obj = set_initial_object(input_dict["algorithms"][str(counter)],DPs,probe[0],input_dict["object_shape"])
            obj, probe, algo_error_rfactor, algo_error_llk, algo_error_mse, probe_positions = PIE(iterations=algo_inputs['iterations'],
                                                                                          step_obj=algo_inputs['step_object'],
                                                                                          step_probe=algo_inputs['step_probe'],
                                                                                          reg_obj=algo_inputs['regularization_object'],
                                                                                          reg_probe=algo_inputs['regularization_probe'],
                                                                                          poscorr_iter=algo_inputs["position_correction"],
                                                                                            rois=probe_positions,
                                                                                            difpads=DPs,
                                                                                            obj=obj,
                                                                                            probe=probe,
                                                                                            probesupp = algo_inputs['probe_support_array'],
                                                                                            wavelength_m=input_dict["wavelength"],
                                                                                            pixelsize_m=input_dict["object_pixel"],
                                                                                            distance_m=input_dict["distance_sample_focus"],
                                                                                            params={'device': input_dict["GPUs"][0:1]})

            # fill errors
            error_rfactor.append(algo_error_rfactor)
            error_nmse.append(algo_error_mse)
            error_llk.append(algo_error_llk)

            #if algo_inputs["position_correction"] > 0:
            # corrected_positions = probe_positions

        else:
            sys.exit('Please select a proper algorithm! Selected: ', input_dict["algorithms"][str(counter)]['name'])

        if counter != len(input_dict['algorithms'].keys()) and plot == True:
            plot_amplitude_and_phase(obj, positions=positions+probe.shape[-1]//2,extent=get_plot_extent_from_positions(positions))

    # at this point, corrected_position should be holding either the corrected version of the probe_positions or the original one,
    # depending on whether algo_inputs["position_correction"]>0 or not,
    corrected_positions = probe_positions

    # change from (Y,X) back to (X,Y) for visualization
    corrected_positions = np.roll(corrected_positions, shift=1, axis=1)

    error_rfactor =  np.concatenate(error_rfactor).ravel()
    error_nmse = np.concatenate(error_nmse).ravel()
    error_llk = np.concatenate(error_llk).ravel()
    error = np.column_stack((error_rfactor, error_nmse, error_llk)) # must be (iterartions,3) array shape

    # save errors in input dict (debug)
    # input_dict['errors'] = error

    return obj, probe, error, corrected_positions, initial_obj, initial_probe

def check_for_nans(*arrays):
    """
    Check for NaN values in an arbitrary number of NumPy arrays and raise a ValueError if any are found.

    Parameters:
    arrays (ndarray): A variable number of NumPy arrays to check for NaNs.

    Raises:
    ValueError: If any NaN values are found in the arrays.

    :meta private:
    """
    for idx, array in enumerate(arrays):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input {idx+1} is not a NumPy array.")

        # Check if there are any NaNs in the array
        if np.isnan(array).any():
            raise ValueError(f"Array {idx+1} contains NaN values.")

def check_shape_of_inputs(DPs,positions,initial_probe):
    """ Checks the size compatibility between the diffractions patterns, probe and positions, and raise a ValuerError if any incompatibility is found. 
     
     Parameters:   
        DPs (ndarray): Measured diffraction patterns organize in (DPs index/number, DPs y-size, DPs x-size).
        positions (ndarray_): Measured positions in metric units (m, mm, um) or pixel, organized in (y,x) list.  
        initial_probe (ndarray): Initial approximation for the probe, see the dictionaty for the options. 

    Raises:
        ValueError: If the number of positions is different from the number of diffraction patterns. 
        ValueError: If the X, Y dimensions of the probe is different from the diffractions patterns one.
    
    :meta private:
    """    
    
    if DPs.shape[0] != positions.shape[0]:
        raise ValueError(f'There is a problem with input data!\nThere are {DPs.shape[0]} diffractiom patterns and {positions.shape[0]} positions. These values should be the same.')

    if initial_probe is not None: # mudei como conversado com Mauro
        if DPs[0].shape[0] != initial_probe.shape[-2] or DPs[0].shape[1] != initial_probe.shape[-1]:
            raise ValueError(f'There is a problem with your input data!\nThe dimensions of input_probe and diffraction pattern differ in the X,Y directions: {DPs.shape} vs {initial_probe.shape}')

def remove_positions_randomly(arr1, arr2, R):

    """Reduce the number of elements in a pair of same size arrays by a choosen value.

    Args:
        arr1 (ndarray): Input array same size as arr2 
        arr2 (ndarray): Input array same size as arr1
        R (int): Number of elements that will be removed

    Raises:
        ValueError: If the number of elements to remove are larger than the number of elements in the arrays

    Returns:
        (ndarray): Reduced array
        (ndarray): Reduced array
    
    :meta private:
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
    """Check the input_dict for the presence of all the demanded parameters, if not present, the parameter is set to a default value. Prints warning for missing parameters. 

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
    
    Returns:
        (dict): Checked and completed dictionary.

    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.   ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
    
    :meta private:
    """    
    
    # Define the default values
    default_values = {
        'datetime': get_datetime(),
        'CPUs': 32,
        'GPUs': [0],
        'regime': 'fraunhoffer', # 'fraunhoffer' or 'fresnel'
        'energy': 10,  # keV
        'detector_distance': 10,  # meters
        'distance_sample_focus': 0,
        'detector_pixel_size': 55e-6, # meters
        'binning': 1,
        'position_rotation': 0,
        'object_padding': 0,
        'incoherent_modes': 1,
        'n_of_positions_to_remove':0,
        'clip_object_magnitude':False,
        'free_log_likelihood':0,
        'fourier_power_bound':0,
        'positions_unit': None,
        'save_restored_data':False,
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
    """_summary_

    Args:
        DPs (_type_): _description_
        initial_obj (_type_): _description_
        initial_probe (_type_): _description_

    Returns:
        _type_: _description_
    
    :meta private:
    """
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


def create_parent_folder(file_path):
    """
    Create the parent folder of the specified file path if it does not exist.

    Args:
        file_path (str): The path of the file for which to create the parent directory.

    :meta private:
    """
    parent_folder = os.path.dirname(file_path)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
        print(f"Created directory: {parent_folder}")
    else:
        # print(f"Directory already exists: {parent_folder}")
        pass

def save_h5_output(input_dict,obj, probe, positions, error,initial_obj=None,initial_probe=None,corrected_positions=None,restored_data=None):
    """Creates the hdf5 output file.

     Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        obj (ndarray): Reconstructed complex object (amplitude and phase).
        probe (ndarray): Reconstructed complex probe (amplitude and phase) for all the incoherent modes organized in (mode index, probe).
        positions (ndarray_): Original positions in pixel unit, organized in (y,x) list.  
        error (ndarray): Reconstruction errors (r_factor: r-factor or residue, mse: normalized mean square, llk: log-likehood ) for all the iterations, organized in (iteration, r-factor, mse, llk).
        initial_obj (ndarray): Initial approximation for the object, see the dictionaty for the options [default: None].
        initial_probe (ndarray, optional): Initial approximation for the probe, see the dictionaty for the options. [default: None].
        corrected_positions (ndarray_): Final positions in pixel unit Same as input without the position correction and the optimized positions with position correction, organized in (y,x) list.  
        restored_data (ndarray, optional): Diffraction pattern that can be saved in the output. [default: None]
    
    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.   ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
    
    :meta private:
    """    

    with  h5py.File(input_dict["hdf5_output"], "w") as h5file:

        # Check if the group "recon" already exists
        if "recon" not in h5file:
            h5file.create_group("recon")

        # Check if the group "metadata" already exists
        if "metadata" not in h5file:
            h5file.create_group("metadata")

        h5file["metadata"].create_dataset('datetime',data=input_dict['datetime'])
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
        h5file["metadata"].create_dataset('n_of_positions_to_remove',data=input_dict['n_of_positions_to_remove'])
        h5file["metadata"].create_dataset('clip_object_magnitude',data=input_dict['clip_object_magnitude'])
        h5file["metadata"].create_dataset('free_log_likelihood',data=input_dict['free_log_likelihood'])
        h5file["metadata"].create_dataset('regime',data=input_dict['regime'])
        h5file["metadata"].create_dataset('fourier_power_bound',data=input_dict['fourier_power_bound'])


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

        if restored_data is not None:
            h5file["recon"].create_dataset('restored_data',data=restored_data)

        h5file["recon"].create_dataset('object',data=obj)
        h5file["recon"].create_dataset('probe',data=probe)
        h5file["recon"].create_dataset('positions',data=positions)
        h5file["recon"].create_dataset('probe_support_array',data=input_dict['probe_support_array'])
        if initial_probe is not None:
            h5file["recon"].create_dataset('initial_probe',data=initial_probe)
        if initial_obj is not None:
            h5file["recon"].create_dataset('initial_obj',data=initial_obj)
        if corrected_positions is not None:
            h5file["recon"].create_dataset('corrected_positions',data=corrected_positions)
        h5file["recon"].create_dataset('error',data=error)

    h5file.close()
    print('Results saved at: ',input_dict["hdf5_output"])

def convert_probe_positions_to_pixels(pixel_size, probe_positions,factor=1):
    """Convert the probe positions measured in metric units (m, mm, um) to pixel and offsets then to the origin.

    Args:
        pixel_size (float): Size of the pixels in the object
        probe_positions (ndarray): Probe position in metric units
        factor (int, optional): Conversion factor (m: 1, mm: 1e-3: mm, um: 1e-6)

    Returns:
        (ndarray): Probe positions in pixel unit.

    :meta private:
    """

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = factor * probe_positions[:, 0] / pixel_size  # convert from metric to pixels
    probe_positions[:, 1] = factor * probe_positions[:, 1] / pixel_size

    return probe_positions

def check_consecutive_keys(algorithms):

    """Checks whether the dictionary keys are consecutive integers. Utilized to check if the algorithms engines are organized correctly.
    Args:
        algorithms (dict): A dictionary with string keys that represent integers.
    
    Returns:
        (bool): True for consecutive integers and False if not. 
    
    :meta private:
    """
    
    keys = list(map(int, algorithms.keys()))
    keys.sort()
    return keys == list(range(1, len(keys) + 1))

def bin_volume(volume, downsampling_factor):
    """ Downsample a 3D volume (N,Y,X) in the Y, X directions by averaging over a specified downsampling factor. 

    Args:
        volume (ndarray): 3D numpy array of shape (N,Y,X).
        downsampling_factor (int): Downsampling factor, where "new_dimension = old_dimension/downsampling_factor".

    Raises:
        ValueError: error in case Y and X dimensions are not divisible by the downsampling factor.

    Returns:
        (ndarray): Downsampled volume, organized in a 3D numpy array of shape (N, Y//downsampling_factor, X//downsampling_factor).
    
    :meta private:
    """

    print('Binning data...')

    def suggest_crop_dimensions(Y, X, downsampling_factor):
        """Calculates the new crop dimensions in the case of X or Y dimensions not divisible by the downsampling factor.

        Args:
            Y (int): Y dimensions of the diffractions patterns.
            X (int): X dimensions of the diffractions patterns.
            downsampling_factor (int): Dimensionality reduction factor for the binning of the diffraction patterns. Must be larger than 1.

        Returns:
            (tuple): New downsampling factor for (X, Y) dimensions. 
        """        
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
        """Downsamples the input data with Numpy reshape and mean.

        Args:
            volume (ndarray): Volume that will be downsampled.
            downsampling_factor (int): Downsampling factor, where "new_dimension = old_dimension/downsampling_factor".

        Returns:
            (ndarray): Downsampled, or binned, data.
        
        :meta private:
        """        
        N, Y, X = volume.shape
        new_shape = (N, Y // downsampling_factor, downsampling_factor, X // downsampling_factor, downsampling_factor)
        downsampled_volume = volume.reshape(new_shape).mean(axis=(2, 4))
        return downsampled_volume

    binned_volume = numpy_downsampling(volume, downsampling_factor)

    print('Binned data to new shape: ', binned_volume.shape)

    # THESE ARE OLD STRATEGIES FOR BINNING
    # if input_dict["binning"] > 0:
    #     DPs = binning_G_parallel(DPs,input_dict["binning"],input_dict["CPUs"]) # binning strategy by G. Baraldi
    # if input_dict["binning"] < 0:
    #     DPs = DPs[:,0::np.abs(input_dict["binning"]),0::np.abs(input_dict["binning"])]
    #     input_dict["binning"] = np.abs(input_dict["binning"])

    return binned_volume

def binning_G(binning,DP):
    """Binning strategy of a 2D diffraction pattern implemented by Giovanni Baraldi. Deprecated. 

    Args:
        binning (int): Binning factor, where "new_dimension = old_dimension/binning". Must be larger than 1, even and positive.
        DP (ndarray): Diffractions pattern, with (X, Y) dimensions. 

    Returns:
        (ndarray): Binned diffraction pattern, with (X/binning, Y/binning) dimensions.
    
    :meta private:
    """    

    if binning % 2 != 0: # no binning
        sys.error(f'Please select an EVEN integer value for the binning parameters. Selected value: {binning}')
    else:
        while binning % 2 == 0 and binning > 0:
            avg = DP + np.roll(DP, -1, -1) + np.roll(DP, -1, -2) + np.roll(np.roll(DP, -1, -1), -1, -2)  # sum 4 neigboors at the top-left value

            div = 1 * (DP >= 0) + np.roll(1 * (DP >= 0), -1, -1) + np.roll(1 * (DP >= 0), -1, -2) + np.roll( np.roll(1 * (DP >= 0), -1, -1), -1, -2)  # Boolean array! Results in the n of valid points in the 2x2 neighborhood

            avg = avg + 4 - div  # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

            avgmask = (DP < 0) & ( div > 0)  # div > 0 means at least 1 neighbor is valid. DP < 0 means top-left values is invalid.

            DP[avgmask] = avg[avgmask] / div[ avgmask]  # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

            DP = DP[:, 0::2] + DP[:, 1::2]  # binning columns
            DP = DP[0::2] + DP[1::2]  # binning lines

            DP[DP < 0] = -1

            DP[div[0::2, 0::2] < 3] = -1  # why div < 3 ? Every neighborhood that had 1 or 2 invalid points is considered invalid?

            binning = binning // 2

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


def binning_G_parallel(DPs,binning, processes):
    """Parallel version of the binning_G utilized for certain number of processes. Deprecated. 

    Args:
        DPs (ndarray): Diffractions pattern, with (X, Y) dimensions. 
        binning (int): Binning factor, where "new_dimension = old_dimension/binning". Must be larger than 1, even and positive.
        processes (int) Number of CPU processed that will be utilized.

    Returns:
        (ndarray): Binned diffraction pattern, with (X/binning, Y/binning) dimensions.
    
    :meta private:
    """    
 
    # def call_binning_parallel(DP):
    #     return binning_G(DP,binning) # binning strategy by G. Baraldi

    def binning_G2(binning,DP):
        return binning_G(DP,binning)

    from functools import partial
    call_binning_parallel = partial(binning_G, binning)

    n_frames = DPs.shape[0]

    binned_DPs = np.empty((DPs.shape[0],DPs.shape[1]//binning,DPs.shape[2]//binning))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(call_binning_parallel,[DPs[i,:,:] for i in range(n_frames)])
        for counter, result in enumerate(results):
            binned_DPs[counter,:,:] = result

    if binned_DPs.shape[1] % 2 != 0: # make shape even
        binned_DPs = binned_DPs[:,0:-1,:]
    if binned_DPs.shape[2] % 2 != 0:
        binned_DPs = binned_DPs[:,:,0:-1]

    return binned_DPs

def append_ones(probe_positions):
    """Adjust shape and column order of positions array to be accepted by Giovanni's code.

    Args:
        probe_positions (ndarray): Initial positions array in (Y,X) shape.

    Returns:
        probe_positions (ndarray): Rearranged probe positions array.
    
    :meta private:
    """
    zeros = np.zeros((probe_positions.shape[0],1))
    probe_positions = np.concatenate((probe_positions,zeros),axis=1)
    probe_positions = np.concatenate((probe_positions,zeros),axis=1) # concatenate columns to use Giovanni's ptychography code

    return probe_positions

def set_initial_probe(input_dict, DPs, incoherent_modes):
    """Created the initial probe for the reconstruction. Probe options are set on the input dictionary as initial_probe (see the dictionary definition) and are: Mean diffraction FFT inverse,
    Fresnel zone plate, Circular, Randon values between 0 and 1, Constant 1s matrix, Load numpy array, Load hdf5 recon.

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        DPs (ndarray): Measured diffraction patterns organize in (DPs index/number, DPs y-size, DPs x-size).
        incoherent_modes (int): Number of incoherent modes for the probe decompostion. 

    Raises:
        ValueError: If no proper probe options has been chosen. See the dictionary initial probe definition for valid options. 

    Returns:
        (ndarray): Initial probe (modes, Y, X).

    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.  ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
    
    :meta private:
    """    
    print('Creating initial probe of type: ',input_dict['initial_probe']["probe"])

    DP_shape = (DPs.shape[1], DPs.shape[2])

    def set_modes(probe, input_dict):
        """Creates all the probe modes by appling a randon modulation to initial single probe mode. 

        Args:
            probe (ndarray): Inital mode (mode = 0) of the probe.
            input_dict (dict): Dictionary with the experiment info and reconstruction parameters.

        Returns:
            (ndarray): Probe with all modes.
        
        :meta private:    
        """        
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
                                    grid_shape = DPs.shape[-1],
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
            probe = h5py.File(path,'r')[input_dict['initial_probe']['h5_tree_path']][()]
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
    """Determines the if the variable is a path, string or array.

    Args:
        variable (str or ndarray): Variable of undefined type that will be tested.

    Raises:
        ValueError: If the variable is not a path, string  or array.

    Returns:
        (str): String that defines the variable type defined as path ("path"), string("standard") or array("array").
    
    :meta private:
    """    
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
    """Created the initial object for the reconstruction. Object options are set on the input dictionary as initial_obj (see the dictionary definition) and are: Random,
       Constant 1s matrix, Numpy array, Load numpy array, Load hdf5 recon.

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        DPs (ndarray): Measured diffraction patterns organize in (DPs index/number, DPs y-size, DPs x-size).
        probe (ndarray): Inital probe organized in (modes,probe).
        obj_shape (tuple): (X, Y) dimensions of the object. 

    Raises:
        ValueError: If no proper object options has been chosen. See the dictionary initial object definition for valid options. 

    Returns:
        (ndarray): Initial object.

    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.  ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
    
    :meta private:
    """    
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
            obj = h5py.File(input_dict['initial_obj']['obj'],'r')[input_dict['initial_obj']['h5_tree_path']] # select first frame of object
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
    Create mask containing probe support region. Probe support options are set on the input dictionary (see the dictionary definition) and options are: Circular, Cross and Numpy array. 

    Args:
        input_dict (dict): Dictionary with the experiment info and reconstruction parameters.
        probe_shape (array): Shape of the probe (mode, Y, X).

    Raises:
        ValueError: If the no valid probe support option has been chosen. See dictionary definition of probe support for options. 

    Returns:
        (ndarray): Mask containing probe support (Y, X).

    Dictionary parameters:
        
        * ``input_dict['energy']`` (float, optional): Incident wave energy utilized in experiment in keV [default: 10]
        * ``input_dict['wavelenght']`` (float, optional): Incident wave wavelenght utilized in experiment in meters. If not in dict will be calculated from the energy value [default: 1.23984e-10]
        * ``input_dict['detector_distance']`` (float, optional): Distance between sample and detector in meters [default: 10]
        * ``input_dict['GPUs']`` (ndarray, optional): List of gpus  [default: 0] 
        * ``input_dict['CPUs']`` (int, optional):  Number of available cpus [default: 32]
        * ``input_dict['hdf5_output']`` (str, optional): Output .hdf5 file for the results [default: None]
        * ``input_dict['regime']`` (str, optional ): Diffraction regime for near-field (fresnel) and far-field (fraunhoffer) [default: fraunhoffer] 
        * ``input_dict['binning']`` (int, optional): Binning of the diffraction patterns prior to processing [default: 1]
        * ``input_dict['n_of_positions_to_remove']`` (int, optional): Number of random positions that will not be included in the reconstruction [default: 0]
        * ``input_dict['object_padding']`` (int, optional): Number of pixels that will be included in the edges of the object. Usefull in position correction to extend the original object [default: 0]  
        * ``input_dict['incoherent_modes']`` (int, optional): Number of incoherent model for the probe [default: 1]
        * ``input_dict['fourier_power_bound']`` (float, optional): Relaxed the wavefront update, 0 is the standard [default: 0]
        * ``input_dict['clip_object_magnitude']`` (bool, optional): Clips the object amplitude between 0 and 1 [default: False]
        * ``input_dict['distance_sample_focus']`` (float, optional): Distance between the incident beam focus and sample (Near-Field only) [default: 0]
        * ``input_dict['probe_support']`` (dict, optional): Mask utilized as support for the probe projection in real space [default: {"type": "circular", "radius": 300, "center_y": 0, "center_x": 0}]
            #. ``Circular``: {"type": "circular",  "radius": float, "center_y": int, "center_x": int}.
            #. ``Cross``:    {"type": "cross",  "center_width": int, "cross_width": int, "border_padding": int }.
            #. ``Numpy array``:  {"type": "array",  "data": myArray}.
        * ``input_dict['initial_obj']`` (dict): Initial guess for the object if initial_obj = None [required]
            #. ``Random``:  {"obj": 'random'}.
            #. ``Constant 1s matrix``: {"obj": 'constant'}.
            #. ``Numpy array``: {"obj": my2darray}.
            #. ``Load numpy array``: {"obj": 'path/to/numpyFile.npy'}.
            #. ``Load hdf5 recon``: {"obj": 'path/to/hdf5File.h5'}, reconstructed object must be in 'recon/object', as default in ssc-cdi.
        *``input_dict['initial_probe']`` (dict) Initial guess for the probe if initial_probe = None [required]
            #.  ``Mean diffraction FFT inverse``: {"probe": 'inverse'}.
            #.  ``Fresnel zone plate``: {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters,'fzp_diameter': diameter in meters, 
                                        'fzp_outer_zone_width': zone width in meters, 'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 
                                        'probe_diameter': diameter, 'probe_normalize': boolean}
            #.  ``Circular``: {"probe": 'circular', "radius": int, "distance": float}. 
            #.  ``Randon values between 0 and 1``: {"probe": 'random'}.
            #.  ``Constant 1s matrix``: {"probe": 'constant'}.
            #.  ``Load numpy array``: {"probe": 'path/to/numpyFile.npy'}.
            #.  ``Load hdf5 recon``: {"probe": 'path/to/hdf5File.h5'}, reconstructed probe must be in 'recon/probe', as default in ssc-cdi.
        *``input_dict['algorithms']['number']`` (dict) Algorithms utilized in the reconstruction and their sequence [0,1,2,...,number][required]
            #. ``PIE (Ptychographic Iterative Engine)``: {'name': 'PIE', 'iterations': int, 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                         'regularization_object': float (min: 0, max: 1), 'regularization_probe': float ((min: 0, max: 1)
                                                         'momentum_obj': float (if > 0, uses mPIE with the given friction value) , momentum_probe': float (if > 0, uses mPIE with the given friction value), 
                                                         'position_correction': int (0: no correction, N: performs correction every N iterations)}
            #. ``AP (Alternating Projections)``: {'name': 'AP', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                 'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                 'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                 'momentum_obj': float , momentum_probe': float, 
                                                'position_correction': (0: no correction, N: performs correction every N iterations)}
            #. ``RAAR (Relaxed Averaged Alternating Reflections)``: {'name': 'RAAR', 'iterations': int, batch: int (define the number of positions to fit into the GPU),
                                                                    'beta': float (wavefront update relaxation, if 1 utilizes DM: Differential Mapping)
                                                                    'step_object': float (min: 0, max: 1), 'step_probe': float (min: 0, max: 1),
                                                                    'regularization_object': float (min: 0, max: 1), 'regularization_probe': float (min: 0, max: 1),
                                                                    'momentum_obj': float , momentum_probe': float, 
                                                                    'position_correction': (0: no correction, N: performs correction every N iterations)}
            # ``Test Engine: PIE_python``: {'name': 'rPIE_python', 'iterations': int, 'step_object': float,  'step_probe': float, 'regularization_object': float,
                                           'regularization_probe': float,'momentum_obj': float, 'momentum_probe': float, 'mPIE_momentum_counter': float} 
            # ``Test Engine: RAAR_python``: {'name': 'RAAR_python', 'iterations': int, 'beta': float, 'regularization_obj': float 'regularization_probe': float} 
            # ``Test Engine: AP_python``: {'name': 'AP_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: DM_python``: {'name': 'DM_python', 'iterations': int, 'regularization_obj': float, 'regularization_probe': float}
            # ``Test Engine: ML_python``: {'name': 'ML_python', 'iterations': int, 'optimizer': 'gradient_descent', 'step_object': float, 'step_probe': float}
    
    :meta private:
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
        probe = input_dict["probe_support"]["data"]

    else:
        raise ValueError(f"Select an appropriate probe support:{input_dict['probe_support']}")

    return probe

def create_circular_mask(mask_shape, radius):
    """" Create circular mask

    Args:
        mask_shape (tuple): Y,X shape of the mask.
        radius (int): Radius of the mask in pixels.

    Returns:
        (ndarray): Circular mask of 1s and 0s.
    
    :meta private:
    """


    """ All values in pixels """
    center_row, center_col = mask_shape
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    return np.where((Xmesh - center_col//2) ** 2 + (Ymesh - center_row//2) ** 2 <= radius ** 2, 1, 0)

def create_cross_mask(mask_shape, cross_width_y=15, border=3, center_square_side = 10, cross_width_x=0):
    """ Create a cross shaped mask
    Args:
        mask_shape (tuple): y and x sizes of the mask
        cross_width_y (int, optional): Cross width along y. [default:  15].
        border (int, optional): Distance from edge of cross mask to the domain border. [default: 3].
        center_square_side (int, optional): Size of the square edge at the center of the cross. [default: 10].
        cross_width_x (int, optional): Cross width along x. [default:  0].

    Returns:
       (ndarray): Cross mask of 1s and 0s.
    
    :meta private:
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
    """Calculate and display the object pixel size

    This function computes the object pixel size based on the provided wavelength, detector distance,
    detector pixel size, diffraction pattern (DP) size, and optional binning factor. It also calculates
    the limit thickness for a resolution of 1 pixel and prints these values.

    Args:
        wavelength (float): Wavelength of the light used in meters.
        detector_distance (float): Distance from the sample to the detector in meters.
        detector_pixel_size (float): Size of a pixel on the detector in meters.
        DP_size (int): Size of the diffraction pattern (number of pixels).
        binning (int, optional): Binning factor. Must be an even number. If 1, no binning occurs. defaults to 1.

    Returns:
        (float): Calculated object pixel size in meters.

    :meta private:
    """

    object_pixel_size = calculate_object_pixel_size(wavelength, detector_distance, detector_pixel_size, DP_size)
    print(f"\tObject pixel size = {object_pixel_size*1e9:.2f} nm")

    PA_thickness = 4 * object_pixel_size ** 2 / (0.61 * wavelength)
    print(f"\tLimit thickness for resolution of 1 pixel: {PA_thickness*1e6:.3f} microns")
    return object_pixel_size

def set_object_shape(object_padding, DP_shape, probe_positions):
    """ Determines shape (Y,X) of object matrix from size of probe and its positions.

    Args:
        object_padding (int): Number of pixels to pad in the border of the object array.
        DP_shape (tuple): Shape of the diffraction patterns array.
        probe_positions (ndarray): Array os probe positiions in pixels.

    Returns:
        (tuple): Object shape (Y,X).
    
    :meta private:
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
    Creates the probe support characteristic of a Fresnel Zone Plate.
    Args:
        wavelength (float): Wavelength of experiment in meters.
        grid_shape (int or list of int): Shape of the grid, either an int (for a square grid) or a list [int, int] for a rectangular grid.
        pixel_size_object (float): Size of a pixel in the object plane in meters.
        beam_type (str): Type of the beam, either 'gaussian' or 'disc'.
        distance_sample_fzpf (float): Distance between the sample and the focus of the FZP in meters.
        fzp_diameter (float): Diameter of the FZP in meters.
        fzp_outer_zone_width (float): Width of the outermost zone of the FZP in meters.
        beamstopper_diameter (float): Diameter of the beamstopper in meters.
        probe_diameter (float): Diameter of the probe in meters.
        probe_normalize (bool): Whether to normalize the probe.
        upsample (int, optional): Upsampling factor for the grid. [default: 10]

    Raises:
        ValueError: If the grid shape is not a integer or list/tuple if integer.
        ValueError: If the beam type is not 'gaussian' or 'disc'.
         
    Returns:
        (ndarray): The initial probe after applying the FZP and beamstopper.
    
    :meta private:
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

