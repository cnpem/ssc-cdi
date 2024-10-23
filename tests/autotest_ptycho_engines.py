import time

t1 = time.time()

print('\n ################################################ STARTING TESTS FOR SSC-CDI ################################################')

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import ndimage

from sscCdi import call_ptychography

def shift_ctr_of_mass_to_img_ctr(image):

    image = np.abs(image)

    # Get the image shape
    height, width = image.shape

    # Calculate the center of mass
    com = ndimage.center_of_mass(image)

    # Calculate the shift needed to center the center of mass
    shift_y = int(height / 2 - com[0])
    shift_x = int(width / 2 - com[1])

    # Apply the shift using np.roll
    shifted_image = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
    return shifted_image, shift_y, shift_x

def compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold, N=50):
    shifted_probe, shift_y, shift_x = shift_ctr_of_mass_to_img_ctr(probe[0])
    shifted_obj = np.roll(obj, shift=(shift_y, shift_x), axis=(0, 1))
    # shifted_obj = shifted_obj[input_dict['object_padding']:-input_dict['object_padding'],input_dict['object_padding']:-input_dict['object_padding']]

    shifted_obj = shifted_obj[N:-N,N:-N]
    model_obj = model_obj[N:-N,N:-N]

    # shifted_obj = shifted_obj/np.max(shifted_obj)
    # model_obj = model_obj/np.max(model_obj)
    # shifted_probe = shifted_probe/np.max(shifted_probe)
    # model_probe = model_probe/np.max(model_probe)

    mean_squared_error_obj = np.abs(np.real(np.sum((model_obj - shifted_obj)**2)))/np.prod(model_obj.shape)
    mean_squared_error_probe = np.abs(np.real(np.sum((model_probe - shifted_probe)**2)))/np.prod(model_probe.shape)

    if mean_squared_error_obj > error_threshold:
        print('ACHO QUE DEU RUIM. Ver imagens!')
        # raise ValueError(f'Error in object reconstruction is higher than the threshold. Error: {mean_squared_error_obj}')
    else:
        print('PASSED!')
        print(f'Error in object reconstruction is within the threshold. Error {mean_squared_error_obj} < {error_threshold}')
        print(f'Error in probe reconstruction : {mean_squared_error_probe}')

    return mean_squared_error_obj, mean_squared_error_probe

def save_results_as_pngs(obj, probe, prefix, output_folder):

    print("Object shape:", obj.shape, "Probe shape:", probe.shape)

    probe = np.squeeze(probe)

    # Create a figure for the 4 side-by-side subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Object Magnitude
    obj_magnitude = np.abs(obj)
    obj_magnitude_normalized = (255 * (obj_magnitude - obj_magnitude.min()) / (obj_magnitude.ptp())).astype(np.uint8)
    axes[0].imshow(obj_magnitude_normalized, cmap='gray')
    axes[0].set_title('Object Magnitude')
    axes[0].axis('off')

    # Object Phase
    obj_phase = np.angle(obj)
    obj_phase_normalized = (255 * (obj_phase - obj_phase.min()) / (obj_phase.ptp())).astype(np.uint8)
    axes[1].imshow(obj_phase_normalized, cmap='gray')
    axes[1].set_title('Object Phase')
    axes[1].axis('off')

    # Probe Magnitude
    probe_magnitude = np.abs(probe)
    probe_magnitude_normalized = (255 * (probe_magnitude - probe_magnitude.min()) / (probe_magnitude.ptp())).astype(np.uint8)
    axes[2].imshow(probe_magnitude_normalized, cmap='gray')
    axes[2].set_title('Probe Magnitude')
    axes[2].axis('off')

    # Probe Phase
    probe_phase = np.angle(probe)
    probe_phase_normalized = (255 * (probe_phase - probe_phase.min()) / (probe_phase.ptp())).astype(np.uint8)
    axes[3].imshow(probe_phase_normalized, cmap='gray')
    axes[3].set_title('Probe Phase')
    axes[3].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{prefix}_results.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Results saved as {output_path}")

print('\n ################################################ LOADING TEST DATA ################################################')

model_obj = np.load('object.npy')
model_probe = np.load('probe.npy')
data = np.load('diff_patterns.npy')
positions = np.load('positions.npy')
positions = np.roll(positions, 1, axis=1)

output_folder = "./" # Paths to output 

error_threshold = 1 # Error threshold for the tests

print('Done. Outputs will be saved in:', output_folder)

print('\n ################################################ QUALITY TEST USING ALL CUDA RAAR+PIE+AP ################################################')

initial_obj = None
initial_probe = None    

input_dict = {}
input_dict['algorithms'] = {} # create a sub-dictionary to store the algorithms

input_dict['CPUs'] = 32  # Number of CPUs to use in parallel execution

input_dict['GPUs'] = [0]  # List of numbers (e.g. [0,1,2]) containing the number of the GPUs

input_dict['hdf5_output'] = output_folder+'test_output.h5'  # Path to hdf5 file to contain all outputs

input_dict['regime'] = 'fraunhoffer'  # Propagation regime. 'fraunhoffer' or 'fresnel'

input_dict['binning'] = 1  # Binning factor (must be an even number). If 1, no binning occurs.

input_dict['n_of_positions_to_remove'] = 1 # number of positions to randomly remove from the scan

input_dict['energy'] = 10  # Energy in keV

input_dict['detector_distance'] = 15  # Detector distance in meters

input_dict['detector_pixel_size'] = 55e-6  # Detector pixel size in meters

input_dict['object_padding'] = 10 # number of pixels to pad the object array. May be necessary if scan area is too large for the initial_object estimate

input_dict['incoherent_modes'] = 1  # Number of incoherent modes to use

input_dict['fourier_power_bound'] = 0  # relaxing the magnitude constraint on the Fourier domain. see equation S2 of Giewekemeyer et al. 10.1073/pnas.0905846107

input_dict['clip_object_magnitude'] = False  # If True, clips the object magnitude between 0 and 1

input_dict['free_log_likelihood'] = 0 # if 0, does not compute free errors. If N, will extract N points from each diffraction pattern for using them for error computation.

input_dict['position_rotation'] = 0 # if 0, does not compute free errors. If N, will extract N points from each diffraction pattern for using them for error computation.

input_dict['fresnel_regime'] = 0 # if 0, does not compute free errors. If N, will extract N points from each diffraction pattern for using them for error computation.

input_dict['distance_sample_focus'] = 0  # Distance in meters between sample and focus or pinhole, used to propagate the probe prior to application of the probe support

input_dict['probe_support'] = {"type": "circular",  "radius": 100,  "center_y": 0, "center_x": 0}  # Support to be applied to the probe matrix after probe update.
                                                                                                   # Options are: 
                                                                                                   # - {"type": "circular",  "radius": 300,  "center_y": 0, "center_x": 0} 
                                                                                                   # - {"type": "cross",  "center_width": 300,  "cross_width": 0, "border_padding": 0} 
                                                                                                   # - {"type": "array",  "data": myArray}

input_dict["initial_obj"] = {"obj": 'random'}  # Initial guess for the object
                                               # Options are: 
                                               # - {"obj": my2darray}, numpy array 
                                               # - {"obj": 'path/to/numpyFile.npy'}, path to .npy, 
                                               # - {"obj": 'path/to/hdf5File.h5'}, path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                                               # - {"obj": 'random'}, random matrix with values between 0 and 1
                                               # - {"obj": 'constant'}, constant matrix of 1s


input_dict['initial_probe'] = { "probe": 'inverse'}  # Initial guess for the probe
                                                     # Options are: 
                                                     # - {"probe": my2darray}, numpy array 
                                                     # - {"probe": 'path/to/numpyFile.npy'}, path to .npy, 
                                                     # - {"probe": 'path/to/hdf5File.h5'}, path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
                                                     # - {"probe": 'random'}, random matrix with values between 0 and 1
                                                     # - {"probe": 'constant'}, constant matrix of 1s
                                                     # - {"probe": 'inverse'}, matrix of the Inverse Fourier Transform of the mean of DPs.
                                                     # - {"probe": 'circular', "radius": 100, "distance":0}, circular mask with a pixel of "radius". 
                                                     #   If a distance (in meters) is given, it propagates the round probe using the ASM method.
                                                     # - {"probe": 'fzp', 'beam_type': 'disc' or 'gaussian', 'distance_sample_fzpf': distance in meters, 
                                                     #   'fzp_diameter': diameter in meters, 'fzp_outer_zone_width': zone width in meters, 
                                                     #   'beamstopper_diameter': diameter in meters (0 if no beamstopper used), 'probe_diameter': diameter, 
                                                     #   'probe_normalize': boolean}


input_dict['algorithms'] = {}

input_dict['algorithms']['1'] = {
    'name': 'RAAR',          # Relaxed Averaged Alternating Reflections
    'batch': 64,             # number of data arrays to fit into the GPU. If the GPU runs out of memory, reduce this number.
    'iterations': 100,
    'beta': 0.5,
    'step_object': 0.5,
    'step_probe': 0.9,
    'regularization_object': 0.1,
    'regularization_probe': 0.1,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0, # 0: no correction. N: performs correction every N iterations
} 

input_dict['algorithms']['2'] = {       # rPIE and mPIE engines. mPIE used if momentum_obj or momentum_probe > 0. Use batch=1 by default.
    'name': 'PIE',                      # Ptychographic Iterative Engine
    'iterations': 100,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.5,
    'regularization_probe': 0.5,
    'momentum_obj': 0.0,                # if > 0, uses mPIE with the given friction value
    'momentum_probe': 0.0,              # if > 0, uses mPIE with the given friction value
    'position_correction': 0,           # 0: no correction. N: performs correction every N iterations
}  

input_dict['algorithms']['3'] = {
    'name': 'AP',                   # Alternating Projections
    'batch': 64,                    # number of data arrays to fit into the GPU. If the GPU runs out of memory, reduce this number.
    'iterations': 100,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.01,
    'regularization_probe': 0.01,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0,       # 0: no correction. N: performs correction every N iterations
}  

print(json.dumps(input_dict, indent=4))
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

np.save(os.path.join(output_folder, 'object.npy'), obj)
np.save(os.path.join(output_folder, 'probe.npy'), probe)

save_results_as_pngs(obj,probe,'00_full_pipeline',output_folder)

print(obj.shape, model_obj.shape)
error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)



print('\n ################################################ Quality control of RAAR_CUDA ################################################')
print("Switching to 1 incoherent mode")
input_dict['incoherent_modes'] = 1  # Number of incoherent modes to use

input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {
    'name': 'RAAR',          # Relaxed Averaged Alternating Reflections
    'batch': 64,             # number of data arrays to fit into the GPU. If the GPU runs out of memory, reduce this number.
    'iterations': 100,
    'beta': 0.9,
    'step_object': 0.9,
    'step_probe': 0.9,
    'regularization_object': 0.1,
    'regularization_probe': 0.1,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0, # 0: no correction. N: performs correction every N iterations
} 
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)
save_results_as_pngs(obj,probe,'01_RAAR_CUDA',output_folder)

error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)

print('\n ################################################Quality control of AP_CUDA ################################################')
input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {
    'name': 'AP',                   # Alternating Projections
    'batch': 64,                    # number of data arrays to fit into the GPU. If the GPU runs out of memory, reduce this number.
    'iterations': 300,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.01,
    'regularization_probe': 0.01,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0,       # 0: no correction. N: performs correction every N iterations
}  
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)
save_results_as_pngs(obj,probe,'02_AP_CUDA',output_folder)

error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)


print('\n ################################################ Quality control of PIE_CUDA ################################################')
input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {       # rPIE and mPIE engines. mPIE used if momentum_obj or momentum_probe > 0. Use batch=1 by default.
    'name': 'PIE',                      # Ptychographic Iterative Engine
    'iterations': 300,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.5,
    'regularization_probe': 0.5,
    'momentum_obj': 0.0,                # if > 0, uses mPIE with the given friction value
    'momentum_probe': 0.0,              # if > 0, uses mPIE with the given friction value
    'position_correction': 0,           # 0: no correction. N: performs correction every N iterations
}  
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)
save_results_as_pngs(obj,probe,'03_PIE_CUDA',output_folder)

error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)

path_npy_obj = 'object.npy'
path_npy_probe = 'probe.npy'

path_hdf5_obj = 'object.h5'
path_hdf5_probe = 'probe.h5'

# probe = np.expand_dims(probe, axis=0)
# def convert_npy_to_hdf5(npy_path, hdf5_path):
#     # Load the .npy file
#     data = np.load(npy_path)
    
#     # Create the HDF5 file and save the data under the 'data' key
#     with h5py.File(hdf5_path, 'w') as hdf5_file:
#         hdf5_file.create_dataset('data', data=data)
    
#     print(f"Data from {npy_path} has been saved to {hdf5_path} under the 'data' dataset.")

# convert_npy_to_hdf5(path_npy_obj, path_hdf5_obj)
# convert_npy_to_hdf5(path_npy_probe, path_hdf5_probe)




print('\n ################################################ Quality control of rPIE_python ################################################')

# PYTHON ENGINES ARE SLOW. SWTICHING TO CORRECT GUESSES JUST TO CHECK IF PYTHON VERSIONS ARE NOT MESSING THEM UP.
input_dict["initial_obj"] = {"obj": path_npy_obj} # path to .npy, 
input_dict['initial_probe'] = {"probe": path_npy_probe} # path to .npy, 


input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {            # rPIE and mPIE engines. mPIE used if mPIE_friction_obj or mPIE_friction_probe > 0. 
    'name': 'rPIE_python',
    'iterations': 10,
    'step_object': 0.5,                      # step size of object update function. 
    'step_probe': 1,                         # step size of probe update function
    'regularization_object': 0.25,           # rPIE regularization parameter. Should be between 0 and 1
    'regularization_probe': 0.5,             # rPIE regularization parameter. Should be between 0 and 1
    'momentum_obj': 0.1,                     # if > 0, uses mPIE with the given friction value
    'momentum_probe': 0.1,                   # if > 0, uses mPIE with the given friction value
    'mPIE_momentum_counter': 10,             # if == N, performs mPIE update every N iterations
}  

obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)
save_results_as_pngs(obj,probe,'04_PIE_python',output_folder)

error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)


print('\n ################################################ Quality control of RAAR_python ################################################')
input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {
    'name': 'RAAR_python',
    'iterations': 10,
    'beta': 0.9,                        # RAAR wavefront beta step
    'regularization_obj': 0.01,      # avoid division by zero in object update
    'regularization_probe': 0.01,       # avoid division by zero in probe update
}  
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)
save_results_as_pngs(obj,probe,'05_RAAR_python',output_folder)


error_obj, error_probe = compare_model_to_recon(obj,probe,model_obj,model_probe,input_dict, error_threshold,N=50)


print('################################################ FIRE TESTS ################################################')

print("\n\n")
print('FIRE TEST: trying all different options of probe support, initial_obj and initial_probe.')
print('Image previews will not be saved!')
print("\n\n")

input_dict['algorithms'] = {}
input_dict['algorithms']['1'] = {
    'name': 'RAAR',          # Relaxed Averaged Alternating Reflections
    'batch': 64,             # number of data arrays to fit into the GPU. If the GPU runs out of memory, reduce this number.
    'iterations': 5,
    'beta': 0.9,
    'step_object': 0.9,
    'step_probe': 0.9,
    'regularization_object': 0.1,
    'regularization_probe': 0.1,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0, # 0: no correction. N: performs correction every N iterations
} 

print("\n################################################ PROBE SUPPORT CIRCULAR ##########################################\n")
input_dict['probe_support'] = {"type": "circular",  "radius": 100,  "center_y": 0, "center_x": 0}  # Support to be applied to the probe matrix after probe update.
print("input_dict['probe_support'] = ", input_dict['probe_support'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ PROBE SUPPORT CROSS ##########################################\n")
input_dict['probe_support'] = {"type": "cross",  "center_width": 300,  "cross_width": 0, "border_padding": 0} 
print("input_dict['probe_support'] = ", input_dict['probe_support'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ PROBE SUPPORT FROM ARRAY ##########################################\n")
input_dict['probe_support'] =  {"type": "array",  "data": np.ones_like(probe,dtype=np.float32)}
print("input_dict['probe_support'] = ", input_dict['probe_support'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL RANDOM OBJECT ##########################################\n")
input_dict["initial_obj"] = {"obj": 'random'}  # Initial guess for the object
print("input_dict['initial_obj'] = ", input_dict['initial_obj'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL OBJECT FROM ARRAY ##########################################\n")
input_dict["initial_obj"] = {"obj": np.ones_like(obj)} # numpy array 
print("input_dict['initial_obj'] = ", input_dict['initial_obj'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL OBJECT FROM .NPY ##########################################\n")
input_dict["initial_obj"] = {"obj": path_npy_obj} # path to .npy, 
print("input_dict['initial_obj'] = ", input_dict['initial_obj'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL OBJECT FROM .HDF5 ##########################################\n")
input_dict["initial_obj"] = {"obj": path_hdf5_obj,'h5_tree_path':'data'} # path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
print("input_dict['initial_obj'] = ", input_dict['initial_obj'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL OBJECT CONSTANT ##########################################\n")
input_dict["initial_obj"] = {"obj": 'constant'} # constant matrix of 1s
print("input_dict['initial_obj'] = ", input_dict['initial_obj'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE IFT ##########################################\n")
input_dict['initial_probe'] = { "probe": 'inverse'}  # IFT of the average data
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE FROM ARRAY ##########################################\n")
input_dict['initial_probe'] = {"probe": np.ones_like(probe)} # numpy array 
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE FROM .NPY ##########################################\n")
input_dict['initial_probe'] = {"probe": path_npy_probe} # path to .npy, 
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE FROM .HDF5 ##########################################\n")
input_dict['initial_probe'] = {"probe": path_hdf5_probe,"h5_tree_path":'data'} # path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE RANDOM ##########################################\n")
input_dict['initial_probe'] = {"probe": 'random'} # random matrix with values between 0 and 1
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE CONSTANT ##########################################\n")
input_dict['initial_probe'] = {"probe": 'constant'} # constant matrix of 1s
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE CIRCULAR ##########################################\n")
input_dict['initial_probe'] = {"probe": 'circular', "radius": 100, "distance":0} # circular mask with a pixel of "radius".  If a distance (in meters) is given, it propagates the round probe using the ASM method.
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n################################################ INITIAL PROBE FZP ##########################################\n")
input_dict['initial_probe'] = {"probe": 'fzp', 
                                'beam_type': 'disc', # or 'gaussian', 
                                'distance_sample_fzpf': 2.9e-3, # in meters,
                                'fzp_diameter': 50e-6, # in meters,
                                'fzp_outer_zone_width': 50e-9, # in meters, 
                                'beamstopper_diameter': 20e-6, # in meters (0 if no beamstopper used), 
                                'probe_diameter': 50e-6,
                                'probe_normalize': False} # boolean
print("input_dict['initial_probe'] = ", input_dict['initial_probe'])
obj, probe, new_positions, input_dict, error = call_ptychography(input_dict.copy(),data, positions, initial_obj=initial_obj, initial_probe=initial_probe)

print("\n##############################################################################################################")
print("################################################ SSC-CDI TESTS PASSED #########################################")
print("###############################################################################################################")
print(f'Total test time: {(time.time()-t1)/60} minutes')
