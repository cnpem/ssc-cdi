import numpy as np
import matplotlib.pyplot as plt
import os,sys, json, time, h5py
from skimage.io import imsave

""" Sirius Scientific Computing Imports """
from sscRadon import radon
from sscCdi.tomo.tomo_processing import angle_mesh_organize, tomography, equalize_frames_parallel, equalize_tomogram, reorder_slices_low_to_high_angle, sort_frames_by_angle
from sscCdi.misc import create_directory_if_doesnt_exist
from sscCdi.processing.unwrap import unwrap_in_parallel

input_dictionary = json.load(open(sys.argv[1])) # LOAD JSON!

"""                  INPUTS                 """
# { # 1 -> True ; 0 -> False
# "Sort":1 ,
# "Crop":1 ,
# "Unwrap":1,
# "Equalize": 1
# "Wiggle":1,
# "Tomo":1
# }

processing_steps = input_dictionary["processing_steps"]
contrast_type = input_dictionary["contrast_type"] # Phase Or Magnitude?
recon_method = input_dictionary["method"] # pwcdi or ptycho?
prefix_string = f'_{contrast_type}'

""" Select data folders  """
sinogram_path = input_dictionary["sinogram_path"]  # folder containing sinogram of 2d projections
raw_angles_path   = input_dictionary["angles_path"]

""" Crop: Select the cropping slices # SLICE MUST HAVE EVEN NUMBER OF POINTS!!! """
top, bottom = input_dictionary["top_crop"], input_dictionary["bottom_crop"] #300, 300 # number of pixels to crops in each direction
left, right = input_dictionary["left_crop"], input_dictionary["right_crop"] #300, 300

""" Phase Unwrapping: remove bad frames and unwrap remaining ones """
bad_frames_before_unwrap = input_dictionary["bad_frames_before_unwrap"] #[7,20,36,65,94,123,152,181,210,239,268,296,324]
phase_unwrap_iterations = input_dictionary["unwrap_iterations"]
phase_unwrap_non_negativity = input_dictionary["unwrap_non_negativity"]
phase_unwrap_gradient_removal = input_dictionary["unwrap_gradient_removal"]

""" Frame Equalization """
equalize_invert        = input_dictionary["equalize_invert"]
equalize_gradient      = input_dictionary["equalize_gradient"]
equalize_outliers      = input_dictionary["equalize_outliers"]
equalize_global_offset = input_dictionary["equalize_global_offset"]
equalize_local_offset  = input_dictionary["equalize_local_offset"]

""" Unwrap + Wiggle: Choose (in the ordered frames) a frame to serve as reference for the alignment. Make sure to select a non-null frame!!! """
wiggle_sinogram_selection = input_dictionary["wiggle_sinogram_selection"]
bad_frames_before_wiggle = input_dictionary["bad_frames_before_wiggle"] # set to zero those frames that are still bad after phase unwrapping or convex Hull
reference_frame = input_dictionary["wiggle_reference_frame"] ## MANUAL!! 
n_of_wiggle_processes = input_dictionary["CPUs"]

""" Regularization: https://doi.org/10.1016/j.rinam.2019.100088  """
do_regularization = input_dictionary["tomo_regularization"]
regularization_parameter = input_dictionary["tomo_regularization_param"]   #tirado do cool, como qqr parametro de reg.

""" Tomo Parameters """
iterations = input_dictionary["tomo_iterations"] # number of iterations of tomographic algorithms
which_reconstruction = input_dictionary["tomo_algorithm"] #"EEM" # "ART", "EM", "EEM", "FBP", "RegBackprojection"
GPUs = input_dictionary["GPUs"] #[0,1] # GPUs to use. GPUs = -1, use default of [0]
threshold_object = input_dictionary["tomo_threshold"]

"""             INPUTS -> SET OUTPUT FILES AND FOLDERS                """
complex_object_file  = input_dictionary["complex_object_filepath"] #os.path.join(sinogram_folder, 'object_' + foldernames[0] + '.npy')

""" Select name of ordered phase unwrapped files """
angles_filepath  = input_dictionary["ordered_angles_filepath"]  #foldernames[0] + '_ordered_angles.npy'
object_filepath  = input_dictionary["ordered_object_filepath"] #foldernames[0] + '_ordered_object.npy'

""" Select output tomogram filenames """
object_tomogram_filepath = input_dictionary["wiggle_sinogram_filepath"] #contrast_type + '_' + foldernames[0] + '_wiggle.npy'

""" Select filenames of reconstructed object """
recon_object_filepath = input_dictionary['reconstruction_filepath'] #contrast_type + '_' + foldernames[0] + f'_reconstruction3D_' + which_reconstruction + '.npy'
recon_object_filepath_thresholded = input_dictionary['reconstruction_equalized_filepath'] #contrast_type + '_' + foldernames[0] + f'_reconstruction3D_' + which_reconstruction + '_thresholded.npy'


if processing_steps["Sort"]:
    """ ########################## ORDENATION ############################## """
    print('Sort datasets by angle ')

    if recon_method == 'ptycho':
        file = h5py.File(sinogram_path, 'r')
        object = file['recon/object']
        angles = file['recon/angles']

    elif recon_method == "pwcdi":
        object = np.load(sinogram_path)
        angles = np.load(raw_angles_path)

    rois =  sort_frames_by_angle(angles)
    np.save(angles_filepath,rois)
    print('\tSorting done')

    object = reorder_slices_low_to_high_angle(object, rois)
    np.save(object_filepath, object) 

if processing_steps["Crop"]: 
    """ ######################## Crop: STILL MANUAL ################################ """

    object = np.load(input_dictionary["complex_object_filepath"]) 

    print(" \tBegin Crop")

    print("\tCropping data")
    object = object[:,top:-bottom,left:-right] # Crop frame

    print('\tShape after cropping:',object.shape)

    np.save(input_dictionary["cropped_sinogram_filepath"],object) # save shaken and padded sorted sinogram

if processing_steps["Unwrap"]:
    """ ######################## Remove Bad Frames and Phase Unwrap: STILL MANUAL ################################ """
    object = np.load(input_dictionary["cropped_sinogram_filepath"])  

    object = np.zeros(object.shape)

    print("Starting Phase Unwrap") 
    # for i in range(object.shape[0]): # is this really needed?
        # object[i,:,:] = -np.angle(RemovePhaseGrad(object[i,:,:]))

    object = unwrap_in_parallel(object,iterations=phase_unwrap_iterations,non_negativity=phase_unwrap_non_negativity,remove_gradient = phase_unwrap_gradient_removal)

    np.save(input_dictionary["unwrapped_sinogram_filepath"],object)  

    print("\tPhase Unwrap done!")

if processing_steps["Equalize Frames"]:

    equalize_invert        = input_dictionary["equalize_invert"]
    equalize_gradient      = input_dictionary["equalize_gradient"]
    equalize_outliers      = input_dictionary["equalize_outliers"]
    equalize_global_offset = input_dictionary["equalize_global_offset"]
    equalize_local_offset  = input_dictionary["equalize_local_offset"]

    print('Loading unwrapped sinogram: ',input_dictionary["unwrapped_sinogram_filepath"] )
    unwrapped_sinogram = np.load(input_dictionary["unwrapped_sinogram_filepath"] )
    equalized_sinogram = equalize_frames_parallel(unwrapped_sinogram,equalize_invert,equalize_gradient,equalize_outliers,equalize_global_offset, ast.literal_eval(equalize_local_offset))
    np.save(input_dictionary["equalized_sinogram_filepath"] ,equalized_sinogram)

if processing_steps["Wiggle"]:
    """ ######################## ZEROING EXTRA FRAMES: MANUAL ################################ """

    if wiggle_sinogram_selection == "unwrapped":
        object = np.load(input_dictionary["unwrapped_sinogram_filepath"]) 
    elif wiggle_sinogram_selection == "convexHull":
        object = np.load(input_dictionary["chull_sinogram_filepath"])
    elif wiggle_sinogram_selection == "cropped":
        object = np.load(input_dictionary["cropped_sinogram_filepath"])
    elif wiggle_sinogram_selection == "equalized":
        object = np.load(input_dictionary["equalized_sinogram_filepath"])
     
    for k in bad_frames_before_wiggle:
        object[k,:,:] = 0

    print('Zeroing extra frames completed!')

    """ ######################## Project Angles and Padding to 180 degrees ################################ """

    angles  = np.load(angles_filepath)
    # angles = (np.pi/180.) * angles

    object, idxP, firstP, projected_angles = angle_mesh_organize(object, angles)
    np.save(input_dictionary["projected_angles_filepath"],projected_angles)

    """ ######################## Wiggle ################################ """
    print('\tStarting Wiggle')
    start = time.time()

    temp_tomogram, shiftv = radon.get_wiggle( object, "vertical", input_dictionary["CPUs"], input_dictionary["wiggle_reference_frame"] )
    temp_tomogram, shiftv = radon.get_wiggle( temp_tomogram, "vertical", input_dictionary["CPUs"], input_dictionary["wiggle_reference_frame"] )
    print('Finished vertical wiggle. Starting horizontal wiggle...')
    tomoP, shifth, wiggle_cmas_temp = radon.get_wiggle( temp_tomogram, "horizontal", input_dictionary["CPUs"], input_dictionary["wiggle_reference_frame"] )
    wiggle_cmas = [[],[]]
    wiggle_cmas[1], wiggle_cmas[0] =  wiggle_cmas_temp[:,1].tolist(), wiggle_cmas_temp[:,0].tolist()
    input_dictionary["wiggle_ctr_of_mas"] = wiggle_cmas


    np.save(object_tomogram_filepath,tomoP)

    elapsed = time.time() - start
    print('Elapsed time for Wiggle (sec):', elapsed )

    print('\tWiggle Complete')

if processing_steps["Tomo"]:
    start = time.time()

    print(f'Starting tomography...')
    reconstruction3D = tomography(input_dictionary,use_regularly_spaced_angles=True)
    elapsed = time.time() - start
    print(f'Reconstruction done!')
    print('Elapsed time for reconstruction (sec):', elapsed )

    print('Saving 3D recon...')
    np.save(input_dictionary["reconstruction_filepath"],reconstruction3D)
    imsave(input_dictionary["reconstruction_filepath"][:-4] + '.tif',reconstruction3D)
    print('\t Saved!')

if processing_steps["Equalize Recon"]:

    tomo_threshold      = input_dictionary["tomo_threshold"]  # max value to be left in reconstructed matrix
    remove_outliers     = input_dictionary["tomo_remove_outliers"]
    remove_local_offset = input_dictionary["tomo_local_offset"]

    print('Equalizing 3D recon...')
    reconstruction = np.load(input_dictionary["reconstruction_filepath"])
    equalized_tomogram = equalize_tomogram(reconstruction,np.mean(reconstruction),np.std(reconstruction),remove_outliers=remove_outliers,threshold=float(tomo_threshold),bkg_window=remove_local_offset)
    print('\t Done!')
    print('Saving equalized 3D recon...')
    np.save(input_dictionary["reconstruction_equalized_filepath"],equalized_tomogram)
    imsave(input_dictionary["reconstruction_equalized_filepath"][:-4] + '.tif',equalized_tomogram)
    print('\t Saved!')