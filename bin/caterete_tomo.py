import numpy as np
import matplotlib.pyplot as plt
import os,sys, json, time
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
# "ConvexHull":1,
# "Wiggle":1,
# "Tomo":1
# }

processing_steps = input_dictionary["processing_steps"]
contrast_type = input_dictionary["contrast_type"] # Phase Or Magnitude?
prefix_string = f'_{contrast_type}'

""" Select data folders  """
sinogram_folder = input_dictionary["sinogram_path"].rsplit('/',1)[0]  #'/ibira/lnls/labs/tepui/proposals/20210062/yuri/Caterete/yuri-ssc-cdi/outputs/microagg_P2_01/reconstruction/' # folder containing sinogram of 2d projections
ibira_path = input_dictionary["ibira_data_path"] #'/ibira/lnls/beamlines/caterete/proposals/20210177/data/ptycho3d/' # folder of 2d projections
foldernames = input_dictionary["folders_list"] # ["microagg_P2_01"] #input

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

""" Output plot folders """
originals_filepath  = [True ,os.path.join(sinogram_folder, '00_frames_original')]
ordered_filepath    = [True,os.path.join(sinogram_folder, '01_frames_ordered')]
cropped_filepath    = [True ,os.path.join(sinogram_folder, '02_frames_cropped')]
unwrapped_filepath  = [True ,os.path.join(sinogram_folder, '03_frames_unwrapped')]
equalized_filepath  = [True ,os.path.join(sinogram_folder, '04_frames_equalized')]
cHull_filepath      = [True ,os.path.join(sinogram_folder, '05_frames_convexHull')]
create_directory_if_doesnt_exist(originals_filepath[1],ordered_filepath[1],cropped_filepath[1],unwrapped_filepath[1],cHull_filepath[1],equalized_filepath[1])


if processing_steps["Sort"]:
    """ ########################## ORDENATION ############################## """
    print('Sort datasets by angle ')

    object = np.load(complex_object_file)

    if originals_filepath[0]: # Save pngs of frames
        for i in range(object.shape[0]):
            plt.figure()
            plt.imshow(np.angle(object[i,:,:]),cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(originals_filepath[1],'original_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    rois =  sort_frames_by_angle(ibira_path,foldernames)

    np.save(angles_filepath,rois)
    print('\tSorting done')

    object = reorder_slices_low_to_high_angle(object, rois)
    np.save(object_filepath, object) 

if processing_steps["Crop"]: 
    """ ######################## Crop: STILL MANUAL ################################ """

    object = np.load(input_dictionary["complex_object_filepath"]) 

    print(" \tBegin Crop")

    if ordered_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.figure()
            plt.imshow(np.angle(object[i,:,:]),cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(ordered_filepath[1], 'ordered_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    print("\tCropping data")
    object = object[:,top:-bottom,left:-right] # Crop frame

    print('Shape after cropping:',object.shape)

    if 1: # Save image preview
        slice_number=0
        figure, subplot = plt.subplots()
        subplot.imshow(np.angle(object[slice_number,:,:]),cmap='gray')#,interpolation='bilinear')
        subplot.set_title('object preview')
        figure.savefig(os.path.join(sinogram_folder,'object_preview.png'))

    print("\tCrop complete!")

    if cropped_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.figure()
            plt.imshow(np.angle(object[i,:,:]),cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(sinogram_folder, cropped_filepath[1], 'cropped_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    np.save(input_dictionary["cropped_sinogram_filepath"],object) # save shaken and padded sorted sinogram

if processing_steps["Unwrap"]:
    """ ######################## Remove Bad Frames and Phase Unwrap: STILL MANUAL ################################ """
    object = np.load(input_dictionary["cropped_sinogram_filepath"])  

    object = np.zeros(object.shape)

    print("Starting Phase Unwrap") 
    # for i in range(object.shape[0]): # is this really needed?
        # object[i,:,:] = -np.angle(RemovePhaseGrad(object[i,:,:]))

    object = unwrap_in_parallel(object,iterations=phase_unwrap_iterations,non_negativity=phase_unwrap_non_negativity,remove_gradient = phase_unwrap_gradient_removal)

    if 1: # Save image preview
        slice_number=0
        figure, subplot = plt.subplots()
        subplot.imshow(object[slice_number,:,:],cmap='gray',interpolation='bilinear')
        subplot.set_title('Phase preview')
        figure.savefig(os.path.join(sinogram_folder,'phaseUnwrap_preview.png'))

    np.save(input_dictionary["unwrapped_sinogram_filepath"],object)  

    if unwrapped_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.figure()
            plt.imshow(object[i,:,:],cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(unwrapped_filepath[1], 'unwrapped_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

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

    if 1: # Plot shake and unshaked object sinograms
        slice = object.shape[1] // 2
        plt.figure()
        plt.imshow(object[:,slice,:])
        plt.colorbar()
        plt.title('No Wiggle')
        plt.savefig(sinogram_folder + 'object_nowiggle.png', format='png', dpi=300)
        plt.clf()
        plt.close()

        plt.figure(0)
        plt.imshow(tomoP[:,slice,:])
        plt.colorbar()
        plt.title('Wiggle')
        plt.savefig(sinogram_folder + 'object_wiggle.png', format='png', dpi=300)
        plt.clf()
        plt.close()

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

    if 1: # Visualize recon slice
        slice = reconstruction3D.shape[0] // 2
        plt.figure(0)
        plt.imshow(reconstruction3D[slice,:,:])
        plt.colorbar()
        plt.savefig(sinogram_folder+f'reconstruction3D_slice{slice}.png', format='png', dpi=300)
        plt.clf()
        plt.close()

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