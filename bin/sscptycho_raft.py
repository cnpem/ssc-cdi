import json
import matplotlib.pyplot as plt
import numpy as np
import os,sys
from time import time
from skimage.io import imsave

from ssc_remote_vis import remote_visualization as rv
from sscRadon import radon

from sscPtycho import RemovePhaseGrad

import sscCdi
from sscCdi.caterete.tomo_processing import angle_mesh_organize, tomography, apply_chull_parallel
from sscCdi.caterete.misc import create_directory_if_doesnt_exist
from sscCdi.caterete.unwrap import unwrap_in_parallel

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


""" Convex Hull """
bad_frames_before_cHull = input_dictionary["bad_frames_before_cHull"] 
chull_invert    = input_dictionary["chull_invert"] 
chull_threshold = input_dictionary["chull_tolerance"] 
chull_opening   = input_dictionary["chull_opening"] 
chull_erosion   = input_dictionary["chull_erosion"] 
convex_hull     = input_dictionary["chull_param"] 

""" Unwrap + Wiggle: Choose (in the ordered frames) a frame to serve as reference for the alignment. Make sure to select a non-null frame!!! """
bad_frames_before_wiggle = input_dictionary["bad_frames_before_wiggle"] # set to zero those frames that are still bad after phase unwrapping or convex Hull
reference_frame = input_dictionary["wiggle_reference_frame"] ## MANUAL!! 
n_of_wiggle_processes = input_dictionary["wiggle_cpus"]

""" Regularization: https://doi.org/10.1016/j.rinam.2019.100088  """
do_regularization = input_dictionary["tomo_regularization"]
regularization_parameter = input_dictionary["tomo_regularization_param"]   #tirado do cool, como qqr parametro de reg.

""" Tomo Parameters """
iterations = input_dictionary["tomo_iterations"] # number of iterations of tomographic algorithms
which_reconstruction = input_dictionary["tomo_algorithm"] #"EEM" # "ART", "EM", "EEM", "FBP", "RegBackprojection"
GPUs = input_dictionary["tomo_n_of_gpus"] #[0,1] # GPUs to use. GPUs = -1, use default of [0]

threshold_object = input_dictionary["tomo_threshold"]

"""             INPUTS -> SET OUTPUT FILES AND FOLDERS                """
complex_object_file  = os.path.join(sinogram_folder, 'object_' + foldernames[0] + '.npy')
output_filesname = foldernames[0]

""" Select name of ordered phase unwrapped files """
angles_filename = output_filesname + '_ordered_angles.npy'
object_filename  = output_filesname + '_ordered_object.npy'

""" Select output tomogram filenames """
object_tomogram_filename = contrast_type + '_' + output_filesname + '_wiggle.npy'

""" Select filenames of reconstructed object """
recon_object_filename = contrast_type + '_' + output_filesname + f'_reconstruction3D_' + which_reconstruction + '.npy'
recon_object_filename_thresholded = contrast_type + '_' + output_filesname + f'_reconstruction3D_' + which_reconstruction + '_thresholded.npy'

""" Output plot folders """
originals_filepath  = [False ,os.path.join(sinogram_folder, '00_frames_original')]
ordered_filepath    = [False,os.path.join(sinogram_folder, '01_frames_ordered')]
cropped_filepath    = [False ,os.path.join(sinogram_folder, '02_frames_cropped')]
unwrapped_filepath  = [False ,os.path.join(sinogram_folder, '03_frames_unwrapped')]
cHull_filepath      = [True ,os.path.join(sinogram_folder, '04_frames_convexHull')]
create_directory_if_doesnt_exist(originals_filepath[1],ordered_filepath[1],cropped_filepath[1],unwrapped_filepath[1],cHull_filepath[1])


if processing_steps["Sort"]:
    """ ########################## ORDENATION ############################## """
    print('Sort datasets by angle ')

    object = np.load(complex_object_file)

    if originals_filepath[0]: # Save pngs of frames
        for i in range(object.shape[0]):
            plt.imshow(np.angle(object[i,:,:]),cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(originals_filepath[1],'original_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    rois =  sscCdi.jupyterTomography.sort_frames_by_angle(ibira_path,foldernames)

    np.save(sinogram_folder + angles_filename,rois)
    print('\tSorting done')

    object = sscCdi.jupyterTomography.reorder_slices_low_to_high_angle(object, rois)
    np.save(sinogram_folder + object_filename, object) 

if processing_steps["Crop"]: 
    """ ######################## Crop: STILL MANUAL ################################ """

    object = np.load(sinogram_folder + object_filename) 

    print(" \tBegin Crop")

    if ordered_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
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
        figure.savefig(sinogram_folder+'object_preview.png')

    print("\tCrop complete!")

    if cropped_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.imshow(np.angle(object[i,:,:]),cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(sinogram_folder, cropped_filepath[1], 'cropped_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    np.save(sinogram_folder + 'crop_' + object_filename,object) # save shaken and padded sorted sinogram

if processing_steps["Unwrap"]:
    """ ######################## Remove Bad Frames and Phase Unwrap: STILL MANUAL ################################ """
    object = np.load(sinogram_folder + 'crop_' + object_filename)  

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
        figure.savefig(sinogram_folder+'phaseUnwrap_preview.png')

    np.save(sinogram_folder + 'unwrap_' + object_filename,object)  

    if unwrapped_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.imshow(object[i,:,:],cmap='gray')
            plt.colorbar()
            plt.savefig( os.path.join(unwrapped_filepath[1], 'unwrapped_frame_' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

    print("\tPhase Unwrap done!")

if processing_steps["ConvexHull"]:
    """ ######################## CONVEX HULL: MANUAL ################################ """

    object = np.load(sinogram_folder + 'unwrap_' + object_filename) 

    for k in bad_frames_before_cHull:
        object[k,:,:] = 0

    output_list = apply_chull_parallel(object,invert=chull_invert,tolerance=chull_threshold,opening_param=chull_opening,erosion_param=chull_erosion,chull_param=convex_hull)
    object = output_list[-1]

    np.save(sinogram_folder + 'unwrap_' + object_filename, object)

    if cHull_filepath[0]: # Save pngs of sorted frames
        for i in range(object.shape[0]):
            plt.imshow(object[i,:,:],cmap='gray')
            plt.colorbar()
            plt.savefig(  os.path.join(cHull_filepath[1], 'cHull_frame' + str(i) + '.png'), format='png', dpi=300)
            plt.clf()
            plt.close()

if processing_steps["Wiggle"]:
    """ ######################## ZEROING EXTRA FRAMES: MANUAL ################################ """

    object = np.load(sinogram_folder + 'unwrap_' + object_filename) # save shaken and padded sorted sinogram

    for k in bad_frames_before_wiggle:
        object[k,:,:] = 0

    print('Zeroing extra frames completed!')

    """ ######################## Project Angles and Padding to 180 degrees ################################ """

    angles  = np.load(sinogram_folder + angles_filename)
    angles = (np.pi/180.) * angles

    object, idxP, firstP, projected_angles = angle_mesh_organize(object, angles)

    projected_angles_filename = angles_filename[:-4]+'_projected.npy'
    np.save(sinogram_folder + projected_angles_filename,projected_angles)

    """ ######################## Wiggle ################################ """
    print('\tStarting Wiggle')
    start = time()

    updateTomoP_0 = radon.get_wiggle( object, 'vertical', n_of_wiggle_processes, reference_frame )
    tomoP = radon.get_wiggle( updateTomoP_0, 'horizontal', n_of_wiggle_processes, reference_frame)
    np.save(sinogram_folder + object_tomogram_filename,tomoP)

    elapsed = time() - start
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
    start = time()
    print(f'Starting tomography...')
    reconstruction3D = tomography(which_reconstruction,contrast_type,angles_filename,iterations,GPUs,do_regularization,regularization_parameter,output_folder,use_regularly_spaced_angles=True)
    elapsed = time() - start
    print(f'Reconstruction done!')
    print('Elapsed time for reconstruction (sec):', elapsed )

    print('Saving 3D recon...')
    np.save(os.path.join(sinogram_folder,recon_object_filename),reconstruction3D)
    imsave(os.path.join(sinogram_folder,recon_object_filename)[:-4] + '.tif',reconstruction3D)
    print('\t Saved!')

    if threshold_object != 0:
        print('Saving thresholded 3D recon...')
        thresholded_recon = np.where(reconstruction3D > threshold_object,0,reconstruction3D)
        np.save(os.path.join(sinogram_folder,recon_object_filename_thresholded),reconstruction3D)
        imsave(os.path.join(sinogram_folder,recon_object_filename_thresholded)[:-4] + '.tif',thresholded_recon)
        print('\t Saved!')


    if 1: # Visualize recon slice
        slice = reconstruction3D.shape[0] // 2
        plt.figure(0)
        plt.imshow(reconstruction3D[slice,:,:])
        plt.colorbar()
        plt.savefig(sinogram_folder+f'reconstruction3D_slice{slice}.png', format='png', dpi=300)
        plt.clf()
        plt.close()

