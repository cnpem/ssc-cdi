import numpy as np
from sscRadon import radon
from ssc_remote_vis import remote_visualization as rv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sscRaft import parallel
from sscOldRaft import *
import time
import sscCdi
from sscCdi.jupyterTomography import sort_frames_by_angle, regularization
from sscPtycho import Show, RemovePhaseGrad
import os
from skimage.restoration import unwrap_phase
import sys

def create_directory_if_doesnt_exist(*args):
    for arg in args:
        if os.path.isdir(arg) == False:
            os.mkdir(arg)

def tomography(algorithm,data,anglesFile,iterations,GPUs):

    angles = np.load(angles_filename) # sorted angles?

    if algorithm == "TEM" or algorithm == "EM":
        data = np.exp(-data)
    elif algorithm == "ART":
        flat = np.ones([1,data.shape[-2],data.shape[-2]],dtype=np.uint16)
        dark = np.zeros(flat.shape[1:],dtype=np.uint16)
        centersino1 = Centersino(frame0=data[0,:,:], frame1=data[-1,:,:], flat=flat[0], dark=dark, device=0) 

    if algorithm != "EEM": # for these
        rays, slices = data.shape[-1], data.shape[-2]
        reconstruction3D = np.zeros((rays,slices,rays))
        for i in range(slices):
            print(f'Reconstructing slice {i}')
            sinogram = data[:,i,:]
            if algorithm == "ART":
                reconstruction3D[:,i,:]= MaskedART( sino=sinogram,mask=flat,niter=iterations ,device=GPUs)
            elif algorithm == "FBP": 
                reconstruction3D[:,i,:]= FBP( sino=sinogram,angs=angles,device=GPUs,csino=centersino1)
            elif algorithm == "RegBackprojection":
                reconstruction3D[:,i,:]= Backprojection( sino=sinogram,device=GPUs)
            elif algorithm == "EM":
                reconstruction3D[:,i,:]= EM(sinogram, flat, iter=iterations, pad=2, device=GPUs, csino=0)
            elif algorithm == "SIRT":
                reconstruction3D[:,i,:]= SIRT_FST(sinogram, iter=iterations, zpad=2, step=1.0, csino=0, device=GPUs, art_alpha=0.2, reg_mu=0.2, param_alpha=0, supp_reg=0.2, img=None)
    elif algorithm == "EEM": #data é o que sai do wiggle! 
        data = np.swapaxes(data, 0, 1) #tem que trocar eixos 0,1 - por isso o swap.
        nangles = data.shape[1]
        recsize = data.shape[2]
        iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
        dic = {'gpu': GPUs, 'blocksize':20, 'nangles': nangles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
        reconstruction3D = parallel.emfs( data, dic )
    else:
        import sys
        sys.exit('Select a proper reconstruction method')

    return reconstruction3D
            

input_dictionary = json.load(open(sys.argv[1])) # LOAD JSON!


if = input_dictionary["run_all_tomo_steps"] == False:
    
    algorithm     = input_dictionary["tomo_algorithm"]
    anglesFile    = input_dictionary["run_all_tomo_steps"]
    iterations    = input_dictionary["tomo_iterations"]
    GPUs          = input_dictionary["tomo_n_of_gpus"]
    output_folder = '/ibira/lnls/beamlines/caterete/apps/jupyter-dev/'#input_dictionary["run_all_tomo_steps"]
    filename      = 'reconstruction3D.npy'#input_dictionary["run_all_tomo_steps"]
    
    print(f'Starting {algo_dropdown.value} tomography...')
    recon3D = np.ones((2,2,2))
    # recon3D = jupyterTomography.tomography(algorithm,data,anglesFile,iterations,GPUs)
    print('\t Finished! \n \t Saving 3D data...')
    np.save(os.path.join(output_folder, {filename}),recon3D)
    print('/t Saved!')
else:


    """                  INPUTS                 """

    processing_steps = { # 1 -> True ; 0 -> False
    "Sort":1 ,
    "Crop":1 ,
    "Unwrap":1,
    "ConvexHull":1,
    "Wiggle":1,
    "Tomo":1
    }

    """ Select data folders  """
    sinogram_folder = '/ibira/lnls/labs/tepui/proposals/20210062/yuri/Caterete/yuri-ssc-cdi/outputs/microagg_P2_01/reconstruction/' # folder containing sinogram of 2d projections
    ibira_path = '/ibira/lnls/beamlines/caterete/proposals/20210177/data/ptycho3d/' # folder of 2d projections
    foldernames = ["microagg_P2_01"] #input

    """ Crop: Select the cropping slices # SLICE MUST HAVE EVEN NUMBER OF POINTS!!! """
    top, bottom = 300, 300 # number of pixels to crops in each direction
    left, right = 300, 300

    """ Phase Unwrapping: remove bad frames and unwrap remaining ones """
    bad_frames = [7,20,36,65,94,123,152,181,210,239,268,296,324]
    phase_unwrap_iterations = 0
    phase_unwrap_non_negativity = False
    phase_unwrap_gradient_removal = False

    """ Zero Frames: Manually set to zero those frames that are still bad after phase unwrapping"""
    frames_to_zero = []

    """ Unwrap + Wiggle: Choose (in the ordered frames) a frame to serve as reference for the alignment. Make sure to select a non-null frame!!! """
    reference_frame = 222 ## MANUAL!! 
    n_of_wiggle_processes = 64

    """ Regularization: https://doi.org/10.1016/j.rinam.2019.100088  """
    do_regularization = True
    regularization_parameter = 0.001   #tirado do cool, como qqr parametro de reg.

    """ Tomo Parameters """
    iterations = 100 # number of iterations of tomographic algorithms
    which_reconstruction = "EEM" # "ART", "EM", "EEM", "FBP", "RegBackprojection"
    GPUs = [0,1] # GPUs to use. GPUs = -1, use default of [0]

    threshold_absol = 0 # if != 0, will apply threshold value to reconstructed object, turning values > threshold to zero 
    threshold_phase = 0 

    """             INPUTS -> SET OUTPUT FILES AND FOLDERS                """
    complex_object_file  = sinogram_folder + 'object_' + foldernames[0] + '.npy'
    output_filesname = foldernames[0]

    """ Select name of ordered phase unwrapped files """
    angles_filename = output_filesname + '_ordered_angles.npy'
    phase_filename  = output_filesname + '_ordered_phase.npy'
    absol_filename  = output_filesname + '_ordered_amplitude.npy'
    object_filename = output_filesname + '_ordered_object.npy'

    """ Select output tomogram filenames """
    phase_tomogram_filename     = output_filesname + '_phase_wiggle.npy'
    amplitude_tomogram_filename = output_filesname + '_amplitude_wiggle.npy'

    """ Select filenames of reconstructed object """
    recon_phase_filename = output_filesname + f'_recon_' + which_reconstruction + '_phase.npy'
    recon_absol_filename = output_filesname + f'_recon_' + which_reconstruction + '_absol.npy'

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

        rois =  jupyterTomography.sort_frames_by_angle(ibira_path,foldernames)

        np.save(sinogram_folder + angles_filename,rois)
        print('\tSorting done')

        reorder_slices_low_to_high_angle(object, rois)
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
            subplot.set_title('Phase preview')
            figure.savefig(sinogram_folder+'phase_preview.png')

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

        phase = np.zeros(object.shape)
        absol = np.zeros(object.shape)

        print("Starting Phase Unwrap") 
        for i in range(object.shape[0]):
            if i in bad_frames:
                print('Ignore frame' + str(i))
            else:
                print('\tPerforming phase unwrap of slice ', i)
                phase[i,:,:] = -np.angle(RemovePhaseGrad(object[i,:,:]))
                absol[i,:,:] = -np.abs(RemovePhaseGrad(object[i,:,:]))
                phase[i,:,:] = sscCdi.unwrap.phase_unwrap(phase[i,:,:],phase_unwrap_iterations,non_negativity=phase_unwrap_non_negativity,remove_gradient = phase_unwrap_gradient_removal)
                absol[i,:,:] = sscCdi.unwrap.phase_unwrap(absol[i,:,:],phase_unwrap_iterations,non_negativity=phase_unwrap_non_negativity,remove_gradient = phase_unwrap_gradient_removal)

        if 1: # Save image preview
            slice_number=0
            figure, subplot = plt.subplots()
            subplot.imshow(phase[slice_number,:,:],cmap='gray',interpolation='bilinear')
            subplot.set_title('Phase preview')
            figure.savefig(sinogram_folder+'phaseUnwrap_preview.png')

        np.save(sinogram_folder + 'unwrap_' + phase_filename,phase)  
        np.save(sinogram_folder + 'unwrap_' + absol_filename,absol)

        if unwrapped_filepath[0]: # Save pngs of sorted frames
            for i in range(phase.shape[0]):
                plt.imshow(phase[i,:,:],cmap='gray')
                plt.colorbar()
                plt.savefig( os.path.join(unwrapped_filepath[1], 'unwrapped_frame_' + str(i) + '.png'), format='png', dpi=300)
                plt.clf()
                plt.close()

        print("\tPhase Unwrap done!")

    if processing_steps["ConvexHull"]:
        """ ######################## CONVEX HULL: MANUAL ################################ """

        phase = np.load(sinogram_folder + 'unwrap_' + phase_filename) # save shaken and padded sorted sinogram
        absol = np.load(sinogram_folder + 'unwrap_' + absol_filename)

        def _operator_T(u):
            d   = 1.0
            uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
            uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
            uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
            uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
            delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
            z = np.sqrt( delta )
            return z

        from skimage.morphology import square, erosion, opening, convex_hull_image, dilation

        for f in range(phase.shape[0]):
            img = phase[f,:,:]
            img2 = absol[f,:,:]
            where = _operator_T(img).real
            new = np.copy(img)
            new[ new > 0] = _operator_T(new).real[ img > 0]
            tol = 1e-5
            mask = (np.abs( new - img) < tol) * 1.0
            mask = opening(mask, square(10))
            mask = erosion(mask, square(30))
            chull = dilation( convex_hull_image(mask), square(40) ) # EXPAND CASCA DA MASCARA
            img = img * chull  #nova imagem apenas com o suporte
            img2 = img2 * chull
            phase[f,:,:] = img
            absol[f,:,:] = img2

        np.save(sinogram_folder + 'unwrap_' + phase_filename, phase)
        np.save(sinogram_folder + 'unwrap_' + absol_filename, absol)

        if cHull_filepath[0]: # Save pngs of sorted frames
            for i in range(phase.shape[0]):
                plt.imshow(phase[i,:,:],cmap='gray')
                plt.colorbar()
                plt.savefig(  os.path.join(cHull_filepath[1], 'cHull_frame' + str(i) + '.png'), format='png', dpi=300)
                plt.clf()
                plt.close()

    if processing_steps["Wiggle"]:
        """ ######################## ZEROING EXTRA FRAMES: MANUAL ################################ """

        phase = np.load(sinogram_folder + 'unwrap_' + phase_filename) # save shaken and padded sorted sinogram
        absol = np.load(sinogram_folder + 'unwrap_' + absol_filename)

        for k in frames_to_zero: # AFTER PHASE UNWRAP!!!
            phase[k,:,:] = 0
            absol[k,:,:] = 0

        print('Zeroing extra frames completed!')

        """ ######################## Wiggle ################################ """
        print('\tStarting Wiggle')
        start = time.time()

        updateTomoP_0 = radon.get_wiggle( phase, 'vertical', n_of_wiggle_processes, reference_frame )
        tomoP = radon.get_wiggle( updateTomoP_0, 'horizontal', n_of_wiggle_processes, reference_frame)
        np.save(sinogram_folder + phase_tomogram_filename,tomoP)

        updateTomoA_0 = radon.get_wiggle( absol, 'vertical', n_of_wiggle_processes, reference_frame )
        tomoA = radon.get_wiggle( updateTomoA_0, 'horizontal', n_of_wiggle_processes, reference_frame )
        np.save(sinogram_folder + amplitude_tomogram_filename, tomoA)

        elapsed = time.time() - start
        print('Elapsed time for Wiggle (sec):', elapsed )

        print('\tWiggle Complete')

        if 1: # Plot shake and unshaked phase sinograms
            slice = phase.shape[1] // 2
            plt.figure()
            plt.imshow(phase[:,slice,:])
            plt.colorbar()
            plt.title('No Wiggle')
            plt.savefig(sinogram_folder + 'phase_nowiggle.png', format='png', dpi=300)
            plt.clf()
            plt.close()

            plt.figure(0)
            plt.imshow(tomoP[:,slice,:])
            plt.colorbar()
            plt.title('Wiggle')
            plt.savefig(sinogram_folder + 'phase_wiggle.png', format='png', dpi=300)
            plt.clf()
            plt.close()

    if processing_steps["Tomo"]:
        """ ######################## Regularization ################################ """
        phase = np.load(sinogram_folder + phase_tomogram_filename)
        absol = np.load(sinogram_folder + amplitude_tomogram_filename)

        # Padded zeros for completion of missing wedge:  from (-70,70) - 140 degrees, to (-90,90) - 180 degrees
        angles  = np.load(sinogram_folder + angles_filename)
        angles = angles[:,1] # get the angles
        anglesmax, anglesmin = angles[-1],  angles[0]     # max and min angles
        angles = np.insert(angles, 0, -90)     # Insert the first angle as -90. Why I do that? Beacause I assume that the first angles is always zero, in order to correctly find the angle step size inside the EM algorithm fro all angles.
        phase = np.pad(phase,((1,0),(0,0),(0,0)),'constant') # Pad zeros corresponding to the extra -90 value
        absol = np.pad(absol,((1,0),(0,0),(0,0)),'constant')
        angles = (angles + 90) # Transform the angles from (-90,90) to (0,180)

        if do_regularization: # If which_reconstruction == "EEM" MIQUELES
            print('\tBegin Regularization')
            #regularized data
            for k in range(phase.shape[1]):
                phase[:,k,:] = jupyterTomography.regularization( phase[:,k,:], regularization_parameter)
                absol[:,k,:] = jupyterTomography.regularization( absol[:,k,:], regularization_parameter)

            print('\tRegularization Done')

        """ ######################## RECON ################################ """
        print('\tBegin Reconstruction with', which_reconstruction)
        start = time.time()

        if which_reconstruction == "TEM" or which_reconstruction == "EM":
            phase = np.exp(-phase)
            absol = np.exp(-absol)
        elif which_reconstruction == "ART":
            flat = np.ones([1,phase.shape[-2],phase.shape[-2]],dtype=np.uint16)
            dark = np.zeros(flat.shape[1:],dtype=np.uint16)
            angles = np.load(angles_filename)
            centersino1 = Centersino(frame0=phase[0,:,:], frame1=phase[-1,:,:], flat=flat[0], dark=dark, device=0) 
            centersino2 = Centersino(frame0=absol[0,:,:], frame1=absol[-1,:,:], flat=flat[0], dark=dark, device=0)

        if which_reconstruction != "EEM": # for these
            rays, slices = phase.shape[-1], phase.shape[-2]
            reconP = np.zeros((rays,slices,rays))
            reconA = np.zeros((rays,slices,rays))
            for i in range(slices):
                print(f'Reconstructing slice {i}')
                sinP = phase[:,i,:]
                sinA = absol[:,i,:]
                if which_reconstruction == "ART":
                    reconP[:,i,:]= MaskedART( sino=sinP,mask=flat,niter=iterations ,device=GPUs)
                    reconA[:,i,:]= MaskedART( sino=sinA,mask=flat,niter=iterations ,device=GPUs)
                elif which_reconstruction == "FBP": 
                    reconP[:,i,:]= FBP( sino=sinP,angs=angles,device=GPUs,csino=centersino1)
                    reconA[:,i,:]= FBP( sino=sinA,angs=angles,device=GPUs,csino=centersino2)
                elif which_reconstruction == "RegBackprojection":
                    reconP[:,i,:]= Backprojection( sino=sinoP,device=GPUs)
                    reconA[:,i,:]= Backprojection( sino=sinoA,device=GPUs)
                elif which_reconstruction == "EM":
                    reconP[:,i,:]= EM(sinP, flat, iter=iterations, pad=2, device=GPUs, csino=0)
                    reconA[:,i,:]= EM(sinA, flat, iter=iterations, pad=2, device=GPUs, csino=0)
                elif which_reconstruction == "SIRT":
                    reconP[:,i,:]= SIRT_FST(sinP, iter=iterations, zpad=2, step=1.0, csino=0, device=GPUs, art_alpha=0.2, reg_mu=0.2, param_alpha=0, supp_reg=0.2, img=None)
                    reconA[:,i,:]= SIRT_FST(sinA, iter=iterations, zpad=2, step=1.0, csino=0, device=GPUs, art_alpha=0.2, reg_mu=0.2, param_alpha=0, supp_reg=0.2, img=None)
        elif which_reconstruction == "EEM": #data é o que sai do wiggle! 
            sinoP = np.swapaxes(phase, 0, 1) #tem que trocar eixos 0,1 - por isso o swap.
            sinoA = np.swapaxes(absol, 0, 1)
            nangles = sinoP.shape[1]
            recsize = sinoP.shape[2]
            iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
            dic = {'gpu': GPUs, 'blocksize':20, 'nangles': nangles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
            reconP = parallel.emfs( sinoP, dic )
            reconA = parallel.emfs( sinoA, dic )
        else:
            import sys
            sys.exit('Select a proper reconstruction method')
        elapsed = time.time() - start

        print(f'Reconstruction done!')
        print('Elapsed time for reconstruction (sec):', elapsed )

        reconA = reconA.astype(np.float32)
        reconP = reconP.astype(np.float32)

        if threshold_absol != 0:
            reconA[reconA > threshold_absol] = 0
        if threshold_phase != 0:
            reconP[reconP > threshold_phase] = 0

        np.save(sinogram_folder + recon_phase_filename,reconP)
        np.save(sinogram_folder + recon_absol_filename,reconA)

        if 1: # Visualize recon slice
            slice = reconP.shape[0] // 2
            plt.figure(0)
            plt.imshow(reconP[slice,:,:])
            plt.colorbar()
            plt.savefig(sinogram_folder+f'reconP_slice{slice}.png', format='png', dpi=300)
            plt.clf()
            plt.close()