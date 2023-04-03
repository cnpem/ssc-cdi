import os, sys, json, time
import numpy as np
import uuid
import SharedArray as sa
import multiprocessing
import multiprocessing.sharedctypes
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.fft import fftshift, fft2, ifft2

""" Sirius Scientific Computing Imports """
import sscCdi, sscPimega, sscRaft, sscRadon
from sscPimega import pi540D
from sscResolution import frc


""" sscCdi relative imports"""
from ..misc import create_directory_if_doesnt_exist, list_files_in_folder, select_specific_angles, export_json, wavelength_from_energy, create_circular_mask, delete_files_if_not_empty_directory, estimate_memory_usage
from ..ptycho.ptychography import  call_G_ptychography
from ..processing.unwrap import unwrap_in_parallel

##### ##### ##### #####                  PTYCHOGRAPHY                 ##### ##### ##### ##### ##### 

def cat_ptychography(input_dict,restoration_dict_list,restored_data_info_list,strategy="serial"):

    if strategy == "serial":

        angles_file = []
        for folder_number, acquisitions_folder in enumerate(input_dict['acquisition_folders']):  # loop when multiple acquisitions were performed for a 3D recon
    
            filepaths, filenames = list_files_in_folder(os.path.join(input_dict['data_folder'], acquisitions_folder,input_dict['scans_string']), look_for_extension=".hdf5")

            for file_number, (filepath,filename) in enumerate(zip(filepaths,filenames)):

                frame = file_number + folder_number*len(filenames) # attribute singular value to each angle

                restoration_dict = restoration_dict_list[folder_number]
                restored_data_info = restored_data_info_list[folder_number]

                """ Read Diffraction Patterns for one angle """
                if len(filepaths) > 1:
                    DPs = pi540D.ioGetM_Backward540D( restoration_dict, restored_data_info, file_number)
                else:
                    DPs, DP_avg, DP_raw_avg = pi540D.ioGet_Backward540D( restoration_dict, restored_data_info[0],restored_data_info[1])
                
                DPs = DPs.astype(np.float32) # convert from float64 to float32 to save memory

                DPs = DPs[1::] # DEBUG
                print(f"\tFinished reading diffraction data! DPs shape: {DPs.shape}")
                
                if frame == 0: 
                   size_of_single_restored_DP = estimate_memory_usage(DPs)[3]
                   estimated_size_for_all_DPs = len(filepaths)*size_of_single_restored_DP
                   print(f"\tEstimated size for {len(filepaths)} DPs of type {DPs.dtype}: {estimated_size_for_all_DPs:.2f} GBs")
                   
                   print(f"\tSaving mean of DPs...")
                   np.save(input_dict['output_path'] + '/DPs_raw_mean.npy',DP_raw_avg)
                   np.save(input_dict['output_path'] + '/DPs_mean.npy',DP_avg)

                """ Read positions """
                probe_positions, angle = read_probe_positions(input_dict, acquisitions_folder,filename , DPs.shape)
                print(f"\tFinished reading probe positions. Shape: {probe_positions.shape}")

                if file_number == 0 and folder_number == 0: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                    input_dict = set_object_shape(DPs.shape,input_dict, probe_positions, input_dict["object_padding"])
                    sinogram = np.zeros((len(input_dict["projections"]),input_dict["object_shape"][0],input_dict["object_shape"][1])) 
                    probes   = np.zeros((len(input_dict["projections"]),1,DPs.shape[-2],DPs.shape[-1]))
                
                run_ptycho = np.any(probe_positions)  # check if probe_positions == null matrix. If so, won't run current iteration

                """ Call Ptycho """
                if not run_ptycho:
                    print(f'\t\t WARNING: Frame #{(folder_number,file_number)} being nulled because number of positions did not match number of diffraction pattern!')
                    input_dict['ignored_scans'].append((folder_number,file_number))
                    sinogram[frame, :, :]  = np.zeros((input_dict["object_shape"][0],input_dict["object_shape"][1])) # build 3D Sinogram
                    probes[frame, :, :, :] = np.zeros((1,DPs.shape[-2],DPs.shape[-1]))
                    angles_file.append([frame,True,angle,angle*180/np.pi])
                else:
                    sinogram[frame, :, :], probes[frame, :, :] = call_G_ptychography(input_dict,DPs,probe_positions) # run ptycho
                    angles_file.append([frame,False,angle,angle*180/np.pi])

                """ Clean DPs temporary data """
                if len(filepaths) > 1:
                    pi540D.ioCleanM_Backward540D( restoration_dict, restored_data_info )
                else:
                    pi540D.ioClean_Backward540D( restoration_dict, restored_data_info[0] )

        np.savetxt(os.path.join(input_dict["output_path"],"angles.txt"),angles_file,delimiter='\t',header = "frame\tbad\tangle_radians\tangle_degrees")

    return sinogram, np.squeeze(probes), input_dict, probe_positions


##### ##### ##### #####                  DATA PREPARATION                 ##### ##### ##### ##### ##### 

def define_paths(input_dict):
    if 'PreviewGCC' not in input_dict: input_dict['PreviewGCC'] = [False,""] # flag to save previews of interest only to GCC, not to the beamline user
    
    #=========== Set Parameters and Folders =====================
    print('\tProposal path: ',input_dict['data_folder'] )
    print('\tAcquisition folder: ',input_dict["acquisition_folders"][0])
 
    input_dict["00_versions"] = f"sscCdi={sscCdi.__version__},sscPimega={sscPimega.__version__},sscResolution={sscResolution.__version__},sscRaft={sscRaft.__version__},sscRadon={sscRadon.__version__}"

    beamline_outputs_path = os.path.join(input_dict['data_folder'] .rsplit('/',3)[0], 'proc','recons',input_dict["acquisition_folders"][0]) # standard folder chosen by CAT for their outputs
    print("\tOutput path:", beamline_outputs_path)
    input_dict["output_path"]  = os.path.join(beamline_outputs_path,input_dict["custom_output_folder"])
    input_dict["temporary_output"]  = os.path.join(input_dict["output_path"],'temp/')

    create_output_directories(input_dict) # create all output directories of interest
    delete_files_if_not_empty_directory(input_dict["temporary_output"])


    input_dict['scans_string'] = 'scans'
    input_dict['positions_string']  = 'positions'

    input_dict['ignored_scans'] = [('folder_number','file_number')]

    images_folder    = os.path.join(input_dict["acquisition_folders"][0],'images')
    
    mdata_dict = json.load(open(os.path.join(input_dict['data_folder'] ,input_dict["acquisition_folders"][0],'mdata.json')))
    input_dict["energy"]               = mdata_dict['/entry/beamline/experiment']["energy"]
    input_dict["detector_distance"]    = mdata_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
    input_dict["restored_pixel_size"]  = mdata_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
    input_dict["detector_exposure"]    = [None,None]
    input_dict["detector_exposure"][1] = mdata_dict['/entry/beamline/detector']['pimega']["exposure time"]
    input_dict["flatfield"]            = os.path.join(input_dict['data_folder'] ,images_folder,'flat.hdf5')
    input_dict["mask"]                 = os.path.join(input_dict['data_folder'] ,images_folder,'mask.hdf5')
    return input_dict


def create_output_directories(input_dict):
    if input_dict["output_path"] != "": # if no path is given, create directory
        create_directory_if_doesnt_exist(input_dict["output_path"])
        create_directory_if_doesnt_exist(input_dict["temporary_output"])


def get_files_of_interest(input_dict,acquistion_folder=''):

    if acquistion_folder != '':
            filepaths, filenames = list_files_in_folder(os.path.join(input_dict['data_folder'] , acquistion_folder,input_dict['scans_string'] ), look_for_extension=".hdf5")
    else:
        filepaths, filenames = list_files_in_folder(os.path.join(input_dict['data_folder'] , input_dict["acquisition_folders"][0],input_dict['scans_string'] ), look_for_extension=".hdf5")

    if input_dict['projections'] != []:
        filepaths, filenames = select_specific_angles(input_dict['projections'], filepaths, filenames)

    return filepaths, filenames


def set_object_shape(DP_shape,input_dict,probe_positions,offset_bottomright):

    DP_size_y = DP_shape[1]
    DP_size_x = DP_shape[2]

    maximum_probe_coordinate_x = int(np.max(probe_positions[:,1])) 
    object_shape_x  = DP_size_x + maximum_probe_coordinate_x + offset_bottomright

    maximum_probe_coordinate_y = int(np.max(probe_positions[:,0])) 
    object_shape_y  = DP_size_y + maximum_probe_coordinate_y + offset_bottomright

    # input_dict["object_shape"] = (object_shape_y, object_shape_x)
    input_dict["object_shape"] = (object_shape_x,object_shape_x)

    return input_dict


def set_object_pixel_size(input_dict,DP_size):

    wavelength = wavelength_from_energy(input_dict["energy"])
    input_dict["wavelength"] = wavelength
    
    object_pixel_size = wavelength * input_dict['detector_distance'] / (input_dict['restored_pixel_size'] * DP_size * input_dict['binning'])
    input_dict["object_pixel"] = object_pixel_size # in meters

    print(f"\tObject pixel size = {object_pixel_size*1e9:.2f} nm")
    PA_thickness = 4*object_pixel_size**2/(0.61*wavelength)
    print(f"\tLimit thickness for resolution of 1 pixel: {PA_thickness*1e6} microns")
    return input_dict


def convert_probe_positions_meters_to_pixels(offset_topleft, dx, probe_positions):

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / dx  # convert from microns to pixels
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / dx 
    
    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    return probe_positions


def read_probe_positions(input_dict, acquisitions_folder,measurement_file, sinogram_shape):

    def rotate_coordinate_system(angle_rad,px,py):
        px_rotated = np.cos(angle_rad) * px - np.sin(angle_rad) * py
        py_rotated = np.sin(angle_rad) * px + np.cos(angle_rad) * py
        return px_rotated, py_rotated
    
    print('Reading probe positions...')
    probe_positions = []
    positions_file = open( os.path.join(input_dict["data_folder"],acquisitions_folder, input_dict["positions_string"], measurement_file[:-5] + '.txt'))

    n_of_DPs = sinogram_shape[0]
    DP_size  = sinogram_shape[1]
    for line_counter, line in enumerate(positions_file):
        line = str(line)
        if line_counter < 1:
            angle = float(line.split(':')[1].split('\t')[0]) # get rotation angle for that frame
        elif line_counter == 1:
            pass
        else:  # skip first line, which is the header;

            positions_x = float(line.split()[1])
            positions_y = float(line.split()[0])

            positions_x, positions_y = rotate_coordinate_system(input_dict["position_rotation"],positions_x, positions_y) # rotate whole coordinate system (correct misalignment of scan and detector axes)

            #TODO: rolate relative angle between scan x and y positions

            probe_positions.append([positions_y, positions_x])

    probe_positions = np.asarray(probe_positions) # convert list of lists to numpy array

    n_of_positions = probe_positions.shape[0]

    if n_of_positions == n_of_DPs:  # check if number of recorded beam positions in txt matches the number of diff. patterns saved in the hdf5
        pass
    else:
        print("\t\tProblem when reading positions. Number of positions {0} is different from number of diffraction patterns {1}".format(n_of_positions, n_of_DPs))
        print('\t\tSetting object as null array with correct shape... New probe positions shape:', probe_positions.shape)
        probe_positions = np.zeros((n_of_DPs-1, 2))

    input_dict = set_object_pixel_size(input_dict,DP_size) 
    probe_positions = convert_probe_positions_meters_to_pixels(input_dict["object_padding"],input_dict["object_pixel"], probe_positions)

    np.savetxt(os.path.join(input_dict["output_path"],"probe_positions_pxls.txt"),probe_positions) # save positions in pixels

    return probe_positions, angle



##### ##### ##### #####                 PROCESSING               ##### ##### ##### ##### ##### 

def match_cropped_frame_dimension(sinogram,frame):
    """ Match the new incoming frame to the same squared shape of the sinogram. Sinogram should have shape (M,N,N)!

    Args:
        sinogram : sinogram of shape (M,N,N)
        frame : frame of shape (A,B)

    Returns:
        frame : frame of shape (N,N)
    """

    if sinogram.shape != frame.shape:
        print('Frame shape do not match the sinogram. Applying correction')
        print(f'\t Sinogram shape: {sinogram.shape}. Frame shape: {frame.shape}')

    if sinogram.shape[1] < frame.shape[0]:
        frame = frame[0:sinogram.shape[1],:]
    elif sinogram.shape[1] > frame.shape[0]:
        frame = np.concatenate((frame,frame[-1,:]),axis=0) # appended a repeated last line

    if sinogram.shape[2] < frame.shape[1]:
        frame = frame[:,0:sinogram.shape[2]]
    elif sinogram.shape[2] > frame.shape[1]:
        frame = np.concatenate((frame,frame[:,-1]),axis=1) # appended a repeated last line

    if sinogram.shape != frame.shape:
        print(f'\t Corrected drame shape: {frame.shape}')

    return frame


def make_1st_frame_squared(frame):
    """ Crops frame of dimension (A,B) to (A,A) or (B,B), depending if A or B is smaller

    Args:
        frame: 2D frame

    Returns:
        frame: cropped frame with smalelr dimension
    """
    if frame.shape[0] != frame.shape[1]:
        smallest_shape = min(frame.shape[0],frame.shape[1])
        frame = frame[0:smallest_shape,0:smallest_shape]
    return frame


def crop_sinogram(sinogram, input_dict,probe_positions): 

    cropped_sinogram = sinogram
    if input_dict['crop'] != []: 
        if isinstance(input_dict['crop'],list):        
            cropped_sinogram = sinogram[input_dict['crop'][0]:input_dict['crop'][1],input_dict['crop'][2]:input_dict['crop'][3]]
        elif isinstance(input_dict['crop'],str):        
            print('\tAuto cropping frames...')
            if input_dict['crop'] == "positions": # Miqueles approach using scan positions
                frame = 0
                cropped_frame = autocrop_using_scan_positions(sinogram[frame,:,:],input_dict,probe_positions) # crop
                if frame == 0: 
                    cropped_frame =  make_1st_frame_squared(cropped_frame)
                    cropped_sinogram = np.empty((sinogram.shape[0],cropped_frame.shape[0],cropped_frame.shape[1]),dtype=complex)
                
                cropped_frame = match_cropped_frame_dimension(cropped_sinogram,cropped_frame)
                cropped_sinogram[frame,:,:] = cropped_frame
                frame += 1
        
            if input_dict['crop'] == "operator_T": # Miqueles approach using T operator
                for frame in range(sinogram.shape[0]):
                    cropped_sinogram[frame,:,:] = autocrop_miqueles_operatorT(sinogram[frame,:,:])
                
            if input_dict['crop'] == "local_entropy": # Yuri approach using local entropy
                for frame in range(sinogram.shape[0]):
                    min_crop_value = []
                    best_crop = auto_crop_noise_borders(sinogram[frame,:,:])
                    min_crop_value.append(best_crop)
                min_crop = min(min_crop_value)
                cropped_sinogram = sinogram[:, min_crop:-min_crop-1, min_crop:-min_crop-1]

        if cropped_sinogram.shape[1] % 2 != 0:  # object array must have even number of pixels to avoid bug during the phase unwrapping later on
            cropped_sinogram = cropped_sinogram[:,0:-1, :]
        if cropped_sinogram.shape[2] % 2 != 0:
            cropped_sinogram = cropped_sinogram[:,:, 0:-1]
        
    return cropped_sinogram


def autocrop_using_scan_positions(image,input_dict,probe_positions):

    probe_positions = 1e-6 * probe_positions / input_dict['object_pixel']     #scanning positions @ image domain

    n         = image.shape[0]
    x         = (n//2 - probe_positions [:,1]).astype(int)
    y         = (n//2 - probe_positions [:,0]).astype(int)
    pinholesize = 0 #tirado do bolso! 
    xmin      = x.min() - pinholesize//2
    xmax      = x.max() + pinholesize//2
    ymin      = y.min() - pinholesize//2
    ymax      = y.max() + pinholesize//2
    new = image[ ymin:ymax, xmin:xmax] 
    new = new + abs(new.min())

    return new


def autocrop_miqueles_operatorT(image):

    def _operator_T(u):
        d   = 1.0
        uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
        uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
        uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
        uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
        delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
        return np.sqrt( delta )
    
    def removeBorders(img, n):
        r,c = img.shape
        img[0:n,:] = 0
        img[r-n:r,:] = 0
        img[:,0:n] = 0
        img[:,c-n:c] = 0
        return img

    from skimage.morphology import square, erosion, convex_hull_image

    img = np.angle(image) ### um certo frame, que vem direto de ptycho
    img_gradient = skimage.filters.scharr(img)
    img_gradient = skimage.util.img_as_ubyte(img_gradient / img_gradient.max())
    img_gradient = gaussian_filter(img_gradient, sigma=10)
    where = _operator_T(img_gradient).real
    new = np.copy(img_gradient)
    new[ new > 0] = _operator_T(new).real[ img_gradient > 0]
    tol = 1e-6
    mask = ( np.abs( new - img_gradient) < tol ) * 1.0
    mask = erosion(mask, square(5))
    mask = removeBorders(mask, 100)
    chull = convex_hull_image(mask)
    image[ chull == 0 ] = 0 #cropando
    return image


def calculate_FRC(image, input_dict):

    frame     = input_dict["FRC"][0]
    padding   = input_dict["FRC"][1]
    sharpness = input_dict["FRC"][2]
    radius    = input_dict["FRC"][3]

    wimg = frc.window( image, padding, [sharpness, radius] )
    dic  = frc.computep( wimg , input_dict["CPUs"] ) 
    frc.plot(dic, {'label': "Resolution", 'unit': "nm", 'pxlsize': input_dict["object_pixel"]},savepath=os.path.join(input_dict["output_path"],'frc.png') )


def auto_crop_noise_borders(complex_array):
    """ Crop noisy borders of the reconstructed object using a local entropy map of the phase

    Args:
        complex_array : reconstructed object

    Returns:
        cropped_array : object without noisy borders
    """    
    import skimage.filters
    from skimage.morphology import disk

    img = np.angle(complex_array)  # get phase to perform cropping analysis

    img_gradient = skimage.filters.scharr(img)
    img_gradient = skimage.util.img_as_ubyte(img_gradient / img_gradient.max())
    local_entropy_map = skimage.filters.rank.entropy(img_gradient, disk(5)) # disk gives size of the region used to calculate local entropy

    smallest_img_dimension = 200
    max_crop = img.shape[0] // 2 - smallest_img_dimension // 2  # smallest image after cropping will have 2*100 pixels in each direction

    crop_sizes = range(1, max_crop, 10)

    mean_list = []
    for c in (crop_sizes):
        """
        mean is a good metric since we expect it to decrease as high entropy border
        gets cropped, and increase again as the low entropy smooth background gets cropped.
        it may become an issue if the sample is displaced from the center, though
        """
        mean = (local_entropy_map[c:-c, c:-c].ravel()).mean()
        mean_list.append(mean)

    best_crop = crop_sizes[np.where(mean_list == min(mean_list))[0][0]]

    return best_crop


def masks_application(difpad, input_dict):

    center_row, center_col = input_dict["DP_center"]

    if input_dict["detector_exposure"][0]: 
        print("\t\tRemoving pixels above detector pile-up threshold")
        mask = np.zeros_like(difpad)
        difpad_region = np.zeros_like(difpad)
        half_size = 128 # 128 pixels halfsize mean the region has 256^2, i.e. the size of a single chip
        mask[center_row-half_size:center_row+half_size,center_col-half_size:center_col+half_size] = 1
        difpad_region = np.where(mask>0,difpad,0)        
        detector_pileup_count = 350000  # counts/sec; value according to Kalile
        detector_exposure_time = input_dict["detector_exposure"][1]
        difpad_rescaled = difpad_region / detector_exposure_time # apply threshold
        difpad[difpad_rescaled > detector_pileup_count] = -1
    elif input_dict["central_mask"][0]:  # circular central mask to block center of the difpad
        radius = input_dict["central_mask"][1] # pixels
        central_mask = create_circular_mask((center_row,center_col), radius, difpad.shape)
        difpad[central_mask > 0] = -1

    return difpad


