import os, sys, json
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

""" sscCdi relative imports"""
from ...misc import create_directory_if_doesnt_exist, delete_files_if_not_empty_directory, estimate_memory_usage, add_to_hdf5_group, wavelength_meters_from_energy_keV, list_files_in_folder, select_specific_angles
from ...ptycho.ptychography import call_ptychography, set_object_pixel_size, set_object_shape
from ...processing.restoration import binning_G_parallel
from ...ptycho.plots import plot_iteration_error, plot_ptycho_corrected_scan_points

from ... import event_start, event_stop, log_event

##### ##### ##### #####                  PTYCHOGRAPHY                 ##### ##### ##### ##### ##### 

@log_event
def cat_ptychography(input_dict,restoration_dict,restored_data_info, filepaths, filenames, folder_names_list, folder_numbers_list, strategy="serial"):
    """ 
    Read restored diffraction data, read probe positions, calculate object parameters, calls ptychography and returns recostruction arrays
    
    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
        restoration_dict: parameters from restoration function, listed for each acquisition folder
        restored_data_info: parameters from restoration function, listed for each acquisition folder
        filepaths: path's list of required projections
        filenames: name's list of required projections
        folder_names_list: list with the folders containing the required projections
        folder_numbers_list: list with the folders' number containing the requires projections
        strategy: flag to select ptychography mode. Only had "serial" mode for now.

    Returns:
        input_dict (dict): updated input dictionary
        sinogram: numpy array containing reconstructed frames
        probe: numpy array containing reconstructed probes
        probe_positions: numpy array containing probe positions in pixels
    """

    import sscPimega

    input_dict["restoration_id"] = restored_data_info
    total_number_of_angles = len(filepaths)
    frame_index = input_dict["projections"]
    corrected_positions_list = []

    if strategy == "serial": #TODO: implement second parallel strategy

        event_start("Read and reconstruct", {"num_of_files": len(filenames)})

        for file_number_index, filename in enumerate(filenames):
            if frame_index == []:
                file_number = file_number_index
            else:
                file_number = frame_index[file_number_index]
            folder_number = folder_numbers_list[file_number_index]
            acquisitions_folder = folder_names_list[file_number_index]

            print(f"\nReading diffraction data for angle: {file_number}")
            event_start("Read restored data")
            if len(input_dict["projections"]) > 1 or len(input_dict["projections"]) == 0: 
                DPs = sscPimega.pi540D.ioGetM_Backward540D( restoration_dict, restored_data_info, file_number_index) # read restored DPs from temporary folder
            else:
                DPs = sscPimega.pi540D.ioGet_Backward540D( restoration_dict, restored_data_info[0],restored_data_info[1])

            DPs = DPs.astype(np.float32) # convert from float64 to float32 to save memory
            event_stop() # read restored data

            if np.abs(input_dict["binning"]) > 1:
                print('Binning data...')
                if input_dict["binning"] > 0:
                    DPs = binning_G_parallel(DPs,input_dict["binning"],input_dict["CPUs"]) # binning strategy by G. Baraldi
                if input_dict["binning"] < 0:
                    DPs = DPs[:,0::np.abs(input_dict["binning"]),0::np.abs(input_dict["binning"])]
                    input_dict["binning"] = np.abs(input_dict["binning"])
                    
                if DPs.shape[1] % 2 != 0: # make shape even 
                    DPs = DPs[:,0:-1,:]
                if DPs.shape[2] % 2 != 0:    
                    DPs = DPs[:,:,0:-1]

            if 'save_restored_data' in input_dict:
                event_start("Save numpy file restored data")
                print(f"Saving restored diffraction patterns...")
                if input_dict['save_restored_data'] == True:
                    np.save(os.path.join(input_dict['output_path'],f"{folder_number:03d}_restored_data.npy"),DPs)
                event_stop() # save restored data

            print(f"\tFinished reading diffraction data! DPs shape: {DPs.shape}")

            event_start("Ptycho preprocessing")

            """ Read positions """
            probe_positions, angle = read_probe_positions(input_dict, acquisitions_folder,filename , DPs.shape)
            print(f"\tFinished reading probe positions. Shape: {probe_positions.shape}")

            if file_number_index == 0:
                input_dict["object_shape"] = set_object_shape(input_dict["object_padding"], DPs.shape, probe_positions)
                sinogram = np.zeros((total_number_of_angles,input_dict["object_shape"][0],input_dict["object_shape"][1]),dtype=np.complex64)

                if input_dict["incoherent_modes"] < 1:
                    incoherent_modes = 1
                else:
                    incoherent_modes = input_dict["incoherent_modes"]
                probes = np.zeros((total_number_of_angles,incoherent_modes,DPs.shape[-2],DPs.shape[-1]),dtype=np.complex64)

                print(f"\tInitial object shape: {sinogram.shape}\t Initial probe shape: {probes.shape}")

                size_of_single_restored_DP = estimate_memory_usage(DPs)[3]
                estimated_size_for_all_DPs = len(filepaths)*size_of_single_restored_DP
                print(f"\tEstimated size for {len(filepaths)} DPs of type {DPs.dtype}: {estimated_size_for_all_DPs:.2f} GBs")

            event_stop()

            """ Call Ptychography """
            sinogram[file_number_index, :, :], probes[file_number_index, :, :], corrected_positions, error, metadata = call_ptychography(input_dict,DPs,probe_positions) # run ptycho

            if corrected_positions is not None:
                corrected_positions_list.append(corrected_positions[:,0,0:2])
            angle = np.array([file_number_index,0,angle,angle*180/np.pi])

            """ Save single frame of object and probe to temporary folder"""

            event_start("Save numpy ptychography files")

            np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_object.npy"),sinogram[file_number_index])
            np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_probe.npy"),probes[file_number_index])
            np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_angle.npy"),angle)
            np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_positions.npy"),np.expand_dims(probe_positions,axis=0))
            np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_error.npy"),error)

            if corrected_positions is not None:
                corrected_positions = corrected_positions[:,0,0:2]
                corrected_positions[:,[0,1]] = corrected_positions[:,[1,0]]
                np.save(os.path.join(input_dict["temporary_output_recons"],f"{file_number:04d}_corrected_positions.npy"),np.expand_dims(corrected_positions,axis=0))

            event_stop() # save numpy ptychography files

        event_stop() # read and reconstruct

        event_start("clean restoration data")
        """ Clean restored DPs temporary data """
        if len(input_dict['projections']) == 1:
            sscPimega.pi540D.ioClean_Backward540D( restoration_dict, restored_data_info[0] )
        else:
            sscPimega.pi540D.ioCleanM_Backward540D( restoration_dict, restored_data_info )
        event_stop() # clean restoration data



    return input_dict, sinogram, probes, probe_positions


##### ##### ##### #####                  DATA PREPARATION                 ##### ##### ##### ##### ##### 

def define_paths(input_dict):
    """ 
    Defines paths of interest for the ptychographic reconstruction and adds them to dictionary variable. Creates folders of interest and instantiates hdf5 output file

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json

    Returns:
        input_dict: updated input dictionary
    """
    if 'PreviewGCC' not in input_dict: input_dict['PreviewGCC'] = [False,""] # flag to save previews of interest only to GCC, not to the beamline user
    
    #=========== Set Parameters and Folders =====================
    print('\tProposal path: ',input_dict['data_folder'] )
    print('\tAcquisition folder: ',input_dict["acquisition_folders"][0])
 
    import sscCdi
    input_dict["versions"] = f"sscCdi={sscCdi.__version__}"

    beamline_outputs_path = os.path.join(input_dict['data_folder'].rsplit('/',3)[0], 'proc','recons',input_dict["acquisition_folders"][0]) # standard folder chosen by CAT for their outputs
    print("\tOutput path:", beamline_outputs_path)
    input_dict["output_path"]  = os.path.join(beamline_outputs_path)
    input_dict["temporary_output"]  = os.path.join(input_dict["output_path"],'temp')
    input_dict["temporary_output_recons"]  = os.path.join(input_dict["output_path"],'recons')

    create_output_directories(input_dict) # create all output directories of interest
    delete_files_if_not_empty_directory(input_dict["temporary_output"])

    input_dict['scans_string'] = 'scans'
    input_dict['positions_string']  = 'positions'

    input_dict['ignored_scans'] = [('folder_number','file_number')]

    images_folder    = os.path.join(input_dict["acquisition_folders"][0],'images')
    
    mdata_dict = json.load(open(os.path.join(input_dict['data_folder'] ,input_dict["acquisition_folders"][0],'mdata.json')))
    input_dict["energy"]               = mdata_dict['/entry/beamline/experiment']["energy"]
    input_dict["detector_distance"]    = mdata_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
    input_dict["detector_pixel_size"]  = mdata_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
    input_dict["detector_exposure"]    = [None,None]
    input_dict["detector_exposure"][1] = mdata_dict['/entry/beamline/detector']['pimega']["exposure time"]
    
    if "flatfield" not in input_dict:
        input_dict["flatfield"] = os.path.join(input_dict['data_folder'] ,images_folder,'flat.hdf5')
    elif input_dict["flatfield"] == "":
        input_dict["flatfield"] = os.path.join(input_dict['data_folder'] ,images_folder,'flat.hdf5')
    input_dict["mask"]          = os.path.join(input_dict['data_folder'] ,images_folder,'mask.hdf5')
    input_dict["empty"]         = os.path.join(input_dict['data_folder'] ,images_folder,'empty.hdf5')
    input_dict["dbeam"]         = os.path.join(input_dict['data_folder'] ,images_folder,'dbeam.hdf5')

    posflat_path = os.path.join(input_dict['data_folder'] ,images_folder,'posflat.hdf5')
    if os.path.exists(posflat_path): # restored flatfield
        input_dict["posflat"] = posflat_path

    posmask_path = os.path.join(input_dict['data_folder'] ,images_folder,'posmask.hdf5')
    if os.path.exists(posflat_path): # restored mask
        input_dict["posmask"] = posmask_path        

    input_dict["datetime"] = get_datetime(input_dict)

    input_dict["hdf5_output"] = os.path.join(input_dict["output_path"],input_dict["datetime"]+".hdf5") # create output hdf5 file
    hdf5_output = h5py.File(input_dict["hdf5_output"], "w")
    hdf5_output.create_group("recon")
    hdf5_output.create_group("log")
    hdf5_output.create_group("frc")

    return input_dict


def get_datetime(input_dict):
    """ 
    Get custom str with acquisition name and current datetime to use as filename

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code

    Returns:
        datetime (str): custom date-time string
    """
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    name = input_dict["acquisition_folders"][0]
    datetime = dt_string + "_" + name.split('.')[0]
    return datetime 

def create_output_directories(input_dict):
    """ 
    Create output directory and temporary folders in it

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code

    """
    if input_dict["output_path"] != "": # if no path is given, create directory
        create_directory_if_doesnt_exist(input_dict["output_path"])
        create_directory_if_doesnt_exist(input_dict["temporary_output"])
        create_directory_if_doesnt_exist(input_dict["temporary_output_recons"])



def convert_probe_positions_meters_to_pixels(offset_topleft, pixel_size, probe_positions):
    """
    Subtratcs minimum of position in each direction, converts from microns to pixels and then apply desired offset 
    """

    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / pixel_size  # convert from microns to pixels
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / pixel_size 
    
    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    return probe_positions


def read_probe_positions(input_dict, acquisitions_folder,measurement_file, sinogram_shape):
    """ 
    Read raw probe positions file (in microns) and convert them to pixels. Also read the rotation angle of the frame for tomography measurements.

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
        acquisitions_folder (str): specific sample/acquisition folder
        measurement_file (str): name of the file contained in the positions file
        sinogram_shape (tuple): shape of diffraction patterns array

    Returns:
        probe_positions (numpy array): array (Y,X) of probe positions in pixels
        angle (float): rotation angle in radians
    """

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
        # elif line_counter == 1:
            # pass
        else:  # skip first line, which is the header;

            positions_x = float(line.split()[1])
            positions_y = float(line.split()[0])

            positions_x, positions_y = rotate_coordinate_system(input_dict["position_rotation"],positions_x, positions_y) # rotate whole coordinate system (correct misalignment of scan and detector axes)

            #TODO: rolate relative angle between scan x and y positions

            probe_positions.append([positions_y, positions_x])

    probe_positions = np.asarray(probe_positions).astype(np.float32) # convert list of lists to numpy array

    n_of_positions = probe_positions.shape[0]

    if n_of_positions == n_of_DPs:  # check if number of recorded beam positions in txt matches the number of diff. patterns saved in the hdf5
        pass
    else:
        print("\t\tProblem when reading positions. Number of positions {0} is different from number of diffraction patterns {1}".format(n_of_positions, n_of_DPs))
        print('\t\tSetting object as null array with correct shape... New probe positions shape:', probe_positions.shape)
        probe_positions = np.zeros((n_of_DPs-1, 2))

    input_dict["wavelength"] = wavelength_meters_from_energy_keV(input_dict["energy"])
    input_dict = set_object_pixel_size(input_dict,DP_size) 
    probe_positions = convert_probe_positions_meters_to_pixels(input_dict["object_padding"],input_dict["object_pixel"], probe_positions)

    return probe_positions, angle



##### ##### ##### #####                 PROCESSING               ##### ##### ##### ##### ##### 

def match_cropped_frame_dimension(sinogram,frame):
    """ 
    Match the new incoming frame to the same squared shape of the sinogram. Sinogram should have shape (M,N,N)!

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
    """ 
    Crops frame of dimension (A,B) to (A,A) or (B,B), depending if A or B is smaller

    Args:
        frame: 2D frame

    Returns:
        frame: cropped frame with smalelr dimension
    """
    if frame.shape[0] != frame.shape[1]:
        smallest_shape = min(frame.shape[0],frame.shape[1])
        frame = frame[0:smallest_shape,0:smallest_shape]
    return frame


def crop_sinogram(input_dict,sinogram ,probe_positions):
    """ 
    Crop sinogram of 2D images manually or via automatic methods

    Args:
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
        sinogram (array): array with N reconstructed images. shape (N,Y,X)
        probe_positions (array): probe positions used to crop the array automatically

    Returns:
        cropped_sinogram (array): cropped sinogram
    """

    if isinstance(input_dict['crop'],list):        
        cropped_sinogram = sinogram[:,input_dict['crop'][0]:input_dict['crop'][1],input_dict['crop'][2]:input_dict['crop'][3]]
    elif isinstance(input_dict['crop'],str):        
        if input_dict['crop'] == "positions": # Miqueles approach using scan positions
            frame = 0
            cropped_frame = autocrop_using_scan_positions(sinogram[frame,:,:],input_dict,probe_positions) # crop
            if frame == 0: 
                cropped_frame =  make_1st_frame_squared(cropped_frame)
                cropped_sinogram = np.empty((sinogram.shape[0],cropped_frame.shape[0],cropped_frame.shape[1]),dtype=complex)
            
            cropped_frame = match_cropped_frame_dimension(sinogram,cropped_frame)
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
    """ 
    Automatic crop of the field of view using the probe positions during scan
    """
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
    """ 
    Approach to automatic cropping field of view by Eduardo Miqueles
    """

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
    import skimage
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


def auto_crop_noise_borders(complex_array):
    """ 
    Crop noisy borders of the reconstructed object using a local entropy map of the phase

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


def calculate_FRC(img, input_dict):
    """ 
    Calculate Fourier Ring Correlation (FRC) via sscResolution package. Saves FRC result to the output hdf5 file

    img (array): image of interest for calculating the FRC
        input_dict (dict): input dictionary of CATERETE beamline loaded from json and modified along the code
    
    The input_dict entrance input_dict["FRC"] is the one of interest for this function. This is a list, where each item does one specific aspect of the FRC:
        input_dict["FRC][0]: Selects which frame of the sinogram to calculate the FRC

    The matrix inputted for the FRC must be squared. Therefore, in case it is not, it will use the following parameters to select a start pixel (start, start) and return a matrix with a certain side
        input_dict["FRC][1]: start:(pixel_row,pixel_column) values to crop the image to squared format. starting positions at the top-left corner
        input_dict["FRC][2]: size: size of the square side

    These are parameters for filtering the image so that it is appropriate for FRC calculation
        input_dict["FRC][3]: padding: padding on the original image (not necessary, can be = 0);
        input_dict["FRC][4]: sharpness: sharpness of the sigmoidal window. The higher the value, the sharper the edge;
        input_dict["FRC][5]: radius: radius of the window from the center of the image;
    """

    import sscResolution

    start, size = input_dict["FRC"][1], input_dict["FRC"][2]

    if img.shape[0] != img.shape[1]:
        img = img[start:start+size,start:start+size]

    padding = input_dict["FRC"][3]
    sharpness = input_dict["FRC"][4]
    radius = input_dict["FRC"][5]
    wimg = sscResolution.frc.window( img, padding, [sharpness, radius] )
    dic = sscResolution.frc.computep( wimg , input_dict["CPUs"] ) 
    
    halfbit  = dic['x']['even']['halfbit']
    resolution = 1e9*input_dict["object_pixel"]/halfbit
    print(f"\tResolution via halfbit criterion: {resolution:.2f} nm")

    add_to_hdf5_group(input_dict["hdf5_output"],'frc','img',img)
    add_to_hdf5_group(input_dict["hdf5_output"],'frc','filtered_img',wimg)
    add_to_hdf5_group(input_dict["hdf5_output"],'frc','halfbit',halfbit)
    add_to_hdf5_group(input_dict["hdf5_output"],'frc','resolution',resolution)


def save_input_dictionary(input_dict,folder_path = "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/inputs/"):
    """
    Saves input_dict in a predetermied folder_path as a json file
    """

    import getpass 

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    filepath = os.path.join(folder_path, getpass.getuser()+'_ptycho_input.json')
    out_file = open(filepath, "w")
    json.dump(input_dict, out_file, indent = 3)
    out_file.close()
    return filepath
