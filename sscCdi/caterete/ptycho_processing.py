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
import sscResolution
import sscPtycho
import sscCdi
from sscPimega import pi540D

""" sscCdi relative imports"""
from ..misc import create_directory_if_doesnt_exist, read_hdf5, export_json, wavelength_from_energy, create_circular_mask, create_rectangular_mask, create_cross_mask
from .cat_restoration import Geometry
from ..processing.ptycho_fresnel import calculate_fresnel_number

def cat_ptychography(jason,restoration_dict_list,restored_data_info_list,strategy="serial"):

    total_n_of_frames = 0
    for acquisitions_folder in jason['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon
        _, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(jason['data_folder'], acquisitions_folder,jason['scans_string']), look_for_extension=".hdf5")
        total_n_of_frames += len(filenames)

    if strategy == "serial":

        for folder_number, acquisitions_folder in enumerate(jason['acquisition_folders']):  # loop when multiple acquisitions were performed for a 3D recon
    
            filepaths, filenames = sscCdi.misc.misc.list_files_in_folder(os.path.join(jason['data_folder'], acquisitions_folder,jason['scans_string']), look_for_extension=".hdf5")

            for file_number, (filepath,filename) in enumerate(zip(filepaths,filename)):

                frame = file_number + folder_number*len(filenames) # attribute singular value to each angle

                restoration_dict = restoration_dict_list[folder_number]
                restored_data_info = restored_data_info_list[folder_number]

                """ Read Diffraction Patterns for one angle """
                DPs = pi540D.ioGetM_Backward540D( restoration_dict, restored_data_info, file_number)
                
                if file_number == 0 and folder_number == 0: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                    object_shape, jason = sscCdi.caterete.ptycho.ptycho_processing.set_object_shape(DPs,jason, [filename], [filepath], acquisitions_folder)
                    sinogram = np.zeros((total_n_of_frames,object_shape[0],object_shape[1])) 
                    probes   = np.zeros((total_n_of_frames,1,DPs.shape[-2],DPs.shape[-1]))

                """ Read positions """
                probe_positions = read_probe_positions(jason, filename , DPs.shape[0])
                
                run_ptycho = np.any(probe_positions)  # check if probe_positions == null matrix. If so, won't run current iteration

                """ Call Ptycho """
                if not run_ptycho:
                    print(f'\t\t WARNING: Frame #{(folder_number,file_number)} being nulled because number of positions did not match number of diffraction pattern!')
                    jason['ignored_scans'].append((folder_number,file_number))
                    sinogram[frame, :, :]  = np.zeros((object_shape[0],object_shape[1])) # build 3D Sinogram
                    probes[frame, :, :, :] = np.zeros((1,DPs.shape[-2],DPs.shape[-1]))
                else:
                    # run ptycho
                    pass

                """ Clean DPs temporary data """
                pi540D.ioCleanM_Backward540D( restoration_dict, restored_data_info )

    return sinogram, probes, jason

##### ##### ##### #####                  DATA PREPARATION                 ##### ##### ##### ##### ##### 


def define_paths(jason):
    if 'PreviewGCC' not in jason: jason['PreviewGCC'] = [False,""] # flag to save previews of interest only to GCC, not to the beamline user
    
    #=========== Set Parameters and Folders =====================
    print('Proposal path: ',jason['data_folder'] )
    print('Acquisition folder: ',jason["acquisition_folders"][0])
 
    beamline_outputs_path = os.path.join(jason['data_folder'] .rsplit('/',3)[0], 'proc','recons',jason["acquisition_folders"][0]) # standard folder chosen by CAT for their outputs
    print("Output path:", beamline_outputs_path)
    jason["output_path"]  = beamline_outputs_path
    jason["temporary_output"]  = os.path.join(jason["output_path"],'temp')

    create_output_directories(jason) # create all output directories of interest

    if jason['initial_obj_path'] in jason and jason['initial_obj_path']   != "": jason['initial_obj_path']   = os.path.join(jason['ReconsPath'], jason['initial_obj_path']) # append initialObj filename to path
    if jason['initial_obj_path'] in jason and jason['initial_probe_path'] != "": jason['initial_probe_path'] = os.path.join(jason['ReconsPath'], jason['initial_probe_path'])
    if jason['initial_obj_path'] in jason and jason['InitialBkg']   != "": jason['InitialBkg']   = os.path.join(jason['ReconsPath'], jason['InitialBkg'])

    jason['scans_string'] = 'scans'
    jason['positions_string']  = 'positions'

    jason['ignored_scans'] = [('folder_number','file_number')]

    images_folder    = os.path.join(jason["acquisition_folders"][0],'images')

    input_dict = json.load(open(os.path.join(jason['data_folder'] ,jason["acquisition_folders"][0],'mdata.json')))
    jason["Energy"]               = input_dict['/entry/beamline/experiment']["energy"]
    jason["detector_distance"]    = input_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
    jason["restored_pixel_size"]  = input_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
    jason["detector_exposure"][1] = input_dict['/entry/beamline/detector']['pimega']["exposure time"]
    jason["EmptyFrame"]           = os.path.join(jason['data_folder'] ,images_folder,'empty.hdf5')
    jason["FlatField"]            = os.path.join(jason['data_folder'] ,images_folder,'flat.hdf5')
    jason["Mask"]                 = os.path.join(jason['data_folder'] ,images_folder,'mask.hdf5')
    return jason


def create_output_directories(jason):
    if jason["output_path"] != "": # if no path is given, create directory
        create_directory_if_doesnt_exist(jason["output_path"])
        create_directory_if_doesnt_exist(jason["temporary_output"])


def get_files_of_interest(jason,acquistion_folder=''):

    if acquistion_folder != '':
            filepaths, filenames = sscCdi.caterete.misc.misc.list_files_in_folder(os.path.join(jason['data_folder'] , acquistion_folder,jason['scans_string'] ), look_for_extension=".hdf5")
    else:
        filepaths, filenames = sscCdi.caterete.misc.misc.list_files_in_folder(os.path.join(jason['data_folder'] , jason["acquisition_folders"][0],jason['scans_string'] ), look_for_extension=".hdf5")

    if jason['projections'] != []:
        filepaths, filenames = sscCdi.caterete.misc.misc.select_specific_angles(jason['projections'], filepaths, filenames)

    return filepaths, filenames


def set_object_shape(DP_size,jason,probe_positions,offset_topleft = 20):

    dx, jason = set_object_pixel_size(jason,DP_size) 

    probe_positions, offset_bottomright = convert_probe_positions_meters_to_pixels(dx, probe_positions, offset_topleft = offset_topleft)

    maximum_probe_coordinate = int(np.max(probe_positions)) 
    object_shape  = DP_size + maximum_probe_coordinate + offset_bottomright

    jason["object_shape"] = object_shape

    return (object_shape,object_shape), jason


def set_object_pixel_size(jason,DP_size):

    wavelength = wavelength_from_energy(jason["Energy"])
    jason["wavelength"] = wavelength
    
    object_pixel_size = wavelength * jason['detector_distance'] / (jason['restored_pixel_size'] * DP_size * jason['binning'])
    jason["object_pixel"] = object_pixel_size # in meters

    return object_pixel_size, jason


def convert_probe_positions_meters_to_pixels(dx, probe_positions, offset_topleft = 20):\
    
    probe_positions[:, 0] -= np.min(probe_positions[:, 0]) # Subtract the probe positions minimum to start at 0
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / dx  # convert from microns to pixels
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / dx 

    probe_positions[:, 0] += offset_topleft # shift probe positions to account for the padding
    probe_positions[:, 1] += offset_topleft 

    print("Check positions here. are values rounded or not?", probe_positions)

    return probe_positions, offset_topleft


def read_probe_positions(jason, measurement_file, n_of_DPs):

    def rotate_coordinate_system(angle_rad,pxl,pyl):
        px = pxl * np.cos(angle_rad) - np.sin(angle_rad) * pyl
        py = pxl * np.sin(angle_rad) + np.cos(angle_rad) * pyl
        return px, py
    
    print('Reading probe positions (probe_positions)...')
    probe_positions = []
    positions_file = open( os.path.join(jason["data_folder"], jason["positions_string"], measurement_file[:-5] + '.txt'))

    for line_counter, line in enumerate(positions_file):
        line = str(line)
        if line_counter >= 1:  # skip first line, which is the header
            
            positions_x = float(line.split()[1])
            positions_y = float(line.split()[0])
            
            #TODO: rotate whole coordinate system (correct misalignment of scan and detector coordiante systems)

            #TODO: rolate relative angle between scan x and y positions

            probe_positions.append([positions_x, positions_y, 1, 1])

    probe_positions = np.asarray(probe_positions) # convert list of lists to numpy array

    n_of_positions = probe_positions.shape[0] + 1

    if n_of_positions[0] == n_of_DPs[0]:  # check if number of recorded beam positions in txt matches the number of diff. patterns saved in the hdf5
        pass
    else:
        print("\t\tProblem when reading positions. Number of positions {0} is different from number of diffraction patterns {1}".format(n_of_positions, n_of_DPs))
        print('\t\tSetting object as null array with correct shape... New probe positions shape:', probe_positions.shape)
        probe_positions = np.zeros((n_of_DPs[0]-1, 4))

    probe_positions, _ = convert_probe_positions_meters_to_pixels(jason["object_pixel"], probe_positions)

    return probe_positions


def set_initial_probe(jason,DP_shape):

    def set_modes(probe, jason):

        mode = probe.shape[0]
        print('\tNumber of modes:', mode)
        # Adicionar modulos incoerentes
        if jason['incoherent_modes'] > mode:
            add = jason['incoherent_modes'] - mode
            probe = np.pad(probe, [[0, int(add)], [0, 0], [0, 0]])
            for i in range(add):
                probe[i + mode] = probe[i + mode - 1] * np.random.rand(*probe[0].shape)

        print("\tProbe shape ({0},{1}) with {2} incoherent modes".format(probe.shape[-2], probe.shape[-1], probe.shape[0]))

        return probe

    print('Creating initial probe...')

    if isinstance(jason['initial_probe_path'],list): # if no path to file given
        
        type = jason['initial_probe_path'][0]

        if type == 'circular':
            probe = create_circular_mask(jason["DP_center"],jason['initial_probe_path'][1],DP_shape)
        elif type == 'squared':
            probe = create_rectangular_mask(DP_shape,jason["DP_center"],jason['initial_probe_path'][1])
        elif type == 'rectangular':
            probe = create_rectangular_mask(DP_shape,jason["DP_center"],jason['initial_probe_path'][1],jason['initial_probe_path'][2])
        elif type == 'cross':
            probe = create_rectangular_mask(DP_shape,jason["DP_center"],jason['initial_probe_path'][1],jason['initial_probe_path'][2])
        elif type == 'constant':
            probe = np.ones(*DP_shape)
        elif type == 'random':
            probe = np.random.rand(*DP_shape)
        else:
            sys.error("Please select an appropriate type for probe initial guess: circular, squared, rectangular, cross, constant, random")

    elif isinstance(jason['initial_probe_path'],str):
        probe = np.load(jason['initial_probe_path'])[0][0] # load guess from file
        probe = probe.reshape((1,1,*probe.shape))
    else:
        sys.error("Please select an appropriate path or type for probe initial guess: circular, squared, cross, constant")

    print("\tProbe shape:", probe.shape)

    if jason['incoherent_modes'] > 1:
        print(f"\tSetting initial incoherent modes: {jason['incoherent_modes']} modes")
        probe = set_modes(probe, jason) # add incoherent modes 

    return probe


def set_initial_object(jason, DP):

        print('Creating initial object...')

        if isinstance(jason['initial_obj_path'],list):
            type = jason['initial_obj_path'][0]
            if type == 'autocorrelation':
                obj = fftshift(ifft2(fftshift(np.sqrt(DP))))
            elif type == 'constant':
                pass
            elif type == 'random':
                pass
            elif type == 'initialize':
                pass #TODO: implement method from https://doi.org/10.1364/OE.465397
        elif isinstance(jason['initial_obj_path'],str): 
            obj = np.load(jason['initial_obj_path'])[0]
        else:
            sys.error("Please select an appropriate path or type for object initial guess: autocorrelation, constant, random")

        return obj

##### ##### ##### #####                  PTYCHO                 ##### ##### ##### ##### ##### 

def ptycho_main(difpads, args, _start_, _end_,gpu):
    t0 = time.time()

    jason               = args[0]
    filenames           = args[1]
    filepaths           = args[2]
    acquisitions_folder = args[3]
    half_size           = args[4]
    object_shape        = args[5]
    sinogram            = args[7]
    probe3d             = args[8]
    backg3d             = args[9]
    geometry            = args[10]

    ibira_datafolder  = jason['data_folder']
    positions_string  = jason['positions_string']

    for i in range(_end_ - _start_):
        
        measurement_file     = filenames[_start_ + i]
        measurement_filepath = filepaths[_start_ + i]
        
        if i == 0:
            current_frame = str(0).zfill(4)  # start at 0. this variable will name the output preview images of the object and probe
        else:
            current_frame = str(int(current_frame) + 1).zfill(4)  # increment one
        
        frame = int(current_frame)

        probe_positions_file = os.path.join(acquisitions_folder, positions_string, measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
        probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)
        probe_positions, _ = convert_probe_positions_meters_to_pixels(jason["object_pixel"], probe_positions)

        run_ptycho = np.any(probe_positions)  # check if probe_positions == null matrix. If so, won't run current iteration. #TODO: output is null when #difpads != #positions. How to solve this?

        if i == 0: t1 = time.time()

        if run_ptycho == True:
                
            if i == 0: t2 = time.time()

            probe_support_radius, probe_support_center_x, probe_support_center_y = jason["probe_support"]

            print(f'Object shape: {object_shape}. Detector half-size: {half_size}')

            datapack, _, sigmask = set_initial_parameters(jason,difpads[frame],probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,object_shape,jason["object_pixel"])

            if i == 0: t3 = time.time()

            run_algorithms = True
            loop_counter = 1
            while run_algorithms:  # run Ptycho:
                try:
                    algorithm = jason['Algorithm' + str(loop_counter)]
                except:
                    run_algorithms = False

                if run_algorithms:
                    if algorithm['Name'] == 'GL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                    probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'],
                                                    epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                    probefresnel_number=jason['fresnel_number'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'positioncorrection':
                        datapack['bkg'] = None
                        datapack = sscPtycho.PosCorrection(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                               probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], 
                                                               epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                               probefresnel_number=jason['fresnel_number'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'RAAR':
                        datapack = sscPtycho.RAAR(iter=algorithm['Iterations'], beta=algorithm['Beta'],
                                                      probecycles=algorithm['ProbeCycles'], batch=algorithm['Batch'],
                                                      epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'],
                                                      sigmask=sigmask, probefresnel_number=jason['fresnel_number'], data=datapack,params={'device':gpu})

                    elif algorithm['Name'] == 'GLL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'],
                                                    probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'],
                                                    epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask,
                                                    probefresnel_number=jason['fresnel_number'], data=datapack,params={'device':gpu})

                    loop_counter += 1
                    RF = datapack['error']

            print('Original object shape:', datapack['obj'].shape)

            if i == 0: t4 = time.time()

            sinogram[frame, :, :] = datapack['obj']  # build 3D Sinogram
            probe3d[frame, :, :, :]  = datapack['probe']
            backg3d[frame, :, :]  = datapack['bkg']

        else:
            print('CAUTION! Zeroing frame:',frame,' for error in position file.')
            sinogram[frame, :, :]   = np.zeros((object_shape[0],object_shape[1])) # build 3D Sinogram
            probe3d[frame, :, :, :] = np.zeros((1,difpads.shape[-2],difpads.shape[-1]))
            backg3d[frame, :, :]    = np.zeros((difpads.shape[-2],difpads.shape[-1]))

        if i == 0: t5 = time.time()

    print(f'\nElapsed time for reconstruction of 1st frame: {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes')
    print(f'Total time iteration: {t5 - t0:.2f} seconds = {(t5 - t0) / 60:.2f} minutes')

    return sinogram, probe3d, backg3d


def _worker_batch_frames_(params, idx_start, idx_end, gpu):
    
    output_object       = params[0]
    output_probe        = params[1]
    output_backg        = params[2]
    difpads             = params[3]
    jason               = params[4] 
    filenames           = params[5]
    filepaths           = params[6]
    acquisitions_folder = params[7]
    half_size           = params[8]
    object_shape        = params[9]
    total_frames        = params[10]   

    _start_ = idx_start
    _end_   = idx_end

    args = (jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames,output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:])

    output_object[_start_:_end_,:,:], output_probe[_start_:_end_,:,:,:], output_backg[_start_:_end_,:,:] = ptycho_main( difpads[_start_:_end_,:,:,:], args, _start_, _end_,gpu)
    

def _build_batch_of_frames_(params):

    total_frames = params[9]
    threads      = len(params[4]['GPUs']) 
    
    b = int( np.ceil( total_frames/threads )  ) 
    
    processes = []
    for k in range( threads ):
        begin_ = k*b
        end_   = min( (k+1)*b, total_frames )
        gpu = [k]

        p = multiprocessing.Process(target=_worker_batch_frames_, args=(params, begin_, end_, gpu))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    

def ptycho3d_batch( difpads, params):
    
    name         = str( uuid.uuid4())
    name1        = str( uuid.uuid4())
    name2        = str( uuid.uuid4())


    jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames = params

    try:
        sa.delete(name)
    except:
        pass
    try:
        sa.delete(name1)
    except:
        pass
    try:
        sa.delete(name2)
    except:
        pass
            
    output_object = sa.create(name,[total_frames, object_shape[0], object_shape[1]], dtype=np.complex64)
    output_probe  = sa.create(name1,[total_frames, 1, difpads.shape[-2],difpads.shape[-1]], dtype=np.complex64)
    output_backg  = sa.create(name2,[total_frames, difpads.shape[-2],difpads.shape[-1]], dtype=np.float32)

    _params_ = ( output_object, output_probe, output_backg, difpads, jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,total_frames)
    
    _build_batch_of_frames_ ( _params_ )

    sa.delete(name)
    sa.delete(name1)
    sa.delete(name2)

    return output_object,output_probe,output_backg


def cat_ptycho_3d(difpads,jason):
    sinogram = []
    probe = []
    background = [] 

    count = -1
    for acquisitions_folder in jason['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon

        count += 1

        print('Starting restoration for acquisition: ', acquisitions_folder)

        filepaths, filenames = sscCdi.caterete.ptycho.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

        print('\nFilenames: ', filenames)

        if count == 0: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
            object_shape, half_size, object_pixel_size, jason =sscCdi.caterete.ptycho.ptycho_processing.set_object_shape(difpads[count],jason,filenames,filepaths,acquisitions_folder)
            jason["object_pixel"] = object_pixel_size

        args = (jason,filenames,filepaths,acquisitions_folder,half_size,object_shape,len(filenames))
        sinogram3d ,probe3d, background3d = sscCdi.caterete.ptycho.ptycho_processing.ptycho3d_batch(difpads[count], args) # Main ptycho iteration over ALL frames in threads

        sinogram.append(sinogram3d)
        probe.append(probe3d)
        background.append(background3d)
    
    return sinogram,probe,background,jason


def cat_ptycho_serial(jason):

    sinogram_list   = []
    probe_list      = []
    background_list = []
    counter = 0
    first_iteration = True
    first_of_folder = True
    time_elasped_restoration = 0
    time_elasped_ptycho = 0

    z1 = float(jason["detector_distance"]) * 1000  # Here comes the distance Geometry(Z1):
    geometry = sscCdi.caterete.Geometry(z1,susp=jason["suspect_border_pixels"],fill = jason["fill_blanks"]) 

    for acquisitions_folder in jason['acquisition_folders']:  
        print('Acquisiton folder: ',acquisitions_folder)
        filepaths, filenames = sscCdi.caterete.ptycho.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

        for measurement_file, measurement_filepath in zip(filenames, filepaths):   
            print('File: ',measurement_file)
            t_start = time.time()

            if 0: # old restoration approach with sscIO
                args1 = (jason,acquisitions_folder,measurement_file,measurement_filepath,len(filenames),geometry)
                difpads, _ , jason = sscCdi.caterete.cat_restoration.restoration_cat_2d(args1,first_run=first_iteration) # restoration of 2D Projection (difpads - real, is a ndarray of size (1,:,:,:))
            else:
                distance = jason["detector_distance"]*1000

                geometry = Geometry(distance)
                
                dic = {}
                dic['path']     = measurement_filepath #"/home/ABTLUS/eduardo.miqueles/test/SS61/scans/0000_SS61_001.hdf5"
                dic['outpath']  = jason["ReconsPath"]+ '/' # "/home/ABTLUS/eduardo.miqueles/test/"
                dic['order']    = "yx" 
                dic['rank']     = "ztyx" # order of axis
                dic['dataset']  = "entry/data/data"
                dic['nGPUs']    = len(jason["GPUs"])
                dic['GPUs']     = jason["GPUs"]
                dic['init']     = 0
                dic['final']    = -1 # -1 to use all DPs
                dic['saving']   = 1 # save or not
                dic['timing']   = 0 # print timers 
                dic['blocksize']= 10
                dic['geometry'] = geometry
                dic['roi']      = jason["detector_ROI_radius"]#512
                dic['center']   = jason["DP_center"] #[1400,140

                dic['flat']     = read_hdf5(jason["FlatField"])[()][0, 0, :, :] # numpy.ones([3072, 3072])
                dic['empty']    = read_hdf5(jason['EmptyFrame']).squeeze().astype(np.float32) # numpy.zeros([3072,3072]) 

                start = time.time()
                uid, nimgs  = pi540D.ioSet_Backward540D( dic ) # read hdf5 and save temporary restored DPs
                difpads = pi540D.ioGet_Backward540D( dic, uid, nimgs ) # read temporary DPs
                pi540D.ioClean_Backward540D( dic, uid ) # remove temporary files
                elapsed = time.time() - start
                # difpads = np.expand_dims(difpads,axis=0)
                difpads = difpads.reshape((1,*difpads.shape))
                difpads = difpads[:,1::,:,:]
                print('Elapsed: {}'.format(elapsed))
 
            time_elasped_restoration += time.time() - t_start
            
            if first_iteration: # Compute object size, object pixel size for the first frame and use it in all 3D ptycho
                object_shape, half_size, object_pixel_size, jason = sscCdi.caterete.ptycho.ptycho_processing.set_object_shape(difpads,jason, [measurement_file], [measurement_filepath], acquisitions_folder)
                jason["object_pixel"] = object_pixel_size
                first_iteration = False

            object_dummy     = np.zeros((1,object_shape[1],object_shape[0]),dtype = complex) # build 3D Sinogram
            probe_dummy      = np.zeros((1,1,difpads.shape[-2],difpads.shape[-1]),dtype = complex)
            background_dummy = np.zeros((1,difpads.shape[-2],difpads.shape[-1]), dtype=np.float32)
            
            args2 = (jason,[measurement_file], [measurement_filepath], acquisitions_folder,half_size,object_shape,len([measurement_file]),object_dummy,probe_dummy,background_dummy,geometry)

            t_start2 = time.time()
            object2d, probe2d, background2d = sscCdi.caterete.ptycho.ptycho_processing.ptycho_main(difpads, args2, 0, 1,jason['GPUs'])   # Main ptycho iteration on ALL frames in threads
            # object2d, probe2d, background2d = object_dummy,probe_dummy,background_dummy
            time_elasped_ptycho += time.time() - t_start2

            if first_of_folder:
                object = object2d
                probe  = probe2d
                background    = background2d
                first_of_folder = False
            else:
                object = np.concatenate((object,object2d), axis = 0)
                probe  = np.concatenate((probe,probe2d),  axis = 0)
                background = np.concatenate((background,background2d), axis = 0)
            counter +=1

        first_of_folder = True

        sinogram_list.append(object)
        probe_list.append(probe)
        background_list.append(background)

    return sinogram_list, probe_list, background_list, time_elasped_restoration, time_elasped_ptycho, jason


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


def crop_sinogram(sinogram, jason): 

    cropped_sinogram = sinogram
    if jason['autocrop'] == True: # automatically crop borders with noise
        print('Auto cropping frames...')
        
        if 1: # Miqueles approach using scan positions
            frame = 0
            ibira_datafolder = jason["data_folder"]
            for acquisitions_folder in jason['acquisition_folders']:  # loop when multiple acquisitions were performed for a 3D recon
                
                filepaths, filenames = sscCdi.caterete.ptycho.ptycho_processing.get_files_of_interest(jason,acquisitions_folder)

                for measurement_file, measurement_filepath in zip(filenames, filepaths):

                    if sinogram.shape[0] == len(filenames): print("SHAPES MATCH!")

                    probe_positions_file = os.path.join(acquisitions_folder, jason['positions_string'], measurement_file[:-5] + '.txt')  # change .hdf5 to .txt extension
                    probe_positions = read_probe_positions(os.path.join(ibira_datafolder,probe_positions_file), measurement_filepath)

                    cropped_frame = autocrop_using_scan_positions(sinogram[frame,:,:],jason,probe_positions) # crop
                    if frame == 0: 
                        cropped_frame =  make_1st_frame_squared(cropped_frame)
                        cropped_sinogram = np.empty((sinogram.shape[0],cropped_frame.shape[0],cropped_frame.shape[1]),dtype=complex)
                    
                    cropped_frame = match_cropped_frame_dimension(cropped_sinogram,cropped_frame)
                    print("SHAPES:",len(filenames),sinogram.shape, cropped_frame.shape)
                    cropped_sinogram[frame,:,:] = cropped_frame
                    frame += 1
       
        if 0: # Miqueles approach using T operator
            for frame in range(sinogram.shape[0]):
                cropped_sinogram[frame,:,:] = autocrop_miqueles_operatorT(sinogram[frame,:,:])
            
        if 0: # Yuri approach using local entropy
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
        print('\t Done!')
        
    return cropped_sinogram


def autocrop_using_scan_positions(image,jason,probe_positions):

    probe_positions = 1e-6 * probe_positions / jason['object_pixel']     #scanning positions @ image domain

    n         = image.shape[0]
    x         = (n//2 - probe_positions [:,0]).astype(int)
    y         = (n//2 - probe_positions [:,1]).astype(int)
    pinholesize = 0 #tirado do bolso! 
    xmin      = x.min() - pinholesize//2
    xmax      = x.max() + pinholesize//2
    ymin      = y.min() - pinholesize//2
    ymax      = y.max() + pinholesize//2
    new = image[xmin:xmax, ymin:ymax] 
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


def apply_phase_unwrap(sinogram, jason):

    if jason['phase_unwrap'][2] != [] and jason['phase_unwrap'][3] != []:
        print('Manual cropping of the data')
        """ Fine manual crop of the reconstruction for a proper phase unwrap
        jason['phase_unwrap'][2] = [upper_crop,lower_crop]
        jason['phase_unwrap'][3] = [left_crop,right_crop] """
        sinogram = sinogram[:,jason['phase_unwrap'][2][0]: -jason['phase_unwrap'][2][1], jason['phase_unwrap'][3][0]: -jason['phase_unwrap'][3][1]]
    
    print('Cropped object shape:', sinogram.shape)

    print('Phase unwrapping the cropped image')
    n_iterations = jason['phase_unwrap'][1]  # number of iterations to remove gradient from unwrapped image.
    
    phase = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))
    absol = np.zeros((sinogram.shape[0],sinogram.shape[-2],sinogram.shape[-1]))

    for frame in range(sinogram.shape[0]):
        original_object = sinogram[frame,:,:]  # create copy of object
        absol[frame,:,:] = np.abs(sinogram[frame,:,:])
        # phase[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.angle(sscPtycho.RemovePhaseGrad(sinogram[frame,:,:])))#, n_iterations, non_negativity=0, remove_gradient=0)
        phase[frame,:,:] = sscCdi.unwrap.phase_unwrap(-np.angle(sinogram[frame,:,:]))

        if 1:  # plot original and cropped object phase and save!
            figure, subplot = plt.subplots(1, 2,dpi=300,figsize=(5,5))
            subplot[0].imshow(-np.angle(original_object),cmap='gray')
            subplot[1].imshow(phase[frame,:,:],cmap='gray')
            subplot[0].set_title('Original')
            subplot[1].set_title('Cropped and Unwrapped')

    return phase,absol


def calculate_FRC(sinogram, jason):

    def resolution_frc(data, pixel, plot=False,plot_output_folder="./outputs/preview",savepath='./outputs/reconstruction'):
        """     
        Fourier Ring Correlation for 2D images:
        The routine inputs are besides the two images for correlation
        Resolution threshold curves desired: "half" for halfbit, "sigma" for 3sigma, "both" for them both
        Pixelsize of the object

        # Output is a dictionary with the resolution values in the object pixelsize unit, and the FRC, frequency and threshold arrays
        # resolution['halfbit'] :  resolution values in the object pixelsize unit
        # resolution['curve']   :  FRC array
        # resolution['freq']    :  frequency array
        # resolution['sthresh'] :  threshold array
        # resolution['hthresh'] :  threshold array

        Args:
            data : image to calculate resolution
            pixel : effective pixel size of the object
            plot_output_folder (str, optional): _description_. Defaults to "./outputs/preview".
            savepath : folder to save FRC dictionary with outputs. Defaults to './outputs/reconstruction'.

        Returns:
            resolution: FRC output dictionary
        """    
        
        print('Calculating resolution by Fourier Ring Correlation...')

        sizex = data.shape[-1]
        sizey = data.shape[-2]

        data = data.reshape((1,sizey,sizex)) # reshape so that fourier_ring_correlation interprets data correctly

        # For this case we will use the odd/odd even/even divisions of one image in a dataset
        data1 = data[:,0:sizey:2, 0:sizex:2]  # even
        data2 = data[:,1:sizey:2, 1:sizex:2]  # odd
        
        resolution = sscResolution.fourier_ring_correlation(data1,data2,pixel)
        if plot:
            sscResolution.get_fourier_correlation_fancy_plot(resolution, plot_output_folder, plot=True)

        if savepath != '':
            export_json(resolution,savepath+'/frc_outputs.txt')

        return resolution

    if sinogram.shape[1]%2!=0:
        sinogram = sinogram[:,0:-1,:]
    if sinogram.shape[2]%2!=0:
        sinogram = sinogram[:,:,0:-1]

    object_pixel_size = jason["object_pixel"] 

    frame = 0 # selects first frame of the sinogram to calculate resolution

    if jason['FRC'] == True:
        print('Estimating resolution via Fourier Ring Correlation')
        resolution = resolution_frc(sinogram[frame,:,:], object_pixel_size, plot=True,plot_output_folder=os.path.join(jason["output_path"]+'/'),savepath=jason["output_path"])
        try:
            print('\tResolution for frame ' + str(frame) + ':', resolution['halfbit_resolution'])
            jason["halfbit_resolution"] = resolution['halfbit_resolution']
        except:
            print('Could not calculate halfbit FRC resolution')
        try:
            print('\tResolution for frame ' + str(frame) + ':', resolution['sigma_resolution'])
            jason["3sigma_resolution"] = resolution['sigma_resolution']
        except:
            print('Could not calculate 3sigma FRC resolution')

    return jason


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

    # cropped_array = complex_array[best_crop:-best_crop, best_crop:-best_crop]  # crop original complex image

    # if 0:  # debug / see results
    #     figure, subplot = plt.subplots(1, 3, figsize=(10, 10), dpi=200)
    #     subplot[0].imshow(img)
    #     subplot[1].imshow(local_entropy_map)
    #     subplot[2].imshow(np.angle(cropped_array))
    #     subplot[0].set_title('Original')
    #     subplot[1].set_title('Local entropy')

    #     subplot[2].set_title('Cropped')

    #     figure, subplot = plt.subplots()
    #     subplot.plot(crop_sizes, mean_list)
    #     subplot.set_xlabel('Crop size')
    #     subplot.set_ylabel('Mean')
    #     subplot.grid()

    # if cropped_array.shape[0] % 2 != 0:  # object array must have even number of pixels to avoid bug during the phase unwrapping later on
    #     cropped_array = cropped_array[0:-1, :]
    # if cropped_array.shape[1] % 2 != 0:
    #     cropped_array = cropped_array[:, 0:-1]

    return best_crop



def masks_application(difpad, jason):

    center_row, center_col = jason["DP_center"]

    if jason["detector_exposure"][0]: 
        print("Removing pixels above detector pile-up threshold")
        mask = np.zeros_like(difpad)
        difpad_region = np.zeros_like(difpad)
        half_size = 128 # 128 pixels halfsize mean the region has 256^2, i.e. the size of a single chip
        mask[center_row-half_size:center_row+half_size,center_col-half_size:center_col+half_size] = 1
        difpad_region = np.where(mask>0,difpad,0)        
        detector_pileup_count = 350000  # counts/sec; value according to Kalile
        detector_exposure_time = jason["detector_exposure"][1]
        difpad_rescaled = difpad_region / detector_exposure_time # apply threshold
        difpad[difpad_rescaled > detector_pileup_count] = -1
    elif jason["central_mask"][0]:  # circular central mask to block center of the difpad
        radius = jason["central_mask"][1] # pixels
        central_mask = create_circular_mask((center_row,center_col), radius, difpad.shape)
        difpad[central_mask > 0] = -1

    return difpad


##### ##### ##### ##### #####               GIOVANNI'S  ALGORITHMS              ##### ##### ##### ##### ##### 

def set_initial_parameters_for_G_algos(jason, difpads, probe_positions, radius, center_x, center_y, object_size, dx):

    def set_sigmask(difpads):
        """Create a mask for invalid pixels

        Args:
            difpads (array): measured diffraction patterns

        Returns:
            sigmask (array): 2D-array, same shape of a diffraction pattern, maps the invalid pixels
            0 for negative values, intensity measured elsewhere
        """    
        # Mask of 1 and 0:
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

    if jason['fresnel_number'] == -1:  # Manually choose wether to find Fresnel number automatically or not
        jason['fresnel_number'] = calculate_fresnel_number(dx, pixel=jason['restored_pixel_size'], energy=jason['Energy'], z=jason['detector_distance'])
        jason['fresnel_number'] = -jason['fresnel_number']
    print('\tF1 value:', jason['fresnel_number'])

    # Compute probe: initial guess:
    probe = set_initial_probe(difpads, jason)

    # Object initial guess:
    obj = set_initial_object(jason, object_size, probe, difpads)

    # Mask of 1 and 0:
    sigmask = set_sigmask(difpads)

    background = np.ones(difpads[0].shape) # dummy

    # Compute probe support:
    probesupp = probe_support(probe, half_size, radius, center_x, center_y)

    probe_positionsi = probe_positions + 0  # what's the purpose of declaring probe_positionsi?

    # Set data for Ptycho algorithms:
    datapack = set_datapack(obj, probe, probe_positions, difpads, background, probesupp)

    return datapack, probe_positionsi, sigmask

def call_G_ptychography(jason,sinogram,probes, probe_positions):

    probe_support_radius, probe_support_center_x, probe_support_center_y = jason["probe_support"]

    datapack, _, sigmask = set_initial_parameters_for_G_algos(jason,difpads[frame],probe_positions,probe_support_radius,probe_support_center_x,probe_support_center_y,jason["object_shape"],jason["object_pixel"])

    run_algorithms = True
    loop_counter = 1
    while run_algorithms:  # run Ptycho:
        try:
            algorithm = jason['Algorithm' + str(loop_counter)]
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
                                        params    = {'device':jason["GPUs"]},
                                        probefresnel_number=jason['fresnel_number'])

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
                                                    params    = {'device':jason["GPUs"]},
                                                    probefresnel_number=jason['fresnel_number'])

            elif algorithm['Name'] == 'RAAR':
                datapack = sscPtycho.RAAR(iter         = algorithm['Iterations'],
                                           beta        = algorithm['Beta'],
                                           probecycles = algorithm['ProbeCycles'],
                                           batch       = algorithm['Batch'],
                                           epsilon     = algorithm['Epsilon'], 
                                           tvmu        = algorithm['TV'],
                                           sigmask     = sigmask,
                                           data        = datapack,
                                           params      = {'device':jason["GPUs"]}, 
                                           probefresnel_number=jason['fresnel_number']) 

            loop_counter += 1
            RF = datapack['error']

    sinogram[frame, :, :] = datapack['obj']  # build 3D Sinogram
    probes[frame, :, :, :]  = datapack['probe']