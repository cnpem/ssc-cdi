import numpy as np
import matplotlib.pyplot as plt
import os, h5py


def miqueles_colormap(img):
    """ Definition of a colormap created by Miquele's for better visualizing diffraction patterns.

    Args:
        img : image to get the maximum value for proper colormap definition

    Returns:
        cmap: colormap
        colors: list of colors 
        bounds: colormap bounds
        norm: normalized colors
    """    

    import matplotlib as mpl
    import numpy

    colors = [ 'white', '#FFC0CB', '#0000FF' , '#00FFFF', 'green', 'gold', 'orange', 'red', '#C20078', 'maroon', 'black' ]
    
    cmap = mpl.colors.ListedColormap(colors)
    
    maxv    = img.max()
    epsilon = -0.1 
    
    bounds =  numpy.zeros([12,])
    bounds[0] = -10
    bounds[1] = epsilon
    bounds[2] = 0
    for k in range(3,11):
        bounds[k] = 10**( (k-2) * numpy.log(maxv) / (9 * numpy.log(10) ))
        bounds[11] = maxv
        
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, colors, bounds, norm

def plotshow_cmap2(image,title=None,figsize=(20,20),savepath=None,show=False):
    """ Function to plot and save figures using Miquele's colormap

    Args:
        image (_type_): 2d image to plot
        title (_type_, optional): Defaults to None.
        figsize (tuple, optional): Defaults to (20,20).
        savepath (_type_, optional): output path to save figure. Defaults to None.
        show (bool, optional): if true, plt.show(). Defaults to False.
    """    
    figure, subplot = plt.subplots(dpi=300,figsize=figsize)
    cmap, colors, bounds, norm = miqueles_colormap(image)
    handle = subplot.imshow(image, interpolation='nearest', cmap = cmap, norm=norm)
    figure.colorbar(handle, boundaries=bounds,ax=subplot)
    if title != None:
        subplot.set_title(title)
    if savepath != None:
        figure.savefig(savepath)
    if show:
        plt.show()

def delete_files_if_not_empty_directory(directory):
    """ Checks if directory is empty and, if not, deletes files within it

    Args:
        directory (str): absolute path to directory
    """
    for root, dirs, files in os.walk(directory):
        if files != []:
            # print("\t\tCleaning directory:", directory)
            for file in files: # For each file in the directory
                file_path = os.path.join(root, file) # Construct the full path to the file
                os.remove(file_path)  # Delete the file

def list_files_in_folder(data_directory,look_for_extension=""):
    """ Function to list all files contained in folder with a certain extension
    
    Args:
        data_directory (string) :path to the directory containing files you want to list
        look_for_extension (string, optional): string containing the file termination string, e.g. '.datx' or '.txt'. The default is "".

    Args:
        filepaths, filenames: two lists, one contaning the complete path of all files, the second containing a list of all file names
    """
    from os.path import isfile, join
    from os import listdir
    filepaths = []
    filenames = []
    if look_for_extension != "": 
        for file in listdir(data_directory):
            if isfile(join(data_directory, file)) and file.endswith(look_for_extension):        
                filepaths.append(join(data_directory, file))
                filenames.append(file)
    else:
        for file in listdir(data_directory):
            if isfile(join(data_directory, file)):
                filepaths.append(join(data_directory, file))
                filenames.append(file)

    filenames.sort(key = lambda x: x.split('_')[0]) # sort according to first four digitis
    filepaths.sort(key = lambda x: x.split('/')[-1].split('_')[0])

    return filepaths, filenames

def select_specific_angles(frames,filepaths,filenames, folders, folders_number):
    """ Function to filter lists, keeping only those with a certain frame number in the string. This is used to select only the desired frames in a 3D ptychography.

    Args:
        frames : list of frames to select
        filepaths (list): input list with full filepaths
        filenames (list): input list with full filenames
        folders (list): input list with full folders names
        folders_number (list): input list with full folders numbers

    Returns:
        filepaths: filtered filepaths list
        filenames: filtered filenames list
        folders: filtered folders names
        folders_number: foltered folders numbers
    """    
    filepaths = list( filepaths[i] for i in frames)
    filenames = list( filenames[i] for i in frames)
    folders = list( folders[i] for i in frames)
    folders_number = list( folders_number[i] for i in frames)

    return filepaths, filenames, folders, folders_number

def save_json_logfile(input_dict):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        input_dict (dic): input_dict dictionary
    """    
    import json, os

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    path = input_dict["output_path"]

    datetime = input_dict["datetime"]
    name = datetime+".json"

    filepath = os.path.join(path,name)
    file = open(filepath,"w")
    json_string = json.dumps(input_dict,indent=2,separators=(', ',': '),sort_keys=True,cls=NpEncoder)
    file.write(json_string)
    file.close()

    add_to_hdf5_group(input_dict["hdf5_output"],'log','logfile',filepath)

def save_json_logfile_tomo(input_dict):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        input_dict (dic): input_dict dictionary
    """    
    import json, os

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    path = input_dict["output_folder"]

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    name = input_dict["filename"]
    datetime = dt_string + "_" + name.split('.')[0]
    name = datetime+".json"

    filepath = os.path.join(path,name)
    file = open(filepath,"w")
    json_string = json.dumps(input_dict,indent=2,separators=(', ',': '),sort_keys=True,cls=NpEncoder)
    file.write(json_string)
    file.close()

def create_directory_if_doesnt_exist(*args):
    """ Create directories from a list of paths if they do not already exist

    Args:
        *args: multiple absolute path to directories
    """
    for arg in args:
        if os.path.isdir(arg) == False:
            print("\tCreating directory: ",arg)
            os.makedirs(arg)

def read_hdf5(path,inner_path = 'entry/data/data'):
    """ Read hdf5 file from path

    Args:
        path (str): absolute path to hdf5 file
        inner_path (str, optional): Inner path of hdf5 file structure to data. Defaults to 'entry/data/data'.

    Returns:
        (h5py File): h5py File object 
    """
    try:
        os.system(f"h5clear -s {path}")
    except:
        pass
    return h5py.File(path, 'r')[inner_path]
    
def debug(func): # decorator function for debugging
    def _debug(*args):
        result = func(*args) # call function
        print(f"{func.__name__}(args: {args}) -> {result}") # print function with arguments and result
        return result
    return _debug

def export_json(params,output_path):
    """ Exports a dictionary to a json file

    Args:
        params : dictionary
        output_path : path to output file
    """    
    import json, numpy
    export = {}
    for key in params:
        export[key] = params[key]
        if isinstance(params[key], numpy.ndarray):
            export[key] = export[key].tolist()
    json.dumps(export)

    out_file = open(output_path, "w")
    json.dump(export,out_file)
    return 0

def wavelength_from_energy(energy_keV):
    """ Calculate wavelenth from energy

    Args:
        energy_keV (float): energy in keV

    Returns:
        wavelength (float): wavelength in meters
    """

    speed_of_light = 299792458        # Speed of Light [m/s]
    planck         = 4.135667662E-18  # Plank constant [keV*s]
    return planck * speed_of_light / energy_keV

def get_array_size_bytes(array):
    """ Calculate size of array in multiples units

    Args:
        array (numpy.ndarrray): n-dimensional numpy array

    Returns:
        (tuple): tuple containing the size of the array in multiple units
    """
    bytes = array.itemsize*array.size
    kbytes = bytes/1e3
    Mbytes = bytes/1e6
    Gbytes = bytes/1e9
    kibytes = bytes/1024
    Mibytes = bytes/1024/1024
    Gibytes = bytes/1024/1024/1024
    return (bytes,kbytes,Mbytes,Gbytes,kibytes,Mibytes,Gibytes)

def estimate_memory_usage(*args):
    """ Estimate total size of multiple arrays in bytes and corresponding units

    Returns:
        (tuple): tuple containing the size of the array in multiple units
    """
    
    bytes = 0
    for arg in args:
        bytes += get_array_size_bytes(arg)[0]
        
    kbytes = bytes/1e3
    Mbytes = bytes/1e6
    Gbytes = bytes/1e9
    kibytes = bytes/1024
    Mibytes = bytes/1024/1024
    Gibytes = bytes/1024/1024/1024
    return (bytes,kbytes,Mbytes,Gbytes,kibytes,Mibytes,Gibytes)

def get_RGB_wheel():
    import matplotlib
    V, H = np.mgrid[0:1:100j, 0:1:300j]
    S = np.ones_like(V)
    HSV = np.dstack((H,S,V))
    RGB = matplotlib.colors.hsv_to_rgb(HSV)
    return RGB, H, S, V
    
def save_plots(complex_array,title='',path=''):

    from sscMisc import convert_complex_to_RGB
    complex_array = np.squeeze(complex_array)

    data_rgb = convert_complex_to_RGB(complex_array)
    magnitude = np.abs(complex_array)
    phase = np.angle(complex_array)
    
    figure = plt.figure(dpi=300)
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)
    ax1.imshow(data_rgb), ax1.set_title(title)
    ax2.imshow(magnitude,cmap='gray'), ax2.set_title("Magnitude")
    ax3.imshow(phase,cmap='hsv'), ax3.set_title("Phase")
    figure.tight_layout()
    if path != '':
        plt.savefig(path)

def plot_error(error,path='',log=False):
    fig, ax = plt.subplots(dpi=150)
    ax.plot(error, 'o-')
    ax.set_xlabel('Iterations') 
    ax.set_ylabel('Error')
    ax.set_title(f'Final error = {error[-1]:.2e}')
    ax.grid()
    if log:
        ax.set_yscale('log')
    if path != '':
        fig.savefig(path)
    
def save_variable(input_dict,variable, name = 'FLAG', group='recon'):
    add_to_hdf5_group(input_dict["hdf5_output"],group,name,variable)

def add_to_hdf5_group(path,group,name,data,mode="a"):
    """ Add data to hdf5 file. Creates a dataset with certain name inside a pre-existing group

    Args:
        path (str): absolute path to hdf5 file
        group (str): group name
        name (str): dataset name
        data: metadata to be saved
        mode (str, optional): h5py.File option for selecting interaction mode. Defaults to "a".

    """
    hdf5_output = h5py.File(path, mode)
    hdf5_output[group].create_dataset(name,data=data)
    hdf5_output.close()

def open_or_create_h5_dataset(path,group,dataset,data,create_group=False):
    """ Open hdf5 file and checks if certain dataset exists. If not, creates it.

    Args:
        path (_type_): _description_
        group (_type_): _description_
        dataset (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """

    group_dataset = group+"/"+dataset

    h5file = h5py.File(path,'a')
    dataset_exists = group_dataset in h5file
    
    if dataset_exists:
        pass
    else:
        if create_group:
            h5file.create_group(group)
        h5file[group].create_dataset(dataset,data=data)

    return h5file, dataset_exists, group_dataset

def concatenate_array_to_h5_dataset(path,group,dataset,data,concatenate = True):
    """_summary_

    Args:
        path (_type_): _description_
        group (_type_): _description_
        dataset (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    h5file, dataset_exists, group_dataset = open_or_create_h5_dataset(path,group,dataset,data)
    
    if dataset_exists and concatenate:
        array = h5file[group_dataset] # save current array
        del h5file[group_dataset] # delete 
        new_data = np.concatenate((array,data),axis=0) # create new array
        h5file[group].create_dataset(dataset,data=new_data) # add new array to h5 file
    
    h5file.close()
    
def combine_volume(*args):                                                                                      
    shape = np.load(args[0]).shape
    data_0 = np.load(args[0])
    volume = np.empty((len(args),*shape),dtype=data_0.dtype) 
    for i, arg in enumerate(args):                                                                                            
        data = np.load(arg)                                                                                     
        volume[i, ...] = data                                                                                   
    return volume

def save_volume_from_parts(input_dict):
    
    print("Combining and saving objects into single file...")
    objects = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="object.npy")[0]
    object = combine_volume(*objects)
    save_variable(input_dict, object,name='object')

    print("Combining and saving probes into single file...")
    probes = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="probe.npy")[0]
    probes = combine_volume(*probes)
    save_variable(input_dict,probes,name='probe')

    print("Combining and saving angles into single file...")
    angles = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="angle.npy")[0]
    angles = combine_volume(*angles)
    save_variable(input_dict,angles,name='angles')

    print("Combining and saving probe positions into single file...")
    positions = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="positions.npy")[0]
    positions = combine_volume(*positions)
    save_variable(input_dict,positions,name='positions')

    print("Combining and saving errors into single file...")
    errors = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="error.npy")[0]
    errors = combine_volume(*errors)
    save_variable(input_dict,errors,name='error',group='log')

    corrected_positions = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="corrected_positions.npy")[0]
    if len(corrected_positions) > 0:
        print("Combining and saving probe final positions into single file...")
        corrected_positions = combine_volume(*corrected_positions)
        save_variable(input_dict,corrected_positions,name='corrected_positions')

    print("Deleting temporary object and probe files...")
    delete_files_if_not_empty_directory(input_dict["temporary_output_recons"])

    save_json_logfile(input_dict) 
    delete_temporary_folders(input_dict)

    return object, probes, angles, positions, errors

def delete_temporary_folders(input_dict):
    if os.path.isdir(input_dict["temporary_output_recons"]): os.rmdir(input_dict["temporary_output_recons"])
    if os.path.isdir(input_dict["temporary_output"]): os.rmdir(input_dict["temporary_output"])

def deploy_visualizer(data,axis=0,type='',title='',cmap='jet',aspect_ratio='',norm="normalize",limits=()):
    """

    data (ndarray): real valued data
    axis (int): slice direction
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm

    import ipywidgets as widgets
    from ipywidgets import fixed

    def get_vol_slice(volume, axis, frame):
        selection = [slice(None)]*3
        selection[axis] = frame
        frame_data = volume[(*selection,)]
        if type == '':
            pass
        elif type == 'real':
            frame_data = np.real(frame_data)
        elif type == 'imag':
            frame_data = np.imag(frame_data)
        elif type == 'amplitude':
            frame_data = np.abs(frame_data)
        elif type == 'phase':
            frame_data = np.angle(frame_data)
        return frame_data

    def get_colornorm(frame, limits, norm):
        if norm == None:
            return None
        elif norm == "normalize":
            if limits:
                return colors.Normalize(vmin=limits[0], vmax=limits[1])
            else:
                return colors.Normalize(vmin=frame.min(), vmax=frame.max())
        elif norm == "LogNorm":
            return colors.LogNorm()
        else:
            raise ValueError("Invalid norm value: {}".format(norm))


    output = widgets.Output()
    with output:
        volume_slice = get_vol_slice(data, axis=0, frame=0)
        figure, ax = plt.subplots(dpi=100)
        ax.imshow(volume_slice, cmap='gray')
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False
        colorbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=colors.SymLogNorm(1,vmin=np.min(volume_slice),vmax=np.max(volume_slice)),
                cmap=cmap))
        plt.show()


    def update_imshow(figure,subplot,frame_number,axis=0,title="",cmap='gray',norm=None,aspect_ratio=''):
        subplot.clear()

        volume_slice = get_vol_slice(data, axis, frame_number)
        colornorm = get_colornorm(volume_slice, limits, norm)
        im = subplot.imshow(volume_slice, cmap=cmap, norm=colornorm)

        if title != "":
            subplot.set_title(f'{title}')
        figure.canvas.draw_idle()

        if aspect_ratio != '':
            subplot.set_aspect(aspect_ratio)

        colorbar.update_normal(im)


    slider_layout = widgets.Layout(width='25%')
    selection_slider = widgets.IntSlider(min=0,max=data.shape[axis],step=1, description="Slice",value=0,layout=slider_layout)

    selection_slider.max, selection_slider.value = data.shape[axis] - 1, data.shape[axis]//2
    widgets.interactive_output(update_imshow, {'figure':fixed(figure),'title':fixed(title),'subplot':fixed(ax),'axis':fixed(axis), 'cmap':fixed(cmap), 'norm':fixed(norm),'aspect_ratio':fixed(aspect_ratio),'frame_number': selection_slider})
    box = widgets.VBox([selection_slider,output])

    return box

def visualize_magnitude_and_phase(data,axis=0,cmap='jet',aspect_ratio=''):

    import numpy as np
    import matplotlib.pyplot as plt

    import ipywidgets as widgets
    from ipywidgets import fixed
    
    def update_imshow(volume,figure,ax1,ax2,frame_number,axis=0,cmap='jet',aspect_ratio=''):
        
        ax1.clear()
        ax2.clear()        
        
        if cmap=='gray':
            cmap1, cmap2 = 'gray', 'gray'
        else:
            cmap1, cmap2 = 'viridis', 'hsv'

        if axis == 0:
            ax11 = ax1.imshow(np.abs(volume[frame_number,:,:]),cmap=cmap1)
            ax22 = ax2.imshow(np.angle(volume[frame_number,:,:]),cmap=cmap2)
        elif axis == 1:
            ax11 = ax1.imshow(np.abs(volume[:,frame_number,:]),cmap=cmap1)
            ax22 = ax2.imshow(np.angle(volume[:,frame_number,:]),cmap=cmap2)
        elif axis == 2:
            ax11 = ax1.imshow(np.abs(volume[:,:,frame_number]),cmap=cmap1)
            ax22 = ax2.imshow(np.angle(volume[:,:frame_number]),cmap=cmap2)
            
        ax11.set_title(f'Magnitude')
        ax22.set_title(f'Phase') 
        figure.canvas.draw_idle()

        if aspect_ratio != '':
            ax1.set_aspect(aspect_ratio)
            ax2.set_aspect(aspect_ratio)

    output = widgets.Output()
    
    with output:
        figure, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5),dpi=100)
        ax1.imshow(np.abs(data[0,:,:]),cmap='viridis')
        ax2.imshow(np.angle(data[0,:,:]),cmap='hsv')
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False
        plt.show()   

    slider_layout = widgets.Layout(width='25%')
    selection_slider = widgets.IntSlider(min=0,max=data.shape[axis],step=1, description="Slice",value=0,layout=slider_layout)

    selection_slider.max, selection_slider.value = data.shape[axis] - 1, data.shape[axis]//2
    widgets.interactive_output(update_imshow, {'volume':fixed(data),'figure':fixed(figure),'ax1':fixed(ax1),'ax2':fixed(ax2),'axis':fixed(axis), 'cmap':fixed(cmap),'aspect_ratio':fixed(aspect_ratio),'frame_number': selection_slider})    
    box = widgets.VBox([selection_slider,output])
    return box

def plot_probe_modes(probe,contrast='phase',frame=0):
    if contrast == 'phase':
        probe_plot = np.angle(probe)[frame]
    else:
        probe_plot = np.abs(probe)[frame]
    
    fig, ax = plt.subplots(1,probe.shape[1],figsize=(15,3),dpi=150)
    
    for i, ax in enumerate(ax):
        ax.imshow(probe_plot[i],cmap='jet')
        ax.set_title(f'Mode {i}')

def plot_volume_histogram(volume,bins=100):
    
    maximum = np.max(volume)
    minimum = np.min(volume)
    mean    = np.mean(volume)
    stddev  = np.std(volume)
    
    fig, ax = plt.subplots()
    ax.hist(volume.flatten(),bins=bins)
    ax.grid()
    ax.set_title(f'Max={maximum:.2f}   Min={minimum:.2f}   Mean={mean:.2f}   StdDev={stddev:.2f}')


def convert_complex_to_RGB(ComplexImg,bias=0.01):
        
    def MakeRGB(Amps,Phases,bias=0): 	# Make RGB image from amplitude and phase
        from matplotlib.colors import hsv_to_rgb
        """ Both amplitude (value) and phase (hue) should be adjusted to range [0,1] """ 
        HSV = np.zeros((Amps.shape[0],Amps.shape[1],3),dtype=np.float32)
        normalizer = (1.0-bias)/Amps.max()
        HSV[:,:,0] = Phases[:,:]
        HSV[:,:,1] = 1
        HSV[:,:,2] = Amps[:,:]*normalizer + bias
        return hsv_to_rgb(HSV)

    def SplitComplex(ComplexImg):
        Phases = np.angle(ComplexImg)	# Phases in range [-pi,pi]
        Phases = Phases*0.5/np.pi + 0.5
        Amps = np.absolute(ComplexImg)
        return Amps,Phases

    Amps,Phases = SplitComplex(ComplexImg)
    return MakeRGB(Amps,Phases,bias)
