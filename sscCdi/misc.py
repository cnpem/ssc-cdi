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

def select_specific_angles(frames,filepaths,filenames):
    """ Function to filter lists, keeping only those with a certain frame number in the string. This is used to select only the desired frames in a 3D ptychography.

    Args:
        frames : list of frames to select
        filepaths (list): inpurt list with full filepaths
        filenames (list): inpurt list with full filenames

    Returns:
        filepaths: filtered filepaths list
        filenames: filtered filenames list
    """    
    filepaths = list( filepaths[i] for i in frames)
    filenames = list( filenames[i] for i in frames)

    return filepaths, filenames

def save_json_logfile(input_dict):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        input_dict (dic): input_dict dictionary
    """    
    import json, os

    path = input_dict["output_path"]

    datetime = input_dict["datetime"]
    name = datetime+".json"

    filepath = os.path.join(path,name)
    file = open(filepath,"w")
    file.write(json.dumps(input_dict,indent=3,sort_keys=True))
    file.close()

    add_to_hdf5_group(input_dict["hdf5_output"],'log','logfile',filepath)

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
    os.system(f"h5clear -s {path}")
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

def create_circular_mask(mask_shape, radius):
    """ All values in pixels """
    center_row, center_col = mask_shape
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    return np.where((Xmesh - center_col//2) ** 2 + (Ymesh - center_row//2) ** 2 <= radius ** 2, 1, 0)

def create_rectangular_mask(mask_shape,center, length_y, length_x=0):
    if length_x == 0: length_x = length_y
    """ All values in pixels """
    center_row, center_col = center
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    mask = np.zeros(*mask_shape)
    mask[center_row-length_y//2:center_row+length_y//2,center_col-length_x//2:center_col+length_x//2] = 1
    return mask 

def create_cross_mask(mask_shape,center, length_y, length_x=0):
    if length_x == 0: length_x = length_y
    """ All values in pixels """
    center_row, center_col = center
    y_array = np.arange(0, mask_shape[0], 1)
    x_array = np.arange(0, mask_shape[1], 1)
    Xmesh, Ymesh = np.meshgrid(x_array, y_array)
    mask = np.zeros(*mask_shape)
    mask[center_row-length_y//2:center_row+length_y//2,:] = 1
    mask[:,center_col-length_x//2:center_col+length_x//2] = 1
    return mask 

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
    
def save_variable(input_dict,variable, flag = 'FLAG'):
    add_to_hdf5_group(input_dict["hdf5_output"],'recon',flag,variable)

def add_to_hdf5_group(path,group,name,data,mode="a"):
    hdf5_output = h5py.File(path, mode)
    hdf5_output[group].create_dataset(name,data=data)
    hdf5_output.close()

def combine_volume(*args):
    shape = np.load(args[0]).shape
    volume = np.empty((0,*shape),dtype=np.float32)
    for arg in args:
        data = np.load(arg)
        data = np.expand_dims(data,axis=0)
        volume = np.concatenate((volume,data),axis=0)
    return volume

def save_volume_from_parts(input_dict):
    
    print("Combining objects into single file...")
    objects = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="object.npy")[0]
    object = combine_volume(*objects)

    print("Combining probes into single file...")
    probes = list_files_in_folder(input_dict["temporary_output_recons"],look_for_extension="probe.npy")[0]
    probes = combine_volume(*probes)

    print("Deleting temporary object and probe files...")
    delete_files_if_not_empty_directory(input_dict["temporary_output_recons"])

    return object, probes 

def delete_temporary_folders(input_dict):
    if os.path.isdir(input_dict["temporary_output_recons"]): os.rmdir(input_dict["temporary_output_recons"])
    if os.path.isdir(input_dict["temporary_output"]): os.rmdir(input_dict["temporary_output"])