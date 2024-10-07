# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import os, h5py

from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from IPython.display import display


def calculate_object_pixel_size(wavelength,detector_distance, detector_pixel_size,n_of_pixels):
    return wavelength * detector_distance / (detector_pixel_size * n_of_pixels)

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

    add_to_hdf5_group(input_dict["hdf5_output"],'metadata','logfile',filepath)

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

def wavelength_meters_from_energy_keV(energy_keV):
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


    
def save_plots(complex_array,title='',path=''):

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
    save_variable(input_dict,errors,name='error',group='metada')

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

def update_slice_visualizer(obj, extent=None, cmap='viridis', vmin=None, vmax=None, norm=None, figsize=(10, 7), title=''):
    if len(obj.shape) == 2:
        obj = np.expand_dims(obj, axis=0)
    
    N, Y, X = obj.shape  # N modes
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, N, width_ratios=[9] * N)

    for i in range(N):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(obj[i], cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, norm=norm)
        fig.colorbar(im, ax=ax, orientation='vertical')
        if extent is None:
            ax.set_ylabel('Y [pxls]')
            ax.set_xlabel('X [pxls]')
        else:
            ax.set_ylabel('Y [m]')
            ax.set_xlabel('X [m]')
        ax.set_title(f'{title}')

    plt.tight_layout()
    plt.show()

def slice_visualizer2(objects, extent=None, plot_type='magnitude', cmap='viridis', use_log_norm=False, figsize=(10, 7), title=''):
    """
    Display an interactive plot to visualize different slices of multiple objects.

    Parameters:
        objects (ndarray): 4D complex-valued array with shape (M, N, Y, X) where M is the number of objects,
                        N is the number of modes, Y and X are the dimensions of each mode.
        extent (tuple): Extent of the plot for x and y axes. Default is None.
        plot_type (str): Type of plot to display. Options are 'real', 'imag', 'amplitude', 'phase', or 'magnitude'. Default is 'magnitude'.
        cmap (str): Colormap for imshow. Default is 'viridis'.
        use_log_norm (bool): Whether to use LogNorm for the norm parameter in imshow. Default is False.
        figsize (tuple): Size of the figure. Default is (10, 7).
        title (str): Title for the imshow plot. Default is ''.
    """
    if np.iscomplexobj(objects):
        if plot_type == 'real' or plot_type == 'r':
            objects = np.real(objects)
        elif plot_type == 'imag' or plot_type == 'imaginary' or plot_type == 'i':
            objects = np.imag(objects)
        elif plot_type == 'amplitude' or plot_type == 'abs' or plot_type == 'magnitude':
            objects = np.abs(objects)
        elif plot_type == 'phase' or plot_type == 'angle':
            objects = np.angle(objects)
        else:
            objects = np.abs(objects)  # Default to magnitude if no valid plot_type is provided

    num_objects = objects.shape[0]
    from ipywidgets import interact, IntSlider, Play, jslink, FloatRangeSlider, HBox
    from IPython.display import display

    vmin = objects.min()
    vmax = objects.max()

    norm = LogNorm(vmin=vmin, vmax=vmax) if use_log_norm else None

    def update_plot(obj_index, value_range):
        vmin, vmax = value_range
        norm = LogNorm(vmin=vmin, vmax=vmax) if use_log_norm else None
        update_slice_visualizer(objects[obj_index], extent, cmap, vmin, vmax, norm, figsize, title)
    
    slider = IntSlider(min=0, max=num_objects-1, step=1, description='Slice #')
    play = Play(value=0, min=0, max=num_objects-1, step=1, interval=500)
    jslink((play, 'value'), (slider, 'value'))

    range_slider = FloatRangeSlider(value=[vmin, vmax], min=vmin, max=vmax, step=(vmax-vmin)/100, description='Color Range')

    display(HBox([play]))
    interact(update_plot, obj_index=slider, value_range=range_slider)

def slice_visualizer(data, axis=0, type='', title='', cmap='gray', aspect_ratio='', norm=None, vmin=None, vmax=None, show_ticks=True):
    """
    Deploy a visualizer for exploring different slices of a 3D volume data array interactively in Jupyter notebooks.

    Parameters:
    - data (ndarray): The 3D volume data to visualize.
    - axis (int, optional): The axis along which the slices will be taken. Default is 0.
    - type (str, optional): The type of data representation in the visualization. It can be one of the following:
        - '': No transformation, displays the data as is.
        - 'real': Displays the real part of complex data.
        - 'imag': Displays the imaginary part of complex data.
        - 'amplitude': Displays the amplitude of complex data.
        - 'phase': Displays the phase of complex data.
    - title (str, optional): The title of the visualization window. Default is an empty string.
    - cmap (str, optional): The colormap used for rendering the slices. Default is 'gray'.
    - aspect_ratio (str, optional): The aspect ratio of the plot. Can be a string (e.g., 'equal', 'auto') or a numeric value.
    - norm (str, optional): The normalization of color scaling, it can be 'normalize', 'LogNorm', or None. Default is None.
    - vmin (float, optional): The minimum data value for normalization. If None, it is automatically calculated from the data.
    - vmax (float, optional): The maximum data value for normalization. If None, it is automatically calculated from the data.
    - show_ticks (bool, optional): Whether to show tick labels on the image. Default is True.

    Returns:
    - box (ipywidgets.VBox): A VBox widget containing the visualization with an interactive slider to control the slice shown.

    The function uses Matplotlib for plotting, ipywidgets for interactivity, and NumPy for data manipulation.
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
        if np.iscomplexobj(frame_data):
            if type in ['real', 'r']:
                frame_data = np.real(frame_data)
            elif type in ['imag', 'imaginary', 'i']:
                frame_data = np.imag(frame_data)
            elif type in ['amplitude', 'abs', 'magnitude']:
                frame_data = np.abs(frame_data)
            elif type in ['phase', 'angle']:
                frame_data = np.angle(frame_data)
            else:
                frame_data = np.abs(frame_data)  # Default to magnitude if no valid type is provided
        return frame_data

    def get_colornorm(frame, vmin, vmax, norm):
        if norm is None:
            return None
        elif norm == "normalize":
            return colors.Normalize(vmin=vmin if vmin is not None else frame.min(), vmax=vmax if vmax is not None else frame.max())
        elif norm == "LogNorm":
            return colors.LogNorm()
        else:
            raise ValueError("Invalid norm value: {}".format(norm))

    output = widgets.Output()
    with output:
        volume_slice = get_vol_slice(data, axis=0, frame=0)
        figure, ax = plt.subplots(dpi=100)
        im = ax.imshow(volume_slice, cmap=cmap, norm=get_colornorm(volume_slice, vmin, vmax, norm))

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        colorbar = plt.colorbar(im, ax=ax)  # Link the colorbar to the imshow object
        plt.show()

    def update_imshow(figure, subplot, frame_number, axis=0, title="", cmap='gray', norm=None, aspect_ratio='', vmin=None, vmax=None, show_ticks=True):
        subplot.clear()

        volume_slice = get_vol_slice(data, axis, frame_number)
        colornorm = get_colornorm(volume_slice, vmin, vmax, norm)
        im = subplot.imshow(volume_slice, cmap=cmap, norm=colornorm,vmin=vmin,vmax=vmax)

        if title != "":
            subplot.set_title(f'{title}')
        figure.canvas.draw_idle()

        if aspect_ratio != '':
            subplot.set_aspect(aspect_ratio)
        
        if not show_ticks:
            subplot.set_xticks([])
            subplot.set_yticks([])

        colorbar.update_normal(im)

    slider_layout = widgets.Layout(width='20%')
    selection_slider = widgets.IntSlider(min=0, max=data.shape[axis], step=1, description="Slice", value=0, layout=slider_layout)
    play = widgets.Play(min=0, max=data.shape[axis]-1, step=1, interval=500, description="Press play", layout=widgets.Layout(width='20%'))

    widgets.jslink((play, 'value'), (selection_slider, 'value'))

    selection_slider.max, selection_slider.value = data.shape[axis] - 1, data.shape[axis] // 2
    widgets.interactive_output(update_imshow, {'figure': fixed(figure), 'title': fixed(title), 'subplot': fixed(ax), 'axis': fixed(axis), 'cmap': fixed(cmap), 'norm': fixed(norm), 'aspect_ratio': fixed(aspect_ratio), 'vmin': fixed(vmin), 'vmax': fixed(vmax), 'show_ticks': fixed(show_ticks), 'frame_number': selection_slider})
    box = widgets.VBox([widgets.VBox([play,selection_slider]), output])

    return box




def amplitude_and_phase_slice_visualizer(data, pixel_values, axis=0, title='', cmap1='viridis', cmap2='hsv', aspect_ratio='', norm="normalize", vmin=None, vmax=None, extent=None):
    """
    Parameters:
        data (ndarray): complex valued data
        pixel_values (ndarray): 2D array of pixel values with shape (N, 2), where the first column is Y and the second column is X
        axis (int): slice direction
        extent (tuple): extent of the images in the format (xmin, xmax, ymin, ymax)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm

    import ipywidgets as widgets
    from ipywidgets import fixed

    def get_vol_slice(volume, axis, frame):
        selection = [slice(None)] * 3
        selection[axis] = frame
        frame_data = volume[tuple(selection)]
        return frame_data

    def get_colornorm(frame, vmin, vmax, norm):
        if norm is None:
            return None
        elif norm == "normalize":
            if vmin is not None or vmax is not None:
                return colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                return colors.Normalize(vmin=frame.min(), vmax=frame.max())
        elif norm == "LogNorm":
            return colors.LogNorm()
        else:
            raise ValueError("Invalid norm value: {}".format(norm))

    def draw_rectangle(ax, pixel_values):
        y_min, y_max = pixel_values[:, 0].min(), pixel_values[:, 0].max()
        x_min, x_max = pixel_values[:, 1].min(), pixel_values[:, 1].max()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    output = widgets.Output()
    with output:
        volume_slice_amplitude = np.abs(get_vol_slice(data, axis, 0))
        volume_slice_phase = np.angle(get_vol_slice(data, axis, 0))

        figure, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))

        im1 = ax1.imshow(volume_slice_amplitude, cmap=cmap1, norm=get_colornorm(volume_slice_amplitude, vmin, vmax, norm), extent=extent)
        ax1.set_title('Amplitude')
        cbar1 = figure.colorbar(im1, ax=ax1, format='%.2e')
        draw_rectangle(ax1, pixel_values)

        im2 = ax2.imshow(volume_slice_phase, cmap=cmap2, norm=get_colornorm(volume_slice_phase, vmin, vmax, norm), extent=extent)
        ax2.set_title('Phase')
        cbar2 = figure.colorbar(im2, ax=ax2, format='%.2e')
        draw_rectangle(ax2, pixel_values)

        figure.canvas.draw_idle()
        plt.show()

    def update_imshow(frame_number, axis=0, cmap1='viridis', cmap2='hsv', aspect_ratio='auto', norm=None, extent=None):
        nonlocal im1, im2, cbar1, cbar2

        ax1.clear()
        ax2.clear()

        volume_slice_amplitude = np.abs(get_vol_slice(data, axis, frame_number))
        volume_slice_phase = np.angle(get_vol_slice(data, axis, frame_number))

        im1 = ax1.imshow(volume_slice_amplitude, cmap=cmap1, norm=get_colornorm(volume_slice_amplitude, vmin, vmax, norm), extent=extent)
        ax1.set_title('Amplitude')
        draw_rectangle(ax1, pixel_values)

        im2 = ax2.imshow(volume_slice_phase, cmap=cmap2, norm=get_colornorm(volume_slice_phase, vmin, vmax, norm), extent=extent)
        ax2.set_title('Phase')
        draw_rectangle(ax2, pixel_values)

        # Update the colorbars
        cbar1.update_normal(im1)
        cbar2.update_normal(im2)

        figure.canvas.draw_idle()

        if aspect_ratio != '':
            ax1.set_aspect(aspect_ratio)
            ax2.set_aspect(aspect_ratio)

    slider_layout = widgets.Layout(width='50%')
    selection_slider = widgets.IntSlider(min=0, max=data.shape[axis] - 1, step=1, description="Slice", value=data.shape[axis] // 2, layout=slider_layout)

    interactive_output = widgets.interactive_output(update_imshow, {
        'frame_number': selection_slider,
        'axis': fixed(axis),
        'cmap1': fixed(cmap1),
        'cmap2': fixed(cmap2),
        'aspect_ratio': fixed(aspect_ratio),
        'norm': fixed(norm),
        'extent': fixed(extent)
    })

    box = widgets.VBox([selection_slider, output])
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

def convert_complex_to_RGB(ComplexImg,bias=0.01):
    """ Convert complex image into RGB image with amplitude encoded by intensity and phase encoded by color

    Args:
        ComplexImg (array): 2d complex array
        bias (float, optional): _description_. Defaults to 0.01.
    """    
        
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

def get_RGB_wheel():
    import matplotlib
    V, H = np.mgrid[0:1:100j, 0:1:300j]
    S = np.ones_like(V)
    HSV = np.dstack((H,S,V))
    RGB = matplotlib.colors.hsv_to_rgb(HSV)
    return RGB, H, S, V

def save_as_hdf5(filepath,data,tag='data'):
    with h5py.File(filepath,'a') as h5file:
        h5file.create_dataset(tag,data=data, dtype=data.dtype)
        print('File created at',filepath)

def create_propagation_video(path_to_probefile,
                             starting_f_value=1e-3,
                             ending_f_value=9e-4,
                             number_of_frames=100,
                             frame_rate=10,
                             mp4=False, 
                             gif=False,
                             jupyter=False):
    
    """ 
    Propagates a probe using the fresnel number to multiple planes and create an animation of the propagation
    #TODO: change this function to create propagation as a function of distance
    """

    probe = np.load(path_to_probefile)[0] # load probe
    
    # delta = -1e-4
    # f1 = [starting_f_value + delta*i for i in range(0,number_of_frames)]
    
    f1 = np.linspace(starting_f_value,ending_f_value,number_of_frames)
    
    # Create list of propagated probes
    b =  [np.sqrt(np.sum([abs(Propagate(a,f1[0]))**2 for a in probe],0))]
    for i in range(1,number_of_frames):
            b += [np.sqrt(np.sum([abs(Propagate(a,f1[i]))**2 for a in probe],0))]
    

    image_list = []
    for j, probe in enumerate(tqdm(b)):
            if jupyter == False:
                animation_fig, subplot = plt.subplots(dpi=300)
                img = subplot.imshow(probe,cmap='jet')#,animated=True)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.set_title(f'f#={f1[j]:.3e}')
            if jupyter == False:
                image_list.append(mplfig_to_npimage(animation_fig))
            else:    
                image_list.append(probe)
            if jupyter == False: plt.close()

    if mp4 or gif:  
        clip = ImageSequenceClip(image_list, fps=frame_rate)
        if mp4:
            clip.write_videofile("propagation.mp4",fps=frame_rate)
        if gif:
            clip.write_gif('propagation.gif', fps=frame_rate)

    return image_list, f1 




def select_real_data(complex_data, real_type='real'):
    if real_type == 'abs' or real_type == 'amplitude':
        return np.abs(complex_data)
    elif real_type == 'phase':
        return np.angle(complex_data)
    elif real_type == 'real':
        return np.real(complex_data)
    elif real_type == 'imaginary':
        return np.imag(complex_data)
    elif real_type == 'amplitude+phase':
        return np.abs(complex_data), np.angle(complex_data)
    else:
        raise ValueError('Select a valid type to plot your complex data: abs, phase, real, imaginary, or amplitude+phase')

def evaluate_shape(volume):
    if len(volume.shape) == 2:
        return np.expand_dims(volume, axis=0)
    elif len(volume.shape) != 3:
        raise ValueError('Your volume shape is wrong. Select a 3D or 2D dataset:', {volume.shape})
    else:
        return volume



def draw_rectangles(array):
    class MultiRectangleDrawer:
        def __init__(self, array):
            self.array = array
            self.mask = np.zeros_like(array, dtype=np.uint8)
            self.fig, self.ax = plt.subplots()
            self.ax.imshow(self.array, cmap='gray')
            self.rect_selector = RectangleSelector(
                self.ax, self.on_select, drawtype='box',
                useblit=True, button=[1],  # only respond to left mouse button
                minspanx=5, minspany=5, spancoords='pixels',
                interactive=False, props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True)
            )
            self.rectangles = []

        def on_select(self, eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            if x1 != x2 and y1 != y2:  # Ensure a valid rectangle
                self.mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 1
                width, height = abs(x2 - x1), abs(y2 - y1)
                rect = Rectangle((min(x1,x2), min(y1,y2)), width, height, fill=False, edgecolor='red', linewidth=2)
                self.ax.add_patch(rect)
                self.rectangles.append(rect)
                self.fig.canvas.draw_idle()
                print(f"Rectangle from ({x1}, {y1}) to ({x2}, {y2})")

        def show(self):
            display(self.fig)

    drawer = MultiRectangleDrawer(array)
    # drawer.show()
    return drawer.mask


def print_h5_tree(name, obj):
    """
    Print the structure of the HDF5 file.
    
    Parameters:
        name : str
            The name of the current group or dataset.
        obj : h5py.Group or h5py.Dataset
            The current group or dataset object.

    """
    if isinstance(obj, h5py.Group):
        print(f"{name}/ (Group)")
        for key in obj.keys():
            print_h5_tree(f"{name}/{key}", obj[key])
    elif isinstance(obj, h5py.Dataset):
        print(f"{name} (Dataset, shape: {obj.shape}, dtype: {obj.dtype})")

def list_h5_file_tree(file_path):
    """
    List the tree structure of an HDF5 file.
    
    Parameters:
        file_path : str
            The path to the HDF5 file.

    """
    with h5py.File(file_path, 'r') as h5file:
        print_h5_tree("/", h5file)

def create_image_from_text(text):
    from matplotlib.backends.backend_agg import FigureCanvas
    fig = plt.Figure(figsize=(2.56, 2.56), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    t = ax.text(0.5, 0.5, text, fontsize=30, fontweight='heavy', ha='center', va='center')
    ax.axis('off')
    canvas.draw()
    img = 1- np.array(canvas.renderer.buffer_rgba())[:, :, 0]/255
    return img