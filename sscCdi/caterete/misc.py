import numpy 
import matplotlib.pyplot as plt

def miqueles_colormap(img):
    """ Definition of a colormap created by Miquele's for better visualizing diffraction patterns.

    Args:
        img : image to get the maximum value for proper colormap definition

    Returns:
        cmap: colormap
        colors: list of colors
        bounds:
        norm:
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
    """ Function to filter lists, keeping only those with a certain frame number in the string. This is used to select only the desired frames in a 3D recon.

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

def save_json_logfile(path,jason):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        jason (dic): jason dictionary
    """    
    import json, os
    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    
    name = jason["Acquisition_Folders"][0]

    name = dt_string + "_" + name.split('.')[0]+".json"

    filepath = os.path.join(path,name)
    file = open(filepath,"w")
    file.write(json.dumps(jason,indent=3,sort_keys=True))
    file.close()
