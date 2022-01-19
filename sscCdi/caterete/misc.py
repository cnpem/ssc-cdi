import numpy 
import matplotlib.pyplot as plt

def miqueles_colormap(img):

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
    """
    Parameters
    ----------
    data_directory : string
        path to the directory containing files you want to list
    look_for_extension : string, optional
        string containing the file termination string, e.g. '.datx' or '.txt'. The default is "".

    Returns
    -------
    filepaths, filenames.
    two lists, one contaning the complete path of all files. the seconds containing list of all file names
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

    filepaths = list( filepaths[i] for i in frames)
    filenames = list( filenames[i] for i in frames)

    return filepaths, filenames

def save_json_logfile(path,jason):
    import json, os
    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    
    if jason["3D_Acquisition_Folders"] != [""]:
        name = jason["3D_Acquisition_Folders"][0]
    else:
        name = jason[ "SingleMeasurement"]
    
    name = dt_string + "_" + name.split('.')[0]+".json"

    filepath = os.path.join(path,name)
    file = open(filepath,"w")
    file.write(json.dumps(jason))
    file.close()
    