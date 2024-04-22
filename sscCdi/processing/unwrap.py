import numpy as np
import os 
from tqdm import tqdm
import scipy
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from skimage.restoration import unwrap_phase

######################### UNWRAP #################################################

def unwrap_in_parallel(sinogram,processes=0):
    """
    Unwraps phase of sinogram slices in parallel using a certain number of processes
    """

    n_frames = sinogram.shape[0]

    if processes == 0:
        processes = min(os.cpu_count(),32)

    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        unwrapped_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(unwrap_phase,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            # if counter % 100 == 0: print('Populating results matrix...',counter)
            unwrapped_sinogram[counter,:,:] = result

    return unwrapped_sinogram


def unwrap_sinogram(sinogram,unwrapped_savepath=''):
    """ Calls unwrapping algorithms in multiple processes for the sinogram frames
    """
    start = time.time()

    sinogram = unwrap_in_parallel(sinogram)
    if unwrapped_savepath != '':
        print('Saving unwrapped volume..')
        open_or_create_h5_dataset(unwrapped_savepath,'entry','data',sinogram,create_group=True)
        print('Saved unwrapped sinogram at: ',unwrapped_savepath)
 
    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return sinogram

def plane_fit_inside_mask(img, mask, epsilon = 1e-3):
    """
    Fits a plane to the 3D points inside a binary mask region in an image.

    This function computes the parameters of a plane that best fits the 3D points
    inside a binary mask region in an image. The mask defines the region of interest,
    and the image provides the values of each point within the mask.

    Args:
        img (numpy.ndarray): The input image containing intensity values.
        mask (numpy.ndarray): A binary mask defining the region of interest.
                             Points with non-zero values are considered for plane fitting.
        epsilon (float, optional): Regularization parameter to stabilize matrix inversion.
                                  Set to zero for no regularization. Default is 1e-3.

    Returns:
        tuple: A tuple containing the parameters of the fitted plane (a, b, c),
               where 'a', 'b', and 'c' represent the coefficients of the plane's equation:
               ax + by + c = z.
    """

    xy = np.argwhere( mask > 0)
    n = len(xy)
    y = xy[:,0].reshape([n,1])
    x = xy[:,1].reshape([n,1])
    F = np.array([ img[y[k],x[k]] for k in range(n) ]).reshape([n,1])
    mat = np.zeros([3,3])
    vec = np.zeros([3,1])
   
    mat[0,0] = (x*x).sum()
    mat[0,1] = (x*y).sum()
    mat[0,2] = (x).sum()
    mat[1,0] = mat[0,1]
    mat[1,1] = (y*y).sum()
    mat[1,2] = (y).sum()
    mat[2,0] = mat[0,2]
    mat[2,1] = mat[1,2]
    mat[2,2] = n
    vec[0,0] = (x*F).sum()
    vec[1,0] = (y*F).sum()
    vec[2,0] = (F).sum()
    eye = np.eye(mat.shape[0])

    if epsilon > 0: # with regularization
        abc = np.dot( np.linalg.inv(mat + epsilon * eye), vec).flatten() 
    else: # without regularization
        abc = np.dot( np.linalg.inv(mat), vec).flatten()
   
    a = abc[0]
    b = abc[1]
    c = abc[2]
   
    return a,b,c

def remove_phase_gradient(img, mask, loop_count_limit=5, epsilon = 1e-3):
    """ 
    Finds a best fit plane inside a masked region of image and subtracts. 
    This process is repeated "loop_count_limit" times or until angular coefficients a,b are smaller than 1e-8

    Args:
        img (numpy.ndarray): 2D image to remove a gradient
        mask (numpy.ndarray): binary mask to indicate region of interest to extract. 
                            Points with non-zero values are considered for plane fitting.
        loop_count_limit (int, optional): Number of times to extract plane fit. Defaults to 5.
        epsilon (float, optional): Regularization parameter to stabilize matrix inversion.
                                  Set to zero for no regularization. Default is 1e-3.

    Returns:
        img (numpy.ndarray): image with subtracted phase gradient
    """

    row   = img.shape[0]
    col   = img.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))

    a = b = c = 1e9
    counter = 0
    while np.abs(a) > 1e-8 or np.abs(b) > 1e-8 or counter < loop_count_limit:
        a,b,c = plane_fit_inside_mask( img, mask, epsilon )
        img = img - ( a*XX + b*YY + c ) # subtract plane from whole image
        counter += 1
    
    return img

######################### EQUALIZATION #################################################

def equalize_sinogram(dic, sinogram,save=True):
    """ Calls equalization algorithms in multiple processes for sinogram frames

    Args:
        dic (dict): dictionary of inputs  
            keys:
                "bad_frames_before_equalization"
                "equalized_sinogram_filepath"
        object (array): frames to be equalized

    """
    start = time.time()

    equalized_sinogram = equalize_frames_parallel(sinogram,dic)
    if save:
        print('Saving equalized sinogran...')
        open_or_create_h5_dataset(dic["equalized_sinogram_filepath"],'entry','data',equalized_sinogram,create_group=True)
        print('Saved equalized object at: ',dic["equalized_sinogram_filepath"])

    print(f'Time elapsed: {time.time() - start:.2f} s' )
    return equalized_sinogram


def equalize_frame(dic,frame):
    """ Performs a series of processing steps over a 2D array, namely:

        1) Removes a gradient (i.e. the phase ramp) for the image as a whole
        2) Makes any NaN values null
        3) Removes outlier values above/below a certain sigma
        4) Removes the global offset of the array, making the smallest value null
        5) Removes a local offset of the array, subtracting the mean value of a desired region from the entire array

    Args:
        dic (dict): dictionary of inputs
            keys:
                "equalize_invert": boolean
                "equalize_remove_phase_gradient": boolean
                "equalize_ROI"
                "equalize_remove_phase_gradient_iterations"
                "equalize_local_offset"
                "equalize_set_min_max"
                "equalize_non_negative"
        frame (array): 2D image/frame to be equalized

    Returns:
        frame (array): equalized frame
    """

    if dic["equalize_invert"] == True:
        frame = -frame

    # Remove phase ramp
    if dic["equalize_remove_phase_gradient"] == True:
        if dic["equalize_ROI"] == []:
            mask = np.ones_like(frame,dtype=bool)
        else:
            mask = np.zeros_like(frame,dtype=bool)
            mask[dic["equalize_ROI"][0]:dic["equalize_ROI"][1],dic["equalize_ROI"][2]:dic["equalize_ROI"][3]] = True
        
        if "equalize_remove_phase_gradient_iterations" in dic:
            iterations = dic["equalize_remove_phase_gradient_iterations"]
        else:
            iterations = 5 
        frame = remove_phase_gradient(frame, mask,loop_count_limit=iterations)

    # Check for NaNs
    whereNaN = np.isnan(frame)
    if whereNaN.any():
        print("NaN values found in frame after removing gradient. Removing them!")
        frame = np.where(whereNaN,0,frame)

    # OBSOLETE: Remove outliers
    # if remove_outlier != 0:
    #     frame = remove_outliers(frame,remove_outlier)

    # OBSOLETE: Remove global offset
    # if remove_global_offset:
    #     frame -= np.min(frame)

    # Remove average offset from specific region
    if dic["equalize_local_offset"]:
        mean = np.mean(frame[dic["equalize_ROI"][0]:dic["equalize_ROI"][1],dic["equalize_ROI"][2]:dic["equalize_ROI"][3]])
        frame -= mean

    if dic["equalize_set_min_max"] != []:
        frame = np.where(frame<dic["equalize_set_min_max"][0],0,np.where(frame>dic["equalize_set_min_max"][1],0,frame))


    if dic["equalize_non_negative"]:    
        frame = np.where(frame<0,0,frame) # put remaining negative values to zero

    return frame

def equalize_frames_parallel(sinogram,dic):
    """ Calls function equalize_frame in parallel at multiple threads for each frameo of the sinogram

    Args:
        sinogram (array): sinogram to be equalized
        dic (dict): dictionary of inputs
            keys:
                "CPUs": number of CPUs

    Returns:
        equalized_sinogram: equalized sinogram
    """

    minimum, maximum, mean, std = np.min(sinogram), np.max(sinogram), np.mean(sinogram), np.std(sinogram)

    # Remove NaNs
    whereNaN = np.isnan(sinogram)
    if whereNaN.any():
        print("NaN values found in unwrapped sinogram. Removing them!")
        sinogram = np.where(whereNaN,0,sinogram)

    # Call parallel equalization
    equalize_frame_partial = partial(equalize_frame, dic)
    print('Sinogram shape to equalize: ', sinogram.shape)

    n_frames = sinogram.shape[0]

    processes = dic["CPUs"]
    print(f'Using {processes} parallel processes')

    with ProcessPoolExecutor(max_workers=processes) as executor:
        equalized_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(equalize_frame_partial,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            # minimum, maximum, mean, std = np.min(result), np.max(result), np.mean(result), np.std(result)
            equalized_sinogram[counter,:,:] = result

    minimum1, maximum1, mean1, std1 = np.min(equalized_sinogram), np.max(equalized_sinogram), np.mean(equalized_sinogram), np.std(equalized_sinogram)
    print(f'Min \t Mean-3*sigma \t Mean \t Mean+3*sigma \t Max ')
    print(f'Old {minimum:.2f}, {mean-3*std:.2f}, {mean:.2f}, {mean+3*std:.2f},{maximum:.2f}')
    print(f'New: {minimum1:.2f}, {mean1-3*std1:.2f},{mean1:.2f}, {mean1+3*std1:.2f},{maximum1:.2f}')

    return equalized_sinogram

def equalize_scipy_optimization(mask,img,initial_guess=(1,1,1),method='Nelder-Mead',max_iter = 1,stop_criteria=(1e-5,1e-5,1e-2)):
    
    def equalization_cost_function(params,img,mask,x,y):
        a, b, c = params
        return np.linalg.norm(mask*(img- (a*x+b*y+c)))**2
    
    if img.shape != mask.shape:
        sys.exit('Image and Mask do not have the same shape')

    y, x = np.indices(img.shape)
    
    for i in range(0,max_iter):
    
        result = scipy.optimize.minimize(equalization_cost_function, initial_guess, args=(img,mask,x,y),method=method)

        success = result.success
        if success == False:
            print('Convergence failed. Scipy.minimize message: ',result.message)
        else:
            # print('Minization done')
            pass

        a,b,c = result.x
        plane = a*x+b*y+c
        img = img - plane
    
    return img, plane, (a,b,c)
    
def equalize_scipy_optimization_parallel(sinogram,mask,initial_guess=(0,0,0),method='Nelder-Mead',max_iter = 1,stop_criteria=(1e-5,1e-5,1e-2),processes=10):

    equalize_scipy_optimization_partial = partial(equalize_scipy_optimization, mask,initial_guess=initial_guess,method=method,max_iter = max_iter,stop_criteria=stop_criteria)
    n_frames = sinogram.shape[0]
    equalized_sinogram = np.empty_like(sinogram)
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(tqdm(executor.map(equalize_scipy_optimization_partial,sinogram),total=n_frames,desc='Equalizing frames'))
        for counter, result in tqdm(enumerate(results),desc="Populating result matrix"):
            equalized_sinogram[counter,:,:] = result[0]
            
    return equalized_sinogram

