import numpy as np
import os 
from skimage.restoration import unwrap_phase
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

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
    """ Finds a best fit plane inside a masked region of image and subtracts. 
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



