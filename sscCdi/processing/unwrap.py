import numpy as np
import os 
from skimage.restoration import unwrap_phase
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def plane_fit_inside_mask( img, mask, epsilon = 1e-3, regularization=True ):
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

    if regularization: # with regularization
        abc = np.dot( np.linalg.inv(mat + epsilon * eye), vec).flatten() 
    else: # without regularization
        abc = np.dot( np.linalg.inv(mat), vec).flatten()
   
    a = abc[0]
    b = abc[1]
    c = abc[2]
   
    return a,b,c

def remove_phase_gradient(img, mask,loop_count_limit=5,epsilon = 1e-3, regularization=True):

    row   = img.shape[0]
    col   = img.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))

    a = b = c = 1e9
    counter = 0
    while np.abs(a) > 1e-8 or np.abs(b) > 1e-8 or counter < loop_count_limit:
        a,b,c = plane_fit_inside_mask( img, mask, epsilon, regularization )
        img = img - ( a*XX + b*YY + c ) # subtract plane from whole image
        counter += 1
    return img

def get_best_plane_fit_inside_mask(mask2, loop_count_limit = 5 ):

    def plane(variables,u,v,a):
        Xmesh,Ymesh = variables
        return np.ravel(u*Xmesh+v*Ymesh+a)

    new   = np.zeros(frame.shape)
    row   = new.shape[0]
    col   = new.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))

    a = b = c = 1e9
    counter = 0
    while np.abs(a) > 1e-8 or np.abs(b) > 1e-8 or counter < loop_count_limit:
        grad_removed, (a,b,c) = remove_phase_gradient(frame,mask2)
        plane_fit = plane((XX,YY),a,b,c).reshape(XX.shape)
        frame = frame - plane_fit
        counter += 1
    return frame


def unwrap_in_parallel(sinogram):

    n_frames = sinogram.shape[0]

    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        unwrapped_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(unwrap_phase,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            # if counter % 100 == 0: print('Populating results matrix...',counter)
            unwrapped_sinogram[counter,:,:] = result

    return unwrapped_sinogram

def unwrap_in_sequence(sinogram, remove_gradient):
    if remove_gradient == []:
        mask = None
    else:
        mask = np.zeros(sinogram[0].shape)
        mask[remove_gradient[0]:remove_gradient[1],remove_gradient[2]:remove_gradient[3]] = 1
    
    unwrapped_sinogram = np.empty_like(sinogram)
    for i in range(sinogram.shape[0]):
        if i%25==0: print(f"Unwrapping frame {i}")
        unwrapped_sinogram[i] = unwrap_phase(sinogram[i],mask)

    return unwrapped_sinogram





