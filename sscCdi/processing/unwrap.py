import numpy as np
import numpy
import os 

from skimage.restoration import unwrap_phase

from functools import partial

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def RemoveGrad(img,mask):
    """ Giovanni Baraldi function for removing a gradient.

    Args:
        img 
        mask 
    """  
    
    hs1 = img.shape[-1]//2
    hs2 = img.shape[-2]//2 
    if img.shape[-1] % 2 == 0 and img.shape[-2] % 2 == 0:  
        xx,yy = np.meshgrid(np.arange(-hs1,hs1) / float(hs1),np.arange(-hs2,hs2) / float(hs2))
    elif img.shape[-1] % 2 == 0 and img.shape[-2] % 2 != 0:
        xx,yy = np.meshgrid(np.arange(-hs1,hs1) / float(hs1),np.arange(-hs2-1,hs2) / float(hs2))
    elif img.shape[-1] % 2 != 0 and img.shape[-2] % 2 == 0:
        xx,yy = np.meshgrid(np.arange(-hs1-1,hs1) / float(hs1),np.arange(-hs2,hs2) / float(hs2))
    else:
        xx,yy = np.meshgrid(np.arange(-hs1-1,hs1) / float(hs1),np.arange(-hs2-1,hs2) / float(hs2))
    #img = 3*xx**2 + 2*yy**2 + 4*xx + 7 * yy - 1
    
    xxm = xx[mask]
    yym = yy[mask]
    dLdD2 = np.average(xxm**2)
    dLdE2 = np.average(yym**2)
    dLdDE = np.average(xxm*yym)

    dLdDF = np.average(xxm)
    dLdEF = np.average(yym)
    dLdF2 = 1

    mat = np.asarray([
        [dLdD2,dLdDE,dLdDF],
        [dLdDE,dLdE2,dLdEF],
        [dLdDF,dLdEF,dLdF2]
        ])
    inv = np.linalg.inv(mat)

    #Res = A*xxm**2 + B*yym**2 + D*xxm + E*yym + F - img[mask]
    Res = img[mask]

    dLdD = np.average(Res*xxm)
    dLdE = np.average(Res*yym)
    dLdF = np.average(Res)

    gradient = -np.matmul(inv,[dLdD,dLdE,dLdF])
    return img + gradient[0]*xx + gradient[1]*yy + gradient[2]
    #Show([img + gradient[0]*xx**2 + gradient[1]*yy**2 + gradient[2]*xx + gradient[3]*yy + gradient[4]])

def RemoveGrad_new( img, mask ):
    xy = numpy.argwhere( mask > 0)
    n = len(xy)
    y = xy[:,0].reshape([n,1])
    x = xy[:,1].reshape([n,1])
    F = numpy.array([ img[y[k],x[k]] for k in range(n) ]).reshape([n,1])
    mat = numpy.zeros([3,3])
    vec = numpy.zeros([3,1])
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
    eye = numpy.eye(mat.shape[0])
    eps = 1e-3 # valor tirado do *
    if 1: # com regularização
        abc = numpy.dot( numpy.linalg.inv(mat + eps * eye), vec).flatten() 
    else: # sem regularização
        abc = numpy.dot( numpy.linalg.inv(mat), vec).flatten()
    a = abc[0]
    b = abc[1]
    c = abc[2]
    new   = numpy.zeros(img.shape)
    row   = new.shape[0]
    col   = new.shape[1]
    XX,YY = numpy.meshgrid(numpy.arange(col),numpy.arange(row))
    new[y, x] = img[ y, x] - ( a*XX[y,x] + b*YY[y,x] + c )
    #for k in range(n):
    #    new[y[k], x[k]] = img[ y[k], x[k]] - ( a*x[k] + b*y[k] + c )
    return new

def unwrap_in_parallel(sinogram,remove_gradient):

    n_frames = sinogram.shape[0]

    print('Sinogram shape to unwrap: ', sinogram.shape)
    if remove_gradient == []:
        mask = None
    else:
        mask = np.zeros(sinogram[0].shape)
        mask[remove_gradient[0]:remove_gradient[1],remove_gradient[2]:remove_gradient[3]] = 1
    phase_unwrap_partial = partial(phase_unwrap,mask)

    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        unwrapped_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(phase_unwrap_partial,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            if counter % 100 == 0: print('Populating results matrix...',counter)
            unwrapped_sinogram[counter,:,:] = result

    return unwrapped_sinogram

def phase_unwrap(img,mask=None):
    if mask != None:
        img = RemoveGrad(img,mask=mask)
    return unwrap_phase(img)


