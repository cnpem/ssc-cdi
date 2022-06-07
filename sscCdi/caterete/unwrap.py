import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase
import numpy

from IPython.display import display
from ipywidgets import *
from skimage.io import imsave

from functools import partial

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def RemoveGrad( img, mask ):
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

def RemoveZernike(img,mask):
    """ Giovanni's function for removing zernikes. Not well understood yet.

    Args:
        img 
        mask 

    """    
    hs1 = img.shape[-1]//2
    hs2 = img.shape[-2]//2
    #hs = 256
    xx,yy = np.meshgrid(np.arange(-hs1,hs1) / float(hs1),np.arange(-hs2,hs2) / float(hs2))
    #img = 3*xx**2 + 2*yy**2 + 4*xx + 7 * yy - 1

    xxm = xx[mask]
    yym = yy[mask]

    dLdA2 = np.average(xxm**4)
    dLdB2 = np.average(yym**4)
    dLdC2 = np.average(xxm**2*yym**2)
    dLdAB = np.average(xxm**2*yym**2)
    dLdAC = np.average(xxm**3*yym)
    dLdBC = np.average(xxm*yym**3)

    dLdD2 = np.average(xxm**2)
    dLdE2 = np.average(yym**2)
    dLdDE = np.average(xxm*yym)

    dLdAD = np.average(xxm**3)
    dLdBE = np.average(yym**3)
    dLdAE = np.average(xxm**2*yym)
    dLdBD = np.average(xxm*yym**2)

    dLdAF = np.average(xxm**2)
    dLdBF = np.average(yym**2)
    dLdDF = np.average(xxm)
    dLdEF = np.average(yym)
    dLdF2 = 1

    dLdCD = np.average(xxm**2*yym)
    dLdCE = np.average(xxm*yym**2)
    dLdCF = np.average(xxm*yym)

    mat = np.asarray([
        [dLdA2,dLdAB,dLdAC,dLdAD,dLdAE,dLdAF],
        [dLdAB,dLdB2,dLdBC,dLdBD,dLdBE,dLdBF],
        [dLdAC,dLdBC,dLdC2,dLdCD,dLdCE,dLdCF],
        [dLdAD,dLdBD,dLdCD,dLdD2,dLdDE,dLdDF],
        [dLdAE,dLdBE,dLdCE,dLdDE,dLdE2,dLdEF],
        [dLdAF,dLdBF,dLdCF,dLdDF,dLdEF,dLdF2]
        ])
    inv = np.linalg.inv(mat)

    #Res = A*xxm**2 + B*yym**2 + D*xxm + E*yym + F - img[mask]
    Res = img[mask]

    dLdA = np.average(Res*xxm**2)
    dLdB = np.average(Res*yym**2)
    dLdC = np.average(Res*xxm*yym)
    dLdD = np.average(Res*xxm)
    dLdE = np.average(Res*yym)
    dLdF = np.average(Res)

    gradient = -np.matmul(inv,[dLdA,dLdB,dLdC,dLdD,dLdE,dLdF])
    #print(gradient)
    return img + gradient[0]*xx**2 + gradient[1]*yy**2 + gradient[2]*xx*yy + gradient[3]*xx + gradient[4]*yy + gradient[5]
    #Show([img + gradient[0]*xx**2 + gradient[1]*yy**2 + gradient[2]*xx + gradient[3]*yy + gradient[4]])

def RemoveGradOld(img,mask):
    """ Giovanni's function for removing a gradient.

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

def unwrap_in_parallel(sinogram,iterations=0,non_negativity=True,remove_gradient = True):

    n_frames = sinogram.shape[0]

    print('Sinogram shape to unwrap: ', sinogram.shape)

    phase_unwrap_partial = partial(phase_unwrap,iterations=iterations,non_negativity=non_negativity,remove_gradient = remove_gradient)

    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        unwrapped_sinogram = np.empty_like(sinogram)
        results = list(tqdm(executor.map(phase_unwrap_partial,[sinogram[i,:,:] for i in range(n_frames)]),total=n_frames))
        for counter, result in enumerate(results):
            if counter % 25 == 0: print('Populating results matrix...',counter)
            unwrapped_sinogram[counter,:,:] = result

    return unwrapped_sinogram

def phase_unwrap(img,iterations=0,non_negativity=True,remove_gradient = True):
    """ Function for phase unwrapping reconstructed object. 

    Args:
        img : 2d image data
        iterations : number of iterations for gradient removal. If 0, no removal happens.
        non_negativity (bool, optional): boolean to make all negative values zero.
        remove_gradient (bool, optional): _description_. boolean to select gradient removal of final unwrapped image

    Returns:
        zernike: phase unwrapped image
    """

    zernike = unwrap_phase(img)
    
    mask = zernike < 0
    for j in range(iterations):
        zernike = RemoveGrad(zernike,mask=mask)
        mask = abs(zernike) < 2**-j

    if non_negativity == True:    
        zernike[zernike<0] = 0

    if remove_gradient == True:
        zernike = RemoveGrad(zernike,mask=mask)
    return zernike

def phase_unwrap_new(new):
    new = image[xmin:xmax, ymin:ymax] 
    new = new + abs(new.min()) # wanted masked region is where new > 0
    new = RemoveGrad(new, new > 0 )
    new = unwrap_phase(new)
    return new

def unwrapInterface(recon_folder,recon_filename,frame_number):

    path_to_recon = os.path.join(recon_folder,recon_filename)
    image = numpy.load(path_to_recon)[frame_number]
    
    vsize, hsize = image.shape[0], image.shape[1]

    fig = plt.figure(figsize=(10,5))
    ax1  = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax2  = fig.add_subplot(1, 2, 2)
    ax2.imshow(image)
    ax1.set_title('Original image')
    ax2.set_title('Unwrapped image')

    def update(top, bottom,left,right):
        ax1.clear()
        ax1.set_title('Original image')
        ax1.imshow(image[top:-bottom,left:-right])
        fig.canvas.draw_idle()
        return top, bottom,left,right

    def on_button_clicked(b):
        global unwrapped_image 
        unwrapped_image = phase_unwrap(image[top.value:-bottom.value,left.value:-right.value],iterations.value,non_negativity=non_negativity_checkbox,remove_gradient = remove_gradient_checkbox)
        ax2.imshow(unwrapped_image)
        return unwrapped_image

    def savefig_button(b):
        fig.savefig('figure.png',dpi=300)
        numpy.save('unwrapped.npy',unwrapped_image)
        imsave('unwrapped.tif',unwrapped_image)


    # Sliders     
    top    = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Top')
    bottom = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Bottom')
    left   = widgets.IntSlider(min=1, max=hsize//2, step=1, value = 1,description='Left')
    right  = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Right')
    ui = widgets.HBox([top, bottom, left, right])    
    out = widgets.interactive_output(update, {'top': top, 'bottom': bottom, 'left': left, 'right': right})

    # Button 1
    button = widgets.Button(description="Unwrap!")
    output = widgets.Output()
    button.on_click(on_button_clicked)

    # Button 2
    save_button = widgets.Button(description="Save figure")
    output2 = widgets.Output()
    save_button.on_click(savefig_button)

    # CheckBox 1 
    non_negativity_checkbox = widgets.Checkbox(value=False,description='Non-negativity')
    # CheckBox 2
    remove_gradient_checkbox= widgets.Checkbox(value=False,description='Remove Gradient')

    # Input 1
    iterations = widgets.BoundedIntText(value=0,  min=0,step=1, description='Iterations:', disabled=False)

    # DISPLAY
    display(ui, out)
    display(iterations)
    display(non_negativity_checkbox)
    display(remove_gradient_checkbox)
    display(button)
    display(save_button)