import numpy as np
import matplotlib.pyplot as plt 
import cupy as cp

from sscPimega import misc 
from matplotlib.colors import LogNorm

import time

t0 = time.perf_counter()

def plotshow(imgs, legend=[],cmap='jet',nlines=1, bLog = False, interpolation='bilinear'): # legend = plot titles
        num = len(imgs)

        for j in range(num):
                if type(cmap) == str:
                        colormap = cmap
                elif len(cmap) == len(imgs):
                        colormap = cmap[j]
                else:
                        colormap = cmap[j//(len(imgs)//nlines)]

                sb = plt.subplot(nlines,(num+nlines-1)//nlines,j+1)
                if type(imgs[j][0,0]) == np.complex64 or type(imgs[j][0,0]) == np.complex128:
                        sb.imshow(CMakeRGB(imgs[j]),cmap='hsv',interpolation=interpolation)
                elif bLog:
                        sb.imshow(np.log(1+np.maximum(imgs[j],-0.1))/np.log(10),cmap=colormap,interpolation=interpolation)
                else:
                        sb.imshow(imgs[j],cmap=colormap,interpolation=interpolation)

                if len(legend)>j:
                        sb.set_title(legend[j])

                sb.set_yticks([])
                sb.set_xticks([])
                sb.set_aspect('equal')

def set_parameters(difpads,jason,rois):
        print('Setting parameters hsize, dx and maxroi ...')
        # Compute half size of diffraction patterns:
        hsize = difpads.shape[-1]//2

        c = 299792458    # Velocity of Light [m/s]
        planck = 4.135667662E-18  # Plank constant [keV*s]
        # Compute/convert pixel size: 
        dx = planck*c/jason['Energy'] * jason['DetDistance'] / (jason['Binning']*jason['RestauredPixelSize']*hsize*2)
        print('\tConverted to pixel size:',dx)
        
        # Get rois minimum and a threshold.
        rois[:,0] -= np.min(rois[:,0])
        rois[:,1] -= np.min(rois[:,1])

        rois[:,0] = 1E-6*rois[:,0]/dx + 32 # why sum 32?
        rois[:,1] = 1E-6*rois[:,1]/dx + 32

        # Compute max rois and size obj (2*hsize+maxroi)
        maxroi = int(np.max(rois))+32 # result needs to be > than 1280 for maxroi to equal 1280 and then obj shape = 2048
        # maxroi = maxroi - maxroi%128

        print('\tmaxroi:',maxroi)
        print("\tObject shape: 2*hsize+maxroi=",2*hsize+maxroi)
        
        return hsize,dx,maxroi, rois

def create_circular_mask(center_row, center_col,radius,mask_shape):
        print('Using manually set circular mask to the diffraction pattern...')
        """ All values in pixels """
        mask = np.zeros(mask_shape)
        y_array = np.arange(0,mask_shape[0],1)
        x_array = np.arange(0,mask_shape[1],1)

        Xmesh, Ymesh = np.meshgrid(x_array,y_array)

        mask = np.where( (Xmesh-center_col)**2 + (Ymesh-center_row)**2 <= radius**2, 1, 0)
        return mask

def pad_and_shift(img,pad,shiftx,shifty):
    
    img2 = np.pad(img,((pad,pad),(pad,pad)))
    
    img2 = np.roll(img2,(shiftx,shifty),axis=(1,0))
    
    img2 = np.where(img2 == 0,-1, img2)

    return img2

def roll_and_crop(img2,shift):
    
    # img2 = np.roll(img2,(shift,0),axis=(1,0))
    # img2 = np.roll(img2,(-shift,0),axis=(1,0))
    # img2 = np.roll(img2,(shift,0),axis=(0,1))
    # img2 = np.roll(img2,(-shift,0),axis=(0,1))

    # crop and pad afterwards to keep the same dimensions as input image
    img2 = img2[shift:-shift,shift:-shift]
    img2 = np.pad(img2,((shift,shift),(shift,shift)))

    return img2

def get_central_region(difpad, center_estimate, radius):
    center_estimate = np.round(center_estimate)
    center_r, center_c = int(center_estimate[0]), int(center_estimate[1])
    region_around_center = difpad[center_r - radius:center_r + radius + 1, center_c - radius:center_c + radius + 1]
    return region_around_center

def refine_center_estimate2(difpad, center_estimate, radius=20):
    from scipy.ndimage import center_of_mass

    """
    Finds a region of radius around center of mass estimate. 
    The position of the max gives a displacement to correct the center of mass estimate
    """

    region_around_center = get_central_region(difpad, center_estimate, int(radius))

    center_displaced = np.where(region_around_center == np.max(region_around_center))
    centerx, centery = center_displaced[0][0], center_displaced[1][0]

    deltaX, deltaY = (region_around_center.shape[0] // 2 - round(centerx)), (
                region_around_center.shape[1] // 2 - round(centery)),

    if 0:  # plot for debugging
        figure, subplot = plt.subplots(1, 2)
        subplot[0].imshow(region_around_center, cmap='jet', norm=LogNorm())
        subplot[0].set_title('Central region preview')
        region_around_center[centerx, centery] = 1e9
        subplot[1].imshow(region_around_center, cmap='jet', norm=LogNorm())

    center = (round(center_estimate[0]) - deltaX, round(center_estimate[1]) - deltaY)

    return center


def get_difpad_center(difpad, refine=True, fit=False, radius=20):
    from scipy.ndimage import center_of_mass
    center_estimate = center_of_mass(difpad)
    if refine:
        center = refine_center_estimate2(difpad, center_estimate, radius=radius)
    else:
        center = (round(center_estimate[0]), round(center_estimate[1]))
    return center

mPIE = True
use_rPIE_update_function = True
correct_momentum = True
simulate_data = True
iterations = 50

# folder = 'datasets/data/'
# # folder = 'datasets/data_cx1404_cy1402/'
# jason = {
# "Energy": 3.8,
# "DetDistance": 27.98,
# "RestauredPixelSize": 55.55E-6,
# 'Binning': 4
# }

# folder = 'datasets/siemens210918/'
# jason = {
# "Energy": 3.8,
# "DetDistance": 13.98,
# "RestauredPixelSize": 55.55E-6,
# 'Binning': 4
# }

# folder = 'datasets/siemens211107/'
# jason = {
# "Energy": 3.8,
# "DetDistance": 6.98,
# "RestauredPixelSize": 55.55E-6,
# 'Binning': 4
# }

folder = 'datasets/siemens210918_16/'
jason = {
"Energy": 3.8,
"DetDistance": 13.98,
"RestauredPixelSize": 55.55E-6,
'Binning': 4
}

pad_shift = False
pad,shiftx,shifty = 20,10,10
shift=1

if 1: # suggested min from paper
    alpha, beta = 0.05, 0.5
    gamma_obj, gamma_probe = 0.1, 0.2
    eta_obj, eta_probe = 0.5, 0.75
    T_lim = 10
else: #suggested max
    alpha, beta = 0.25, 5
    gamma_obj, gamma_probe = 0.5, 1
    eta_obj, eta_probe = 0.9, 0.99
    T_lim = 100 

# alpha, beta = 0.2, 1
# gamma_obj, gamma_probe = 0.15, 0.5
# eta_obj, eta_probe = 0.9, 0.9
# T_lim = 30

if simulate_data == True:
    N = 128
    x = np.linspace(-1,1,N)
    X,Y = np.meshgrid(x,x)

    P = X**2 + Y**2 < 0.9   # Probe

    img = np.load('/ibira/lnls/labs/tepui/proposals/20210062/yuri/yuri_ssc-cdi/other_scripts/xPIE/image.npy') # Load Imagem

    if 1: # Plot initial guesses
        figure, subplot = plt.subplots(1,2,dpi=300)
        subplot[0].imshow(np.abs(P))
        subplot[0].set_title('Probe Model')
        subplot[1].imshow(np.abs(img))
        subplot[1].set_title('Object Model')

    pos = [2,16,32,64,96,126]
    posX,posY = np.meshgrid(pos,pos)
    posX = posX.flatten()
    posY = posY.flatten()
    
    medidas = []

    for px,py in zip(posX,posY):
        W = img[py:py+N,px:px+N]*P

        medida = np.fft.fft2(W)
        medida = np.fft.fftshift(medida)
        
        medida = np.absolute(medida)**2

        if 0:# add invalid grid to data
            delta = 25
            for i in range(0,medida.shape[0]):
                medida[0:medida.shape[0]:delta] = -1

            for i in range(0,medida.shape[1]):
                medida[:,0:medida.shape[1]:delta] = -1
                
        misc.imshow(np.abs(medida),(5,5),savename='difpadgrid.png')

        if pad_shift: # shift difpad to evaluate how much it worsens the reconstruction
            # medida = pad_and_shift(medida,pad,shiftx,shifty)
            # medida = roll_and_crop(medida,1)
            misc.imshow(np.abs(medida),(5,5),savename='difpadshift.png')

        medida = np.fft.fftshift(medida)

        medidas.append(medida)

    O = np.random.rand(*img.shape)+0j     # Gerar um dado inicial (ruido) para objeto (O) e probe (P - caso queremos recupera-la tbm)
    P = P + np.random.rand(*P.shape)*1j

    medidas = cp.asarray(medidas)

else:
    
    rois = np.load(folder+'rois.npy').astype(int) # Load probe positions
    difpads = np.load(folder+'difpads.npy') 
    P = np.load(folder+'probe.npy')
    print('\tLoad done!')

    if 1: # mask from file to binning = 4 difpad
        mask = np.load('mask_2021-11-11_2.npy')
        difpads[:, mask > 0] = -1

    if 0: # shift difpad by n pixels
        n = 20
        for i in range(0,difpads.shape[0]):
            difpads[i,:,:] = roll_and_crop(difpads[i,:,:],n)

    if 0: # insert beamstop
        center_row, center_col = get_difpad_center(difpads[0,:,:])
        central_mask = create_circular_mask(center_row, center_col, 10, difpads[0, :, :].shape)
        difpads[:, central_mask > 0] = -1


    print(f'Probe shape = {P.shape} ')
    P = P[0,0,:,:]

    hsize,dx, maxroi, _ = set_parameters(difpads,jason,rois)
    print('hsize,dx, maxroi',hsize,dx, maxroi)
    obj_shape = 2*hsize+maxroi

    # Gerar um dado inicial (ruido) para objeto (O) 
    O = np.random.rand(obj_shape,obj_shape)+0j

    if 1: # Plot probe
        figure, subplot = plt.subplots()
        subplot.imshow(np.abs(P),cmap='jet')
        subplot.set_title('Probe initial guess')
        figure.savefig('probeinitial.png')

    if 1: # Plot diffraction pattern
        # figure, subplot = plt.subplots(dpi=300)
        # subplot.imshow(np.abs(difpads[0]),cmap='jet',norm=LogNorm())
        # subplot.set_title('0th diffraction pattern')
        misc.imshow(difpads[0],(20,20),savename='difpad.png')
        # misc.imshow(np.sum(difpads,axis=0),(20,20))#,savename='imagem2.png')

    if 1: # Plot initial guesses
        figure, subplot = plt.subplots(1,2,dpi=300)
        subplot[0].imshow(np.abs(P),cmap='jet',norm=LogNorm())
        subplot[0].set_title('Probe guess')
        subplot[1].imshow(np.abs(O))
        subplot[1].set_title('Object guess')
    plt.show()
    
    print(f'Rois shape = {rois.shape} \t rois[0]={rois[0]}')
    print(f'Probe shape = {P.shape} ')
    print(f'Diffraction pattern shape = {difpads.shape} ')
    print(f'Object shape = {O.shape}')

    posY = rois[:,0]
    posX = rois[:,1]

    medidas = cp.asarray(difpads)
    ########################################################################

print('posX; posY',posX.shape,posY.shape)
print('len medidas',len(medidas))

offset = P.shape[1]
probeVelocity = 0
objVelocity = 0
T_counter = 0

obj = cp.asarray(O)
probe = cp.asarray(P)

error_list = []
for j in range(iterations):

    print(f'Iteration {j+1}/{iterations}')
    error = 0
    O_aux = obj+0 
    P_aux = probe+0 
    for i in np.random.permutation(len(medidas)):  
        px = posX[i]
        py = posY[i]
        med = medidas[i]

        W = obj[py:py+offset,px:px+offset]*probe

        Pmm = cp.fft.fft2(W)
        error += cp.sum((cp.abs(Pmm[med >= 0]) - cp.sqrt(med[med >= 0]))**2)/ (cp.sum(cp.sqrt(med[med >= 0]**2)))  
        
        Pmm[med >= 0] = ((Pmm/cp.abs(Pmm))*cp.sqrt(med))[med >= 0]
        
        Pm = cp.fft.ifft2(Pmm)
        
        Diff = Pm - W

        #TODO: update only where difpad is valid?
        if use_rPIE_update_function: # rPIE update function
            obj[py:py+offset,px:px+offset] = obj[py:py+offset,px:px+offset] + gamma_obj*Diff*probe.conj()/ ( (1-alpha)*cp.abs(probe)**2+alpha*(cp.abs(probe)**2).max() )
            # P = P + gamma_probe*Diff*obj[py:py+offset,px:px+offset].conj()/ ( (1-beta)*cp.abs(P)**2+beta*(cp.abs(P)**2).max() )
            probe = probe + gamma_probe*Diff*obj[py:py+offset,px:px+offset].conj()/ ( (1-beta)*cp.abs(obj[py:py+offset,px:px+offset])**2+beta*(cp.abs(obj[py:py+offset,px:px+offset])**2).max() )
        else: #ePIE update function
            obj[py:py+offset,px:px+offset] = obj[py:py+offset,px:px+offset] + alpha*Diff*probe.conj()/(cp.abs(probe)**2).max()
            probe = probe + beta*Diff*obj[py:py+offset,px:px+offset].conj()/(cp.abs(obj)**2).max()
        
        if correct_momentum:
            if mPIE == True: # momentum addition
                T_counter += 1 
                if T_counter == T_lim : # T parameter in mPIE paper
                    probeVelocity  = probeVelocity*eta_probe + (probe - P_aux)
                    objVelocity = objVelocity*eta_obj  + (obj - O_aux)  
                    obj = O_aux + objVelocity
                    probe = P_aux + probeVelocity 
                    
                    O_aux = obj
                    P_aux = probe            
                    T_counter = 0
                    # T_counter = T_lim - 1
    
    error_list.append(error.get())

    if correct_momentum == False:
        if mPIE == True: # old momentum addition
            if j > 10 : # mPIE step?
                gradO = obj - O_aux
                gradP = probe - P_aux
                probeVelocity = probeVelocity*eta_probe + gradP
                objVelocity   = objVelocity*eta_obj + gradO
                obj = O_aux + objVelocity
                probe = P_aux + probeVelocity 
        
        
    if j%100 == 0: # Print iteration
        figure, subplot = plt.subplots(1,2)
        subplot[0].imshow(np.abs(obj.get()),cmap='jet')
        subplot[1].imshow(np.abs(probe.get()),cmap='jet')
        subplot[0].set_title('Object')
        subplot[1].set_title('Probe')
        figure.suptitle(f'Iteration #{j}')
        # plt.savefig(f'iteration_{j}.png')
        plt.show()
        plt.close()

probe = probe.get() # get from cupy to numpy
obj = obj.get()

import datetime
now = datetime.datetime.now()
now = now.strftime("%H_%M_%S")

#np.save('probe_chute.npy',P)
figure, subplot = plt.subplots(1,2,dpi=300)
subplot[0].imshow(np.absolute(probe))
subplot[1].imshow(np.angle(probe))
subplot[0].set_title('Magnitude')
subplot[1].set_title('Phase')
plt.savefig(f"{now}_probe.png",format='png')

# obj = obj[50:-50,50:-50]#shift:-shift,shift:-shift]

figure, subplot = plt.subplots(1,2,dpi=300)
subplot[0].imshow(np.absolute(obj))
subplot[0].imshow((np.absolute(obj)))
subplot[1].imshow(np.angle(obj))
subplot[0].set_title('Magnitude')
subplot[1].set_title('Phase')
figure.suptitle('Object Reconstruction')
plt.savefig(f"{now}_obj.png",format='png')
# plt.close()
plt.show()

print(error_list)
figure, subplot = plt.subplots()
subplot.plot(error_list[1::])
subplot.set_yscale('log')
subplot.grid()
plt.savefig(f"{now}_error.png")

t1 = time.perf_counter()

print(f'Total time = {(t1-t0)/60} min' )