#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:24:35 2020

@author: francesco
"""

# update pos tools
import numpy as np
import torch
#from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
#from generic_tools import contraststretch, blur_edges
#from fft_tools import fft2t, ifft2t, fftshiftt, ifftshiftt

#from torch.fft import fftfreq

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def blur_mask(shape, low=0, high=0.8):
    dy, dz = shape
    rows, cols = np.mgrid[:dy, :dz]
    rad = np.sqrt((rows - dy / 2)**2 + (cols - dz / 2)**2)
    mask = np.zeros((dy, dz))
    rmin, rmax = low * rad.max(), high * rad.max()
    mask[rad < rmin] = 1
    mask[rad > rmax] = 0
    zone = np.logical_and(rad >= rmin, rad <= rmax)
    zone = gaussian(zone, sigma=10)
    return zone


def checkposcorr(ptyobj, xpostrue, ypostrue):
    oldx, oldy = xpostrue, ypostrue
    currx, curry = ptyobj.x_pos.copy(), ptyobj.y_pos.copy()
    distvec = np.sqrt((oldx - currx)**2 + (oldy - curry)**2)
    return np.median(distvec)


def corrpos(prj, sim, x_pos, y_pos, i, adamobj):

    #if cstretch: # contrast strecth
    #    prj, sim = np.abs(contraststretch(prj)), np.abs(contraststretch(sim))

    # Estimate shift
    shift = phase_corr(prj, sim,upsample_factor=200)
    poserrnorm = np.sqrt(shift[0]**2 + shift[1]**2)

    pars = adamobj.backward_pass(shift)

    # Update positions
    # x_pos[i] -= shift[0]
    # y_pos[i] -= shift[1]
    # adam
    x_pos[i] -= pars[0]
    y_pos[i] -= pars[1]
    return poserrnorm


def checkpos(currx, curry, dimr, dimc, maxr, maxc):
    ''' check that each view stays in the object windows'''
    diffr, diffc = currx + dimr, curry + dimc
    okr = maxr - dimr if diffr > maxr or diffr < 0 else currx
    okc = maxc - dimc if diffc > maxc or diffc < 0 else curry
    return max(0,okr), max(0,okc)


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):

    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    im2pi = 2j * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:

        kernel = ((torch.arange(ups_size) - ax_offset)[:, None]
                  * torch.fft.fftfreq(n_items, upsample_factor))
        kernel = torch.exp(-im2pi * kernel).to(device).type(data.dtype)


        #data = np.tensordot(kernel, data, axes=(1, -1))
        data = torch.tensordot(kernel,data.T, dims=1)

    return data


def maxinar(arr, arrshape):
    return np.unravel_index(torch.argmax(torch.abs(arr)).item(), arrshape)



def phase_corr_gpu(ref, mov, upsample_factor=1):
    src_freq, target_freq = torch.fft.fft2(ref+0j), torch.fft.fft2(mov+0j)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = torch.fft.ifft2(image_product)

    # Locate maximum
    maxima = maxinar(cross_correlation, cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor> 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                            upsampled_region_size,
                                            int(upsample_factor),
                                            sample_region_offset).conj()
        # Locate maximum and map back to original pixel grid
        maxima = maxinar(cross_correlation, cross_correlation.shape)
        maxima = np.stack(maxima).astype(np.float64) - dftshift
        shifts = shifts + maxima / upsample_factor

    return shifts


if __name__  == '__main__':
    from skimage import data, img_as_float
    from skimage.transform import rescale, resize
    from skimage.registration import phase_cross_correlation
    import time
    import matplotlib.pyplot as plt

    imgsizes = [128,256,512,1024,2048]
    upscalefact =[ 1,2,4,8,16,32,64,128, 256]
    img = img_as_float(data.camera()).astype(np.float32)
    #imgt = torch.from_numpy(img)
    tempi = []
    print(img.shape)
    for ii in imgsizes: # image dim
        tempi.append([])
        imgs = resize(img, (ii,ii))
        imgst = torch.from_numpy(imgs).cuda()
        for rri,rr in enumerate(upscalefact): # upsample factors
            t0 = time.time()
            for pp in range(5):
                #out = phase_cross_correlation(imgs, imgs,upsample_factor=rr)
                out = phase_corr_gpu(imgst, imgst,upsample_factor=rr)
            t1 = time.time()
            tt = (t1-t0)/5
            tempi[-1].append(tt)
            #plt.plot(rr, , c=plt.cm.RdYlBu(rri))

    plt.figure(dpi=150)
    for it, tempo in enumerate(tempi):
        plt.semilogy(upscalefact, tempo, label=str(imgsizes[it]), alpha=0.9, marker='*', linewidth=0.5)
        #plt.scatter(upscalefact, np.log10(tempo))
    plt.grid(); plt.legend(title='img size', bbox_to_anchor=(1.04,1), loc="upper left"); plt.xlabel('Upsample factor'); plt.ylabel('time [s]')
    plt.title('GPU'); plt.ylim([0, 0.5])
    plt.show()

    print('end')




