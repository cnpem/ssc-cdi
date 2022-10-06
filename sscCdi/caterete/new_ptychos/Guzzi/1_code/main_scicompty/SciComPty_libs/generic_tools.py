#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:17:47 2020

@author: francesco
"""
#generic tools

import numpy as np
#import torch
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import h5py

from skimage.restoration import unwrap_phase

def genCircleMask(imsizepx,phyradius,deltax,deltay):
    # generates centered circle
    x, y = np.indices((imsizepx, imsizepx))
    x = x * deltax
    y = y * deltay
    x1, y1 = (imsizepx/2.0, imsizepx/2.0)
    x1 = x1*deltax
    y1 = y1*deltay
    r1 = phyradius
    mask1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask = mask1/np.amax(mask1)
    #io.imshow(mask)
    #io.show()
    return mask.astype(np.float32)


def composeAgainImage(d, PosArray, flipIt=True, twinMicexp=True, weightFun=None):
    # corrected for tomato, default is 1,0

    if twinMicexp:
        x_pos = PosArray[:, 0]  # = ShiftPosX[i] #ShiftPosX[ShiftPosX.shape[0]-i-1]
        y_pos = PosArray[:, 1]  # = ShiftPosY[j]
    else:
        x_pos = PosArray[:, 1]  # = ShiftPosX[i] #ShiftPosX[ShiftPosX.shape[0]-i-1]
        y_pos = PosArray[:, 0]  # = ShiftPosY[j]

    fn = np.zeros(x_pos.size)
    x_pos = np.abs(x_pos); y_pos = np.abs(y_pos)
    x_pos -= x_pos.min();  y_pos -= y_pos.min()
    x_pos = x_pos.astype(int); y_pos = y_pos.astype(int)

    xmax = np.asarray([x_pos[i] + d[i].shape[0] for i in range(len(fn))]).max()
    ymax = np.asarray([y_pos[i] + d[i].shape[1] for i in range(len(fn))]).max()
    #print(xmax,ymax)
    c = np.zeros(shape=(xmax, ymax), dtype=d.dtype)
    s = np.zeros_like(c)  # sample density
    m = np.mean(d, axis=0) if weightFun is None else weightFun*1.0
    mimg = np.mean(m)

    if flipIt:
        m = np.flipud(m)
        d = np.asarray([np.flipud(cd) for cd in d])

    for i in range(len(fn)):
        #sys.stdout.write('.'), sys.stdout.flush(),
        c[x_pos[i]:x_pos[i] + d[i].shape[0], y_pos[i]:y_pos[i] + d[i].shape[1]] += d[i]
        s[x_pos[i]:x_pos[i] + d[i].shape[0], y_pos[i]:y_pos[i] + d[i].shape[1]] += m

    fst = c / (s+1e-6)
    #fst = np.nan_to_num(fst,mimg)
    fst[np.isnan(fst)] = mimg
    fst[np.isinf(fst)] = mimg
    #fst = contraststretch(np.abs(fst))
    #fst = fstabs * np.exp(1j*np.angle(fst))

    return fst, c, s, x_pos, y_pos, d


def contraststretch(img, fac=0.125):
    mag, phase = np.abs(img), np.angle(img)
    N,M = int(mag.shape[0] * fac), int(mag.shape[1]* fac)
    p2, p98 = np.percentile(mag[N:-N, M:-M], (5, 95))
    mag = exposure.rescale_intensity(mag, in_range=(p2, p98))
    return mag*np.exp(1j*phase) if np.iscomplexobj(img) else mag


def blur_edges(prj, low=0, high=0.8):
    _prj = prj.copy()
    dy, dz = _prj.shape
    rows, cols = np.mgrid[:dy, :dz]
    rad = ((rows - dy / 2)**2 + (cols - dz / 2)**2)**0.5
    mask = np.zeros((dy, dz))
    rmin, rmax = low * rad.max(), high * rad.max()
    mask[rad < rmin] = 1
    mask[rad > rmax] = 0
    zone = np.logical_and(rad >= rmin, rad <= rmax)
    mask[zone] = (rmax - rad[zone]) / (rmax - rmin)
    #feathered = np.empty((dy, dz), dtype=np.uint8)
    _prj *= mask
    return _prj


def plot_curr(obj, ccprobe, it, err, frac=1/16, cstr=True, unwrap=False):
    scicomplogo = cv2.imread('SciComPty_libs/scicomp.png')
    maxor, maxoc = obj.shape
    l1,l2 = int(frac*maxor),int(frac*maxoc)
    objmag = contraststretch(np.abs(obj[l1:-l1,l2:-l2])) if cstr else np.abs(obj[l1:-l1,l2:-l2])
    if unwrap:
        objphase = unwrap_phase(contraststretch(np.angle(obj[l1:-l1,l2:-l2])) if cstr else np.angle(obj[l1:-l1,l2:-l2]))
    else:
        objphase = contraststretch(np.angle(obj[l1:-l1,l2:-l2]) if cstr else np.angle(obj[l1:-l1,l2:-l2]))            

    fig = plt.figure(dpi=180, figsize=(5,5))
    plt.subplot(221); plt.imshow(objmag, cmap='gray'); plt.colorbar(fraction=0.046)
    plt.subplot(222); plt.imshow(objphase, cmap='gray');  plt.colorbar(fraction=0.046)
    plt.subplot(223); plt.imshow(contraststretch(np.abs(ccprobe)), cmap='gray'); plt.colorbar(fraction=0.046)
    plt.subplot(224); plt.imshow(contraststretch(np.angle(ccprobe)), cmap='gray'); plt.colorbar(fraction=0.046)
    plt.suptitle('SciComPty\nIt:{} Err:{}'.format(it, np.round(err, decimals=3).astype(np.int)))
    plt.tight_layout(); fig.canvas.draw()
    #winre = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    winre = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    winre = winre.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    winout = winre + 0
    winout[:100,-100:] = scicomplogo
    cv2.imshow('curr', winout)
    cv2.waitKey(100)




def load_positions(pospath):
    if '.h5' in pospath or '.hdf5' in pospath:
        f = h5py.File(pospath,'r')
        # positions im microns
        xpos = f['/sample_motors/sample_x_pos'][:]
        ypos = f['/sample_motors/sample_y_pos'][:]
        f.close()
        PosArray = np.asarray([np.asarray([x,y]) for x,y in zip(xpos,ypos) ]).reshape(len(xpos),2)*1e-6
    else:
        PosArray = np.load(pospath)
    return PosArray

from skimage.color import hsv2rgb
def cplxtohsv(cin, vmin=0, vmax=None):
    h,v  = .5*np.angle(cin)/np.pi + .5, np.abs(cin)
    vmax = v.max() if vmax is None else vmax
    v = (v.clip(vmin,vmax)-vmin)/(vmax-vmin)
    out = np.ones((h.shape[0], h.shape[1], 3))
    out[:,:,0], out[:,:,2] = h, v
    return hsv2rgb(out)

