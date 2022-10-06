#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:15:32 2020

@author: francesco
"""

print('SciComPty 2022 nadia prop')
import torch

from torch.fft import fft2 as fft2t
from torch.fft import ifft2 as ifft2t
from torch.fft import fftshift as fftshiftt
from torch.fft import ifftshift as ifftshiftt

import numpy as np

#from fft_tools import propagator, fft2t, ifft2t, fftshiftt, ifftshiftt, propagator
#from posupdate_tools import checkpos, corrpos, blur_mask

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.filters import gaussian


def blur_mask(shape, low=0, high=0.8):
    dy, dz = shape
    rows, cols = np.mgrid[:dy, :dz]
    rad = np.sqrt((rows - dy / 2)**2 + (cols - dz / 2)**2)
    mask = np.zeros((dy, dz))
    # mask = torch.zeros((dy,dz))
    rmin, rmax = low * rad.max(), high * rad.max()
    mask[rad < rmin] = 1
    mask[rad > rmax] = 0
    zone = np.logical_and(rad >= rmin, rad <= rmax)
    # zone = torch.logical_and(rad >= rmin, rad <= rmax)
    zone = gaussian(zone, sigma=10)
    return zone


def checkposcorr(ptyobj, xpostrue, ypostrue):
    oldx, oldy = xpostrue, ypostrue
    oldx -= oldx.mean(axis=0)
    oldy -= oldy.mean(axis=0)
    currx, curry = ptyobj.x_pos.copy(), ptyobj.y_pos.copy()
    currx -= currx.mean(axis=0)
    curry -= curry.mean(axis=0)
    distvec = np.sqrt((oldx - currx)**2 + (oldy - curry)**2)
    return np.median(distvec)


def corrpos(prj, sim, x_pos, y_pos, i, adamobj, fixpesi):

    #if cstretch: # contrast strecth
    #    prj, sim = np.abs(contraststretch(prj)), np.abs(contraststretch(sim))

    # Estimate shift
    shift = phase_corr(prj, sim,upsample_factor=100)
    # poserrnorm = np.sqrt(shift[0]**2 + shift[1]**2)

    pars = adamobj.backward_pass(shift)

    # Update positions
    # x_pos[i] -= shift[0]
    # y_pos[i] -= shift[1]
    # adam
    x_pos[i] -= fixpesi[i] * pars[0] # 1e-1
    y_pos[i] -= fixpesi[i] * pars[1] # 1e-1

    poserrnorm = np.sqrt(pars[0]**2 + pars[1]**2)
    return poserrnorm


def checkpos(currx, curry, dimr, dimc, maxr, maxc):
    ''' check that each view stays in the object windows'''
    currx, curry = abs(max(0, currx)), abs(max(0, curry)) # sempre maggiore di 0

    diffr, diffc = currx + dimr, curry + dimc
    okr = maxr-dimr if diffr > maxr else currx
    okc = maxc-dimc if diffc > maxc else curry
    # if okr<0:
    #     print(maxr, dimr)
    # if okc <0:
    #     print(maxc, dimc)
    # okr = maxr - dimr if diffr > maxr or diffr < 0 else currx
    # okc = maxc - dimc if diffc > maxc or diffc < 0 else curry
    # return max(0,okr), max(0,okc)

    return okr, okc


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



def phase_corr(ref, mov, upsample_factor=1):
    #src_freq, target_freq = torch.fft.fft2(ref+0j), torch.fft.fft2(mov+0j)
    src_freq, target_freq = torch.fft.fft2(ref), torch.fft.fft2(mov)
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



def gen_linspace(N):
    if N%2 == 0:
        return torch.linspace(-N/2.,N/2. -1,N)
    else:
        return torch.linspace(-(N-1)/2.,(N-1)/2. -1,N)

# tipo nadia multi
class propagator:
    def __init__(self, fd, fs, ps, N, wavelength, dim=1):
        self.zsd = fd - fs
        self.factor = ps*ps*np.pi/(wavelength)*((1./fd) - (1./self.zsd))
        l1 = gen_linspace(N)
        x1, y1 = torch.meshgrid(l1, l1)
        qradius = (x1**2 + y1**2)*self.factor
        self.coefficient = torch.exp(1j*qradius).to(device)
        if dim >1:
            self.coefficient = self.coefficient.unsqueeze(0).repeat(dim,1,1)

        #print(self.coefficient.shape)
            

    def tosampleplane(self, uin):
        return ifftshiftt(ifft2t(uin * self.coefficient))


    def todetector(self, uin):
        return fft2t(ifftshiftt(uin)) * self.coefficient.conj()




def sphericalWaveT(N, ilambda, dx, z0):
    xx = gen_linspace(N) * dx
    uu,vv = torch.meshgrid(xx,xx)
    xydst = uu**2 + vv**2
    lz = ilambda*z0
    p = torch.exp(1j*np.pi*xydst/lz)#/np.sqrt(xydst)
    return p


class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        return self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))


def toTensor(data):
    return data if isinstance(data, torch.Tensor) else torch.from_numpy(data)


def abs2(data):
    return data.real**2 + data.imag**2


# versione 2021 (tipo maiden)
# versione 2022 proviamo a convertirlo in tipo nadia
class PtyRecon:
    def __init__(self, data, x_pos,y_pos,
                  fd, fs, ps, wavelength,
                 probeguess=None, objguess=None,
                 posupalpha=5., canvasmask=None, detectormask=None):

        self.n, self.h,self.w = probeguess.shape
        self.x_pos, self.y_pos = x_pos.astype(np.float), y_pos.astype(np.float)
        self.oldxpos, self.oldypos = self.x_pos.copy(), self.y_pos.copy()

        self.data = toTensor(data).to(device).reshape(len(data), 1, self.h, self.w)

        self.probeguess, self.objguess = toTensor(probeguess).to(device), toTensor(objguess).to(device).reshape(1,objguess.shape[0], objguess.shape[1])

        # propagator (tipo nadia)
        self.prop = propagator(fd, fs, ps, data.shape[1], wavelength)

        self.adamobj = [AdamOptimizer(np.asarray(self.x_pos[ii], self.y_pos[ii]), alpha=posupalpha) for ii in range(len(x_pos))]
        self.blurmask = torch.from_numpy(blur_mask((self.h, self.w), 0.1, 0.7)).to(device) #0.8

        if canvasmask is None:
            self.pesimask = np.ones_like(x_pos)
        else:
            self.pesimask = canvasmask.reshape(-1)

        self.okdet = torch.ones(len(probeguess), probeguess.shape[1], probeguess.shape[2], dtype=torch.bool) if detectormask is None else torch.from_numpy(detectormask)
        self.okdet = self.okdet.to(device)


    def updateRpieMulti(self, upobj=1.0, upill=1.0, uppos=False, alpha=.04, beta=1.):
        x_pos, y_pos = self.x_pos, self.y_pos

        errnorm, poserrcum = [],[]
        idxar = np.random.choice(np.arange(len(self.data)), size=len(self.data), replace=False)
        for kk, (xx, yy) in enumerate(zip(x_pos, y_pos)):
            i = idxar[kk]
            okpos = checkpos(x_pos[i], y_pos[i], self.h, self.w, self.objguess.shape[1], self.objguess.shape[2]) if uppos else (x_pos[i], y_pos[i])
            xposokint, yposokint = map(int, okpos)

            # get obj view
            objview = self.objguess[0,xposokint:xposokint + self.h, yposokint:yposokint + self.w]

            # exit wave
            exitwaves = self.probeguess * objview

            # corrrect exit wave at detector
            #exitwavesdet = fft2t(exitwaves) # tipo maiden
            exitwavesdet = self.prop.todetector(exitwaves)#.reshape(1, 1300, 1300)
                        
            normfac = abs2(exitwavesdet).sum(axis=0)**0.5
            #correxitwavesdet = (self.data[i] * exitwavesdet)/ (1e-6 + normfac)
            correxitwavesdet = exitwavesdet/ (1e-6 + normfac)
            correxitwavesdet[self.okdet] *= self.data[i][self.okdet]
            correxitwavesdet[~self.okdet] *= torch.abs(exitwavesdet[~self.okdet])
            

            #correxitwaves = ifft2t(correxitwavesdet) # tipo maiden
            correxitwaves = self.prop.tosampleplane(correxitwavesdet)#.reshape(1,1300,1300)

            phidiffs = correxitwaves - exitwaves

            errnorm.append(torch.abs(phidiffs).sum())

            # calculate updates
            absobjview, absprobe = abs2(objview), abs2(self.probeguess).sum(axis=0)
            denomP, denomO = beta*absobjview.max() + (1-beta)*absobjview, alpha*absprobe.max() + (1-alpha)*absprobe
            probeupd, objupd = objview.conj()*phidiffs/(1e-6 + denomP), (phidiffs *  self.probeguess.conj()).sum(axis=0)/(1e-6 + denomO)

            objupdcand = self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] + upobj * objupd

            if uppos:
                # prj, sim = self.data[i], uncorrAtDet # detector plane
                prj, sim = objupdcand*self.blurmask, objview*self.blurmask #objplane
                poserrnorm = corrpos(prj, sim, self.x_pos, self.y_pos, i, self.adamobj[kk], self.pesimask)
                poserrcum.append(poserrnorm)

            # update estimates
            # self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] += upobj * objupd
            self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] = objupdcand
            self.probeguess += upill * probeupd #0.1 buono


        return torch.tensor(errnorm).mean().item(), np.array(poserrcum).mean()



    def updateEpiemulti(self, upobj=1.0, upill=1.0, uppos=False):
        x_pos, y_pos = self.x_pos, self.y_pos

        errnorm, poserrcum = [],[]
        idxar = np.random.choice(np.arange(len(self.data)), size=len(self.data), replace=False)
        for kk, (xx, yy) in enumerate(zip(x_pos, y_pos)):
            i = idxar[kk] # get obj view
            okpos = checkpos(x_pos[i], y_pos[i], self.h, self.w, self.objguess.shape[1], self.objguess.shape[2]) if uppos else (x_pos[i], y_pos[i])
            xposok, yposok = okpos
            xposokint, yposokint = int(xposok), int(yposok)
            objview = self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w]

            exitwaves = self.probeguess * objview

            # corrrect exit wave at detector
            exitwavesdet = fft2t(exitwaves)
            normfac = abs2(exitwavesdet).sum(axis=0)**0.5
            correxitwavesdet = self.data[i] * (exitwavesdet / (1e-6 + normfac))
            correxitwaves = ifft2t(correxitwavesdet)

            phidiffs = correxitwaves - exitwaves

            toterr = torch.abs(phidiffs).sum()
            errnorm.append(toterr)

            probeupd = (phidiffs * objview.conj())/(1e-6 + abs2(objview).max()) # 1e-3 buono
            objupd = (phidiffs *  self.probeguess.conj()).sum(axis=0)/(1e-6 + abs2(self.probeguess).sum(axis=0).max())

            objupdcand = self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] + upobj * objupd

            if uppos:
                # prj, sim = self.data[i], uncorrAtDet # detector plane
                prj, sim = objupdcand*self.blurmask, objview*self.blurmask #objplane
                poserrnorm = corrpos(prj, sim, self.x_pos, self.y_pos, i, self.adamobj[kk], self.pesimask)
                poserrcum.append(poserrnorm)

            # update estimates
            # self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] += upobj * objupd
            self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] = objupdcand
            self.probeguess += upill * probeupd #0.1 buono

        return torch.tensor(errnorm).mean().item(), np.array(poserrcum).mean()


    def updateRpieHolo(self, upobj=1.0, upill=1.0, uppos=False, alpha=.04, beta=1.):
        x_pos, y_pos = self.x_pos, self.y_pos

        errnorm, poserrcum = [],[]
        idxar = np.random.choice(np.arange(len(self.data)), size=len(self.data), replace=False)
        for kk, (xx, yy) in enumerate(zip(x_pos, y_pos)):
            # get obj view
            i = idxar[kk]
            okpos = checkpos(x_pos[i], y_pos[i], self.h, self.w, self.objguess.shape[0], self.objguess.shape[1]) if uppos else (x_pos[i], y_pos[i])
            xposok, yposok = okpos
            xposokint, yposokint = int(xposok), int(yposok)
            objview = self.objguess[xposokint:xposokint + self.h, yposokint:yposokint + self.w]

            currexwave = self.probeguess * objview

            # correct exit wave and find differece
            corrctd_exw, uncorrAtDet = self.updateExitWaveHolo(currexwave, self.data[i], self.probeguess)
            phidiff = corrctd_exw - currexwave
            errnorm.append(torch.abs(phidiff).sum())

            # calculate updates
            absobjview, absprobe = torch.abs(objview)**2, torch.abs(self.probeguess)**2
            denomP, denomO = beta*absobjview.max() + (1-beta)*absobjview, alpha*absprobe.max() + (1-alpha)*absprobe
            probeupd, objupd = objview.conj()*phidiff/(1e-6 + denomP), self.probeguess.conj()*phidiff/(1e-6 + denomO)

            # update estimates
            self.objguess[0, xposokint:xposokint + self.h, yposokint:yposokint + self.w] += upobj * objupd
            self.probeguess += upill * probeupd #0.1 buono

            # position refinement
            if uppos:
                #prj, sim = self.data[i].cpu().numpy(), uncorrAtDet.cpu().numpy()
                prj, sim = self.data[i], uncorrAtDet
                poserrnorm = corrpos(prj, sim, self.x_pos, self.y_pos, i, cstretch=False, blur=True, rin=0.5, rout=0.8, fixpesi=sself.pesimask)
                #prj, sim = np.abs(corrctd_exw/(1e-3+self.probeguess)), np.abs(objview)
                poserrcum.append(poserrnorm)


        return torch.tensor(errnorm).mean().item(), np.array(poserrcum).mean()


if __name__ == '__main__':
    pass
