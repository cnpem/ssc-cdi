#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:52:47 2020

@author: francesco
"""
import torch
import torch.fft
import numpy as np
from skimage import exposure

import warnings
warnings.filterwarnings('ignore')

from posupdate_tools import corrpos, checkpos, blur_mask, corrposfixed, calccorrfac

print('pytorch', torch.__version__)
print('gpu',torch.cuda.get_device_name(torch.cuda.current_device()))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
    mimg = np.mean(d)
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
    fst = contraststretch(np.abs(fst))
    #fst = fstabs * np.exp(1j*np.angle(fst))

    return fst, c, s, x_pos, y_pos, d


def contraststretch(img):
    mag, phase = np.abs(img), np.angle(img)
    p2, p98 = np.percentile(mag, (5, 95))
    mag = exposure.rescale_intensity(mag, in_range=(p2, p98))
    return mag*np.exp(1j*phase) if np.iscomplexobj(img) else mag


def gen_linspace(N):
    if N%2 == 0:
        return torch.linspace(-N/2.,N/2. -1,N)
    else:
        return torch.linspace(-(N-1)/2.,(N-1)/2. -1,N)


def abs2(dd):
    return dd.real**2 + dd.imag**2



fftshiftt = torch.fft.fftshift
ifftshiftt = torch.fft.ifftshift
fft2t = torch.fft.fft2
ifft2t = torch.fft.ifft2


def ft2(g, delta):
    return fftshiftt(fft2t(fftshiftt(g))) * delta**2


def ift2(G, delta_f):
    N = G.shape[0]
    return ifftshiftt(ifft2t(ifftshiftt(G))) * (N * delta_f)**2


class angprop:
    def __init__(self, d1, d2, Dz, wvl, N):
        k = 2*np.pi/wvl    # optical wavevector
        x1, y1 = torch.meshgrid(gen_linspace(N)*d1, gen_linspace(N)*d1)
        r1sq = x1**2 + y1**2;
        self.df1 = 1 / (N*d1);
        fX, fY = torch.meshgrid(gen_linspace(N)*self.df1, gen_linspace(N)*self.df1)
        fsq = fX**2 + fY**2
        # scaling parameter
        self.m = d2/d1
        # observation-plane coordinates
        x2, y2 = torch.meshgrid(gen_linspace(N)*d2, gen_linspace(N)*d2)
        r2sq = x2**2 + y2**2;
        #% quadratic phase factors
        self.Q1 = torch.exp(1j*k/2*(1-self.m)/Dz*r1sq).to(device)
        self.Q2 = torch.exp(-1j*np.pi**2*2*Dz/self.m/k*fsq).to(device)
        self.Q3 = torch.exp(1j*k/2*(self.m-1)/(self.m*Dz)*r2sq).to(device)
        self.d1 = d1

    def Propagate(self, Uin):
        return self.Q3* ift2(self.Q2 * ft2(self.Q1 * Uin / self.m , self.d1), self.df1)



class fresnprop:
    def __init__(self, nx, ny, dx, wvl, z, cuda=True):
        self.wvl, self.z = wvl, z
        x,y = gen_linspace(nx)*dx, gen_linspace(ny)*dx
        xx,yy = torch.meshgrid(x,y)
        rr = (xx**2 + yy**2).to(device)
        self.fac = torch.exp(rr*1j*np.pi/(self.wvl*self.z))


    def propagate(self, xin):
        F = fftshiftt(fft2t(fftshiftt(xin)))
        output = ifftshiftt(ifft2t(ifftshiftt(F*self.fac)))
        return output


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



class PtyRecon:
    def __init__(self, data, x_pos,y_pos, dz, wvl, delta1, probeguess=None, objguess=None, posupalpha=1.):

        self.h,self.w = probeguess.shape
        self.x_pos, self.y_pos = x_pos.astype(np.float), y_pos.astype(np.float)
        self.oldxpos, self.oldypos = self.x_pos.copy(), self.y_pos.copy() # used for crosscorr

        self.data = torch.from_numpy(data).to(device)

        self.probeguess, self.objguess = torch.from_numpy(probeguess).to(device), torch.from_numpy(objguess).to(device)

        # propagators
        self.forward =  angprop(delta1, delta1, dz, wvl, self.h)
        self.back =  angprop(delta1, delta1, -dz, wvl, self.h)

        self.blurmask = torch.from_numpy(blur_mask((self.h, self.w), 0.1, 0.8)).to(device)
        #self.datamasked = self.blurmask.unsqueeze(0) * self.data
        self.adamobj = [AdamOptimizer(np.asarray(self.x_pos[ii], self.y_pos[ii]), alpha=posupalpha) for ii in range(len(x_pos))]
        
        self.poskx, self.posky = 100,100
        
        self.old_poserrx, self.old_poserry = np.zeros_like(x_pos, dtype=np.float32), np.zeros_like(y_pos, dtype=np.float32)
        self.currposerrx, self.currposerry = np.zeros_like(x_pos, dtype=np.float32), np.zeros_like(y_pos, dtype=np.float32)
        
    
    def updatecoeff(self, setth=0.3, incfac=1.1, decfac = 0.9): #fixed crosscorrs
        # print(self.old_poserrx, self.currposerrx, self.old_poserry, self.currposerry)
        cx, cy = calccorrfac(self.old_poserrx, self.currposerrx, self.old_poserry, self.currposerry)
        if cx > setth: self.poskx *= incfac
        if cy > setth: self.posky *= incfac

        if cx < -setth: self.poskx *= decfac
        if cy < -setth: self.posky *= decfac
        
        print(self.poskx, self.posky, cx, cy)


    def updateExitWave(self, exw, obsdata):
        detfield = self.forward.Propagate(exw)
        detcorr = obsdata * torch.exp(1j*torch.angle(detfield))
        phicorr = self.back.Propagate(detcorr)
        return phicorr, detfield


    def updateEpie(self, upill, uppos=False):
        x_pos, y_pos = self.x_pos, self.y_pos

        errnorm, poserrcum = [],[]
        idxar = np.random.choice(np.arange(len(self.data)), size=len(self.data), replace=False)
        for kk, (xx, yy) in enumerate(zip(x_pos, y_pos)):
            # get obj view
            i = idxar[kk]
            xposok, yposok = x_pos[i], y_pos[i]
            xposokint, yposokint = int(xposok), int(yposok)
            okpos = checkpos(x_pos[i], y_pos[i], self.h, self.w, self.objguess.shape[0], self.objguess.shape[1]) if uppos else (x_pos[i], y_pos[i])
            xposok, yposok = okpos
            xposokint, yposokint = int(xposok), int(yposok)
            objview = self.objguess[xposokint:xposokint + self.h, yposokint:yposokint + self.w]

            # gen exit wave
            currexwave = self.probeguess * objview

            # correct exit wave and find differece
            corrctd_exw, uncorrAtDet = self.updateExitWave(currexwave, self.data[i])
            phidiff = corrctd_exw - currexwave
            errnorm.append(float(torch.abs(phidiff).sum()))

            # calc update factor
            objupd = self.probeguess.conj()*phidiff/(1e-3 + abs2(self.probeguess).max())
            probeupd = objview.conj()*phidiff/(1e-3 + abs2(self.objguess).max())

            objupdcand = self.objguess[xposokint:xposokint + self.h, yposokint:yposokint + self.w] + objupd

            # position refinement
            if uppos:
                # prj, sim = self.data[i], uncorrAtDet # detector plane
                prj, sim = objupdcand*self.blurmask, objview*self.blurmask #objplane 
                # poserrnorm = corrpos(prj, sim, self.x_pos, self.y_pos, i, self.adamobj[kk]) # adam method
                #prj, sim = np.abs(corrctd_exw/(1e-3+self.probeguess)), np.abs(objview)
                poserrnorm = corrposfixed(prj, sim, self.x_pos, self.y_pos, self.poskx, self.posky, i, self.currposerrx, self.currposerry) # Zhang2013 method
                poserrcum.append(poserrnorm)

            # update estimates
            self.objguess[xposokint:xposokint + self.h, yposokint:yposokint + self.w] = objupdcand
            #self.objguess[xposokint:xposokint + self.h, yposokint:yposokint + self.w] += shift_partial(objupd, xposokint,yposokint,xposok, yposok)

            if upill:
                self.probeguess += probeupd
        
        if uppos: # fixed posscorr
            print(self.currposerrx, self.currposerry)
            print(self.old_poserrx, self.old_poserry)
            self.updatecoeff()
            self.old_poserrx, self.old_poserry = self.currposerrx.copy(), self.currposerry.copy() # used for crosscorr

        return sum(errnorm)/float(self.data.shape[0]), sum(poserrcum)/float(self.data.shape[0])


if __name__ == '__main__':
    import tifffile as tf
    import matplotlib.pyplot as plt
    from skimage.filters import gaussian
    import cv2

    import time

    def plot_curr(obj, probe, it, err):

        fig = plt.figure(dpi=180, figsize=(5,5))
        plt.subplot(221); plt.imshow(contraststretch(np.abs(obj)), cmap='gray'); plt.colorbar(fraction=0.046)
        plt.subplot(222); plt.imshow(np.angle(obj), cmap='gray');  plt.colorbar(fraction=0.046)
        plt.subplot(223); plt.imshow(contraststretch(np.abs(ccprobe)), cmap='gray'); plt.colorbar(fraction=0.046)
        plt.subplot(224); plt.imshow(np.angle(ccprobe), cmap='gray'); plt.colorbar(fraction=0.046)
        plt.suptitle('It:{} Err:{}'.format(it, np.round(err).astype(np.int)))
        plt.tight_layout(); fig.canvas.draw()
        # plt.savefig('tmp.png')
        winre = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        sizes = fig.canvas.get_width_height()[::-1]
        winre = winre.reshape(sizes[0], sizes[1],3)

        plt.close()
        # winre = cv2.imread('tmp.png')

        cv2.imshow('curr', winre)
        cv2.waitKey(1)

    def plot_err(losshist, gtposerrhist, estposerrhist):
        fig = plt.figure(dpi=90)
        plt.plot(losshist[:]/np.amax(losshist[:]), label='loss max:{}'.format(np.round(np.amax(losshist),2)))
        plt.plot(gtposerrhist[:]/np.amax(gtposerrhist[:]), label='gtposerr avg max:{}'.format(np.round(np.amax(gtposerrhist),2)))
        plt.plot(estposerrhist[:]/np.amax(estposerrhist[:]), label='est pos err avg max:{}'.format(np.round(np.amax(estposerrhist),2)))
        plt.grid(); plt.legend(); plt.tight_layout(); fig.canvas.draw()
        winre = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        sizes = fig.canvas.get_width_height()[::-1]
        winre = winre.reshape(sizes[0], sizes[1],3)
        plt.close()

        cv2.imshow('err', cv2.cvtColor(winre, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


    def checkposcorr(ptyobj, xpostrue, ypostrue):
        oldx, oldy = xpostrue, ypostrue
        currx, curry = ptyobj.x_pos.copy(), ptyobj.y_pos.copy()
        distvec = np.sqrt((oldx - currx)**2 + (oldy - curry)**2)
        return np.median(distvec)



    DIMSAMPLE = 128
    #delta1 = (1300.0 / DIMSAMPLE) * 1.03e-8
    delta1 = 9.7656e-7
    Dz = 0.1
    #en = 1020
    #wvl = (6.62606957e-34 * 299792458.) / (en * 1.602176565e-19) # wavelength in meter
    wvl = 1e-9

    datapath = '/mnt/mydata/cose_quarantena/img_dataset/datasets/mars/mag_near.tif'
    pospath = '/mnt/mydata/cose_quarantena/img_dataset/datasets/mars/pos_near.npy'

    #white, mask, data = tf.imread(whitepath)/2**16, tf.imread(maskpath), tf.imread(datapath)/2**16
    data = tf.imread(datapath)**0.5
    data = data/np.amax(data)
    white = np.mean(data, axis=0)
    mask = white > 0.5*np.amax(white)
    pixpos = np.load(pospath)/delta1

    #data = data[:49]
    #pixpos = pixpos[:49, :]

    print(pixpos.shape)

    # add position error
    __, __,__, xpostrue, ypostrue, ___ = composeAgainImage(data, pixpos, flipIt=False, twinMicexp=False, weightFun=white)
    # pixpos += 2.5*np.random.randn(*pixpos.shape)
    # np.save('corruptedpixpos5mars2.npy', pixpos)
    # np.save('corruptedpos_22ott21_25.npy', pixpos)
    #print('Loading old')
    # pixpos = np.load('corruptedpixpos5mars2.npy')
    pixpos = np.load('corruptedpos_22ott21_25.npy')

    # init full obj
    canvas, c,s, xpos, ypos, data = composeAgainImage(data, pixpos, flipIt=False, twinMicexp=False, weightFun=white)
    plt.figure(); plt.imshow(np.abs(canvas), cmap='gray'); plt.title('Init canvas'); plt.show()

    # init illumination
    mask = genCircleMask(data.shape[1], data.shape[1] *0.5, 1, 1)

    # set init
    objguess = np.ones_like(canvas).astype(np.complex)
    probeguess = mask.astype(np.complex)

    # init pty obj
    Ptyobj = PtyRecon(data, xpos, ypos, Dz, wvl, delta1, probeguess=probeguess, objguess=objguess, posupalpha=.3)

    losshist, gtposerrhist, estposerrhist = [], [], []
    for ep in range(201):
        # current update
        t0 = time.time()
        rloss, est_poserr = Ptyobj.updateEpie(upill=True, uppos=True)
        gtposerrmedian = checkposcorr(Ptyobj, xpostrue, ypostrue)
        t1 = time.time()
        # show update
        print('It: {} Err: {} PosErr: {} PosErrEst: {} Time: {}'.format(ep, np.round(rloss,decimals=1), np.round(gtposerrmedian, decimals=4), np.round(est_poserr, decimals=3), np.round(t1-t0,decimals=1)))
        obj, ccprobe = Ptyobj.objguess.cpu().numpy(), Ptyobj.probeguess.cpu().numpy()
        if ep % 5==0:
            plot_curr(obj, ccprobe, ep, rloss)
            if ep > 0:
                plot_err(losshist, gtposerrhist, estposerrhist)

        # save hist
        losshist.append(rloss)
        gtposerrhist.append(gtposerrmedian)
        estposerrhist.append(est_poserr)



    plt.figure(dpi=600)
    plt.plot(losshist[:]/np.amax(losshist[:]), label='loss max:{}'.format(np.round(np.amax(losshist),2)))
    plt.plot(gtposerrhist[:]/np.amax(gtposerrhist[:]), label='gtposerr avg max:{}'.format(np.round(np.amax(gtposerrhist),2)))
    plt.plot(estposerrhist[:]/np.amax(estposerrhist[:]), label='est pos err avg max:{}'.format(np.round(np.amax(estposerrhist),2)))
    plt.grid(); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.show()

    print ('end')







