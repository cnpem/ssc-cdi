# Python Wrapper for libpsicc
# Author: Giovanni L. Baraldi

import numpy as np
import ctypes
import matplotlib.pyplot as plt

from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_int as int32
from ctypes import POINTER
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t

from time import time
from platform import processor

import os
binary_path = os.path.dirname(os.path.realpath(__file__)) + '/'

if processor() == 'ppc64le':
        psiccdll = ctypes.CDLL(binary_path+'libpsicc_ppc.so')
elif processor() == 'x86_64':
        import subprocess
        gpumodel = str(subprocess.check_output(['nvidia-smi', '--query-gpu=name','--format=csv','--id=0']).decode("utf-8")).split('\n')[1]
        #print(gpumodel)
        if 'A100' in gpumodel[:4]:
                psiccdll = ctypes.CDLL(binary_path+'libpsicc_x86_dgx.so')
        else:
                psiccdll = ctypes.CDLL(binary_path+'libpsicc_x86_upc.so')
else:
        raise Exception('Unsupported Platform!')

def nice(f): # scientific notation + 2 decimals
        return "{:.2e}".format(f)

def Show(imgs, legend=[],cmap='jet',nlines=1, bLog = False, interpolation='bilinear'): # legend = plot titles
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

        #plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.show()

defdevice = [0]

from copy import deepcopy
def SetDevices(devices):
        global defdevice
        defdevice = deepcopy(devices)
        print(defdevice)

def MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,_funcc,regularization,objsupp,probesupp,sigmask,epsilon,bkg,probef1,data,params):
        #if device==-1:
        #        device = defdevice

        if obj is None:
                obj = data['obj']
        if probe is None:
                probe = data['probe']
        if difpads is None:
                difpads = data['difpads']
        if rois is None:
                rois = data['rois']
        if objsupp is None:
                try:
                        objsupp = data['objsupp']
                except:
                        pass
        if probesupp is None:
                try:
                        probesupp = data['probesupp']
                except:
                        pass
        if sigmask is None:
                try:
                        sigmask = data['sigmask']
                except:
                        pass
        if bkg is None:
                try:
                        bkg = data['bkg']
                except:
                        pass

        if iter is None:
                iter = data['iter']
        if batch is None:
                try:
                        batch = data['batch']
                except:
                        batch = 16
        if objbeta is None:
                try:
                        objbeta = data['objbeta']
                except:
                        objbeta = 0.95
        if probebeta is None:
                try:
                        probebeta = data['probebeta']
                except:
                        probebeta = 0.9
        if regularization is None:
                try:
                        regularization = data['regularization']
                except:
                        regularization = -1
        if epsilon is None:
                try:
                        epsilon = data['epsilon']
                except:
                        epsilon = 1E-3
        if probef1 is None:
                try:
                        probef1 = data['probef1']
                except:
                        probef1 = 0


        if rois.shape[0] != difpads.shape[0]:
                print("Error:",difpads.shape[0],"difpads  v ",rois.shape[0]," rois mismatch")
                quit()
        if rois.shape[-1] != 4:
                print("Incorrect roi specification: shape=" , rois.shape, ' rois has 4 attribs: x,y,exptime,I0')
                quit()
        
        if probe.shape[-1]%difpads.shape[-1] != 0 or probe.shape[-2]%difpads.shape[-2] != 0:
                print("Error:",probe.shape,"probe  v ",difpads.shape," difpads shape mismatch")
                quit()

        if len(rois.shape) == 2:
                rois = rois[:,None]

        for acqui in rois:
                for roi in acqui:
                        if probe.shape[-1] + roi[0] > obj.shape[-1] or probe.shape[-2] + roi[1] > obj.shape[-2]:
                                print("Error: Roi",roi,"is outside object bounds:",roi,"+",probe.shape,"=",obj.shape)
                                quit()
                        elif roi[0] < 0 or roi[1] < 0:
                                print("Error: Roi",roi,"has negative indexing.")
                                quit()

        if probesupp is not None:
                assert(probesupp.shape[-1] == probe.shape[-1] and probesupp.shape[-2] == probe.shape[-2] and probesupp.size == probe.size)
                probesupp = np.ascontiguousarray(probesupp.astype(np.float32))
                probesuppptr = probesupp.ctypes.data_as(void_p)
        else:
                probesuppptr = void_p(0)

        if objsupp is not None:
                assert(objsupp.size >= obj.size)
                objsupp = np.ascontiguousarray(objsupp.astype(np.float32))
                objsuppptr = objsupp.ctypes.data_as(void_p)
                if len(objsupp.shape) >= 3:
                        numobjsupport = int32(objsupp.shape[0])
                else:
                        numobjsupport = int32(1)
        else:
                objsuppptr = void_p(0)
                numobjsupport = int32(0)

        if bkg is not None:
                assert(bkg.shape[-1] == difpads.shape[-1] & bkg.shape[-2] == difpads.shape[-2])
                bkg = np.ascontiguousarray(bkg.astype(np.float32))
                bkgptr = bkg.ctypes.data_as(void_p)
        else:
                bkgptr = void_p(0)

        obj = np.ascontiguousarray(obj.astype(np.complex64))
        objptr = obj.ctypes.data_as(void_p)

        probe = np.ascontiguousarray(probe.astype(np.complex64))
        probeptr = probe.ctypes.data_as(void_p)

        difpads = np.ascontiguousarray(difpads.astype(np.float32))
        difpadsptr = difpads.ctypes.data_as(void_p)

        if sigmask is not None:
                assert(sigmask.shape[-1] == difpads.shape[-1] & sigmask.shape[-2] == difpads.shape[-2])
                sigmask = np.ascontiguousarray(sigmask.astype(np.float32))
        else:
                sigmask = np.ones(difpads.shape[-2:],dtype=np.float32)
        sigmaskptr = sigmask.ctypes.data_as(void_p)

        rois = np.ascontiguousarray(rois.astype(np.float32))
        roisptr = rois.ctypes.data_as(void_p)
        

        rfactor = np.zeros((iter),dtype=np.float32)
        rfactorptr = rfactor.ctypes.data_as(void_p)

        numdev = int32(len(params['device']))
        devices = np.ascontiguousarray(np.asarray(params['device']).astype(np.int32))
        devicesptr = devices.ctypes.data_as(void_p)

        psizex = int32(probe.shape[-1])
        osizex = int32(obj.shape[-1])
        dsizex = int32(difpads.shape[-1])
        iter = int32(iter)
        batch = int32(batch)
        objbeta = float32(objbeta)
        probebeta = float32(probebeta)
        numrois = int32(rois.shape[0])
        flyscansteps=int32(rois.shape[1])
        epsilon=float32(epsilon)
        probef1=float32(probef1)

        nummodes = int32(probe.shape[0])
        if len(probe.shape) < 3:
                nummodes = int32(1)
        
        print("Algo:", difpads.shape, obj.shape, probe.shape, batch, iter, regularization)
        time0 = time()

        _funcc(objptr, probeptr, difpadsptr, psizex, osizex, dsizex, roisptr, numrois, batch, iter, 
                numdev, devicesptr, rfactorptr, objbeta, probebeta, nummodes, float32(regularization), 
                        objsuppptr, probesuppptr, numobjsupport, sigmaskptr, flyscansteps, epsilon, bkgptr, probef1)

        print("Done in:",time()-time0,"s.")

        mydict = {}
        mydict['obj'] = obj
        mydict['probe'] = probe
        mydict['error'] = rfactor
        mydict['bkg'] = bkg
        mydict['rois'] = rois
        mydict['difpads'] = difpads
        mydict['probesupp'] = probesupp
        mydict['objsupp'] = objsupp
        return mydict

def RAAR(obj=None,probe=None,difpads=None,rois=None,iter=None,beta=None,probecycles=None,batch=None,tvmu=None,objsupp=None,probesupp=None,sigmask=None,epsilon=None,bkg=None,probef1=None,data=None,params=None):
        if probecycles is None and params is not None:
                try:
                        params['probecycles']
                except:
                        params['probecycles'] = 3
        return MAKENICE(obj,probe,difpads,rois,iter,batch,beta,probecycles,psiccdll.Raar,tvmu,objsupp,probesupp,sigmask,epsilon,bkg,probef1,data,params)
        #return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.Raar,-1)

def GL(obj=None,probe=None,difpads=None,rois=None,iter=None,objbeta=None,probebeta=None,batch=None,tvmu=None,objsupp=None,probesupp=None,sigmask=None,epsilon=None,bkg=None,probef1=None,data=None,params=None):
        return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.GL,tvmu,objsupp,probesupp,sigmask,epsilon,bkg,probef1,data,params)

def GlobalPSF(obj=None,probe=None,difpads=None,rois=None,iter=None,objbeta=None,probebeta=None,batch=None,tvmu=None,objsupp=None,probesupp=None,sigmask=None,epsilon=None,bkg=None,probef1=None,data=None,params=None):
        return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.PSF,tvmu,objsupp,probesupp,sigmask,epsilon,bkg,probef1,data,params)

def PosCorrection(obj=None,probe=None,difpads=None,rois=None,iter=None,objbeta=None,probebeta=None,batch=None,tvmu=None,objsupp=None,probesupp=None,sigmask=None,epsilon=None,bkg=None,probef1=None,data=None,params=None):
        return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.PosCorr,tvmu,objsupp,probesupp,sigmask,epsilon,bkg,probef1,data,params)

def PIE(obj,probe,difpads,rois,iter,batch=1):
        return MAKENICE(obj,probe,difpads,rois,iter,batch,psiccdll.PIE,-1)

def LocalPSF(obj,probe,difpads,rois,iter,objbeta=0.97,probebeta=0.93,batch=32,tvmu=0,objsupp=None,probesupp=None,sigmask=None,epsilon=1E-3):
        return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.LPSF,tvmu,objsupp,probesupp,sigmask,epsilon)

'''def LSQML(obj,probe,difpads,rois,iter,noisemodel,beta=1.0,probealpha=0.05,batch=-1):
        assert(noisemodel>=0)
        return MAKENICE(obj,probe,difpads,rois,iter,batch,beta,probealpha,psiccdll.LSQML,noisemodel=noisemodel)'''


def LSML(obj,probe,difpads,rois,iter,objbeta=0.97,probebeta=0.93,batch=-1,tvmu=0,objsupp=None,probesupp=None,
        sigmask=None,epsilon=1E-3,noisemodel=0,bkg=None,flat=None,bkgmask=None,flatmask=None):
        #if device==-1:
        #        device = defdevice

        if rois.shape[0] != difpads.shape[0]:
                print("Error:",difpads.shape[0],"difpads  v ",rois.shape[0]," rois mismatch")
                quit()
        if rois.shape[-1] != 4:
                print("Incorrect roi specification: shape=" , rois.shape, ' rois has 4 attribs: x,y,exptime,I0')
                quit()
        
        if probe.shape[-1]%difpads.shape[-1] != 0 or probe.shape[-2]%difpads.shape[-2] != 0:
                print("Error:",probe.shape,"probe  v ",difpads.shape," difpads shape mismatch")
                quit()

        if len(rois.shape) == 2:
                rois = rois[:,None]

        for acqui in rois:
                for roi in acqui:
                        if probe.shape[-1] + roi[0] > obj.shape[-1] or probe.shape[-2] + roi[1] > obj.shape[-2]:
                                print("Error: Roi",roi,"is outside object bounds:",roi,"+",probe.shape,"=",obj.shape)
                                quit()
                        elif roi[0] < 0 or roi[1] < 0:
                                print("Error: Roi",roi,"has negative indexing.")
                                quit()

        if probesupp is not None:
                assert(probesupp.shape[-1] == probe.shape[-1] and probesupp.shape[-2] == probe.shape[-2] and probesupp.size == probe.size)
                probesupp = np.ascontiguousarray(probesupp.astype(np.float32))
                probesuppptr = probesupp.ctypes.data_as(void_p)
        else:
                probesuppptr = void_p(0)

        if objsupp is not None:
                assert(objsupp.size >= obj.size)
                objsupp = np.ascontiguousarray(objsupp.astype(np.float32))
                objsuppptr = objsupp.ctypes.data_as(void_p)
                if len(objsupp.shape) >= 3:
                        numobjsupport = int32(objsupp.shape[0])
                else:
                        numobjsupport = int32(1)
        else:
                objsuppptr = void_p(0)
                numobjsupport = int32(0)

        if flat is not None:
                assert(flat.shape[-1] == difpads.shape[-1] & flat.shape[-2] == difpads.shape[-2])
                flat = np.ascontiguousarray(flat.astype(np.float32))
        else:
                flat = np.ones(difpads.shape[-2:],dtype=np.float32)
        flatptr = flat.ctypes.data_as(void_p)

        if bkg is not None:
                assert(bkg.shape[-1] == difpads.shape[-1] & bkg.shape[-2] == difpads.shape[-2])
                bkg = np.ascontiguousarray(bkg.astype(np.float32))
                bkgptr = bkg.ctypes.data_as(void_p)
        else:
                bkgptr = void_p(0)

        if flatmask is not None:
                assert(flatmask.shape[-1] == difpads.shape[-1] & flatmask.shape[-2] == difpads.shape[-2])
                flatmask = np.ascontiguousarray(flatmask.astype(np.float32))
        else:
                flatmask = np.zeros(difpads.shape[-2:],dtype=np.float32)
        flatmaskptr = flatmask.ctypes.data_as(void_p)

        if bkgmask is not None:
                assert(bkgmask.shape[-1] == difpads.shape[-1] & bkgmask.shape[-2] == difpads.shape[-2])
                bkgmask = np.ascontiguousarray(bkgmask.astype(np.float32))
        else:
                bkgmask = np.zeros(difpads.shape[-2:],dtype=np.float32)
        bkgmaskptr = bkgmask.ctypes.data_as(void_p)

        if sigmask is not None:
                assert(sigmask.shape[-1] == difpads.shape[-1] & sigmask.shape[-2] == difpads.shape[-2])
                sigmask = np.ascontiguousarray(sigmask.astype(np.float32))
        else:
                sigmask = np.ones(difpads.shape[-2:],dtype=np.float32)
        sigmaskptr = sigmask.ctypes.data_as(void_p)

        obj = np.ascontiguousarray(obj.astype(np.complex64))
        objptr = obj.ctypes.data_as(void_p)

        probe = np.ascontiguousarray(probe.astype(np.complex64))
        probeptr = probe.ctypes.data_as(void_p)

        difpads = np.ascontiguousarray(difpads.astype(np.float32))
        difpadsptr = difpads.ctypes.data_as(void_p)

        rois = np.ascontiguousarray(rois.astype(np.float32))
        roisptr = rois.ctypes.data_as(void_p)
        
        rfactor = np.zeros((iter),dtype=np.float32)
        rfactorptr = rfactor.ctypes.data_as(void_p)

        numdev = int32(len(defdevice))
        devices = np.ascontiguousarray(np.asarray(defdevice).astype(np.int32))
        devicesptr = devices.ctypes.data_as(void_p)

        psizex = int32(probe.shape[-1])
        osizex = int32(obj.shape[-1])
        dsizex = int32(difpads.shape[-1])
        iter = int32(iter)
        batch = int32(batch)
        objbeta = float32(objbeta)
        probebeta = float32(probebeta)
        numrois = int32(rois.shape[0])
        flyscansteps=int32(rois.shape[1])
        epsilon=float32(epsilon)
        noisemodel = int32(noisemodel)
        
        nummodes = int32(probe.shape[0])
        if len(probe.shape) < 3:
                nummodes = int32(1)
        
        print("LSML:", difpads.shape, obj.shape, probe.shape, batch, iter, tvmu)
        time0 = time()

        psiccdll.LSQML(objptr, probeptr, difpadsptr, psizex, osizex, dsizex, roisptr, numrois, batch, iter, 
                numdev, devicesptr, rfactorptr, objbeta, probebeta, nummodes, float32(tvmu), 
                objsuppptr, probesuppptr, numobjsupport, sigmaskptr, flyscansteps, epsilon, noisemodel,
                bkgptr,flatptr,bkgmaskptr,flatmaskptr)

        print("Done in:",time()-time0,"s.")

        return obj, probe, rfactor, bkg, flat

def RemovePhaseGrad(img2):
    img = img2+0
    ft = np.fft.fft2(np.fft.fftshift(img))
    maxx = np.argmax(abs(ft))
    mx = maxx % ft.shape[1]
    my = maxx // ft.shape[1]

    ax = -2j * np.pi / ft.shape[1] * np.arange(ft.shape[1])
    ay = -2j * np.pi / ft.shape[0] * np.arange(ft.shape[0])

    xx,yy = np.meshgrid(ax,ay)
    img *= np.exp(mx*xx + my*yy)

    for k in range(1,10):
        ax = -2j * np.pi / ft.shape[1] * np.arange(-ft.shape[1]//2,ft.shape[1]//2)
        ay = -2j * np.pi / ft.shape[0] * np.arange(-ft.shape[0]//2,ft.shape[0]//2)

        xx,yy = np.meshgrid(ax,ay)
        eps = 0.5/1.4**k

        kxx = np.exp(xx*eps)
        kyy = np.exp(yy*eps)
        cxx = kxx.conj()
        cyy = kyy.conj()

        pg = abs(np.sum(img))
        x0 = abs(np.sum(img*kxx))
        x1 = abs(np.sum(img*cxx))

        if x0 > pg and x0 > x1:
            img *= kxx
        elif x1 > pg and x1 > x0:
            img *= cxx
        
        pg = abs(np.sum(img))
        y0 = abs(np.sum(img*kyy))
        y1 = abs(np.sum(img*cyy))

        if y0 > pg and y0 > y1:
            img *= kyy
        elif y1 > pg and y1 > y0:
            img *= cyy

    phase = np.sum(img)
    img *= phase.conj()/abs(phase)

    return img

def FRC(img1,img2):
        ft1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img1)))
        ft2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img2)))

        fg = ft1*ft2.conj()
        ft1 = abs(ft1)**2
        ft2 = abs(ft2)**2

        aar = np.arange(-ft1.shape[0]//2,ft1.shape[0]//2)
        xx,yy = np.meshgrid(aar,aar)

        rr = (np.sqrt(xx**2+yy**2)+0.5).astype(np.int32)
        frc = np.zeros((ft1.shape[0]//2),dtype=np.complex64)

        for r in range(ft1.shape[0]//2):
                frc[r] = np.sum(fg[rr==r])/np.sqrt(1E-7+np.sum(ft1[rr==r])*np.sum(ft2[rr==r]))

        return frc


def GetDifpads(phantom,probe,rj,photons=-1):
    sx = probe.shape[1]
    sy = probe.shape[0]
    difpads = []
    for r in rj:
        ew = phantom[r[1]:sy+r[1],r[0]:sx+r[0]] * probe
        difpad = abs(np.fft.fft2(ew)) / np.sqrt(ew.size)
        
        if photons > 0:
            difpad *= difpad
            ratt = photons/np.sum(difpad)
            difpad *= ratt

            difpad = np.random.poisson(difpad)
            difpad = np.sqrt(difpad / ratt)

        difpad = np.fft.fftshift(difpad)

        #hole = 5
        #difpad[sy//2-hole:sy//2+hole,sx//2-hole:sx//2+hole] = -1

        difpads.append( difpad.astype(np.float32) )

    #psicc.Show([np.fft.fftshift(np.log(1+difpads[78]))/np.log(10)])
    return difpads

def Error(obj,phantom):
    stx = psize
    sty = psize
    enx = 2048-psize
    eny = 2048-psize

    obj *= np.sum(obj[sty:eny,stx:enx].conj()*phantom[sty:eny,stx:enx])/np.sum(abs(phantom[sty:eny,stx:enx])**2)
    print('Total:',np.sqrt(np.sum(abs(obj[sty:eny,stx:enx]-phantom[sty:eny,stx:enx])**2)/(np.sum(abs(phantom[sty:eny,stx:enx])**2))+1E-10))

    angobj = np.angle(obj[sty:eny,stx:enx])
    angpha = np.angle(phantom[sty:eny,stx:enx])

    angobj -= np.average(angobj)
    angpha -= np.average(angpha)

    print('Phase:',np.sqrt(np.average((angpha-angobj)**2)))

from matplotlib.colors import hsv_to_rgb
def MakeRGB(Amps,Phases,bias=0): 	# Make RGB image from amplitude and phase
	HSV = np.zeros((Amps.shape[0],Amps.shape[1],3),dtype=np.float32)
	normalizer = (1.0-bias)/Amps.max()
	HSV[:,:,0] = Phases[:,:]
	HSV[:,:,1] = 1
	HSV[:,:,2] = Amps[:,:]*normalizer + bias
	return hsv_to_rgb(HSV)

def SplitComplex(ComplexImg):
	Phases = np.angle(ComplexImg)	# Phases in range [-pi,pi]
	Phases = Phases*0.5/np.pi + 0.5
	Amps = np.absolute(ComplexImg)
	return Amps,Phases

def CMakeRGB(ComplexImg,bias=0.01):
	Amps,Phases = SplitComplex(ComplexImg)
	return MakeRGB(Amps,Phases,bias)

def RemovePhaseGrad(img):
    objsize = img.shape[-1]//2
    ftimg = np.fft.fftshift(abs(np.fft.fft2(img)))
    x = 0
    y = 0
    bCont = False
    while not bCont:
        ref = ftimg[objsize+y,objsize+x]
        maxnei = max(ftimg[objsize+y+1,objsize+x],ftimg[objsize+y-1,objsize+x],ftimg[objsize+y,objsize+x+1],ftimg[objsize+y,objsize+x-1])
        if ref >= maxnei:
            bCont = True
        else:
            if ftimg[objsize+y+1,objsize+x] == maxnei:
                y += 1
            elif ftimg[objsize+y-1,objsize+x] == maxnei:
                y -= 1
            elif ftimg[objsize+y,objsize+x+1] == maxnei:
                x += 1
            elif ftimg[objsize+y,objsize+x-1] == maxnei:
                x -= 1
            else:
                print('error!')
                exit(-1)

    y = np.argmax(ftimg)//ftimg.shape[-1] - ftimg.shape[-2]//2
    x = np.argmax(ftimg)%ftimg.shape[-1] - ftimg.shape[-1]//2

    objsize = img.shape[-1]
    ftimg = np.fft.fftshift(abs(np.fft.fft2(np.pad(img,[[0,objsize],[0,objsize]]))))
    x *= 2
    y *= 2
    bCont = False
    while not bCont:
        ref = ftimg[objsize+y,objsize+x]
        maxnei = max(ftimg[objsize+y+1,objsize+x],ftimg[objsize+y-1,objsize+x],ftimg[objsize+y,objsize+x+1],ftimg[objsize+y,objsize+x-1])
        if ref >= maxnei:
            bCont = True
        else:
            if ftimg[objsize+y+1,objsize+x] == maxnei:
                y += 1
            elif ftimg[objsize+y-1,objsize+x] == maxnei:
                y -= 1
            elif ftimg[objsize+y,objsize+x+1] == maxnei:
                x += 1
            elif ftimg[objsize+y,objsize+x-1] == maxnei:
                x -= 1
            else:
                print('error!')
                exit(-1)
                
    #img = np.fft.ifft2(np.roll(np.roll(np.fft.fft2(img),-x,1),-y,0))

    xx,yy = np.meshgrid(np.arange(-img.shape[-1]//2,img.shape[-1]//2),np.arange(-img.shape[-2]//2,img.shape[-2]//2))

    print(x,y)
    img *= np.exp(-1j*np.pi*(xx*x/float(img.shape[-1])+yy*y/float(img.shape[-2])))
    
    globalphase = np.sum(img/np.sqrt(abs(img)+1E-20))
    img *= globalphase.conj() / abs(globalphase)

    phi = np.angle(img)
    phi -= np.average(phi)
    A = np.average(phi*xx)/np.average(xx**2)
    B = np.average(phi*yy)/np.average(yy**2)

    print(A,B)

    img *= np.exp(-1j*(xx*A+yy*B))

    return img#[objsize//3:2*objsize//3,objsize//3:2*objsize//3]

def CoherentModes(obj,probe,difpads,rois,iter,batch=1,objbeta=0.97,probebeta=0.93,tvmu=-1,weights=None,
objsupp=None,probesupp=None,sigmask=None,epsilon=1E-3,bkg=None):
        #if device==-1:
        #        device = defdevice

        if rois.shape[0] != difpads.shape[0]:
                print("Error:",difpads.shape[0],"difpads  v ",rois.shape[0]," rois mismatch")
                quit()
        if rois.shape[-1] != 4:
                print("Incorrect roi specification: shape=" , rois.shape, ' rois has 4 attribs: x,y,exptime,I0')
                quit()
        
        if probe.shape[-1]%difpads.shape[-1] != 0 or probe.shape[-2]%difpads.shape[-2] != 0:
                print("Error:",probe.shape,"probe  v ",difpads.shape," difpads shape mismatch")
                quit()

        if len(rois.shape) == 2:
                rois = rois[:,None]

        for acqui in rois:
                for roi in acqui:
                        if probe.shape[-1] + roi[0] > obj.shape[-1] or probe.shape[-2] + roi[1] > obj.shape[-2]:
                                print("Error: Roi",roi,"is outside object bounds:",roi,"+",probe.shape,"=",obj.shape)
                                quit()
                        elif roi[0] < 0 or roi[1] < 0:
                                print("Error: Roi",roi,"has negative indexing.")
                                quit()

        if probesupp is not None:
                assert(probesupp.shape[-1] == probe.shape[-1] and probesupp.shape[-2] == probe.shape[-2] and probesupp.size == probe.size)
                probesupp = np.ascontiguousarray(probesupp.astype(np.float32))
                probesuppptr = probesupp.ctypes.data_as(void_p)
        else:
                probesuppptr = void_p(0)

        if bkg is not None:
                assert(bkg.shape[-1] == difpads.shape[-1] & bkg.shape[-2] == difpads.shape[-2])
                bkg = np.ascontiguousarray(bkg.astype(np.float32))
                bkgptr = bkg.ctypes.data_as(void_p)
        else:
                bkgptr = void_p(0)

        if objsupp is not None:
                assert(objsupp.size >= obj.size)
                objsupp = np.ascontiguousarray(objsupp.astype(np.float32))
                objsuppptr = objsupp.ctypes.data_as(void_p)
                if len(objsupp.shape) >= 3:
                        numobjsupport = int32(objsupp.shape[0])
                else:
                        numobjsupport = int32(1)
        else:
                objsuppptr = void_p(0)
                numobjsupport = int32(0)

        obj = np.ascontiguousarray(obj.astype(np.complex64))
        objptr = obj.ctypes.data_as(void_p)

        probe = np.ascontiguousarray(probe.astype(np.complex64))
        probeptr = probe.ctypes.data_as(void_p)

        difpads = np.ascontiguousarray(difpads.astype(np.float32))
        difpadsptr = difpads.ctypes.data_as(void_p)

        if sigmask is not None:
                assert(sigmask.shape[-1] == difpads.shape[-1] & sigmask.shape[-2] == difpads.shape[-2])
                sigmask = np.ascontiguousarray(sigmask.astype(np.float32))
        else:
                sigmask = np.ones(difpads.shape[-2:],dtype=np.float32)
        sigmaskptr = sigmask.ctypes.data_as(void_p)

        rois = np.ascontiguousarray(rois.astype(np.float32))
        roisptr = rois.ctypes.data_as(void_p)

        weights = np.ascontiguousarray(weights.astype(np.complex64))
        weightsptr = weights.ctypes.data_as(void_p)
        

        rfactor = np.zeros((iter),dtype=np.float32)
        rfactorptr = rfactor.ctypes.data_as(void_p)

        numdev = int32(len(defdevice))
        devices = np.ascontiguousarray(np.asarray(defdevice).astype(np.int32))
        devicesptr = devices.ctypes.data_as(void_p)

        psizex = int32(probe.shape[-1])
        osizex = int32(obj.shape[-1])
        dsizex = int32(difpads.shape[-1])
        iter = int32(iter)
        batch = int32(batch)
        objbeta = float32(objbeta)
        probebeta = float32(probebeta)
        numrois = int32(rois.shape[0])
        epsilon=float32(epsilon)
        
        nummodes = int32(probe.shape[0])
        if len(probe.shape) < 3:
                nummodes = int32(1)
        
        print("Mixed:", difpads.shape, obj.shape, probe.shape, batch, iter, tvmu)
        
        time0 = time()

        psiccdll.CoherentModes(objptr, probeptr, difpadsptr, psizex, osizex, dsizex, roisptr, numrois, batch, iter, 
                numdev, devicesptr, rfactorptr, objbeta, probebeta, nummodes, float32(tvmu), 
                        objsuppptr, probesuppptr, numobjsupport, sigmaskptr, epsilon, weightsptr, bkgptr)

        print("Done in:",time()-time0,"s.")

        mydict = {}
        mydict['obj'] = obj
        mydict['probe'] = probe
        mydict['error'] = rfactor
        mydict['bkg'] = bkg
        mydict['weights'] = weights
        return mydict

        #return obj, probe, rfactor, weights, bkg
