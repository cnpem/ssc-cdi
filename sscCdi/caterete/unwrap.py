import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from sscPtycho import Show, RemovePhaseGrad
from skimage.restoration import unwrap_phase

def RemoveZernike(img,mask):
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

def RemoveGrad(img,mask):
    hs1 = img.shape[-1]//2
    hs2 = img.shape[-2]//2 
    # print(hs1,hs2)
    #hs = 256
    xx,yy = np.meshgrid(np.arange(-hs1,hs1) / float(hs1),np.arange(-hs2,hs2) / float(hs2))
    #img = 3*xx**2 + 2*yy**2 + 4*xx + 7 * yy - 1
    
    # print('xxm shape:',xx.shape)
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
    # print(gradient)
    return img + gradient[0]*xx + gradient[1]*yy + gradient[2]
    #Show([img + gradient[0]*xx**2 + gradient[1]*yy**2 + gradient[2]*xx + gradient[3]*yy + gradient[4]])



def tt(img,R):
    zernike = unwrap_phase(img)
    
    mask = zernike < 0
    
    for j in range(R):
        zernike = RemoveGrad(zernike,mask=mask)
        mask = abs(zernike) < 2**-j
        # Show([zernike,mask],cmap='gray')

    # zernike[zernike<0] = 0
    # zernike = RemoveGrad(zernike,mask=mask)
    
    return zernike
