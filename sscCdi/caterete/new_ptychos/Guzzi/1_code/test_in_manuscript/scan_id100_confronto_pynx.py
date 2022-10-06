
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tifffile as tf
from skimage.restoration import unwrap_phase
from skimage.transform import rescale, resize

from joblib import Parallel, delayed

from ijshow import view

from pynx.ptycho import Ptycho, PtychoData, shape, ScaleObjProbe, ML, DM, AP
import pynx.wavefront as wavefront

from numpy.fft import fftshift, ifftshift, fft2, ifft2
from skimage.filters import window
from skimage.restoration import unwrap_phase

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


def loadptyd(path):
    with h5py.File(path, 'r') as f:
        data = f['chunks/0/data'][()]
        positions = f['chunks/0/positions'][()]
        weights = f['chunks/0/weights'][()]
        distance = f['meta/distance'][()]
        energy = f['meta/energy'][()]
        psize = f['meta/psize'][()]
        wvl = f['info/scan_info/wavelength'][()]
    return data, positions, weights, distance, energy, psize, wvl


def genCircleMask(imsizepx, pixrad, offsetrow=0, offsetcol=0, dtype=np.float32):
    # generates centered disk as default
    imsizepx = (int(np.ceil(imsizepx)), int(np.ceil(imsizepx)))
    x, y = np.indices(imsizepx).astype(float)
    x1, y1 = imsizepx[0] / 2.0, imsizepx[1] / 2.0
    x1, y1 = x1 - offsetrow, y1 + offsetcol
    mask1 = (x - x1) ** 2 + (y - y1) ** 2 < pixrad ** 2
    return (1.0*mask1).astype(dtype)


def gen_linspace(N):
    if N%2 == 0:
        return np.linspace(-N/2.,N/2. -1,N)
    else:
        return np.linspace(-(N-1)/2.,(N-1)/2. -1,N)


    
def sphericalWave(N, ilambda, dx, z0, ap):
    k = 2.*np.pi/ilambda
    xx = gen_linspace(N) * dx
    uu,vv = np.meshgrid(xx,xx)
    xyrad = uu**2 + vv**2
    mask =  xyrad < ap**2
    mask = mask/mask.max()
    rdist = np.sqrt(xyrad + z0**2)
    p = mask * np.exp(-1j * np.sign(z0) * k * rdist)/rdist
    p = p/np.abs(p).max()
    return p, mask



def cropcenter(img, N, center=None):
    imgshape = img.shape
    h, w = img.shape[-2], img.shape[-1]
    if center is None:
        sr, sc = h/2, w/2
    else:
        sr, sc = center        
    if len(imgshape) == 2:
        return img[int(sr-N/2):int(sr-N/2)+N, int(sc-N/2):int(sc-N/2) + N]
    elif len(imgshape) == 3:
        return img[:, int(sr-N/2):int(sr-N/2)+N, int(sc-N/2):int(sc-N/2) + N]

    

def makeGaussian(size, fwhm = 3, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def rebin(img):
    if len(img.shape) == 3:
        return img[:, ::2,::2] + img[:, 1::2,::2] + img[:, ::2,1::2] + img[:, 1::2,1::2]
    else:
        return img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]


def pad(img, h, w, cval=0):
    #  in case when you have odd number
    top_pad = int(np.floor((h - img.shape[0]) / 2.))
    bottom_pad = int(np.ceil((h - img.shape[0]) / 2.))
    right_pad = int(np.ceil((w - img.shape[1]) / 2.))
    left_pad = int(np.floor((w - img.shape[1]) / 2.))
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=cval))


def phasefac(x, size):
    N,M = size
    u = np.linspace(-N//2, N//2, N)
    v = np.linspace(-M//2, M//2, M)
    u, v = np.meshgrid(u,v)
    return np.exp(-2j*np.pi*(u*x[0] + v*x[1]))


def fixphase(imgc):
    n, m = imgc.shape[0]//2, imgc.shape[1]//2
    X = fftshift(fft2(imgc*window('hann', imgc.shape)))
    maxcoord = np.argmax(np.abs(X))
    mr,mc = np.unravel_index(maxcoord, imgc.shape)
    print('max idx', mr, mc)
    newimg = imgc*phasefac((mc-m, mr-n), imgc.shape)
    return newimg, (mc-m, mr-n)


#############

#darkpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan118_dark/scan118_dark_n_image_02_exp_0.4.tif'
#darkpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan118_dark/scan118_dark_n_image_03_exp_0.7.tif'
#dark = tf.imread(darkpath)*1.0
#dark = dark.mean(axis=0)


diffpath = '/storage/slow/francesco/nuovo_scicomp_rpiebetterthanepie/ssid4_new_darkcorr_autocropped_1284.tif'
diffdata = tf.imread(diffpath)*1.
#diffdata = diffdata - dark
diffdata[diffdata<0] = 0

#diffdata = np.rot90(diffdata, 3, axes=(1,2) )

#ccoord = 643, 645
ccoord = None

DIMSAMPLE = 1250
#diffdata = cropcenter(diffdata, DIMSAMPLE, ccoord)
diffdata = rebin(diffdata)
#diffdata = rebin(diffdata)
#diffdata *= np.expand_dims(makeGaussian(diffdata.shape[-1], fwhm=diffdata.shape[-1]*0.85), 0)


pospath = '/storage/slow/francesco/nuovo_scicomp_rpiebetterthanepie/ssid4pos.npy'

##maskpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan107/mask_ccd_107.tif'
##mask = tf.imread(maskpath)*1.0
##mask /= mask.max()
##mask = cropcenter(mask, DIMSAMPLE, ccoord)
#mask = rebin(mask)
##mask = mask ==0
mask = None

outrecon = '/net/memory4twinmic/store/project/twinmic/scicomp/tesla-storage/software_paper_peerJ/data_real/'

#positions = np.load(pospath).T*1e-6
positions = load_positions(pospath)

plt.figure(); plt.plot(positions[:,0], positions[:,1]); plt.show()

#diffdata = diffdata[::2]
#positions = positions[::2]

psize = 2*20e-6
fs = 350e-6
en = 1495
distance = 0.7515
wvl = (6.62606957e-34 * 299792458.) / (en * 1.602176565e-19) # wavelength in meter
pixel_size_object = wvl * (distance-fs)/(diffdata.shape[-1]*psize)

print('fs', fs)
print('Obj plane res: {:.2f} nm'.format(pixel_size_object*1e9))
print('Pos shape', positions.shape)

# y,x
y,x  = positions[:,0], positions[:,1]
x -= x.min()
y -= y.min()
data = PtychoData(iobs=diffdata/diffdata.max(), positions=(x,y), detector_distance=distance, mask=mask, pixel_size_detector=psize, wavelength=wvl, near_field=False)

nprobes = 1
#probeinit = genCircleMask(diffdata.shape[-1], 0.25*diffdata.shape[-1])
#probeinit = np.expand_dims(probeinit,0)# * np.ones((3,1,1))
#probeinit = probeinit + 0.1*np.random.randn(nprobes ,diffdata.shape[-2], diffdata.shape[-1])
#probeinit *= np.linspace(0.1, 1, nprobes)[::-1].reshape(nprobes, 1,1)

probeinit = sphericalWave(diffdata.shape[-1], wvl, pixel_size_object, -fs, 6e-6)[0]
probeinit = np.expand_dims(probeinit, 0)
if nprobes >1:
    probeinit = probeinit + 1e-3 * np.random.randn(nprobes, probeinit.shape[-2], probeinit.shape[-1])
    probeinit = probeinit * np.linspace(0.1,1, nprobes)[::-1].reshape(nprobes, 1, 1)


##view(probeinit)
print('Probe init', probeinit.shape)

# calculate final obj size
nx0, ny0 = shape.calc_obj_shape(x/pixel_size_object, y/pixel_size_object, (diffdata.shape[-2], diffdata.shape[-1]))
# obj init
obj0 = 0.1* np.exp(1j * np.random.uniform(0, 0.5, (nx0, ny0)))
# obj0 = np.ones((nx0, ny0)).astype(complex)
# main ptycho obj
p = Ptycho(probe=probeinit, obj=obj0, data=data, background=None) # shift false means diffdata are not fftshifted
# initial scaling
p = ScaleObjProbe() * p

# Optimize
# p = DM(update_object=True, update_probe=True, calc_llk=100, show_obj_probe=10) ** 100 * p
p = AP(update_object=True, update_probe=True, update_pos=True, calc_llk=20, show_obj_probe=20) ** 200 * p
p = DM(update_object=True, update_probe=True, update_pos=True, calc_llk=20, show_obj_probe=20) ** 600 * p
#p = ML(update_object=True, update_probe=True, update_pos=True, calc_llk=100, show_obj_probe=25) ** 200 * p

### save output
##objarray, probearray = p._obj[0], p._probe[0]
##objmag, probemag = np.abs(objarray), np.abs(probearray)
##objphase, probeangle = np.angle(objarray), np.angle(probearray)
##
##plt.figure()
##plt.subplot(221); plt.imshow(objmag); plt.title('Obj mag'); plt.colorbar()
##plt.subplot(222); plt.imshow(objphase); plt.title('Obj phase'); plt.colorbar()
##plt.subplot(223); plt.imshow(probemag); plt.title('Probe mag'); plt.colorbar()
##plt.subplot(224); plt.imshow(probeangle); plt.title('Probe phase'); plt.colorbar()
##plt.tight_layout(); plt.show()
imgc = p.get_obj()[0]

n, m = imgc.shape[-2]//2, imgc.shape[-1]//2


X = fftshift(fft2(imgc*window('hann', imgc.shape)))
newimg, traslcoor = fixphase(imgc)
newimgX =  fftshift(fft2(newimg* window('hann', imgc.shape)))
#maxcoord = np.argmax(np.abs(newimgX))
#mr,mc = np.unravel_index(maxcoord, imgc.shape)
#print('max idx', mr, mc)
#view([np.log10(1+np.abs(X)), np.log10(1+np.abs(newimgX))])

view([np.abs(newimg), unwrap_phase(np.angle(newimg))])
#view(p.get_probe())


print('end')

