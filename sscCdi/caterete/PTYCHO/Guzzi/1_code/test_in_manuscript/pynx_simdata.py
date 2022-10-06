
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


#############

#darkpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan118_dark/scan118_dark_n_image_02_exp_0.4.tif'
#darkpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan118_dark/scan118_dark_n_image_03_exp_0.7.tif'
#dark = tf.imread(darkpath)*1.0
#dark = dark.mean(axis=0)

#diffpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan107/scan107_n_image_02_exp_0.4.tif'
#diffpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan107/scan107_n_image_03_exp_0.7.tif'
#diffpath = '/net/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan_80/twinmic_scan_80_darkcorr_autocropped_1256.tif'
diffpath = '/net/memory4twinmic/store/project/twinmic/scicomp/tesla-storage/software_paper_peerJ/simulator_zernike/synthdata/diatom_synth_new.tif'
diffdata = tf.imread(diffpath)*1.
#diffdata = diffdata - dark
diffdata[diffdata<0] = 0

#ccoord = 643, 645
ccoord = None

DIMSAMPLE = 1024


#diffdata = cropcenter(diffdata, DIMSAMPLE, ccoord)
#diffdata = rebin(diffdata)
#diffdata = rebin(diffdata)
#diffdata *= np.expand_dims(makeGaussian(diffdata.shape[-1], fwhm=diffdata.shape[-1]*0.85), 0)


#pospath = '/net/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan_80/pos_twinmic_scan_80.npy'
pospath = '/net/memory4twinmic/store/project/twinmic/scicomp/tesla-storage/software_paper_peerJ/simulator_zernike/synthdata/synthpos_new.npy'

##maskpath = '/mnt/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan107/mask_ccd_107.tif'
##mask = tf.imread(maskpath)*1.0
##mask /= mask.max()
##mask = cropcenter(mask, DIMSAMPLE, ccoord)
#mask = rebin(mask)
##mask = mask ==0
mask = None

outrecon = '/net/memory4twinmic/store/project/twinmic/20205474/DONKIORCHESTRA/proc_data/scan_80/out/'

#positions = np.load(pospath).T*1e-6
positions = np.load(pospath)

#diffdata = diffdata[::2]
#positions = positions[::2]




wvl = 4.892e-10; #wavelength
fd = 0.8932; #focal to detector
fs = 2.5e-3; #focal to sample
psize = 13.5e-6; #pixel size
distance = fd -fs
pixel_size_object = wvl * (distance)/(diffdata.shape[-1]*psize) #already rebinned

print('Obj plane res: {:.2f} nm'.format(pixel_size_object*1e9))
print('Pos shape', positions.shape)


x,y = positions[:,0], positions[:,1]
x -= x.min()
y -= y.min()
data = PtychoData(iobs=diffdata/diffdata.max(), positions=(x,y), detector_distance=distance, mask=mask, pixel_size_detector=psize, wavelength=wvl, near_field=False)

nprobes = 7
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
p = AP(update_object=True, update_probe=True, update_pos=False, calc_llk=10, show_obj_probe=10) ** 200 * p
#p = AP(update_object=True, update_probe=False, update_pos=True, calc_llk=100, show_obj_probe=100) **  500 * p
#p = DM(update_object=True, update_probe=True, update_pos=True, calc_llk=100, show_obj_probe=10) ** 1000 * p
#p = ML(update_object=True, update_probe=True, update_pos=False, calc_llk=100, show_obj_probe=25) ** 500 * p

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
    
view(p.get_obj())
view(p.get_probe())


print('end')

