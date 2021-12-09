import sscResolution
import sscPtycho
import sscCdi
import sscIO      
from sscPimega import pi540D

from sys import argv
import h5py
import pandas as pd
import json
import numpy as np

from math import e
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from time import time

from operator import sub
import numpy as np
import math
import os
import pandas as pd
import h5py

from numpy.fft import fftshift as shift
from numpy.fft import ifftshift as ishift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
#
#
# MODULES FOR THE FINAL APPLICATION 
# (see main code below)
#
#
#
#+++++++++++++++++++++++++++++++++++++++++++++++++

def plotshow(imgs, file, subplot_title = [],legend=[],cmap='jet',nlines=1, bLog = False, interpolation='bilinear'): # legend = plot titles
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
                        sb.imshow(sscPtycho.CMakeRGB(imgs[j]),cmap='hsv',interpolation=interpolation)
                elif bLog:
                        sb.imshow(np.log(1+np.maximum(imgs[j],-0.1))/np.log(10),cmap=colormap,interpolation=interpolation)
                else:
                        sb.imshow(imgs[j],cmap=colormap,interpolation=interpolation)

                if len(legend)>j:
                        sb.set_title(legend[j])

                sb.set_yticks([])
                sb.set_xticks([])
                sb.set_aspect('equal')
                if subplot_title != []:
                        sb.set_title(subplot_title[j])
        # plt.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(file + '.png', format='png', dpi=300)
        plt.clf()
        plt.close()

def Geometry( L ):
#     det = pi540D.get_detector_dictionary( L )
    xdet = pi540D.get_project_values_geometry()
    det = pi540D.get_detector_dictionary( xdet, L )
    geo = pi540D.geometry540D( det )
    return geo


def pre_processing_Giovanni(img,args):

        def Restaurate(img,geom):
          return pi540D.backward540D( img, geom )

        def UnRestaurate(img,geom):
                return pi540D._worker_annotation_image( pi540D.forward540D( img, geom ) )

        Binning, empty, flat, cx, cy, hsize, geometry = args

        binning = Binning+0
        img[empty>1] = -1

        img = img * flat

        img = img.astype(np.float32)
        img = Restaurate(img,geometry)
        img0 = img+0

        img[img<0] = -1
        img = img[cy-hsize:cy+hsize,cx-hsize:cx+hsize]

        ###### Binning
        while binning%2 == 0 and binning > 0:
                avg = img+np.roll(img,-1,-1)+np.roll(img,-1,-2)+np.roll(np.roll(img,-1,-1),-1,-2) # sum 4 neigboors at the top-left value

                div = 1*(img>=0)+np.roll(1*(img>=0),-1,-1)+np.roll(1*(img>=0),-1,-2)+np.roll(np.roll(1*(img>=0),-1,-1),-1,-2) # Boolean array! Resulta in the n of valid points in the 2x2 neighborhood

                avg = avg + 4 - div # results in the sum of valid points only. +4 factor needs to be there to compensate for -1 values that exist when there is an invalid neighbor

                avgmask = (img<0) & (div>0) # div > 0 means at least 1 neighbor is valid. 
                
                img[avgmask] = avg[avgmask]/div[avgmask] # sum of valid points / number of valid points IF NON-NULL REGION and IF TOP-LEFT VALUE INVALID. What about when all 4 pixels are valid? No normalization in that case?

                img = img[:,0::2] + img[:,1::2] # Binning columns
                img = img[0::2]   + img[1::2]   # Binning lines

                img[img<0] = -1

                img[div[0::2,0::2]<3] = -1 # why div < 3 ?

                binning = binning//2

        while binning%3 == 0 and binning > 0:
                avg = np.roll(img,1,-1)+np.roll(img,-1,-1)+np.roll(img,-1,-2)+np.roll(img,1,-2)+np.roll(np.roll(img,1,-2),1,-1)+np.roll(np.roll(img,1,-2),-1,-1)+np.roll(np.roll(img,-1,-2),1,-1)+np.roll(np.roll(img,-1,-2),-1,-1)
                div = np.roll(img>0,1,-1)+np.roll(img>0,-1,-1)+np.roll(img>0,-1,-2)+np.roll(img>0,1,-2)+np.roll(np.roll(img>0,1,-2),1,-1)+np.roll(np.roll(img>0,1,-2),-1,-1)+np.roll(np.roll(img>0,-1,-2),1,-1)+np.roll(np.roll(img>0,-1,-2),-1,-1)

                avgmask = (img<0) & (div>0)/div[avgmask]

                img = img[:,0::3] + img[:,1::3] + img[:,2::3]
                img = img[0::3] + img[1::3] + img[2::3]

                img[img<0] = -1
                binning = binning//3

        if binning > 1:
                print('Entering binning > 1 only')
                avg = -img*1.0 + binning**2 - 1
                div = img*0
                for j in range(0,binning):
                        for i in range(0,binning):
                                avg += np.roll(np.roll(img,j,-2),i,-1)
                                div += np.roll(np.roll(img>0,j,-2),i,-1)

                avgmask = (img<0) & (div>0)
                img[avgmask] = avg[avgmask]/div[avgmask]

                imgold = img+0
                img = img[0::binning,0::binning]*0
                for j in range(0,binning):
                        for i in range(0,binning):
                                img += imgold[j::binning,i::binning]

                img[img<0] = -1

        return img
  


def cat_restauration(jason,path,name):
        import os
        fullpath = os.path.join(path,name) 

        # Here comes the distance Geometry(Z1):
        z1 = float(jason["DetDistance"])
        z1 = z1*1000

        geometry = Geometry(z1)

        empty = np.asarray(h5py.File(jason['EmptyFrame'],'r')['/entry/data/data']).squeeze().astype(np.float32)
        # empty = np.load('masks/empty_zeros.npy')
        
        sscCdi.caterete.misc.plotshow_cmap2(empty,title=f"{jason['EmptyFrame'].split('/')[-1]}",savepath=jason["PreviewFolder"]+'/00_empty.png')

        em = empty > 0
        
        cx = 1419 
        cy = 1395 
        hsize = 1280 #(2560/2)

        Binning = int(jason['Binning'])

        flat = np.load(jason['FlatField'])
        flat = np.array(flat)
        flat[np.isnan(flat)] = -1
        flat[flat==0] = 1

        h5f,_ = sscIO.io.read_volume(fullpath,'numpy', use_MPI=True, nprocs=32)

        r_params =  (Binning, empty, flat, cx, cy, hsize, geometry)
        output,_ = pi540D.backward540D_batch( h5f, z1, jason['Threads'], [hsize//2, hsize//2], pre_processing_Giovanni, r_params, 'only')

        return output

def read_probe_positions(probe_positions_filepath,measure):
        print('Reading probe positions (probe_positions)...')
        probe_positions = []
        positions_file = open(probe_positions_filepath)

        line_counter = 0
        for line in positions_file:
            line = str(line)
            if line_counter > 1: # skip first line, which is the header
                T = -3E-3 # why rotate by this amount?
                pxl = float(line.split()[1]) 
                pyl = float(line.split()[0])
                px = pxl*np.cos(T) - np.sin(T)*pyl
                py = pxl*np.sin(T) + np.cos(T)*pyl
                probe_positions.append([px,py,1,1])
            line_counter += 1
        
        probe_positions = np.asarray(probe_positions)

        pshape = pd.read_csv(probe_positions_filepath,sep=' ').shape # why read pshape from file? can it be different from probe_positions.shape+1?
        
        with h5py.File(measure, 'r') as file:
            mshape = file['entry/data/data'].shape
        
        if pshape[0] == mshape[0]: # check if number of recorded beam positions in txt matches the positions saved to the hdf
                print('\tSuccess in read positions file:' + probe_positions_filepath)
                print("\tShape probe_positions:",probe_positions.shape,pshape,mshape)
        else:
                print("\tError in probe_positions shape. {0} is different from diffraction pattern shape {1}".format(probe_positions.shape,mshape))
                print('\t\t Setting object as null array with correct shape.')
                # probe_positions = np.zeros([1,1,1,1]) 
                probe_positions = np.zeros((mshape[0],4))

        return probe_positions


def create_squared_mask(start_row,start_column,height,width,mask_shape):
        """ Create squared mask. Start position is the top-left corner. All values in pixels!
        """
        mask = np.zeros(mask_shape)
        mask[start_row:start_row+height,start_column:start_column+width] = 1
        return mask

def create_circular_mask(center_row, center_col,radius,mask_shape):
        print('Using manually set circular mask to the diffraction pattern...')
        """ All values in pixels """
        mask = np.zeros(mask_shape)
        y_array = np.arange(0,mask_shape[0],1)
        x_array = np.arange(0,mask_shape[1],1)

        Xmesh, Ymesh = np.meshgrid(x_array,y_array)

        mask = np.where( (Xmesh-center_col)**2 + (Ymesh-center_row)**2 <= radius**2, 1, 0)
        return mask


def set_initial_parameters(jason,difpads,probe_positions,radius,center_x,center_y,maxroi,dx):

        hsize = difpads.shape[-1]//2

        if jason['f1'] == -1: # Manually choose wether to find Fresnel number automatically or not
            jason['f1'] = setfresnel(dx, pixel=jason['RestauredPixelSize'], energy=jason['Energy'],z=jason['DetDistance'])
            jason['f1'] = -jason['f1']
        print('\tF1 value:',jason['f1']) 
       
        # Compute probe: initial guess:
        probe = set_initial_probe(difpads,jason)
       
        # Adicionar modulos incoerentes
        probe = set_modes(probe,jason)

        # GPUs selection:
        set_gpus(jason)

        # Object initial guess:
        obj = set_initial_obj(jason,hsize,maxroi,probe,difpads)

        # Mask of 1 and 0:
        sigmask = set_sigmask(difpads)

        # Background: better not use any for now.
        background = set_background(difpads,jason)

        # Compute probe support:
        probesupp = probe_support(probe,hsize,radius,center_x,center_y)

        probe_positionsi = probe_positions+0 # what's the purpose of declaring probe_positionsi?

        # Set data for Ptycho algorithms:
        datapack = set_datapack(obj,probe,probe_positions,difpads,background,probesupp)

        return datapack,probe_positionsi,sigmask,hsize,maxroi, probe_positions

def set_parameters(difpads,jason,probe_positions):
        print('Setting parameters hsize, dx and maxroi ...')
        # Compute half size of diffraction patterns:
        hsize = difpads.shape[-1]//2

        c = 299792458             # Velocity of Light [m/s]
        planck = 4.135667662E-18  # Plank constant [keV*s]

        # Compute/convert pixel size: 
        dx = planck*c/jason['Energy'] * jason['DetDistance'] / (jason['Binning']*jason['RestauredPixelSize']*hsize*2)
        print('\tConverted to pixel size:',dx)
        
        # Get probe_positions minimum and a threshold.
        probe_positions[:,0] -= np.min(probe_positions[:,0])
        probe_positions[:,1] -= np.min(probe_positions[:,1])

        offset_topleft = 10 
        offset_bottomright = offset_topleft 
        probe_positions[:,0] = 1E-6*probe_positions[:,0]/dx + offset_topleft 
        probe_positions[:,1] = 1E-6*probe_positions[:,1]/dx + offset_topleft 

        # Compute max probe_positions and size object (2*hsize+maxroi)
        maxroi = int(np.max(probe_positions)) + offset_bottomright

        print(f'\tmaxroi: {np.max(probe_positions)}, int(maxroi):{maxroi}')
        print("\tObject shape: 2*hsize+maxroi=",2*hsize+maxroi)
        
        return dx,maxroi, probe_positions

def setfresnel(dx=1, pixel=55.55E-6, energy=3.8E3,z=1):
        print('Setting Fresnel number automatically...')
        c = 299792458    # Velocity of Light [m/s]
        plank = 4.135667662E-15  # Plank constant [ev*s]
        const = (plank * c)
        wave =  const / (energy*1000)   # [m]  waveleght 

        magn = pixel / dx   # m 
        F1 = ( (dx * dx) * magn ) / (wave * z )
        
        print('\tFresnel number (F1) - F1:',F1)
        print('\tMagnification:',magn)
        print('\tEffective Pixel size:', dx)

        return F1

def set_initial_probe(difpads,jason):
        print('Setting initial probe...')
        # Compute probe: initial guess:
        if jason['InitialProbe'] == "":
            # Initial guess for none probe:
            probe = np.average(difpads,0)[None] 
            ft = shift(fft2(shift(probe)))
            probe = np.sqrt(shift(ifft2(shift(ft))))
        else:
            # Load probe:
            print(jason['InitialProbe'])
            probe = np.load(jason['InitialProbe'])[0]

        print("\tProbe shape:",probe.shape)
        return probe

def set_modes(probe,jason):
        print('Setting modes...')
        mode = probe.shape[0]
        print('\tNumber of modes:',mode)
        # Adicionar modulos incoerentes
        if jason['Modes'] > mode:
            add = jason['Modes'] - mode
            probe = np.pad(probe,[[0,int(add)],[0,0],[0,0]])
            for i in range(add): 
                probe[i + mode] = probe[i + mode - 1]*np.random.rand(*probe[0].shape)

        print("\tProbe shape ({0},{1}) with {2} incoherent modes".format(probe.shape[-2],probe.shape[-1],probe.shape[0]))

        return probe

def set_gpus(jason):
        print('Setting GPUs...')
        if jason['GPUs'][0] < 0:
                jason['GPUs'] = [0,1,2,3]
        sscPtycho.SetDevices(jason['GPUs'])

def set_initial_obj(jason,hsize,maxroi,probe,difpads):
        print('Setting initial guess for Object...')
        # Object initial guess:
        if jason['InitialObj'] == "":
            obj = np.random.rand(2*hsize+maxroi,2*hsize+maxroi) * (np.sqrt(np.average(difpads)/np.average(abs(np.fft.fft2(probe))**2)))
            #obj = np.random.rand(2048,2048) * (np.sqrt(np.average(difpads)/np.average(abs(np.fft.fft2(probe))**2)))
        else:
            obj = np.load(jason['InitialObj'])
        
        return obj

def set_sigmask(difpads):
        # Mask of 1 and 0:
        sigmask = np.ones(difpads[0].shape)
        sigmask[difpads[0]<0] = 0

        return sigmask


def set_background(difpads,jason):
        print('Setting background...')
        # Background: better not use any for now.
        if jason['InitialBkg'] == "": 
                print('\tUsing no background!')
                background = np.zeros(difpads[0].shape)
        else: 
                try:
                        background = np.maximum(abs(np.load(jason['ReconPath']+jason['InitialBkg'])),1)
                except:
                        background = np.ones(difpads[0].shape)

        return background

def probe_support(probe, hsize, radius,center_x, center_y):
        print('Setting probe support...')
        # Compute probe support:
        ar = np.arange(-hsize,hsize)
        xx,yy = np.meshgrid(ar,ar)
        probesupp = (xx+center_x)**2+(yy+center_y)**2 < radius**2 # offset of 30 chosen by hand?
        probesupp = np.asarray([probesupp for k in range(probe.shape[0])])

        # No support:
        #probesupp = probesupp*0 + 1

        return probesupp

# Propagation:
def Prop(img,f1):
        # See paper "Memory and CPU efficient computation of the Fresnel free-space propagator in Fourier optics simulations". Are terms missing after convolution?
        hs = img.shape[-1]//2
        ar = np.arange(-hs,hs) / float(2*hs)
        xx,yy = np.meshgrid(ar,ar)
        g = np.exp(-1j*np.pi/f1 * (xx**2+yy**2))
        return np.fft.ifft2(np.fft.fft2(img)*np.fft.fftshift(g))


def set_datapack(obj,probe,probe_positions,difpads,background,probesupp):
        print('Creating datapack...')
        # Set data for Ptycho algorithms:
        datapack = {}
        datapack['obj'] = obj
        datapack['probe'] = probe
        datapack['rois'] = probe_positions
        datapack['difpads'] = difpads
        datapack['bkg'] = background 
        datapack['probesupp'] = probesupp

        return datapack

def get_pixel_size( N, du, energy, z):

        energy_     = energy * 1000        # ev
        cvel        = 299792458            # m/s
        planck      = 4.135667662e-15      # ev * s
        wavelength  = cvel * planck/ (energy_)        

        # N * dx * du = dist * lambda

        return (z * wavelength) / ( (du * 1e-6)  * N )

def save_variable(variable,predefined_name,savename=""):
        print(f'Saving variable {predefined_name}...')
        print(len(variable))
        variable = np.asarray(variable,dtype=object)
        for i in range(variable.shape[0]):
                print('shapes',variable[i].shape)
        for i in range(variable.shape[0]): # loop to circumvent problem with nan values
                if math.isnan(variable[i][:,:].imag.sum()):
                        variable[i][:,:] = np.zeros(variable[i][:,:].shape)

        variable = np.asarray(variable,dtype=np.complex64)
        
        if savename != "":
                np.save(savename,variable)
        else:
                np.save(predefined_name,variable)

        print('\t',savename,variable.shape)


def resolution_fsc(data,pixel):
        print('Calculating resolution by Fourier Shell Correlation...')
        # Fourier Shell Correlation for 3D images:
        # The routine inputs are besides the two images for correlation
        # Resolution threshold curves desired: "half" for halfbit, "sigma" for 3sigma, "both" for them both
        # Pixelsize of the object

        sizex = data.shape[-1]
        sizey = data.shape[-2]
        sizez = data.shape[0]

        # For this case we will use the odd/odd even/even divisions of one image in a dataset
        data1 = data[0:sizez:2,0:sizey:2,0:sizex:2] # even
        data2 = data[1:sizez:2,1:sizey:2,1:sizex:2] # odd

        # Output is a dictionary with the resolution values in the object pixelsize unit, and the FSC, frequency and threshold arrays
        # resolution['halfbit'] :  resolution values in the object pixelsize unit
        # resolution['curve']   :  FSC array
        # resolution['freq']    :  frequency array 
        # resolution['sthresh'] :  threshold array
        # resolution['hthresh'] :  threshold array
        resolution = sscResolution.fshell(data1, data2, 'both', pixel)

        return resolution


def resolution_frc(data,pixel,plot_output_folder="./outputs"):
        print('Calculating resolution by Fourier Ring Correlation...')
        # Fourier Ring Correlation for 2D images:
        # The routine inputs are besides the two images for correlation
        # Resolution threshold curves desired: "half" for halfbit, "sigma" for 3sigma, "both" for them both
        # Pixelsize of the object

        sizex = data.shape[-1]
        sizey = data.shape[-2]

        # For this case we will use the odd/odd even/even divisions of one image in a dataset
        data1 = data[0:sizey:2,0:sizex:2] # even
        data2 = data[1:sizey:2,1:sizex:2] # odd

        # Output is a dictionary with the resolution values in the object pixelsize unit, and the FRC, frequency and threshold arrays
        # resolution['halfbit'] :  resolution values in the object pixelsize unit
        # resolution['curve']   :  FRC array
        # resolution['freq']    :  frequency array 
        # resolution['sthresh'] :  threshold array
        # resolution['hthresh'] :  threshold array
        resolution = sscResolution.fring(data1, data2, 'both', pixel)
        sscResolution.display(resolution['curve'],resolution['hthresh'],resolution['sthresh'],resolution['freq'],plot_output_folder,pixel,'both',"ring",plot=False)

        return resolution


#+++++++++++++++++++++++++++++++++++++++++++++++++
#
#
#
# MAIN APPLICATION (for Sirius/caterete beamline)
#
#
#
#
#+++++++++++++++++++++++++++++++++++++++++++++++++

t0 = time()

jason = json.load(open(argv[1]))  # Open jason file

if jason["LogfilePath"] != "":
    sscCdi.caterete.misc.save_json_logfile(jason["LogfilePath"], jason)

# define seed for generation of the same random values
np.random.seed(jason['Seed'])

if jason['InitialObj'] != "":
    jason['InitialObj'] = jason['ObjPath']+jason['InitialObj']
if jason['InitialProbe'] != "":
    jason['InitialProbe'] = jason['ProbePath']+jason['InitialProbe']
if jason['InitialBkg'] != "":
    jason['InitialBkg'] = jason['BkgPath']+jason['InitialBkg']

ibira_datafolder = jason['ProposalPath']
print('ibira_datafolder = ', ibira_datafolder)

empty_detector = h5py.File(jason["EmptyFrame"], 'r')['entry/data/data'][()][0,0,:,:] # raw shape is (1,1,3072,3072)
sscCdi.caterete.misc.plotshow_cmap2(empty_detector,title=f"{jason['EmptyFrame'].split('/')[-1]}",savepath=jason["PreviewFolder"]+'/00_empty.png')

flatfield = np.load(jason["FlatField"])
flatfield[np.isnan(flatfield)] = -1
sscCdi.caterete.misc.plotshow_cmap2(flatfield, title=f"{jason['FlatField'].split('/')[-1]}",
                                    savepath=jason["PreviewFolder"]+'/01_flatfield.png')

if jason["Mask"] != "":
    initial_mask = np.load(jason["Mask"])
    sscCdi.caterete.misc.plotshow_cmap2(initial_mask, title=f"{jason['Mask'].split('/')[-1]}",
                                        savepath=jason["PreviewFolder"]+'/02_mask.png')

sinogram = []
probe3d = []
backg3d = []
first_iteration = True  # flag to save only in the first loop iteration

# loop when multiple acquisitions were performed for a 3D recon
for acquisitions_folder in jason['3D_Acquisition_Folders']:

    print('Starting reconstructiom for acquisition: ', acquisitions_folder)

    if jason["3D_Acquisition_Folders"] != [""]: # if data inside subfolder, list all hdf5 and select the ones you want
        filepaths, filenames = sscCdi.caterete.misc.list_files_in_folder(os.path.join(ibira_datafolder, acquisitions_folder), look_for_extension=".hdf5")
        if jason['Frames'] != []:
            filepaths, filenames = sscCdi.caterete.misc.select_specific_angles(jason['Frames'], filepaths, filenames)
    else:  # otherwise, use directly the .hdf5 measurement file in the proposal path
        filepaths, filenames = [os.path.join(ibira_datafolder, jason["SingleMeasurement"])], [jason["SingleMeasurement"]]

    # loop through each hdf5, one for each sample angle
    for measurement_file, measurement_filepath in zip(filenames, filepaths):

        if first_iteration:
            current_frame = str(0).zfill(4)
        else:
            current_frame = str(int(current_frame)+1).zfill(4)

        if first_iteration:  # plot only for first iteration
            difpad_number = 0
            raw_difpads = h5py.File(measurement_filepath, 'r')['entry/data/data'][()][:, 0, :, :]
            sscCdi.caterete.misc.plotshow_cmap2(raw_difpads[difpad_number, :, :],title=f'Raw Diffraction Pattern #{difpad_number}', savepath= jason['PreviewFolder'] + '/03_difpad_raw.png')

        print('Raw difpad shape',raw_difpads.shape)
        print(raw_difpads[0].shape,raw_difpads[1].shape)

        probe_positions_file = os.path.join(acquisitions_folder, measurement_file[:-5]+'.txt')  # change .hdf5 to .txt extension
        print('probe_positions_file = ', probe_positions_file)

        probe_positions = read_probe_positions(ibira_datafolder+probe_positions_file, measurement_filepath)

        if first_iteration:
            t1 = time()

        # check if probe_positions == null matrix. If so, won't run current iteration. #TODO: output is null when #difpads != #positions. How to solve this?
        run_ptycho = np.any(probe_positions)
        if run_ptycho == True:
            print('Begin Restauration')
            if jason['OldRestauration'] == True:
                print(ibira_datafolder, measurement_file)
                difpads = cat_restauration(jason, os.path.join(ibira_datafolder, acquisitions_folder), measurement_file)

                if 1:  # OPTIONAL: exclude first difpad to match with probe_positions_file list
                    difpads = difpads[1:]
            
            else:
                print('Entering Miqueles Restauration.')
                dic = {}
                dic['susp']     = jason["ChipBorderRemoval"] # parameter to ignore borders of the detector chip
                dic['roi']      = jason["DetectorROI"] # radius of the diffraction pattern wrt to center. Changes according to the binning value!
                dic['binning']  = jason['Binning']
                dic['distance'] = jason['DetDistance'] * 1e+3
                dic['nproc']    = jason["Threads"]
                dic['data']     = ibira_datafolder + measurement_file
                dic['empty']    = jason['EmptyFrame']
                dic['flat']     = jason['FlatField']
                dic['order']    = 'only'
                dic['function'] = sscCdi.caterete.restauration.cat_preproc_ptycho_measurement

                difpads, elapsed_time = sscCdi.caterete.restauration.cat_preproc_ptycho_projections(dic)
            
            print('shape difpads',difpads.shape)
            print(difpads[0].shape,difpads[1].shape)

            if first_iteration:
                sscCdi.caterete.misc.plotshow_cmap2(
                    difpads[difpad_number, :, :], title=f'Restaured Diffraction Pattern #{difpad_number}', savepath= jason['PreviewFolder'] + '/04_difpad_restaured.png')
                sscCdi.caterete.misc.plotshow_cmap2(np.mean(
                    difpads, axis=0), title=f'Mean Restaured Diffraction Pattern #{difpad_number}', savepath= jason['PreviewFolder'] + '/04_difpad_restaured_mean.png')
                if jason["SaveDifpadPath"] != "":
                    np.save(jason["SaveDifpadPath"], np.mean(difpads, axis=0))

            print('Finished Restauration')
            if first_iteration: t2 = time()

            if jason["CircularMask"] != []:  # Circular central mask
                print("Applying circular mask to central pixels")
                radius, center_row, center_col = jason["CircularMask"]
                central_mask = create_circular_mask(center_row, center_col, radius, difpads[0, :, :].shape)
                difpads[:, central_mask > 0] = -1

            if 0:  # low pass
                print("Applying lowpass filter")
                radius, center_row, center_col = 300, 320, 321
                central_mask = create_circular_mask(center_row, center_col, radius, difpads[0, :, :].shape)
                difpads[:, central_mask == 0] = -1

            if jason["DetectorExposure"][0]:
                print("Removing pixels above detector pile-up threshold")
                #TODO: apply threshold only in the chip of interest around central peak
                detector_pileup_count = 300000  # counts/sec; value according to Kalile
                detector_exposure_time = jason["DetectorExposure"][1]
                difpads_rescaled = difpads/detector_exposure_time
                difpads[difpads_rescaled > detector_pileup_count] = -1

            if jason["Mask"] != "":
                print('Applying mask from file to Diffraction Pattern')
                mask = np.load(jason['Mask'])
                print(difpads.shape)
                print(mask.shape)
                difpads[:, mask > 0] = -1

            if first_iteration:
                sscCdi.caterete.misc.plotshow_cmap2(difpads[difpad_number, :, :], title=f'Restaured + Processed Diffraction Pattern #{difpad_number}', savepath= jason['PreviewFolder'] + '/05_difpad_processed.png')
                sscCdi.caterete.misc.plotshow_cmap2(np.mean(difpads, axis=0), title=f"Mean of all difpads: {measurement_filepath.split('/')[-1]}", savepath=jason["PreviewFolder"]+'/05_difpad_processed_mean.png')

            probe_support_radius, probe_support_center_x, probe_support_center_y = jason["ProbeSupport"]

            if first_iteration == True: # maxroi from the first 2D frame will be used to define object size. otherwise, it may vary by 1 pixel and result in bug
                dx,maxroi, probe_positions = set_parameters(difpads,jason,probe_positions)
            else:
                _,_, probe_positions = set_parameters(difpads,jason,probe_positions)

            datapack, probe_positionsi, sigmask, hsize, maxroi, probe_positions = set_initial_parameters(jason, difpads, probe_positions, probe_support_radius, probe_support_center_x, probe_support_center_y,maxroi,dx)

            if first_iteration:  t3 = time()

            run_algorithms = True
            loop_counter = 1
            while run_algorithms:  # Run Ptycho:
                try:
                    algorithm = jason['Algorithm'+str(loop_counter)]
                except:
                    run_algorithms = False

                if run_algorithms:
                    if algorithm['Name'] == 'GL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'],
                                      objbeta=algorithm['ObjBeta'], probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], epsilon=algorithm['Epsilon'],tvmu=algorithm['TV'], sigmask=sigmask, probef1=jason['f1'], data=datapack)

                    elif algorithm['Name'] == 'positioncorrection':
                        datapack['bkg'] = None
                        datapack =  sscPtycho.PosCorrection(iter=algorithm['Iterations'],objbeta=algorithm['ObjBeta'], probebeta=algorithm['ProbeBeta'],  batch=algorithm['Batch'], epsilon=algorithm['Epsilon'],tvmu=algorithm['TV'], sigmask=sigmask, probef1=jason['f1'], data=datapack)

                    elif algorithm['Name'] == 'Mixed':
                        datapack = sscPtycho.CoherentModes(iter=algorithm['Iterations'],objbeta=algorithm['ObjBeta'], probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], epsilon=algorithm['Epsilon'],tvmu=algorithm['TV'], sigmask=sigmask, weights=weights, probef1=jason['f1'], data=datapack)

                    elif algorithm['Name'] == 'RAAR':
                        datapack = sscPtycho.RAAR(iter=algorithm['Iterations'],beta=algorithm['Beta'], probecycles=algorithm['ProbeCycles'], batch=algorithm['Batch'], epsilon=algorithm['Epsilon'], tvmu=algorithm['TV'], sigmask=sigmask, probef1=jason['f1'], data=datapack)

                    elif algorithm['Name'] == 'GLL':
                        datapack = sscPtycho.GL(iter=algorithm['Iterations'], objbeta=algorithm['ObjBeta'], probebeta=algorithm['ProbeBeta'], batch=algorithm['Batch'], epsilon=algorithm['Epsilon'],tvmu=algorithm['TV'], sigmask=sigmask, probef1=jason['f1'], data=datapack)

                    loop_counter += 1
                    RF = datapack['error']
                    print(RF[0:algorithm['Iterations']:max(algorithm['Iterations']//10, 1)]/np.sqrt(np.sum(difpads)))

            print('Original object shape:', datapack['obj'].shape)

            if first_iteration:
                t4 = time()

            if jason['Phaseunwrap'][0] == True:
                """ Crop reconstruction for a proper phase unwrap """
                slice_rows, slice_columns = slice(jason['Phaseunwrap'][2][0],jason['Phaseunwrap'][2][1]), slice(jason['Phaseunwrap'][3][0],jason['Phaseunwrap'][3][1])
                # slice_rows, slice_columns = slice(1*hsize//2,-1*hsize//2), slice(1*hsize//2,-1*hsize//2)
                datapack['obj'] = datapack['obj'][slice_rows,slice_columns]
                print('Cropped object shape:', datapack['obj'].shape)

                tt_parameter = jason['Phaseunwrap'][1] # number of iterations to remove gradient from unwrapped image
                absolute = sscCdi.unwrap.phase_unwrap(-np.abs(sscPtycho.RemovePhaseGrad(datapack['obj'])),tt_parameter)
                angle = sscCdi.unwrap.phase_unwrap(-np.angle(sscPtycho.RemovePhaseGrad(datapack['obj'])),tt_parameter)
                datapack['obj'] = absolute*np.exp(-1j*angle)
            else: 
                pass
            
            sinogram.append(datapack['obj']) # Build 3D Sinogram:
            probe3d.append(datapack['probe'])
            backg3d.append(datapack['bkg'])

            if jason['FRC'] == True:
                # padding = 600
                # obj2 = np.pad(datapack['obj'],((padding,padding),(padding,padding)))
                # resolution = resolution_frc(obj2, dx)
                resolution = resolution_frc(datapack['obj'], dx)
                print('\tResolution for frame ' + str(current_frame) + ':', resolution['halfbit'])

            if jason['Preview']:  # Preview Reconstruction:
                # '''
                plt.figure()
                plt.scatter(probe_positionsi[:, 0], probe_positionsi[:, 1])
                plt.scatter(datapack['rois'][:, 0, 0],datapack['rois'][:, 0, 1])
                plt.savefig( jason['PreviewFolder'] + '/scatter_2d.png', format='png', dpi=300)
                plt.clf()
                plt.close()
                # '''
                # Show probe:
                plotshow([abs(Prop(p, jason['f1'])) for p in datapack['probe']]+[p for p in datapack['probe']],file= jason['PreviewFolder'] + '/probe_2d_' + str(current_frame), nlines=2)

                # Show object:
                ango = np.angle(datapack['obj'])
                abso = np.clip(abs(datapack['obj']), 0.0, np.max( abs(datapack['obj'][hsize:maxroi, hsize:maxroi])))

                plotshow([ango, abso], subplot_title=['Phase', 'Magnitude'], file= jason['PreviewFolder'] + '/object_2d_' + str(current_frame), cmap='gray', nlines=1)
        else:
            continue

        first_iteration = False

    if jason['SaveObj'] == True:
        if jason['SaveObjname'] != "":
            save_variable(sinogram, jason['ObjPath']+'sino',savename=jason['ObjPath'] + jason['SaveObjname'])
        else:
            save_variable(sinogram, jason['ObjPath']+'sino')

    if jason['SaveProbe'] == True:
        if jason['SaveProbename'] != "":
            save_variable(probe3d, jason['ProbePath']+'probe',savename=jason['ProbePath'] + jason['SaveProbename'])
        else:
            save_variable(probe3d, jason['ProbePath']+'probe')

    if jason['SaveBkg'] == True:
        if jason['SaveBkgname'] != "":
            save_variable(backg3d, jason['BkgPath']+'bkg',savename=jason['BkgPath'] + jason['SaveBkgname'])
        else:
            save_variable(backg3d, jason['BkgPath']+'bkg')

t5 = time()

print( f'\nElapsed time for reconstruction of 1st frame: {t4-t0:.2f} seconds = {(t4-t0)/60:.2f} minutes')
print(f'Reading percentual time for 1st frame: {100*(t1-t0)/(t4-t0):.2f}%')
print(f'Restauration percentual time for 1st frame: {100*(t2-t1)/(t4-t0):.2f}%')
print(  f'Pre-Processing percentual time for 1st frame: {100*(t3-t2)/(t4-t0):.2f}% ')
print(f'Reconstruction percentual time for 1st frame: {100*(t4-t3)/(t4-t0):.2f}% ')
print(f'Total time: {t5-t0:.2f} seconds = {(t5-t0)/60:.2f} minutes')
