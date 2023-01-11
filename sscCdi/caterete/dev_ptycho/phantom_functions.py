import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os, json, h5py
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from scipy.ndimage import rotate
from PIL import Image
import sscPhantom

from sscCdi.caterete.ptycho_processing import convert_probe_positions, set_object_shape

""" New functions """

def match_colorbar(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)

def propagate_beam(wavefront, experiment_params,propagator='fourier'):
    """ Propagate a wavefront using fresnel ou fourier propagator

    Args:
        wavefront : the wavefront to propagate
        dx : pixel spacing of the wavefront input
        wavelength : wavelength of the illumination
        distance : distance to propagate
        propagator (str, optional): 'fresenel' or 'fourier'. Defaults to 'fresnel'.

    Returns:
        output: propagated wavefront
    """    
    
    from numpy.fft import fft2, fftshift, ifftshift, ifft2

    dx, wavelength,distance = experiment_params 
    
    if propagator == 'fourier':
        if distance > 0:
            output = fftshift(fft2(fftshift(wavefront)))
        else:
            output = ifftshift(ifft2(ifftshift(wavefront)))            
    
    elif propagator == 'fresnel':
    
        ysize, xsize = wavefront.shape
        x_array = np.linspace(-xsize/2,xsize/2-1,xsize)
        y_array = np.linspace(-ysize/2,ysize/2-1,ysize)

        fx = x_array/(xsize)
        fy = y_array/(ysize)

        FX,FY = np.meshgrid(fx,fy)
        # Calculate approx phase distribution for each plane wave component
        w = FX**2 + FY**2 
        # Compute FFT
        F = fftshift(fft2(fftshift(wavefront)))
        # multiply by phase-shift and inverse transform 
        a = np.exp(-1j*np.pi*( distance*wavelength/dx**2)*w)
        output = ifftshift(ifft2(ifftshift(F*a)))

    return output

def get_object_pixel(N,pixel_size,wavelength,distance):
    return wavelength*distance/(N*pixel_size)

def get_positions(scan_step,size_x,size_y):

    x_positions = np.arange(0,size_x,scan_step)
    y_positions = np.arange(0,size_y,scan_step)

    error = 0.05
    error2 = error
    
    x_pos = x_positions + np.random.normal(scale=error*scan_step,size=x_positions.shape)
    y_pos = y_positions + np.random.normal(scale=error*scan_step,size=y_positions.shape)

    x_pos_error = x_positions + np.random.normal(scale=error2*scan_step,size=x_positions.shape)
    y_pos_error = y_positions + np.random.normal(scale=error2*scan_step,size=y_positions.shape)

    return y_pos,x_pos, y_pos_error,x_pos_error

def set_object_size(x_pos,y_pos,obj_pxl_size,probe_size, gap = 10):
    return (np.int(gap + probe_size[0]+(np.max(y_pos)-np.min(y_pos))//obj_pxl_size),np.int(gap+probe_size[1]+(np.max(x_pos)-np.min(x_pos))//obj_pxl_size))


def calculate_diffraction_pattern(idx,obj,probe,wavelength,distance,obj_pxl):
    y,x = idx
    wavefront_box = probe*obj[y:y+probe.shape[0],x:x+probe.shape[1]]
    DP = np.abs(propagate_beam(wavefront_box, (obj_pxl,wavelength,distance)))**2
    return DP

def save_angles_file_CAT_standard():
    for i, angle_number in enumerate(range(tomogram.shape[0])): 
        filename = str(angle_number).zfill(4)+f"_complex_phantom_001.txt"
        line = f"Ry: {angles[i]}\tPiezoB2\tPiezoB3\tPiezoB1\t"
        with open(os.path.join(path,filename), 'w') as f:
            f.write(line)
            
def save_positions_file_CAT_standard(x,y,path,filename,x_original, y_original):

    line = f"Ry: {0}\tPiezoB2\tPiezoB3\tPiezoB1\t"
    f = open(os.path.join(path,'positions',f"{filename}_001.txt"), 'w')
    y,x = np.meshgrid(y*1e6,x*1e6) # save in microns
    f.write(line)
    columns = np.c_[y.flatten(),x.flatten()]
    for i in range(columns.shape[0]):
        f.write(f"\n{columns[i,0]}\t{columns[i,1]}")
    f.close()

    line = f"Ry: {0}\tPiezoB2\tPiezoB3\tPiezoB1\t"
    f = open(os.path.join(path,'positions',f"{filename}_without_error_001.txt"), 'w')
    y,x = np.meshgrid(y_original*1e6,x_original*1e6) # save in microns
    f.write(line)
    columns = np.c_[y.flatten(),x.flatten()]
    for i in range(columns.shape[0]):
        f.write(f"\n{columns[i,0]}\t{columns[i,1]}")
    f.close()
    
def create_hdf_file(matrix,path,filename):
    difpads = np.ones((10,5,3))
    hdf_file = h5py.File(os.path.join(path,'scans',f'{filename}_001.hdf5'), 'w')
    group = hdf_file.create_group('entry/data/')
    group.create_dataset("data",data=matrix)
    hdf_file.close()

def rotation_Rz(matrix,angle):
    return rotate(matrix,angle,reshape=False,axes=(1,0))
    
def get_projection(angle,magnitude,phase,wavevector):
    return np.exp(-wavevector*np.sum(rotation_Rz(magnitude,angle),axis=0)) * np.exp(-1j*wavevector*np.sum(rotation_Rz(phase,angle),axis=0))
    # return wavevector*np.sum(rotation_Rz(magnitude,angle),axis=0)*np.exp(-1j*wavevector*np.sum(rotation_Rz(phase,angle),axis=0))

def save_hdf_masks(path,shape):
    path = os.path.join(path,'images')

    dbeam = np.zeros((1,1,shape[0],shape[1]))
    hdf_file = h5py.File(os.path.join(path,f'dbeam.hdf5'), 'w')
    group = hdf_file.create_group('entry/data/')
    group.create_dataset("data",data=dbeam)
    hdf_file.close()
    
    empty = np.zeros((1,1,shape[0],shape[1]))
    hdf_file = h5py.File(os.path.join(path,f'empty.hdf5'), 'w')
    group = hdf_file.create_group('entry/data/')
    group.create_dataset("data",data=empty)
    hdf_file.close()
    
    flat = np.zeros((1,1,shape[0],shape[1]))
    hdf_file = h5py.File(os.path.join(path,f'flat.hdf5'), 'w')
    group = hdf_file.create_group('entry/data/')
    group.create_dataset("data",data=flat)
    hdf_file.close()
    
    mask = np.zeros((1,1,shape[0],shape[1]))
    hdf_file = h5py.File(os.path.join(path,f'mask.hdf5'), 'w')
    group = hdf_file.create_group('entry/data/')
    group.create_dataset("data",data=mask)
    hdf_file.close()
    return 0

def add_error_to_positions(positionsX,positionsY,mu=0,sigma=1e-6):
    # sigma controls how big the error is
    deltaX = np.random.normal(mu, sigma, positionsX.shape)
    deltaY = np.random.normal(mu, sigma, positionsY.shape)
    return positionsX+deltaX,positionsY+deltaY     

def create_positions_file(frame,probe_steps_xy,obj_pxl,filename,path,random_shift_range,position_errors=False):

    """ Probe """
    dx, dy = probe_steps_xy # probe step size in each direction
    y_pxls = np.arange(0,frame.shape[0]+1,dy)
    x_pxls = np.arange(0,frame.shape[1]+1,dx)

    if 1: # apply random shifts to avoid ptycho periodic features
        random_shift_y = [np.int(i) for i in (-1)**np.int(2*np.random.rand(1))*random_shift_range*np.random.rand(y_pxls.shape[0])] # generate random shift between -random_shift_range and +random_shift_range
        random_shift_x = [np.int(i) for i in (-1)**np.int(2*np.random.rand(1))*random_shift_range*np.random.rand(x_pxls.shape[0])]
        y_pxls, x_pxls = random_shift_y + y_pxls, random_shift_x + x_pxls
    
    y_pxls, x_pxls = y_pxls - np.min(y_pxls), x_pxls - np.min(x_pxls)
    Y_pxls, X_pxls = np.meshgrid(y_pxls,x_pxls)
    # print('Original pxls',y_pxls,y_pxls.shape,'\n',x_pxls,x_pxls.shape)

    """ Convert to metric units """
    x_meters, y_meters = x_pxls*obj_pxl , y_pxls*obj_pxl
    artificial_shift = x_meters[0] # artificial shift to have value close to the typical ones given by the beamline files (i.e. not starting at zero)
    x_meters, y_meters = x_meters - artificial_shift, y_meters - artificial_shift
    x_meters_original, y_meters_original = x_meters, y_meters # save original correct values

    """ Add position errors"""
    if position_errors == True:
        print( "Adding errors to position")
        x_meters, y_meters = add_error_to_positions(x_meters,y_meters)

    save_positions_file_CAT_standard(x_meters,y_meters,path,filename,x_meters_original, y_meters_original)


def set_object_size_pxls(x_pos,y_pos,probe_size,bottom_right_gap=0):
    return (np.int(bottom_right_gap + probe_size[0]+(np.max(y_pos)-np.min(y_pos))),np.int(bottom_right_gap+probe_size[1]+(np.max(x_pos)-np.min(x_pos))))

def set_object_frame(y_pxls, x_pxls,frame,probe,object_offset,path):
    obj = np.zeros(set_object_size_pxls(x_pxls,y_pxls,probe.shape,object_offset),dtype=complex)
    obj[object_offset:object_offset+frame.shape[0],object_offset:object_offset+frame.shape[1]] = frame

    model_path = os.path.join(path,'model','model_obj.npy')
    np.save(model_path,obj)
    print(f"Calculating diffraction data for object of size {obj.shape}. Used {object_offset} pixel of offset at the border.")
    print(f"\tData saved at: ",model_path)
    return obj

def convert_probe_positions(dx, probe_positions, offset_topleft = 20):
    """Set probe positions considering maxroi and effective pixel size

    Args:
        difpads (3D array): measured diffraction patterns
        jason (json file): file with the setted parameters and directories for reconstruction
        probe_positions (array): each element is an 2-array with x and y probe positions
        offset_topleft (int, optional): [description]. Defaults to 20.

    Returns:
        object pixel size (float), maximum roi (int), probe positions (array)
    """    

    # Subtract the probe positions minimum to start at 0
    probe_positions[:, 0] -= np.min(probe_positions[:, 0])
    probe_positions[:, 1] -= np.min(probe_positions[:, 1])

    offset_bottomright = offset_topleft #define padding width
    probe_positions[:, 0] = 1E-6 * probe_positions[:, 0] / dx + offset_topleft #shift probe positions to account for the padding
    probe_positions[:, 1] = 1E-6 * probe_positions[:, 1] / dx + offset_topleft #shift probe positions to account for the padding

    return np.round(probe_positions).astype(np.int32), offset_bottomright

def get_xy_positions(probe_positions):
    # Get unique X and Y probe posiitons from the file
    first = probe_positions[0,0]
    uniques_list = [first]
    uniques_list2 = []
    for i,item in enumerate(probe_positions[:,0]):
        if item == first:
            uniques_list2.append(probe_positions[i,1])
        else:
            first = item
            uniques_list.append(first)
    uniques_list2 = [*set(uniques_list2)] #remove duplicates  
    return np.asarray(uniques_list), np.sort(np.asarray(uniques_list2))

def convert_positions_to_pixels(pixel_size,probe_positions,offset_topleft):
    probe_positions, offset_bottomright = convert_probe_positions(pixel_size, probe_positions, offset_topleft)
    uniqueX, uniqueY = get_xy_positions(probe_positions)
    Y_pxls, X_pxls = np.meshgrid(uniqueY, uniqueX)
    return Y_pxls, X_pxls

def read_probe_positions_new(path,filename):
    # print('Reading probe_positions...')
    probe_positions = []
    positions_file = open(os.path.join(path,'positions',f"{filename}_001.txt"))

    line_counter = 0
    for line in positions_file:
        line = str(line)
        if line_counter >= 1:  # skip first line, which is the header
            pxl = float(line.split()[1])
            pyl = float(line.split()[0])
            probe_positions.append([pxl, pyl, 1, 1])
        line_counter += 1

    probe_positions = np.asarray(probe_positions)

    return probe_positions

def get_ptycho_diffraction_data(frame,probe,obj_pxl,wavelength,distance,filename,path,probe_steps_xy,position_errors=False,object_offset=10):

    random_shift_range = np.int(0.5*object_offset) # the amount of shift will be only a fraction of the padding borders; that way we guarantee probe scan positions won't required values bigger than the object matrix size
    create_positions_file(frame,probe_steps_xy,obj_pxl,filename,path,random_shift_range)

    probe_positions = read_probe_positions_new(path,filename)

    Y_pxls, X_pxls = convert_positions_to_pixels(obj_pxl,probe_positions,object_offset)
    
    obj = set_object_frame(Y_pxls, X_pxls,frame,probe,object_offset,path)

    """ Loop through positions """ 
    diffraction_patterns = np.zeros((X_pxls.flatten().shape[0],1,probe.shape[0],probe.shape[1]))

    calculate_diffraction_pattern_partial = partial(calculate_diffraction_pattern,obj=obj,probe=probe,wavelength=wavelength,distance=distance,obj_pxl=obj_pxl)
    idx_list = [ [y,x] for y, x in zip(Y_pxls.flatten(),X_pxls.flatten()) ]
    
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(),32)) as executor:
        results = executor.map(calculate_diffraction_pattern_partial,idx_list)
        for counter, result in enumerate(results):
            diffraction_patterns[counter,0,:,:] = result

    return diffraction_patterns


def create_metadata_with_beamline_standard(inputs,beamlime='CAT'):
    
    speed_of_light, planck = 299792458, 4.135667662E-18  # Plank constant [keV*s]; Speed of Light [m/s]
    wavelength = planck * speed_of_light / inputs["energy"] # meters
    wavevector = 2*np.pi/wavelength
    
    mdata = {"/entry/beamline/experiment": {"distance": inputs["distance"]*1e3, "energy": inputs["energy"]},
             "/entry/beamline/detector": {"pimega": {"exposure time": 10.0, "pixel size": inputs["probe_pxl"]*1e6}}}

    inputs["/entry/beamline/experiment"] = mdata["/entry/beamline/experiment"]
    inputs["/entry/beamline/detector"] = mdata["/entry/beamline/detector"]
    inputs["wavelength"] = wavelength
    inputs["wavevector"] = wavevector
    
    json.dump(mdata,open(os.path.join(inputs["path"],"mdata.json"), "w"))
    
    # print("Metadata created successfuly")

    return inputs

def get_phantom(inputs,sample,load):
    
    if sample == 'donut':
    
        if load == False:
            params = { 'HowMany': 3,
                       'radius': 0.02,
                       'Rtorus': 0.2,
                       'rtorus': 0.05}
            phantom1, phantom2 = sscPhantom.donuts.createDonuts( inputs["matrix_size"],inputs["n_cpus"], inputs["energy"], params )
            magnitude = phantom1 + phantom2
            magnitude = np.swapaxes(magnitude,1,0)

            params = { 'HowMany': 10,
                       'radius': 0.07,
                       'Rtorus': 0.5,
                       'rtorus': 0.1}
            phantom1, phantom2 = sscPhantom.donuts.createDonuts( inputs["matrix_size"],inputs["n_cpus"], inputs["energy"], params )
            phase = phantom1 + phantom2
            phase = np.swapaxes(phase,1,0)

            magnitude = 0.016*magnitude/np.max(magnitude)
            phase = 6*phase/np.max(phase) # 3 complete unwraps in total

            np.save(os.path.join(inputs["path"],'model','magnitude.npy'),magnitude)
            np.save(os.path.join(inputs["path"],'model','phase.npy'),phase)
        else:
            phase = np.load(os.path.join(inputs["path"],'model','magnitude.npy'))
            magnitude = np.load(os.path.join(inputs["path"],'model','phase.npy'))
    else:
        pass # no other object for now

    phantom, magnitude_view, phase_view = build_complex_object(magnitude,phase)

    return phantom, magnitude, phase

def build_complex_object(magnitude,phase):
    phantom = np.exp(-magnitude)*np.exp(1j*phase)
    magnitude_view = -np.log(np.abs(phantom))
    phase_view = np.angle(phantom)
    return phantom, magnitude_view, phase_view

def get_sinogram(inputs,magnitude, phase,load):

    if inputs["n_of_angles"] != 0:
        angles = np.linspace(0,180, inputs["n_of_angles"])
    else:
        angles = np.array([0])

    data_path = os.path.join(inputs["path"],'model','complex_sinogram.npy')

    get_projection_partial = partial(get_projection,magnitude=magnitude,phase=phase,wavevector=inputs["wavevector"])

    if load == False: # project 3D object for all angles
        sinogram = np.zeros((angles.shape[0],magnitude.shape[1],magnitude.shape[2]),dtype=complex)
        processes = min(os.cpu_count(),32)
        print(f'Using {processes} parallel processes')
        with ProcessPoolExecutor(max_workers=processes) as executor:
            results = list(tqdm(executor.map(get_projection_partial,angles),total=len(angles)))
            for counter, result in enumerate(results):
                if counter % 100 == 0: print('Populating results matrix...',counter)
                sinogram[counter,:,:] = result
        np.save(data_path,sinogram)
    else:
        sinogram = np.load(data_path)
    
    print(f"Created complex sinogram of shape {sinogram.shape} from object of size {magnitude.shape}")
    print(f"\tData saved at {data_path}")

    return sinogram 

def get_probe(inputs,probe_type='circle',size=50,preview = True):
    
    if probe_type=='CAT': # Realistic Probe """
        probe = np.load(os.path.join(inputs["path"],'model','probe_at_focus_1.25156micros_pixel.npy'))
        half=size//2 # half the size you want in one dimension
        probe = probe[probe.shape[0]//2-half:probe.shape[0]//2+half,probe.shape[1]//2-half:probe.shape[1]//2+half]
        np.save(os.path.join(inputs["path"],'model','processed_probe.npy'),probe)
        
    elif probe_type=='circle': #Round probe """
        probe = np.ones((100,100))
        xprobe = np.linspace(0,probe.shape[0]-1,probe.shape[0])
        xprobe -= probe.shape[0]//2
        Y,X = np.meshgrid(xprobe,xprobe)
        probe = np.where(X**2+Y**2<=45**2,1,0)
    else:
        print("ERROR: Please select 'CAT' or 'circle' for probe type!")
    
    print(f"Loading probe of type -{probe_type}- with shape {probe.shape}")

    return probe

def get_detector_data(inputs,sinogram, probe,position_errors=False):

    obj_pxl   = get_object_pixel(probe.shape[0],inputs["probe_pxl"],inputs["wavelength"],inputs["distance"])

    save_hdf_masks(inputs["path"],probe.shape)
    #TODO: apply masks to DPs

    """ Loop through all frames"""
    for i in range(sinogram.shape[0]):
        if i%25==0: print(f"Creating dataset {i+1}/{sinogram.shape[0]}")
        filename = str(i).zfill(4)+f"_complex_phantom"
        difpads = get_ptycho_diffraction_data(sinogram[i],probe,obj_pxl,inputs["wavelength"],inputs["distance"],filename,inputs["path"],inputs["probe_steps_xy"],position_errors=position_errors)

        """ Save to hdf5 file """
        create_hdf_file(difpads,inputs["path"],filename)

        # save data to numpy
        difpads = np.squeeze(difpads)
        np.save(os.path.join(inputs["path"],f'{filename}_001.hdf5.npy'),difpads)
        np.save(os.path.join(inputs["path"].rsplit('/',4)[0],'proc','recons',inputs["path"].rsplit('/',4)[-2],f'{filename}_001.hdf5.npy'),difpads)
        
    print(f"Detector data created with shape {difpads.shape}")
    
    return difpads
    
def create_ptycho_phantom(inputs,sample="donut",probe_type="circle",position_errors=False,load=False):
    
    create_metadata_with_beamline_standard(inputs)
    
    phantom, magnitude, phase = get_phantom(inputs,sample,load=load)
    
    sinogram = get_sinogram(inputs,magnitude, phase,load=load)
    
    probe = get_probe(inputs,probe_type)
    
    diffraction_data = get_detector_data(inputs,sinogram, probe,position_errors=position_errors)
    
    figure, ax = plt.subplots(1,7,dpi=200,figsize=(15,15))
    ax[0].imshow(np.sum(magnitude,axis=0)), ax[0].set_title('Magnitude')
    ax[1].imshow(np.sum(phase,axis=0)), ax[1].set_title('Phase')
    ax[2].imshow(np.sum(-np.log(np.abs(phantom)),axis=0)), ax[2].set_title('-log(abs(phantom))')
    ax[3].imshow(np.sum(np.angle(phantom),axis=0)), ax[3].set_title('angle(phantom)')
    ax[4].imshow(np.abs(probe),cmap='jet'), ax[4].set_title('abs')
    ax[5].imshow(np.angle(probe),cmap='hsv'), ax[5].set_title('angle')
    ax[6].imshow(np.mean(diffraction_data,axis=0),norm=LogNorm()), ax[6].set_title('DPs mean')
    for axis in ax.ravel(): axis.set_xticks([]), axis.set_yticks([])
    plt.show()
    

    print("Phantom created at",inputs["path"])
    
    return phantom, magnitude, phase, sinogram, probe, diffraction_data

