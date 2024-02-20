import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import scipy

from skimage.registration import phase_cross_correlation
from skimage.transform import pyramid_reduce
from scipy.ndimage import center_of_mass

# import sscCdi

### Cross Correlation ### 
def alignment_variance_field(data, downscaling_factor=0, fft_upsampling=10, return_common_valid_region=True, remove_null_borders = True, use_gradient = True,downscaling_method='skip_pixels'):
    """ Performs alignment of the variance fild of a block images by registering neighboor slices. See https://doi.org/10.1364/OE.27.036637

    Args:
        data (numpy array): block of images to be aligned
        downscaling_factor (int, optional): not used for now. Downsamples the image to performe alignment in a lower resolution version of the image. Defaults to 2.
        fft_upsampling (int, optional): upsampling factor to improve sub-pixel imaging registration. See https://doi.org/10.1364/OL.33.000156. Defaults to 10.
        return_common_valid_region (bool, optional): If true, will return a zeroed border corresponding to the largest shift among all frames in each direction. Defaults to True.
        remove_null_borders (bool, optional): Remove the null borders. Defaults to True.

    Returns:
        aligned_volume (numpy array): aligned volume
    """
    
    _, total_shift = get_shifts_of_local_variance(data,fft_upsampling,downscaling_factor, use_gradient,downscaling_method)

    if downscaling_factor > 1:
        total_shift = total_shift*downscaling_factor # multiply by downsampling factor
    
    aligned_volume = shift_volume_slices(data,total_shift)
    
    if return_common_valid_region:
        masked_volume = np.where(aligned_volume==0,0,1)
        product = np.prod(np.abs(masked_volume),axis=0)
        where_null = np.where(np.abs(product) == 0,0,1)
        aligned_volume[:] = np.where(where_null==1,aligned_volume,0) 
    
    if remove_null_borders:
        aligned_volume = remove_black_borders(aligned_volume)
        
    return aligned_volume
    
def get_shifts_of_local_variance(data,fft_upsampling,downscaling_factor, use_gradient,downscaling_method='pyramid'):
    """ Calculates local variance field of images (in a block) and finds the shift between them.

    Args:
        data (numpy array): block of images to be aligned 
        fft_upsampling (int): upsampling factor to improve sub-pixel imaging registration. See https://doi.org/10.1364/OL.33.000156. 
        downscaling_factor (int, optional): not used for now. Downsamples the image to performe alignment in a lower resolution version of the image. Defaults to 2.

    Returns:
        neighbor_shifts: array of values containing the shift between neighbor images
        total_shifts: shift of each image with respect to the first image
    """
    
    if downscaling_factor > 1:
        print("Downscaling images for alignment...")
        if downscaling_method == 'pyramid_reduce': # slow method! how to speed it up?
            data = pyramid_reduce(np.real(data), downscale=downscaling_factor,order=1,channel_axis=0) + 1j*pyramid_reduce(np.imag(data), downscale=downscaling_factor,order=1,channel_axis=0)
        elif downscaling_method == 'skip_pixels':
            data = data[:,0::downscaling_factor,0::downscaling_factor]
        else:
            sys.exit('Select a proper downscaling method: pyramid_reduce or skip_pixels')
    
    neighbor_shifts = np.empty((data.shape[0],2))
    
    print('Finding shift between neighboor slices...')
    for i in range(0,data.shape[0]-1):
        if i%50==0: print(f"Finding shift between slices #{i}/{data.shape[0]}")

        if use_gradient:
            local_variance1 = calculate_local_variance_field(data[i])
            local_variance2 = calculate_local_variance_field(data[i+1])
        else:
            local_variance1 = data[i]
            local_variance2 = data[i+1]
        
        shift, error, diffphase = phase_cross_correlation(local_variance1, local_variance2, upsample_factor=fft_upsampling)

        neighbor_shifts[i][0] = shift[0]
        neighbor_shifts[i][1] = shift[1]
        
    total_shift = np.cumsum(neighbor_shifts,axis=0)
    
    return neighbor_shifts, total_shift


def shift_volume_slices(data,total_shift):
    """ Shifts each image in the block "data" according to the values in total_shift
    """

    aligned_volume = np.zeros_like(data)
    aligned_volume[0] = data[0]

    for i in range(0,data.shape[0]-1):
        if i%50==0: print(f"Shifting slice #{i}/{data.shape[0]}")
        aligned_volume[i+1] = scipy.ndimage.shift(data[i+1],total_shift[i])

    return aligned_volume

def calculate_local_variance_field(matrix):
    """ Calculate the local variance field of a complex matrix
    
    """
    
    gradient = np.gradient(matrix)
    del_x = gradient[1]
    del_y = gradient[0]
        
    return np.sqrt(np.abs(del_x)**2 + np.abs(del_y)**2)



### VMF ### 

def alignment_vertical_mass_fluctuation(misaligned_volume, filter_sigma = 0, curve_portion = None, use_phase_gradient = False, return_common_valid_region=True, remove_null_borders = True, plot = None):
    """ Performs the alignment via "Vertical Mass Fluctuation" (as presented in https://doi.org/10.1364/OE.27.036637) to refine alignment of images in the vertical direction

    Args:
        misaligned_volume (numpy array): a volume of images, usually already pre-aligned by another method
        use_phase_gradient (bool, optional): whether to use the phase-gradient for alignment instead of the original images. Defaults to False.
        return_common_valid_region (bool, optional): _description_. Defaults to True.
        return_common_valid_region (bool, optional): If true, will return a zeroed border corresponding to the largest shift among all frames in each direction. Defaults to True.
        remove_null_borders (bool, optional): Remove the null borders. Defaults to True.

    Returns:
        aligned_volume (numpy array) : aligned volume
    """
    
    
    curves, aligned_curves, total_shift = align_1D_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion,plot)
    
    print('Aligning volume...')
    aligned_volume = np.zeros_like(misaligned_volume)
    aligned_volume[0] = misaligned_volume[0]
    for i in range(0,misaligned_volume.shape[0]-1):
        if i%50==0: print(f"Aligning slice #{i}/{misaligned_volume.shape[0]}")
        aligned_volume[i+1] = scipy.ndimage.shift(misaligned_volume[i+1],[total_shift[i,0],0])

    if return_common_valid_region:
        masked_volume = np.where(aligned_volume==0,0,1)
        product = np.prod(np.abs(masked_volume),axis=0)
        where_null = np.where(np.abs(product) == 0,0,1)
        aligned_volume[:] = np.where(where_null==1,aligned_volume,0) 

    if remove_null_borders:
        aligned_volume = remove_black_borders(aligned_volume)    

    return aligned_volume, curves, total_shift

def align_1D_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion,plot):

    curves = get_VMF_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion)

    aligned_curves, total_shift, neighbor_shifts = overlap_curves(curves)

    if plot != None:

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(curves.T)
        ax[1].imshow(aligned_curves.T)
        ax[0].set_ylabel('Y')
        ax[0].set_xlabel('Projection #')        
        ax[1].set_xlabel('Projection #')

    return curves, aligned_curves, total_shift

def get_VMF_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion):
    curves = []
    print("Calculating 1D mass distribution...")
    for i in range(misaligned_volume.shape[0]):
        frame = misaligned_volume[i]

        if use_phase_gradient:
            curve = vertical_phase_gradient(frame)
            if filter_sigma>0:
                curve = scipy.ndimage.gaussian_filter1d(curve, filter_sigma)
        else:
            curve = vertical_mass_distribution(frame)

        if curve_portion != None:
            if len(curve_portion) != 2:
                sys.exit('Please insert a tuple of 2 values for curve_portion: (value1, value2)')
            else:
                curve = curve[curve_portion[0]:curve_portion[1]]

        curves.append(curve)

    curves = np.asarray(curves)
    return curves

def vertical_phase_gradient(frame):
    """
    Calculate the vertical phase gradient of a complex image using the analytical formula.
    See equation (6) in https://doi.org/10.1364/OE.27.036637 
    """

    gradient = np.gradient(frame)
    phase_gradient_y = np.imag( frame.conj() * gradient[0] / np.abs(frame)**2  )
    phase_gradient_y = np.sum(phase_gradient_y,axis=1)
    return phase_gradient_y

def vertical_mass_distribution(frame):
    """
    Calculate vertical mass distrbution of image 
    """
    return np.sum(frame,axis=1)
    

def shift_2d_replace(data, dx, dy, constant=False):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def remove_black_borders(volume):
    """
    Remove the null borders of a volume of images 
    """

    not_null = np.argwhere(np.abs(volume[0]))

    # Bounding box of non-black pixels.
    x0, y0 = not_null.min(axis=0)
    x1, y1 = not_null.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    volume = volume[:,x0:x1, y0:y1]

    return volume

def calculate_curve_ctr_of_mass(curve,positions):
    """
    Calculate the center of mass of a 2D curve
    """
    ctr_mass = np.dot(curve,positions)/np.sum(curve)
    return ctr_mass


def overlap_curves(data):
    """
    Overlap 2D curves using subpixel image registration methods.
    See https://doi.org/10.1364/OL.33.000156
    """

    neighbor_shifts = np.empty((data.shape[0],1))

    print('Finding shift between neighboor slices...')
    for i in range(0,data.shape[0]-1):

        shift, error, diffphase = phase_cross_correlation(data[i], data[i+1], upsample_factor=10,normalization=None) # if normalzaition = phase, shifts are barely > 0
        neighbor_shifts[i][0] = shift[0]

    total_shift = np.cumsum(neighbor_shifts,axis=0)
    aligned_curves = np.zeros_like(data)
    aligned_curves[0] = data[0]

    print('Shifting slices...')
    for i in range(0,data.shape[0]-1):
        aligned_curves[i+1] = scipy.ndimage.shift(data[i+1],total_shift[i])
        
    return aligned_curves, total_shift, neighbor_shifts
