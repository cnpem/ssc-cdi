import numpy as np
import cupy as cp
from tqdm import tqdm

""" Relative imports """
from ..misc import wavelength_from_energy

def calculate_fresnel_number(energy,pixel_size,sample_detector_distance,magnification=1,source_sample_distance=0):
    """
    Calculate fresnel number in magnification scenario. 

    Args:
        energy: energy in keV
        pixel_size: object pixel size
        sample_detector_distance: sample to detector distance in meters
        magnification (int, optional): magnification of the optical system. If 1, no magnification is used. Defaults to 1.
        source_sample_distance (int, optional): source to sample distance in meters. Defaults to 0.

    Returns:
        (float): Fresnel number 
    """

    wavelength = wavelength_from_energy(energy) # meters
    if magnification != 1:
        magnification = (source_sample_distance+source_sample_distance)/source_sample_distance
    return -(pixel_size**2) / (wavelength * sample_detector_distance * magnification)

# def Propagate(img, fresnel_number): # Probe propagation
#         """
#         Function for free space propagation of the probe in the Fraunhoffer regime

#         See paper `Memory and CPU efficient computation of the Fresnel free-space propagator in Fourier optics simulations <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28750&id=420820>`_.
#         Args:
#                 img (array): probe
#                 fresnel_number (float): Fresnel number

#         Returns:
#                 [type]: [description]
#         """    
#         hs = img.shape[-1] // 2
#         ar = np.arange(-hs, hs) / float(2 * hs)
#         xx, yy = np.meshgrid(ar, ar)
#         g = np.exp(-1j * np.pi / fresnel_number * (xx ** 2 + yy ** 2))

#         return np.fft.ifft2(np.fft.fft2(img) * np.fft.fftshift(g))


def fresnel_propagator_cone_beam(wavefront, wavelength, pixel_size, sample_to_detector_distance, source_to_sample_distance = 0):

    np = cp.get_array_module(wavefront) # make code agnostic to cupy and numpy
    
    K = 2*np.pi/wavelength # wavenumber
    z2 = sample_to_detector_distance
    z1 = source_to_sample_distance
    
    if z1 != 0:
        M = 1 + (z2/z1)
    else:
        M = 1
    
    gamma_M = 1 - 1/M
        
    FT = np.fft.fftshift(np.fft.fft2(wavefront))

    ny, nx = wavefront.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx,d = pixel_size))#*2*np.pi 2*np.pi factor to calculate angular frequencies 
    fy = np.fft.fftshift(np.fft.fftfreq(ny,d = pixel_size))#*2*np.pi
    FX, FY = np.meshgrid(fx,fy)
    # kernel = np.exp(-1j*(z2/M)/(2*K)*(FX**2+FY**2)) # if using angular frequencies. Formula as in Paganin equation 1.28
    kernel = np.exp(-1j*np.pi*wavelength*(z2/M)*(FX**2+FY**2)) # if using standard frequencies. Formula as in Goodman, Fourier Optics, equation 4.21

    wave_parallel = np.fft.ifft2(np.fft.ifftshift(FT * kernel))*np.exp(1j*K*z2/M)

    if z1 != 0:
        y, x = np.indices(wavefront.shape)
        wave_cone = wave_parallel * (1/M) * np.exp(1j*gamma_M*K*z2) * np.exp(1j*gamma_M*K*(x**2+y**2)/(2*z2))
        return wave_cone
    else:
        return wave_parallel

def create_propagation_video(path_to_probefile,
                             starting_f_value=1e-3,
                             ending_f_value=9e-4,
                             number_of_frames=100,
                             frame_rate=10,
                             mp4=False, 
                             gif=False,
                             jupyter=False):
    
    """ 
    Propagates a probe using the fresnel number to multiple planes and create an animation of the propagation
    #TODO: change this function to create propagation as a function of distance
    """

    probe = np.load(path_to_probefile)[0] # load probe
    
    # delta = -1e-4
    # f1 = [starting_f_value + delta*i for i in range(0,number_of_frames)]
    
    f1 = np.linspace(starting_f_value,ending_f_value,number_of_frames)
    
    # Create list of propagated probes
    b =  [np.sqrt(np.sum([abs(Propagate(a,f1[0]))**2 for a in probe],0))]
    for i in range(1,number_of_frames):
            b += [np.sqrt(np.sum([abs(Propagate(a,f1[i]))**2 for a in probe],0))]
    

    image_list = []
    for j, probe in enumerate(tqdm(b)):
            if jupyter == False:
                animation_fig, subplot = plt.subplots(dpi=300)
                img = subplot.imshow(probe,cmap='jet')#,animated=True)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.set_title(f'f#={f1[j]:.3e}')
            if jupyter == False:
                image_list.append(mplfig_to_npimage(animation_fig))
            else:    
                image_list.append(probe)
            if jupyter == False: plt.close()

    if mp4 or gif:  
        clip = ImageSequenceClip(image_list, fps=frame_rate)
        if mp4:
            clip.write_videofile("propagation.mp4",fps=frame_rate)
        if gif:
            clip.write_gif('propagation.gif', fps=frame_rate)

    return image_list, f1

