import numpy as np

""" Relative imports """
from ..misc import wavelength_from_energy

def calculate_fresnel_number(energy,pixel_size,sample_detector_distance,magnification=1,source_sample_distance=0):
    """_summary_

    Args:
        energy: energy in keV
        pixel_size: object pixel size
        sample_detector_distance: sample to detector distance in meters
        magnification (int, optional): magnification of the optical system. Defaults to 1.
        source_sample_distance (int, optional): source to sample distance in meters. Defaults to 0.

    Returns:
        _type_: _description_
    """

    wavelength = wavelength_from_energy(energy) # meters
    if magnification != 1:
        magnification = (source_sample_distance+source_sample_distance)/source_sample_distance
    return -(pixel_size**2) / (wavelength * sample_detector_distance * magnification)

def Propagate(img, fresnel_number): # Probe propagation
        """ Frunction for free space propagation of the probe in the Fraunhoffer regime

        See paper `Memory and CPU efficient computation of the Fresnel free-space propagator in Fourier optics simulations <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28750&id=420820>`_.
        Args:
                img (array): probe
                fresnel_number (float): Fresnel number

        Returns:
                [type]: [description]
        """    
        hs = img.shape[-1] // 2
        ar = np.arange(-hs, hs) / float(2 * hs)
        xx, yy = np.meshgrid(ar, ar)
        g = np.exp(-1j * np.pi / fresnel_number * (xx ** 2 + yy ** 2))

        return np.fft.ifft2(np.fft.fft2(img) * np.fft.fftshift(g))

def create_propagation_video(path_to_probefile,
                             starting_f_value=1e-3,
                             ending_f_value=9e-4,
                             number_of_frames=100,
                             frame_rate=10,
                             mp4=False, 
                             gif=False,
                             jupyter=False):
    
    """ Propagates a probe using the fresnel number to multiple planes and create an animation of the propagation
    #TODO: change this function to create propagation as a function of distance
    """


    from tqdm import tqdm
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

