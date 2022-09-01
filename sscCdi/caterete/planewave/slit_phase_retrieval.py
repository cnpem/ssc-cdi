from re import I
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cupy as cp

def propagate_beam(wavefront, dx, wavelength,distance,propagator='fourier'):
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

    ysize, xsize = wavefront.shape
    
    x_array = np.linspace(-xsize/2,xsize/2-1,xsize)
    y_array = np.linspace(-ysize/2,ysize/2-1,ysize)
    
    fx = x_array/(xsize)
    fy = y_array/(ysize)
    
    FX,FY = np.meshgrid(fx,fy)

    if propagator == 'fourier':
        if distance > 0:
            output = fftshift(fft2(fftshift(wavefront)))
        else:
            output = ifftshift(ifft2(ifftshift(wavefront)))            
    elif propagator == 'fresnel':
        # % Calculate approx phase distribution for each plane wave component
        w = FX**2 + FY**2 
        # Compute FFT
        F = fftshift(fft2(fftshift(wavefront)))
        # % multiply by phase-shift and inverse transform 
        a = np.exp(-1j*np.pi*( distance*wavelength/dx**2)*w)
        output = ifftshift(ifft2(ifftshift(F*a)))

    return output

def get_object_pixel(N,pixel_size,wavelength,distance):
    return wavelength*distance/(N*pixel_size)

def magnitude_constraint(estimate,data):
    return np.sqrt(data)*estimate/np.abs(estimate)

def pinhole_constraint(wavefront,pinhole):
    wavefront[pinhole<=0] = 0
    wavefront[wavefront<0] = 0
    return wavefront

def support_constraint(wavefront,support,use_half_support=False):
    return wavefront

def object_update_function(object,probe,wavefront,previous_wavefront,alpha):
    return wavefront

def probe_update_function(object,probe,wavefront,previous_wavefront,beta):
    return wavefront

speed_of_light, planck = 299792458, 4.135667662E-18  # Plank constant [keV*s]; Speed of Light [m/s]
N = 100

""" Load model """
model = np.ones((N,N))
calculate_error = False # flag to calculate error against model at each iteration
error = []

""" Input parameters """
distance = 10 # meters
energy = 3 # keV
wavelength = planck * speed_of_light / energy # meters
detector_pixel = 55e-6 # meters
n_of_pixels = 3072 

object_pixel_size = get_object_pixel(n_of_pixels,detector_pixel,wavelength,distance)

apply_support = False

distance_pinhole = 3e-3 # mm

alpha,beta = 0.9, 0.1 # PIE update function parameters

phase_retrieval_iterations = 1
""" Load measurement """
DP = np.ones((N,N)) # load restored diffraction pattern
t = np.linspace(-1,1,N)
x, y = np.meshgrid( t, t)
pinhole = (x**2 + y**2 < 0.01)*1.0 # estimate from ptycho?


""" Initial guess """
probe = propagate_beam(pinhole,object_pixel_size,wavelength,distance_pinhole)
object = np.ones_like(probe,dtype=complex)
wavefront = probe*object


""" Estimate support from auto-correlation """
support = np.ones_like(object)

""" Phase-retrieval loop """
for iter in range(0,phase_retrieval_iterations):

    previous_wavefront = wavefront
    """ Fourier Space-constraint """
    wavefront = propagate_beam(wavefront, object_pixel_size, wavelength,+distance) # propagate
    wavefront = magnitude_constraint(wavefront,DP) # substitute known magnitude
    wavefront = propagate_beam(wavefront, object_pixel_size, wavelength,-distance) # backpropagate

    """ Real Space-constraint """
    if apply_support == True:
        wavefront = support_constraint(wavefront,support)

    probe = wavefront/object

    probe = propagate_beam(probe,object_pixel_size,wavelength,-distance_pinhole,propagator='fresnel') # short distance propagation! is fresnel propagator the correct one?
    probe = pinhole_constraint(probe,pinhole) # set wavefront null outside pinhole; set negative values null inside pinhole
    probe = propagate_beam(probe,object_pixel_size,wavelength,+distance_pinhole,propagater='fresnel')

    object = object_update_function(object,probe,wavefront,previous_wavefront,alpha) # update function of the PIE family
    probe  = probe_update_function(object,probe,wavefront,previous_wavefront,beta)

    wavefront = probe*object

    if calculate_error:
        error.append(calculate_error(wavefront,model))






