# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:30:25 2020

@author: yuri.tonin
"""

import numpy as np

def FFT_filter_1D(positions,signal,cutoff_frequency,debug=False,plot=False):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """ The cutoff frequency should be given in reciprocal units [e.g. 1/mm] to the unit of position being used [e.g. mm]
        Use debug=True to see how the function works with a sinosoidal signal
    """
    
    x = positions
    deltaX = x[1]-x[0]
    if debug:
        start = 100
        x = np.linspace(-start,start,10000)
        deltaX = x[1]-x[0]
        
        # Create signal
        signal = np.zeros(len(x))
        f_max = 4
        for f in range(1,f_max + 1):
            # f = 1/T
            signal += np.sin(2*np.pi*f*x)
        
        cutoff_frequency = 4
    
    f_nyquist = 1 / (2*deltaX)
    # print('\nReal space resolution = {0:.3e}'.format(deltaX))
    # print('Nyquist frequency = {0:.3e}'.format(f_nyquist))

    FT_signal = np.fft.fft(signal)

    FT_signal = np.fft.fftshift(FT_signal)
    frequencies = np.fft.fftfreq(len(signal),d=deltaX)
    frequencies = np.fft.fftshift(frequencies)
    intensity = np.abs(FT_signal)
    
    if f_nyquist < cutoff_frequency:
        import sys
        sys.exit('Cutoff frequency value is bigger than Nyquist frequency of your signal. Choose smaller cutoff.\n')
        
    fspace_filter = np.where( np.abs(frequencies) >= cutoff_frequency, 0, 1 )
    
    convolution = FT_signal*fspace_filter
    
    convolution_shift = np.fft.ifftshift(convolution)
    filtered_signal = np.fft.ifft(convolution_shift)
    
    convolution_intensity = np.abs(convolution)**2
    
    filtered_signal = np.real(filtered_signal)
    
    if plot:
        
        """ PLOT """
        x_lim = 1
        f_lim=5

        figure, subplot = plt.subplots(3,2,dpi=300)
        
        subplot[0,0].plot(x,signal)
        subplot[0,0].set_title('Signal')
        subplot[0,0].set_xlabel('x')
        # subplot[0,0].set_xlim(-x_lim,x_lim)
        subplot[0,0].ticklabel_format(useOffset=False)

        subplot[0,1].plot(frequencies,np.real(FT_signal))
        # subplot[0,1].set_xlim(-f_lim,f_lim)
        subplot[0,1].set_title('Fourier Trasnform')
        subplot[0,1].set_xlabel('f')
        subplot[0,1].set_ylim(np.min(FT_signal),np.max(FT_signal))

        
        subplot[1,0].plot(frequencies,fspace_filter)
        subplot[1,0].set_title('Filter. Cutoff = {0:.3e}'.format(cutoff_frequency))
        subplot[1,0].set_xlabel('f')
        
        subplot[1,1].plot(frequencies,np.real(convolution))
        subplot[1,1].set_title('Convolution')
        subplot[1,1].set_xlabel('f')
        # subplot[1,1].set_xlim(-f_lim,f_lim)
        subplot[1,1].set_ylim(np.min(FT_signal),np.max(FT_signal))
        
        subplot[2,0].plot(x,filtered_signal)
        subplot[2,0].ticklabel_format(axis="both", style="sci",useOffset=False,scilimits=(-3, 3))

        subplot[2,0].set_title('Filtered Signal')
        subplot[2,0].set_xlabel('x')
        # subplot[2,0].set_xlim(-x_lim,x_lim)

        subplot[2,1].set_axis_off()
        
        figure.tight_layout()
        
    return filtered_signal

def ButterworthFilter(x,y,Spatial_Cutoff=1):
    
    from scipy import signal
    
    y_mean = np.mean(y)
    
    y -= y_mean
    
    Nyquist_Freq = 2 * (x[1]-x[0])
    
    Wn = Nyquist_Freq / Spatial_Cutoff
    
    if Spatial_Cutoff < Nyquist_Freq:
        Wn = 1
    
    b, a = signal.butter(1, Wn, 'low')
    output_signal = signal.filtfilt(b, a, y)
    
    return (output_signal+y_mean) 


def FFTLowPassFilter(img,mask_vertical_size,show_filtered_plot=True,show_mask_plots=False): # Fast Fourier Transform 2D Low Pass filter
    
    import matplotlib.pyplot as plt

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    plt.subplot(121),plt.imshow(img, cmap = 'jet')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'jet')
    plt.title('Spatial Frequency Space'), plt.xticks([]), plt.yticks([])
    plt.show()
        
    rows, cols = img.shape
    factor = int(cols/rows)
    crow,ccol = int(rows/2) , int(cols/2)
    
    mask = np.zeros((rows,cols),np.uint8)
    mask_size = int(mask_vertical_size/2)
    mask[crow-mask_size:crow+mask_size, ccol-factor*mask_size:ccol+factor*mask_size] = 1

    fshift = np.multiply(mask,fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift+1))
    
    if show_mask_plots == True:

        plt.figure(figsize=(12,5))
        plt.imshow(magnitude_spectrum)
        plt.title("Masked Fourier Space")
        plt.show()  


    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    
    if show_filtered_plot == True:
        plt.figure(figsize=(12,5))
        plt.imshow(img_back, cmap = 'jet')
        plt.title('Image after Low Pass Filter')
        plt.show()  
    
    return img_back

def piston(dataset): # is this really Zernike piston? check!
    return dataset - dataset.min()

def plane(variables,a,b,c):
    x = variables[0]
    y = variables[1]
    f = a*x+b*y+c
    
    return np.ravel(f)

def linear_equation_1d(x,a,b):
    return a*x + b

def linear_fit(x,y,a=1,b=1):
    import scipy.optimize
    popt, pcov = scipy.optimize.curve_fit(linear_equation_1d, x, y, p0=(a,b)) 
    fit_y = popt[0]*x+popt[1]
    return fit_y

def get_lowest_integer(x):
    import numpy as np
    x = np.int(np.floor(x))
    # x = np.round(x).astype('int')
    return x

def get_closest_integer(x):
    import numpy as np
    x = np.round(x).astype('int')
    return x

def plane(variables,a,b,c):
    
    x = variables[0]
    y = variables[1]
    f = a*x+b*y+c
    
    return np.ravel(f)

def paraboloid(variables,a,b,c):
    x = variables[0]
    y = variables[1]
    f = (x/a)**2+(y/b)**2+c
    
    return np.ravel(f)

def parabola(x, x0, a, b, c):
    return a*(x-x0)**2 + b*(x-x0) +c

def fit_parabola(x,y,initial_guess=[0.0,1.0,1.0, 0]):
    
    # initial_guess = [x0,a,b,c]
    
    from scipy.optimize import curve_fit    

    popt, pcov = curve_fit(parabola, x, y, p0=initial_guess)
    x0 = popt[0]
    a = popt[1]
    b = popt[2]
    c = popt[3]
    
    # print('Best fit parabola: {0:.2e}*(x-{1:.2e})**2 + {2:.2e}*(x-{1:.2e}) + {3:.2e} '.format(a,x0,b,c))
    
    fit = parabola(x, x0, a, b, c)
    
    return fit, x0,a,b,c

def fit_paraboloid(x_array,y_array,data_matrix,fit_guess=(1,1,1)):

    from scipy.optimize import curve_fit    

    x, y = x_array, y_array
    dataset = data_matrix
    X, Y = np.meshgrid(x, y)
    size_to_reshape = X.shape
    
    params, pcov = curve_fit(paraboloid, (X, Y), np.ravel(dataset), fit_guess)
    
    paraboloid_fit = paraboloid(np.array([X, Y]), params[0],params[1],params[2])
    paraboloid_fit = paraboloid_fit.reshape(size_to_reshape)

    return paraboloid_fit

def ellipse_at_mirror_coordinate_system(x,p,q,theta,x0,y0):
    # see Sutter et al - Geometrical and wave-optical effects on the performance of a bent-crystal dispersive X-ray spectrometer.
    y1 = ((p+q)*np.sin(theta))/( (p+q)**2 - (p-q)**2 *np.sin(theta)**2 )
    y2 = 2*p*q+(p-q)*np.cos(theta)*(x-x0) - 2*np.sqrt(p*q)*np.sqrt(p*q+((p-q)*np.cos(theta))*(x-x0)-(x-x0)**2)
    y = y0 + y1 * y2  
    
    # a = np.sqrt(p*q)
    # # a = np.sqrt(p*q+((p-q)*np.cos(theta))*(x-x0)-(x-x0)**2)
    # print(np.min(a))
    # print(a)
    
    return y

def fit_ellipse(x,y,initial_guess,bounds=[]):
    
    from scipy.optimize import curve_fit
    
    if bounds !=[]:

        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
    
        popt, pcov = curve_fit(ellipse_at_mirror_coordinate_system, x, y, p0=initial_guess,bounds=(lower_bounds, upper_bounds))
    
    else:
        popt, pcov = curve_fit(ellipse_at_mirror_coordinate_system, x, y, p0=initial_guess) # there seems to be an Runtime warning when this curve_fit is called due to an invalid value of the sqrt. It does not seem to affect the results, since a fit is obtained afterall
    
    p = popt[0]
    q = popt[1]
    theta = popt[2]
    x0 = popt[3]
    y0 = popt[4]
    
    # print('Ellipse fit parameters: p={0:.2e}, q = {1:.2e}, theta = {2:.2e}'.format(p,q,theta))
    
    fit = ellipse_at_mirror_coordinate_system(x, p, q, theta,x0,y0)
    
    return fit, p,q, theta, x0, y0
    

def calculate_PV_and_RMS(matrix):
    # SUBTRACTS MEAN TO CALCULATE RMS, SO THAT STDDEV == RMS
    matrix = matrix - np.mean(matrix)
    PV = np.max(matrix)-np.min(matrix)
    RMS = np.std(matrix)
    return PV, RMS

def subtract_minimum(matrix):
    minimum = np.min(matrix)
    matrix = matrix - minimum
    return matrix

def derivate(x, y, return_centered_values=True):

    dy_dx = np.diff(y) / np.diff(x)
    if return_centered_values:
        x_new = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    else:
        x_new = x[:-1]

    return x_new, dy_dx


def PSD_1D(x,y):
    
    # as from paper: Using the Power Spectral Density method to characterise the surface topography Using the Power Spectral Density method to characterise the surface topography of optical surfaces
    
    import numpy as np
    
    dx = x[1]-x[0]
    
    PSD_1D = (dx/len(y))*np.abs(np.fft.fft(y))**2
    PSD_1D = np.fft.fftshift(PSD_1D)
    frequencies = np.fft.fftfreq(len(y),d=dx)
    frequencies = np.fft.fftshift(frequencies)
    
    return frequencies, PSD_1D

def get_slope_from_height(position_array,unfiltered_height,apply_filter=True):
    # POSITIONS MUST BE IN METERS!    
    """ Filter data to retain only low frequencies (lower than 10^3 m^-1)"""
    cutoff_frequency = 1e3 # POSITIONS MUST BE IN METERS! THIS WILL FILTER SPATIAL FREQUENCIES GREATEN THAN 1/MM OR 10^3/M
    if apply_filter == True:
        height = FFT_filter_1D(position_array,unfiltered_height,cutoff_frequency)
    else:
        height = unfiltered_height
    positions , slope = derivate(position_array, height, return_centered_values=True)     # positions is the same for all slices
    return positions*1e3, slope

def calculate_RoC(x, y,nominal_RoC=0):
    from scipy.interpolate import interp1d
    
    x1, d1y = derivate(x, y)
    x2, d2y = derivate(x1, d1y)

    d1y_interp = interp1d(x1, d1y)
    d1y_i = d1y_interp(x2) # calculate 1st derivative for same x points from d2y

    R = (1 + d1y_i ** 2) ** (3 / 2) / np.abs(d2y)

    if 1:
        import matplotlib.pyplot as plt
        figure, subplot = plt.subplots(dpi=300)
        if nominal_RoC != 0:
            subplot.axhline(nominal_RoC,color='red',label='Nominal RoC={0:.2e}m'.format(nominal_RoC))
        # subplot.plot(1e3*x2,d2y,label='2nd derivative')
        # subplot.plot(1e3*x1,d1y,label='1st derivative')
        subplot.plot(1e3*x2,R,label='RoC')
        subplot2 = subplot.twinx()
        subplot2.plot(1e3*x,1e9*y,label='Ellipse Height',color='green')
        subplot.grid()
        subplot.legend()
        subplot.set_ylabel('RoC [m]')
        subplot.set_xlabel('Position [mm]')
        subplot2.legend()
        subplot2.set_ylabel('Height [nm]')
    return x2, R

def calculate_radius_difference(x, *args):
    from scipy.interpolate import interp1d
    
    # get x_ell and radius_ell from args
    x_ell = args[0]
    radius_ell = args[1]
    nominal_RoC = args[2]
    
    radius_ell_interp = interp1d(x_ell, radius_ell) 
    
    # manually chack if x is inside x_ell bounds
    if((x <= x_ell.min()) or (x >= x_ell.max())): #THIS 'IF STATEMENT' SOLVES SCIPY ISSUE OF FINDING POINT OUT OF INTERPOLATION RANGE
        # value out of interpolation range. Returning infinity")
        return np.inf # return large value to fail minimization
    else:
        return radius_ell_interp(x) - nominal_RoC 
    
def rotate_curve_around_origin(x,y,rotation_angle):
    x = x*np.cos(rotation_angle)+y*np.sin(rotation_angle)
    y = -x*np.sin(rotation_angle)+y*np.cos(rotation_angle)
    return x, y

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    
def Translation_matrix(x,y):
  import numpy as np
  return np.array([[ 1, 0, x],
                    [ 0, 1, y],
                    [ 0, 0, 1]])

def Rotation2D_around_point(theta,x,y,rotation_point): 
    import numpy as np
    """ theta in radians """   

    x0, y0 = rotation_point
    
    rotation_matrix = np.array( [[np.cos(theta),-np.sin(theta),0],
                                 [np.sin(theta),np.cos(theta) ,0],
                                 [0            ,0             ,1]])

    data = np.array([x,y,np.ones(len(x))])
    
    vector = Translation_matrix(x0,y0)@rotation_matrix@Translation_matrix(-x0,-y0)@data
    
    return vector[0], vector[1]