import numpy as np

""" UNDER DEVELOPMENT: Position Correction """

def update_beta(positions1,positions2, beta):
    
    k = np.corrcoef(positions1,positions2)[0,1]

    if np.isnan(k).any():
        print('Skipping')
    else:
        threshold1 = +0.3
        threshold2 = -0.3
        
        if k > threshold1:
            beta = beta*1.1 # increase by 10%
        elif k < threshold2:
            beta = beta*0.9 #reduce by 10%
        else:
            pass # keep same value
        
    return beta

def get_illuminated_mask(probe,probe_threshold):
    probe = np.abs(probe)
    mask = np.where(probe > np.max(probe)*probe_threshold, 1, 0)
    return mask

def position_correction(i, obj,previous_obj,probe,position_x,position_y, betas, probe_threshold=0.5, upsampling=100):

    beta_x,beta_y = betas

    illumination_mask = get_illuminated_mask(probe,probe_threshold)

    obj = obj*illumination_mask
    previous_obj = previous_obj*illumination_mask

    relative_shift, error, diffphase = phase_cross_correlation(obj, previous_obj, upsample_factor=upsampling)

    # if 0 :
    #     threshold = 5
    #     if np.abs(beta_y*relative_shift[0]) > threshold or np.abs(beta_x*relative_shift[1]) > threshold:
    #         new_position = np.array([position_x,position_y])
    #     else:
    #         new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    #         # new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # else:
    
    # new_position = np.array([position_x + beta_x*relative_shift[1], position_y + beta_y*relative_shift[0]])
    new_position = np.array([position_x - beta_x*relative_shift[1], position_y - beta_y*relative_shift[0]])
    # new_position = np.array([position_x + beta_x*relative_shift[0], position_y + beta_y*relative_shift[1]])

    if i == 0:
        print(position_x, beta_x*relative_shift[1],'\t',position_y,beta_y*relative_shift[0],relative_shift)

    return new_position, relative_shift, illumination_mask

def position_correction2(i,updated_wave,measurement,obj,probe,px,py,offset,betas,experiment_params):
    """ Position correct of the gradient of intensities """ 
    
    beta_x, beta_y = betas
    
    
    # Calculate intensity difference
    updated_intensity_at_detector = np.abs(updated_wave)**2
    intensity_diff = (updated_intensity_at_detector-measurement).flatten()
    
    # Calculate wavefront gradient
    obj_dy = np.roll(obj,1,axis=0)
    obj_dx = np.roll(obj,1,axis=1)
    
    obj_box     = obj[py:py+offset[0],px:px+offset[1]]
    obj_dy_box  = obj_dy[py:py+offset[0],px:px+offset[1]]
    obj_dx_box  = obj_dx[py:py+offset[0],px:px+offset[1]]
    
    wave_at_detector    = propagate_beam(obj_box*probe,    experiment_params,propagator='fourier')
    wave_at_detector_dy = propagate_beam(obj_dy_box*probe, experiment_params,propagator='fourier')
    wave_at_detector_dx = propagate_beam(obj_dx_box*probe, experiment_params,propagator='fourier')

    obj_pxl = experiment_params[0]
    wavefront_gradient_x = (wave_at_detector-wave_at_detector_dx)/obj_pxl
    wavefront_gradient_y = (wave_at_detector-wave_at_detector_dy)/obj_pxl
   
    # Calculate intensity gradient
    intensity_gradient_x = 2*np.real(wavefront_gradient_x*np.conj(wave_at_detector))
    intensity_gradient_y = 2*np.real(wavefront_gradient_y*np.conj(wave_at_detector))
    
    
    # Solve linear system
    A_matrix = np.column_stack((intensity_gradient_x.flatten(),intensity_gradient_y.flatten()))
    A_transpose = np.transpose(A_matrix)
    relative_shift = np.linalg.pinv(A_transpose@A_matrix)@A_transpose@intensity_diff

    # Update positions
    # new_positions = np.array([px - beta_x*relative_shift[0], py - beta_y*relative_shift[1]])
    new_positions = np.array([py - beta_y*relative_shift[1],px - beta_x*relative_shift[0]])
    
    if i == 0:
        print(px, beta_x*relative_shift[1],'\t',py,beta_y*relative_shift[0],relative_shift)
    
    return new_positions

def plot_positions_and_errors(data_folder,dataname,offset,PIE_positions=[],positions_story=[]):
    
    import os, json
    
    metadata = json.load(open(os.path.join(data_folder,dataname,'mdata.json')))
    distance = metadata['/entry/beamline/experiment']['distance']*1e-3
    energy = metadata['/entry/beamline/experiment']['energy']
    pixel_size = metadata['/entry/beamline/detector']['pimega']['pixel size']*1e-6
    wavelength, wavevector = calculate_wavelength(energy)
    
    diffraction_patterns = np.load(os.path.join(data_folder,dataname,f"0000_{dataname}_001.hdf5.npy"))

    n_pixels = diffraction_patterns.shape[1]
    obj_pixel_size = wavelength*distance/(n_pixels*pixel_size)
    
    _,_,measured = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}",obj_pixel_size,offset,0)
    _,_,true = read_probe_positions_in_pxls(os.path.join(data_folder,dataname),f"0000_{dataname}_without_error",obj_pixel_size,offset,0)
    
    colors = np.linspace(0,positions.shape[0]-1,positions.shape[0])
    fig, ax = plt.subplots(dpi=150)
    ax.legend(["True" , "Measured", "Corrected", "Path"],loc=(1.05,0.84))    
    ax.scatter(measured[:,1],measured[:,0],marker='o',c='red')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    if positions_story != []:
        for i in range(PIE_positions.shape[0]):
            y = positions_story[:,i,1]
            x = positions_story[:,i,0]
            ax.scatter(y,x,color='blue',s=2,marker=',',alpha=0.2)
    if PIE_positions != []:
        ax.scatter(PIE_positions[:,1],PIE_positions[:,0],marker='x',color='blue')#,c=np.linspace(0,positions.shape[0]-1,positions.shape[0]),cmap='jet')
    ax.scatter(true[:,1],true[:,0],marker='*',color='green')#,c=colors,cmap='jet')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid()

