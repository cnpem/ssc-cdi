import numpy as np
import matplotlib.pyplot as plt
import json
import sscCdi
import sscRaft

print(f'sscCdi version: {sscCdi.__version__}')
print(f'sscRaft version: {sscRaft.__version__}')

dic = {}

#### Load data

dic["recon_method"]  = "ptycho" # ptycho or pwcdi
dic["contrast_type"] = "complex" # phase, magnitude or complex
dic["sinogram_path"] = "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/proc/recons/glass_sphere/2023-08-17-10h52m_glass_sphere.hdf5"

dic = sscCdi.define_paths(dic)

obj, angles = sscCdi.read_data(dic)
print(f"Object shape = {obj.shape} \t Number of angles: {angles.shape}")

projection = np.sum(obj,axis=0)
print(obj.shape,projection.shape)

#### Crop data

dic["top_crop"]    = 1500
dic["bottom_crop"] = 1400
dic["left_crop"]   = 1400
dic["right_crop"]  = 1500

sscCdi.tomo_crop(dic,obj)

cropped_data = np.load(dic["cropped_sinogram_filepath"])

#### Sort data

sscCdi.tomo_sort(dic,cropped_data, angles)

dic["bad_frames_after_sorting"] = [113]
sscCdi.remove_frames_after_sorting(dic)

sorted_data = np.load(dic["ordered_object_filepath"])

#### Alignment: Cross Correlation (CC) and Vertical Mass Fluctuation (VMF)
#### Comment if not use

aligned_data_CC = sscCdi.alignment_variance_field(sorted_data, fft_upsampling=10,return_common_valid_region=True, remove_null_borders = True)
aligned_data_VMF = sscCdi.alignment_vertical_mass_fluctuation(
                                        aligned_data_CC, use_phase_gradient = True, 
                                        return_common_valid_region=True, 
                                        remove_null_borders = True, plot = False
                                    ) # if data is not equalized, phase gradient should be used

np.save(dic["pre_aligned_sinogram_filepath"],aligned_data_VMF) # select which prealigned dataset to save

#### Unwrap

cropped_data = np.load(dic["cropped_sinogram_filepath"])
dic["bad_frames_before_unwrap"] = [28,30,45,49,65,66,91,113]

""" Select data to be unwrapped """
# data_to_unwrap = np.angle(sorted_data)
# data_to_unwrap = np.angle(cropped_data)
data_to_unwrap = np.angle(aligned_data_CC)
# data_to_unwrap = np.angle(aligned_data_VMF)

sscCdi.tomo_unwrap(dic, data_to_unwrap)

#### 2D Equalization

dic["bad_frames_before_equalization"] = []

dic["CPUs"] = 32

dic["equalize_invert"] = True # invert phase shift signal from negative to positive
dic["equalize_ROI"] = [0,100,0,100] # region of interest of null region around the sample used for phase ramp and offset corrections
dic["equalize_remove_phase_gradient"] = True  # if empty and equalize_ROI = [], will subtract best plane fit from whole image
dic["equalize_remove_phase_gradient_iterations"] = 5
dic["equalize_local_offset"] = False # remove offset of each frame from the mean of ROI 
dic["equalize_set_min_max"]= [-5,8] # [minimum,maximum] threshold values for whole volume
dic["equalize_non_negative"] = True # turn any remaining negative values to zero

sinogram_to_equalize = np.load(dic["unwrapped_sinogram_filepath"])
sscCdi.tomo_equalize(dic,sinogram_to_equalize)

#### Alignment (wiggle) 

dic["wiggle_sinogram_selection"] = dic["equalized_sinogram_filepath"]
dic["step_percentage"] = 0 # need to project irregular angle steps to a regular grid?

sscCdi.preview_angle_projection(dic)

dic["project_angles_to_regular_grid"] = True 
dic["bad_frames_before_wiggle"] = [] 
dic["wiggle_reference_frame"] = 0 

dic = sscCdi.tomo_alignment(dic)

#### Tomography: select dictionary according to tomographic method

dic['using_wiggle'] = True

dic['automatic_regularization'] = 0 # skip if 0; regularization between 0 and 1 to enhance borders prior to recon (https://www.sciencedirect.com/science/article/pii/S2590037419300883?via%3Dihub)

# dic["algorithm_dic"] = { # if FBP: filtered back-projection
#     'algorithm': "FBP",
#     'gpu': [0,1],
#     'filter': 'lorentz', # 'gaussian','lorentz','cosine','rectangle'
#     'regularization': 0.1, # 0 <= regularization <= 1; use for smoothening
# }

dic["algorithm_dic"] = { # if eEM: emission expectation maximization
    'algorithm': "EM",
    'gpu': [0,1],
    'regularization': 0.0001,
    'method': 'eEM', 
    'niterations': [20,0,0,0], # [global iterations, iterations EMTV, iterations2 EMTV, Cone-beam integration points]
    'epsilon': 1e-15, #for EMTV only
    'blocksize': 20, # blocks for parallelization
}

# sinogram = np.load(dic["equalized_sinogram_filepath"])
sinogram = np.load(dic["wiggle_sinogram_filepath"])
recon3D = sscCdi.tomo_recon(dic,sinogram)

#### 3D Equalization

dic["tomo_remove_outliers"] = 0
dic["tomo_threshold"] = 20.0
dic["tomo_local_offset"] = [] # [top,bottom,left,right, axis]
dic["tomo_mask"] = []

sscCdi.tomo_equalize3D(dic)

