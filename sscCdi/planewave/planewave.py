import numpy as np
import os
import time

import ctypes
from   ctypes import c_void_p  as void_p

from ..cditypes_planewave import *


def subformat_shuffle(data, num_gpus):
  """
  @brief Rearranges a 2D array `data` by splitting along the second axis and stacking 
  the segments along the first axis.  It splits `data` into `num_gpus` segments along 
  the columns (axis 1) and then concatenates these segments along the rows (axis 0).
  
  This function is designed to rearrange data in a way that when it is redistributed
  to multiple GPUs in a CUDA cudaLibXtDesc *data pointer, its cufftXtSubFormat subformat 
  variable is exactly CUFFT_XT_FORMAT_INPLACE_SHUFFLED. This is the subformat that a 
  given CUFFT_XT_FORMAT_INPLACE subformatted volume becomes after an FFT execution with 
  th cufftXt library. This function is applied to the measured data in pwCDI just 
  before the ctypes wrapper is called, and by doing this, we can multiply the shuffled
  measured data directly with the Fourier domain volumes that are obtained in shuffled 
  subformat during the iterations. A more detailed explanation can be found in this 
  presentation: https://cnpemcamp.sharepoint.com/:p:/r/sites/GCC-ScientificComputing/Documentos%20Partilhados/Shared%20material/Apresenta%C3%A7%C3%B5es/2024%20-%20performance%20updates%203dpwcdi%20(mauro).pptx?d=w19683560326d4650adc6947174f69cb5&csf=1&web=1&e=vxogDb

  Note that only setting the subformat variable in the struct type cudaLibXtDesc is 
  obviously not enough to actually change the data format. One has to actively 
  rearrange the voxels to obtain the matching subformat. In principle, that subformat 
  change could be done with some cudaMemcpy calls, but I never managed to make it work.
  Be very careful to modify this. 

  The transformation can be represented as:
  
    Given an input array of shape (M, N),
    - It is split into `num_gpus` parts along the column dimension (axis 1).
    - The resulting parts are stacked along the row dimension (axis 0).
  
  @param data        Input 2D NumPy array of shape (M, N) to be shuffled.
  @param num_gpus    Number of GPUs to split the data for.
  
  @return           A reshaped NumPy array with shuffled dimensions.
  
  @note This function assumes `num_gpus` is a valid divisor of `data.shape[1]` to ensure 
  equal splits.
  """
  M = data.shape[0]
  split_data_x = np.array_split(data, num_gpus, axis=1)
  final_data = np.concatenate(split_data_x, axis=0)
  return final_data


 
def pwcdi3d(data, dic):
  """
  @brief Executes the 3D Plane-Wave Coherent Diffraction Imaging (PWCDI) algorithm across multiple GPUs.
  
  This function processes a 3D diffraction dataset using the PWCDI method, distributing computations over multiple GPUs when available
  and needed. It allows support initialization, support constraints, and various algorithmic and optimization options for iterative phase 
  retrieval. The wrapper executes a cascade of algorithms defined by the dic parameter. Each algorithm operates by updating an object
  estimate based on the input diffraction data while applying constraints from a support array. Additional algorithmic parameters, 
  such as error metrics and extra constraints, influence the reconstruction process. The algorithms must be specified in a structured
  way: 0, 1, 2, .... Always starting from 0 and incrementing 1 per algorithm. They will be executed in that order. Example:
  
  dic['method'] = {0:{"name": "HIO",
                    "niter": 10000,                        
                    "shrinkwrap_subiter": 20,
                    "initial_shrinkwrap_subiter": 20,
                    "beta": 0.95,
                    "beta_update": 7,
                    "beta_reset_subiter": 20,
                    "sigma": 6.0/np.pi,
                    "sigma_mult": 0.99,
                    "shrinkwrap_threshold": 6.0,
                    "shrinkwrap_iter_filter": "amplitude",  
                    "extra_constraint": None,                 
                    "extra_constraint_subiter": 1,
                    "initial_extra_constraint_subiter": 0},
                1:{"name": "ER",
                   "niter": 200}}

  @param data: 3D diffraction dataset, provided as a NumPy array of shape (N, N, N) with dtype `float32`.
  @param dic: Dictionary containing algorithmic parameters, including:
      - `obj_input` (np.ndarray, optional): Initial guess for the object, must be (N, N, N) with dtype `complex64`. 
      - `gpus` (list[int]): List of GPU IDs used for computation. Must be a sublist of [0,1,..,M] where M is the number of available GPUs.
      - 'timing' (bool): Flag to enable timing measurements.
      - 'swap_d_x' (bool, default: False): Swap the main iteration HOST<->DEVICES variable during ShrinkWrap operation.
      - 'map_d_signal' (bool, default: False): Keep the measured data variable in CUDA mapped memory or not.
      - 'map_d_support' (bool, default: False): Keep the support variable in CUDA mapped memory or not. 
      - `eps_zeroamp` (float, optional, default: 0): Threshold for zero-amplitude regions in projection operators.
      - 'sthreads' (int) [DEPRECATED]: Number of CPU threads used for parallel copy operations. 
      - `support` (dict, optional): Parameters defining the support constraint:
          - `sup_data` (np.ndarray, int, optional, default: None): Binary support mask (N, N, N, dtype `uint8`). If passed as None, the support 
          will be created automatically using a norm p ball of radius r (see the two parameters below).
          - `p` (int): Norm type of the automatically created support.
          - `r` (float): Ball radius of the automatically created support.  
      - `method` (list[dict]): Sequence of iterative phase retrieval methods to apply, where each method includes:
          - `name` (str): Algorithm name (e.g., 'HIO', 'ER').
          - 'sigma' (float, default: 0.1): Controls the standard deviation for some process. 
          - 'sigma_mult' (float, default: 0.99): Multiplication factor applied to sigma.
          - 'beta' (float, default: 0.9): The relaxation HIO parameter. Ignored if running ER.
          - 'beta_update' (float, default: 7.0): Exponent term of the HIO relaxation parameter. Ignored if running ER.
          - 'beta_reset_subiter' (int, default: -1): Defines when beta should reset; -1 means no reset.
          - `niter` (int): Number of iterations.
          - 'shrinkwrap_subiter' (int, optional, default: 10): Frequency of shrinkwrap updates. 
          - 'initial_shrinkwrap_subiter (int, option, default: 0): Start shrinkwrapping at that iteration. 
          - 'shrinkwrap_iter_filter' (string, optional, default: "amplitude"): What part of the iteration variable will be used to 
          extract the new support during shrinkwrap ("full", "amplitude", "real").
          - 'shrinkwrap_threshold' (float, default: 0.1): Threshold for the shrinkwrap operation.
          - `extra_constraint` (str, optional, default: None): Additional constraint type (e.g., 'LEFT_SEMIPLANE', 'FIRST_QUADRANT').
          - `extra_constraint_subiter` (int, optional, default: 1): Frequency of applying the extra constraint. Must be greater than zero.
          - 'initial_extra_constraint_subiter' (int, optional, default: 0): When to start extra_constraint
      - `err_type` (str, optional, default: None) [NOT FULLY OPERATIONAL]: Type of error metric used (e.g., 'iter_diff'). 
      - `err_subiter` (int, optional, default: 0) [NOT FULLY OPERATIONAL]: Frequency of error metric evaluation.

  @return obj_output (np.ndarray, complex): The output recovered complex object.
  @return finsup_output (np.ndarray, int): The output recovered support. 


  @note This function assumes that `data` and other variables (if provided) are preallocated with correct
  sizes and contiguous.
  """

  # todo: ensure that data is cube (N,N,N), and that support is also (N,N,N), and that object initizalization 
  # is also (N,N,N)
  N = data.shape[0]

  # other main parameters 
  ngpus = len(dic['gpus'])
  gpus = dic['gpus']
  timing = dic['timing']

  # change to subformat to shuffled if necessary
  if ngpus>1:
    data = subformat_shuffle(data, ngpus)
  
  # handle misc parameters 
  eps_zeroamp = dic.get('eps_zeroamp',0.0)
  
  # handle output pointers 
  obj_output = np.zeros((N,N,N), dtype=np.complex64)  
  finsup_output = np.zeros((N,N,N), dtype=np.uint8)

  # handle initial object data 
  obj_input = dic.get("obj_input", None)
  if isinstance(obj_input, np.ndarray):
    if obj_input.shape != (N,N,N):
      raise ValueError("obj_input must have the same shape as the object")
    if obj_input.dtype != np.complex64:
      obj_input = obj_input.astype(np.complex64)
    
    obj_input = np.ascontiguousarray(obj_input)
    obj_input_ptr = obj_input.ctypes.data_as(np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags="C_CONTIGUOUS"))
    # np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS')
  elif obj_input is None:
    # obj_input = ctypes.cast(None, ctypes.POINTER(ctypes.c_void_p))
    obj_input_ptr = ctypes.c_void_p()

    
  # handle the support parameters 
  sup_info = dic.get('support',{'p': 10, 
                                'r': 0.4,
                                'data':None})
  sup_data = sup_info.get('data', None) #.astype(np.uint8)
  if sup_data is None:
    sup_data = ctypes.POINTER(ctypes.c_ubyte)()
  else:
    if isinstance(sup_data, np.ndarray):
      if sup_data.shape != (N,N,N):
        raise ValueError("sup_data must have the same shape as the object")
      if sup_data.dtype != np.uint8:
        sup_data = sup_data.astype(np.uint8)
    sup_data = getPointer(sup_data.flatten(), dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)) 
  
  # handle extra parameters 
  sthreads      = dic.get('sthreads',1) 
  b_ngpu        = ctypes.c_int(ngpus)
  b_gpus        = np.array(gpus, dtype=np.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  b_data        = getPointer(data.flatten(), dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
      
  # handle err parameters 
  err_type = dic.get("err_type",None)
  err_subiter = dic.get("err_subiter", 1)        
  err_type_options = {None:            0,  # NO_ERR
                     "iter_diff":     1}   # ITER_DIFF
  err_type = err_type_options.get(err_type, None)
  
  # handle memory parameters 
  map_d_signal = dic.get("map_d_signal", False)
  map_d_support = dic.get("map_d_support", False)
  swap_d_x = dic.get("swap_d_x", False)
  
  ## common parameters
  PARAMS = make_SSC_PWCDI_PARAMS((timing, 
                                  N, 
                                  sthreads, 
                                  sup_info['p'], 
                                  sup_info['r'],
                                  sup_data, 
                                  eps_zeroamp,
                                  err_type,
                                  err_subiter,
                                  map_d_signal,
                                  map_d_support,
                                  swap_d_x))
  
  ## algorithmic planning 
  num_methods = len(dic['method'])

  SEQ = []
  for k in range(num_methods):      
    # dictionary parsing 
    if isinstance(dic['method'][k], dict):
      # base method choices
      name = dic['method'][k]['name']
      iteration = dic['method'][k]['niter']


      # hio base  parameters 
      shrinkwrap_subiter = dic['method'][k].get('shrinkwrap_subiter',10)
      initial_shrinkwrap_subiter = dic["method"][k].get("initial_shrinkwrap_subiter",0)
      extra_constraint = dic['method'][k].get('extra_constraint',None)
      extra_constraint_subiter = dic['method'][k].get('extra_constraint_subiter', 1)
      initial_extra_constraint_subiter = dic['method'][k].get('initial_extra_constraint_subiter',0)
          
      # decode extra constraint model in hio
      extra_constraint_options = {None:               0,    # NO_EXTRA_CONSTRAINT
                                  "left_semiplane":   1,    # LEFT_SEMIPLANE
                                  "right_semiplane":  2,    # RIGHT_SEMIPLANE
                                  "top_semiplane":    3,    # TOP_SEMIPLANE 
                                  "bottom_semiplane": 4,    # BOTTOM_SEMIPLANE
                                  "first_quadrant":   5,    # FIRST_QUADRANT
                                  "second_quadrant":  6,    # SECOND_QUADRANT
                                  "third_quadrant":   7,    # THIRD_QUADRANT
                                  "fourth_quadrant":  8}    # FOURTH_QUADRANT 
      extra_constraint = extra_constraint_options.get(extra_constraint, None)

      # decode remaining base hio parameters 
      sigma = dic['method'][k].get('sigma',0.1)
      sigma_mult = dic['method'][k].get('sigma_mult',0.99)
      beta = dic['method'][k].get('beta',0.9)
      beta_update = dic['method'][k].get('beta_update',7.0)
      beta_reset_subiter = dic['method'][k].get("beta_reset_subiter", -1) # by default, it doesn't reset 

      # Decode shrinkwrap customizations in hio. These two options are related to how the mask is 
      # multiplied to the iteration variable and what part of the iteration variable is filtered to obtain
      # the new support. 
      # I'm making shrinkwrap_mask_multiply hardcoded full since it seems to be the only logical option.
      # the other ones were developed for debugging purposes and can be removed in future versions. The
      # iter filter type is more debatable as some papers mention to filter only the amplitude of the 
      # iteration variable in order to obtain a support 
      shrinkwrap_iter_filter_options = {'full':0,
                                       'amplitude':1,
                                       'real':2}
      shrinkwrap_mask_multiply_options = {'full':0,
                                         'real':1,
                                         'legacy':2}

      shrinkwrap_threshold = dic['method'][k].get('shrinkwrap_threshold', 0.1)
      shrinkwrap_iter_filter = shrinkwrap_iter_filter_options.get(dic['method'][k].get('shrinkwrap_iter_filter', 'amplitude'))
      # shrinkwrap_mask_multiply = shrinkwrap_mask_multiply_options.get(dic['method'][k].get('shrinkwrap_mask_multiply', 'full'))
      shrinkwrap_mask_multiply = shrinkwrap_mask_multiply_options["full"] 

      # This other customization of the shrinkwrap allows one to define the gaussian kernel already
      # fftshifted, which means we don't need to fftshift it before using it. I'm also making this 
      # hard coded False because it is not completely tested. it can be done in the future 
      # for an optimized execution shrinkwrap_fftshift_gaussian = dic['method'][k].get('shrinkwrap_fftshift_gaussian', True)
      shrinkwrap_fftshift_gaussian = dic["method"][k].get("shrinkwrap_fftshift_gaussian", False)
        
      # append to the sequence
      SEQ.append(make_SSC_PWCDI_METHOD((name, 
                                        iteration, 
                                        shrinkwrap_subiter, 
                                        initial_shrinkwrap_subiter,
                                        extra_constraint, 
                                        extra_constraint_subiter,
                                        initial_extra_constraint_subiter, 
                                        shrinkwrap_threshold,  
                                        shrinkwrap_iter_filter,
                                        shrinkwrap_mask_multiply,
                                        shrinkwrap_fftshift_gaussian,
                                        sigma, 
                                        sigma_mult, 
                                        beta,
                                        beta_update,
                                        beta_reset_subiter))) 
  
  ALGO = (SSC_PWCDI_METHOD*num_methods)(*SEQ)
  b_nalgo = ctypes.c_int(num_methods)


  # call the main pwcdi function   
  libssccdi.pwcdi(obj_output,       
                  finsup_output,   
                  b_data,
                  obj_input_ptr, 
                  b_gpus,
                  b_ngpu,
                  b_nalgo,
                  PARAMS,
                  ALGO)
    
  # returns recovered data
  return obj_output, finsup_output
 
