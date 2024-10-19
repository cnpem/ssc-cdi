import numpy as np
import os
import time

import ctypes
from   ctypes import c_void_p  as void_p

from ..cditypes_planewave import *


def pwcdi3d(data, dic):
  N = dic['dimension']
  ngpus = len(dic['gpus'])
  gpus = dic['gpus']
  timing = dic['timing']


  # sigma = dic.get('sigma',0.1)
  # sigma_mult = dic.get('sigma_mult',0.99)
  # beta = dic['beta']
  # beta_update = dic.get('beta_update',7.0)
  # beta_reset_subiter = dic.get("beta_reset_subiter", -1) # by default, it doesnt reset 

  # handle misc parameters 
  eps_zeroamp = dic.get('eps_zeroamp',0.001)
  
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
                                'data':None,
                                'positive_imag':False})
  sup_data = sup_info.get('data',None) #.astype(np.uint8)
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
      extra_constraint_subiter = dic['method'][k].get('extra_constraint_subiter',0)
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

      # decode shrinkwrap customizations in hio
      shrinkwrap_iter_filter_options = {'full':0,
                                       'amplitude':1,
                                       'real':2}
      shrinkwrap_mask_multiply_options = {'full':0,
                                         'real':1,
                                         'legacy':2}

      shrinkwrap_threshold = dic['method'][k].get('shrinkwrap_threshold', 0.1)
      shrinkwrap_iter_filter = shrinkwrap_iter_filter_options.get(dic['method'][k].get('shrinkwrap_iter_filter', 'amplitude'))
      shrinkwrap_mask_multiply = shrinkwrap_mask_multiply_options.get(dic['method'][k].get('shrinkwrap_mask_multiply', 'full'))
      shrinkwrap_fftshift_gaussian = dic['method'][k].get('shrinkwrap_fftshift_gaussian', True)
      
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
    
       
  return obj_output, finsup_output
 