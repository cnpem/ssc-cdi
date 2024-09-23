import numpy as np
import os
import time

import ctypes
from   ctypes import c_void_p  as void_p

from ..cditypes_planewave import *


def pwcdi3d(data, dic):
  """ Phase-retrieval for a 3D measured data.
  
  Args:
      dic: input dictionary
  
  Returns:
      (None)
  """

  N = dic['dimension']
  ngpus = len(dic['gpus'])
  gpus = dic['gpus']
  timing = dic['timing']
  sigma = dic.get('sigma',0.1)
  sigma_mult = dic.get('sigma_mult',0.99)
  beta = dic['beta']
  beta_update = dic.get('beta_update',7.0)
  betaResetSubiter = dic.get("betaResetSubiter", -1) # by default, it doesnt reset 
  eps_zeroamp = dic.get('eps_zeroamp',0.001)
  outpath  = dic['output']
  
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
    
  # handle the support parameters 
  sup_info = dic.get('support',{'p': 10, 
                                'r': 0.4,
                                'data':None,
                                'positive_imag':False})
  sup_data = sup_info.get('data',None) 
  if sup_data is None:
    sup_data = ctypes.POINTER(ctypes.c_ubyte)()
  else:
    if isinstance(sup_data, np.ndarray):
      if sup_data.shape != (N,N,N):
        raise ValueError("sup_data must have the same shape as the object")
      if sup_data.dtype != np.uint8:
        sup_data = sup_data.astype(np.uint8)
    sup_data = getPointer(sup_data.flatten(), dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
  sup_positive_imag = sup_info.get('positive_imag',False)
    
  # handle shrink wrapping customizations
  # later, this will have to be moved outside sup_info dictionary.
  #todo: create a swinfo dictionary 
  sw_info = dic.get('sw',{'threshold': 10,
                          'iter_filter': 'amplitude',
                          'mask_multiply':'full',
                          })
  sw_iter_filter_options = {'full':0,
                            'amplitude':1,
                            'real':2}
  sw_mask_multiply_options = {'full':0,
                              'real':1,
                              'legacy':2}
  sw_threshold = sw_info.get('threhsold',10)
  sw_iter_filter = sw_iter_filter_options.get(sw_info['iter_filter'],1)
  sw_mask_multiply = sw_mask_multiply_options.get(sw_info['mask_multiply'],0)
  sw_fftshift_gaussian = sw_info.get('fftshift_gaussian',False)
  
  # handle extra parameters 
  sthreads      = dic['sthreads']
  # finsup_path   = dic['finsup_path']
  b_ngpu        = ctypes.c_int(ngpus)
  b_gpus        = np.array(gpus, dtype=np.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  b_data        = getPointer(data.flatten(), dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
      
  # handle err parameters 
  errType = dic.get("errType",None)
  errSubiter = dic.get("errSubiter", 1)        
  errType_options = {None:            0,  # NO_ERR
                     "iter_diff":     1}  # ITER_DIFF
  errType = errType_options.get(errType, None)
  
  # handle memory parameters 
  map_d_signal = dic.get("map_d_signal", False)
  map_d_support = dic.get("map_d_support", False)
  swap_d_x = dic.get("swap_d_x", False)
  
  ## common parameters
  PARAMS = make_SSC_PWCDI_PARAMS((beta, 
                                  timing, 
                                  N, 
                                  sthreads, 
                                  sup_info['p'], 
                                  sup_info['r'],
                                  sup_data,
                                  sup_positive_imag,
                                  sw_threshold, 
                                  sw_iter_filter,
                                  sw_mask_multiply,
                                  sw_fftshift_gaussian,
                                  sigma, 
                                  sigma_mult, 
                                  beta_update,
                                  betaResetSubiter,
                                  eps_zeroamp,
                                  errType,
                                  errSubiter,
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
      shrinkWrap = dic['method'][k]['shrinkWrapSubiter']
      initialShrinkWrapSubiter = dic["method"][k].get("initialShrinkWrapSubiter",0)
      extraConstraint = dic['method'][k].get('extraConstraint',None)
      extraConstraintSubiter = dic['method'][k].get('extraConstraintSubiter',0)
      initialExtraConstraintSubiter = dic['method'][k].get('initialExtraConstraintSubiter',0)
          
            
      # decode extra constraint model
      extraConstraint_options = {None:               0,    # NO_EXTRA_CONSTRAINT
                                 "left_semiplane":   1,    # LEFT_SEMIPLANE
                                 "right_semiplane":  2,    # RIGHT_SEMIPLANE
                                 "top_semiplane":    3,    # TOP_SEMIPLANE 
                                 "bottom_semiplane": 4,    # BOTTOM_SEMIPLANE
                                 "first_quadrant":   5,    # FIRST_QUADRANT
                                 "second_quadrant":  6,    # SECOND_QUADRANT
                                 "third_quadrant":   7,    # THIRD_QUADRANT
                                 "fourth_quadrant":  8}    # FOURTH_QUADRANT 
      extraConstraint = extraConstraint_options.get(extraConstraint, None)
      
      # append to the sequence
      SEQ.append(make_SSC_PWCDI_METHOD((name, 
                                        iteration, 
                                        shrinkWrap, 
                                        initialShrinkWrapSubiter,
                                        extraConstraint, 
                                        extraConstraintSubiter,
                                        initialExtraConstraintSubiter))) 
  
  ALGO = (SSC_PWCDI_METHOD*num_methods)(*SEQ)
  b_nalgo = ctypes.c_int(num_methods)
     
  # call the main pwcdi function   
  libssccdi.pwcdi(obj_output,       
                  finsup_output,   
                  b_data,
                  obj_input, 
                  b_gpus,
                  b_ngpu,
                  b_nalgo,
                  PARAMS,
                  ALGO)
    
       
  return obj_output, finsup_output
 