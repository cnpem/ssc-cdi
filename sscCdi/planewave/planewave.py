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
    beta = dic['beta']
    sigma = dic.get('sigma',0.1)
    sigma_mult = dic.get('sigma_mult',0.99)
    beta_update = dic.get('beta_update',7.0)
    betaResetSubiter = dic.get("betaResetSubiter", -1) # by default, it doesnt reset 
    eps_zeroamp = dic.get('eps_zeroamp',0.001)
    outpath  = dic['output']
    
    
    # unpack initial object data (amplitude and phase)
    amplitude_obj_data = dic.get("amplitude_obj_data", None) 
    phase_obj_data = dic.get("phase_obj_data", None)
    if amplitude_obj_data is None:
        amplitude_obj_data = ctypes.POINTER(ctypes.c_float)()
    else:
        amplitude_obj_data = getPointer(amplitude_obj_data.flatten(),
                                        dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if phase_obj_data is None:
        phase_obj_data = ctypes.POINTER(ctypes.c_float)()
    else:
        phase_obj_data = getPointer(phase_obj_data.flatten(), 
                                    dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
 
    # unpack the support parameters 
    supinfo = dic.get('support',{'p': 10, 
                                 'r': 0.4,
                                 'data':None,
                                 'positive_imag':False})
    swinfo = dic.get('sw',{'threshold': 10,
                           'iter_filter': 'amplitude',
                           'mask_multiply':'full',
                           })
    
    ## unpack support initial data and other parameters 
    sup_data = supinfo.get('data',None) 
    if sup_data is None:
        sup_data = ctypes.POINTER(ctypes.c_float)()
    else:
        sup_data = getPointer(sup_data.flatten(), dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sup_positive_imag = supinfo.get('positive_imag',False)
    
    # unpack shrink wrapping customizations
    # later, this will have to be moved outside supinfo dictionary.
    #todo: create a swinfo dictionary 
    sw_iter_filter_options = {'full':0,
                             'amplitude':1,
                             'real':2}
    sw_mask_multiply_options = {'full':0,
                                'real':1,
                                'legacy':2}
    
    sw_threshold = swinfo.get('threhsold',10)
    sw_iter_filter = sw_iter_filter_options.get(swinfo['iter_filter'],1)
    sw_mask_multiply = sw_mask_multiply_options.get(swinfo['mask_multiply'],0)
    sw_fftshift_gaussian = swinfo.get('fftshift_gaussian',False)
    
    sthreads      = dic['sthreads']
    finsup_path   = dic['finsup_path']
 
    b_ngpu        = ctypes.c_int(ngpus)
    b_gpus        = np.array(gpus, dtype=np.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_outpath     = outpath.encode('utf-8')
 
    
    b_finsup_path = finsup_path.encode('utf-8')
    b_data        = getPointer(data.flatten(), dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
       
        
    errType = dic.get("errType",None)
    errSubiter = dic.get("errSubiter", 1)        
            
            
    # decode err type and subiter
    errType_options = {None:            0,  # NO_ERR
                       "iter_diff":     1}  # ITER_DIFF
    errType = errType_options.get(errType, None)
    
    
    # memory parameters 
    map_d_signal = dic.get("map_d_signal", False)
    map_d_support = dic.get("map_d_support", False)
    swap_d_x = dic.get("swap_d_x", False)
    
    
    ## common parameters
    PARAMS = make_SSC_PWCDI_PARAMS((amplitude_obj_data,
                                    phase_obj_data,
                                    beta, 
                                    timing, 
                                    N, 
                                    sthreads, 
                                    supinfo['p'], 
                                    supinfo['r'],
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
    nmet = len(dic['method'])

    SEQ = []
    for k in range(nmet):      
        # dictionary parsing 
        if isinstance(dic['method'][k],dict):
            # base method choices
            name = dic['method'][k]['name']
            iteration = dic['method'][k]['niter']
            shrinkWrap = dic['method'][k]['shrinkWrapSubiter']
            initialShrinkWrapSubiter = dic["method"][k].get("initialShrinkWrapSubiter",0)
            extraConstraint = dic['method'][k].get('extraConstraint',None)
            extraConstraintSubiter = dic['method'][k].get('extraConstraintSubiter',0)
            initialExtraConstraintSubiter = dic['method'][k].get('initialExtraConstraintSubiter',0)
            
            
        # decode extra constraint model
        extraConstraint_options = {None:              0,    # NO_EXTRA_CONSTRAINT
                                  "left_semiplane":   1,    # LEFT_SEMIPLANE
                                  "right_semiplane":  2,    # RIGHT_SEMIPLANE
                                  "top_semiplane":    3,    # TOP_SEMIPLANE 
                                  "bottom_semiplane": 4,    # BOTTOM_SEMIPLANE
                                  "first_quadrant":   5,    # FIRST_QUADRANT
                                  "second_quadrant":  6,    # SECOND_QUADRANT
                                  "third_quadrant":   7,    # THIRD_QUADRANT
                                  "fourth_quadrant":  8}    # FOURTH_QUADRANT 
        extraConstraint = extraConstraint_options.get(extraConstraint,None)
        
        SEQ.append(make_SSC_PWCDI_METHOD((name, 
                                          iteration, 
                                          shrinkWrap, 
                                          initialShrinkWrapSubiter,
                                          extraConstraint, 
                                          extraConstraintSubiter,
                                          initialExtraConstraintSubiter))) 
    
    ALGO    = (SSC_PWCDI_METHOD * nmet)(* SEQ)
    b_nalgo = ctypes.c_int(nmet)
     
# populate algorithms list  
    algorithms = []
    for k in range(nmet):
        if isinstance(dic['method'][k],list):
            algorithms.append(dic['method'][k][0])
        elif isinstance(dic['method'][k],dict):
            algorithms.append(dic['method'][k]["name"])

    
    print("ssc-cdi: init libsscdi\n")
    
    libssccdi.pwcdi(b_outpath,
                    b_finsup_path, 
                    b_data,
                    b_gpus,
                    b_ngpu,
                    b_nalgo,
                    PARAMS,
                    ALGO)
    
       
def methods(dic):
    #hint: https://stackoverflow.com/questions/36906222/how-do-i-construct-an-array-of-ctype-structures
    nmet = len(dic['method'])
                              
    SEQ = []
    for k in range(nmet): 
        # dictionary parsing 
        if isinstance(dic['method'][k],dict): 
            # base method choices
            name = dic['method'][k]['name']
            iteration = dic['method'][k]['niter']
            shrinkWrap = dic['method'][k]['shrinkWrapSubiter']
            initialShrinkWrapSubiter = dic["method"][k].get("initialShrinkWrapSubiter",0)
            extraConstraint = dic['method'][k].get('extraConstraint',None)
            initialExtraConstraintSubiter = dic['method'][k].get('initialExtraConstraintSubiter',0)
            extraConstraintSubiter = dic['method'][k].get('extraConstraintSubiter',0)
            
            
        # decode extra constraint model
        extraConstraint_options = {None:              0,    # NO_EXTRA_CONSTRAINT
                                  "left_semiplane":   1,    # LEFT_SEMIPLANE
                                  "right_semiplane":  2,    # RIGHT_SEMIPLANE
                                  "top_semiplane":    3,    # TOP_SEMIPLANE 
                                  "bottom_semiplane": 4,    # BOTTOM_SEMIPLANE
                                  "first_quadrant":   5,    # FIRST_QUADRANT
                                  "second_quadrant":  6,    # SECOND_QUADRANT
                                  "thid_quadrant":    7,    # THIRD_QUADRANT
                                  "fourth_quadrant":  8}    # FOURTH_QUADRANT 
        extraConstraint = extraConstraint_options.get(extraConstraint)
        
        
        
        SEQ.append(make_SSC_PWCDI_METHOD((name, 
                                          iteration, 
                                          shrinkWrap, 
                                          extraConstraint, 
                                          extraConstraintSubiter,
                                          initialExtraConstraintSubiter))) 
            
        
    
    PLAN = (SSC_PWCDI_METHOD*nmet)(* SEQ)
    
    libssccdi.methods(PLAN, nmet)

    
# Define the function signatures
libssccdi.test_linear_conv.restype = None
libssccdi.test_linear_conv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex64, flags='C_CONTIGUOUS'),  # output_data
    np.ctypeslib.ndpointer(dtype=np.complex64, flags='C_CONTIGUOUS'),  # input_data
    np.ctypeslib.ndpointer(dtype=np.complex64, flags='C_CONTIGUOUS'),  # kernel
    ctypes.c_int,  # data_width
    ctypes.c_int,  # data_height
    ctypes.c_int,  # data_depth
    ctypes.c_int,  # kernel_width
    ctypes.c_int,  # kernel_height
    ctypes.c_int   # kernel_depth
]


def linconv(input_data, kernel, data_shape, kernel_shape):
    data_width, data_height, data_depth = data_shape
    kernel_width, kernel_height, kernel_depth = kernel_shape

    # Ensure input_data and kernel are contiguous arrays
    input_data = np.ascontiguousarray(input_data, dtype=np.complex64)
    kernel = np.ascontiguousarray(kernel, dtype=np.complex64)

    # Allocate output_data on the host
    # output_data = np.zeros(data_width * data_height * data_depth, dtype=np.complex64)

    # Call the CUDA function via ctypes
    libssccdi.test_linear_conv(input_data,
                                  input_data,
                                  kernel,
                                  data_width,
                                  data_height,
                                  data_depth,
                                  kernel_width,
                                  kernel_height,
                                  kernel_depth)

    return input_data.reshape(data_shape)


 
# Define the data types for the function arguments
libssccdi.separableConvolution3D.restype = None
libssccdi.separableConvolution3D.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # input   C WAS NOT HERE
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # output   C WAS NOT HERE
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags='C_CONTIGUOUS'),  # kernel   C WAS NOT HERE
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int,  # depth
    ctypes.c_int,  # kernelLength
]

def linconv_optimized_sep(input_data, kernel, data_shape, kernel_shape):


    data_width, data_height, data_depth = data_shape
    # kernel_width, kernel_height, kernel_depth = kernel_shape


    # Ensure input_data and kernel are contiguous arrays
    input_data = np.ascontiguousarray(input_data, dtype=np.complex64)
    kernel = np.ascontiguousarray(kernel, dtype=np.complex64)

    # Allocate output_data on the host
    output_data = np.zeros((data_width, data_height, data_depth), dtype=np.complex64)

    # Call the CUDA function via ctypes
    libssccdi.separableConvolution3D(output_data,
                                     input_data,
                                     kernel,
                                     data_width,
                                     data_height,
                                     data_depth,
                                     kernel_shape)

    return output_data





# Update the argtypes to reflect the removal of the gpus parameter
libssccdi.m_test_linear_conv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # output_data
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # input_data
    np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags='C_CONTIGUOUS'),  # kernel
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int,  # depth
    ctypes.c_int,  # kernelLength
    ctypes.c_int,  # nGPUs
    ctypes.POINTER(ctypes.c_int)  # gpuIndexes
    # np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='CONTIGUOUS')  # gpuIndexes
]
 
def m_test_linear_conv_wrapper(input_data, kernel, data_shape, kernel_length, gpuIndexes):
    data_width, data_height, data_depth = data_shape
      # Assuming kernel_shape is a tuple with at least one element

    # Ensure input_data and kernel are contiguous arrays
    input_data = np.ascontiguousarray(input_data, dtype=np.complex64)
    kernel = np.ascontiguousarray(kernel, dtype=np.complex64)

    # Allocate output_data on the host
    output_data = np.zeros((data_width, data_height, data_depth), dtype=np.complex64)

    # Convert gpu_indexes Python list to a ctypes array of integers
    # s_gpuIndexes = (ctypes.c_int * len(gpuIndexes))(*gpuIndexes)
    b_gpuIndexes = np.array(gpuIndexes, dtype=np.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Call the CUDA function via ctypes
    libssccdi.m_test_linear_conv(output_data,
                                 input_data,
                                 kernel,
                                 data_width,
                                 data_height,
                                 data_depth,
                                 kernel_length,
                                 len(gpuIndexes),
                                 b_gpuIndexes)

    return output_data