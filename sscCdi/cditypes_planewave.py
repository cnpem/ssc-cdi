# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


# We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import os
import sys
# from typing import Optional, Tuple
import numpy as np
from time import time
import glob

nthreads = multiprocessing.cpu_count()

def load_library(lib, ext):
#     try:
    _path = glob.glob(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext)
    #print("ssc-cdi: Trying to load library:", _path)
    if isinstance(_path, list):
        _path = _path[0] 
    library = ctypes.CDLL(_path) 
#     except:
#         print("ssc-cdi: library ",lib," was not loaded.\n")
#         library = None

    return library


libstdcpp = ctypes.CDLL(ctypes.util.find_library("stdc++"),
                        mode=ctypes.RTLD_GLOBAL)
libcufft = ctypes.CDLL(ctypes.util.find_library("cufft"),
                       mode=ctypes.RTLD_GLOBAL)
libcublas = ctypes.CDLL(ctypes.util.find_library("cublas"),
                        mode=ctypes.RTLD_GLOBAL)


_lib = "lib/libssccdi"
ext = '.so'
libssccdi  = load_library(_lib, ext) 

def getPointer(darray,dtype=np.float32):
    if darray.dtype != dtype:
        return np.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return np.ascontiguousarray(darray)
    else:
        return darray

    
class SSC_PWCDI_PARAMS(ctypes.Structure):
    """A ssc_pwcdi_params C structure:"""
    _fields_ = [("amplitude_obj_data",ctypes.POINTER(ctypes.c_float)),
                ("phase_obj_data",ctypes.POINTER(ctypes.c_float)),
                # ("obj_output", ctypes.POINTER(ctypes.c_complex64)),  # New field for complex output
                # ("finsup_output", ctypes.POINTER(ctypes.c_int16)), # New field for short output
                ("beta", ctypes.c_float ),
                ("timing", ctypes.c_int ),
                ("N", ctypes.c_int ),
                ("sthreads", ctypes.c_int ),
                ("pnorm", ctypes.c_int ),
                ("radius", ctypes.c_float ),
                ("sup_data", ctypes.POINTER(ctypes.c_float)),
                ("sup_positive_imag", ctypes.c_bool),
                ("sw_threshold", ctypes.c_float ),
                ("sw_iter_filter", ctypes.c_int),
                ("sw_mask_multiply", ctypes.c_int),
                ("sw_fftshift_gaussian", ctypes.c_bool),
                ("sigma", ctypes.c_float),
                ("sigma_mult", ctypes.c_float),
                ("beta_update", ctypes.c_float),
                ("betaResetSubiter", ctypes.c_int),
                ("eps_zeroamp", ctypes.c_float),
                ("errType", ctypes.c_int),
                ("errSubiter", ctypes.c_int),
                ("map_d_signal", ctypes.c_bool),
                ("map_d_support", ctypes.c_bool),
                ("swap_d_x", ctypes.c_bool)]

class SSC_PWCDI_METHOD(ctypes.Structure):
    """A ssc_pwcdi_method C structure:"""
    _fields_ = [("name", ctypes.c_char_p),
                ("iteration", ctypes.c_int),
                ("shrinkWrap", ctypes.c_int),
                ("initialShrinkWrapSubiter", ctypes.c_int),
                ("extraConstraint", ctypes.c_int),
                ("extraConstraintSubiter", ctypes.c_int),
                ("initialExtraConstraintSubiter", ctypes.c_int)]

def make_SSC_PWCDI_PARAMS( tupla ):
    """Make a ssc_pwcdi_params from a Python tuple """
      
    return SSC_PWCDI_PARAMS(tupla[0],                      # amplitude_obj_data
                            tupla[1],                      # phase_obj_data
                            # tupla[2],                      # b_obj_output
                            # tupla[3],                      # b_finsup_output                   
                            ctypes.c_float(tupla[2]),
                            ctypes.c_int(tupla[3]),
                            ctypes.c_int(tupla[4]),
                            ctypes.c_int(tupla[5]),
                            ctypes.c_int(tupla[6]),
                            ctypes.c_float(tupla[7]),
                                           tupla[8],       # sup_data
                            ctypes.c_bool(tupla[9]),       # sup_positive_imag
                            ctypes.c_float(tupla[10]),
                            ctypes.c_int(tupla[11]),
                            ctypes.c_int(tupla[12]),
                            ctypes.c_bool(tupla[13]),       # sw_fftshift_gaussian
                            ctypes.c_float(tupla[14]),
                            ctypes.c_float(tupla[15]),
                            ctypes.c_float(tupla[16]),
                            ctypes.c_int(tupla[17]),       # betaResetSubiter
                            ctypes.c_float(tupla[18]),     # eps_zeroamp
                            ctypes.c_int(tupla[19]),       # errType
                            ctypes.c_int(tupla[20]),       # errSubiter
                            ctypes.c_bool(tupla[21]),      # map_d_signal   
                            ctypes.c_bool(tupla[22]),      # map d_support
                            ctypes.c_bool(tupla[23])       # swap_d_signal
                           )

def make_SSC_PWCDI_METHOD(tupla):
    """Make a ssc_pwcdi_method from a Python tuple """

    return SSC_PWCDI_METHOD(ctypes.c_char_p(tupla[0].encode('utf-8')),   # name 
                            ctypes.c_int(tupla[1]),                      # niter
                            ctypes.c_int(tupla[2]),                      # shrinkWrapSubiter
                            ctypes.c_int(tupla[3]),                      # initialShrinkWrapSubiter
                            ctypes.c_int(tupla[4]),                      # extraConstraint
                            ctypes.c_int(tupla[5]),                      # extraConstraintSubiter
                            ctypes.c_int(tupla[6]))                      # initialExtraConstraintSubiter
 

try:
    libssccdi.pwcdi.argtypes =  [ctypes.c_char_p,
                                 ctypes.c_char_p,  
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_int),
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 SSC_PWCDI_PARAMS,
                                 ctypes.POINTER(SSC_PWCDI_METHOD)]  
    libssccdi.pwcdi.restype  = None

    
except:
    print ('ssc-cdi: No plane-wave CDI 3D Functions compiled!!')
    pass

