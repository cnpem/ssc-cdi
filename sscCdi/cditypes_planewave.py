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
  _fields_ = [("beta", ctypes.c_float ),
              ("timing", ctypes.c_int ),
              ("N", ctypes.c_int ),
              ("sthreads", ctypes.c_int ),
              ("pnorm", ctypes.c_int ),
              ("radius", ctypes.c_float ),
              ("sup_data", ctypes.POINTER(ctypes.c_ubyte)),
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

# Define the make_SSC_PWCDI_PARAMS function
def make_SSC_PWCDI_PARAMS(tupla):
  """Make a ssc_pwcdi_params from a Python tuple"""
  return SSC_PWCDI_PARAMS(tupla[0],                      # beta
                          tupla[1],                      # timing
                          tupla[2],                      # N
                          tupla[3],                      # sthreads
                          tupla[4],                      # pnorm
                          tupla[5],                      # radius
                          tupla[6],                      # sup_data
                          tupla[7],                      # sup_positive_imag
                          tupla[8],                      # sw_threshold
                          tupla[9],                      # sw_iter_filter
                          tupla[10],                     # sw_mask_multiply
                          tupla[11],                     # sw_fftshift_gaussian
                          tupla[12],                     # sigma
                          tupla[13],                     # sigma_mult
                          tupla[14],                     # beta_update
                          tupla[15],                     # betaResetSubiter
                          tupla[16],                     # eps_zeroamp
                          tupla[17],                     # errType
                          tupla[18],                     # errSubiter
                          tupla[19],                     # map_d_signal
                          tupla[20],                     # map_d_support
                          tupla[21])                     # swap_d_x


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
  libssccdi.pwcdi.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_output
                              np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),      # finsup_output
                              ctypes.POINTER(ctypes.c_float),                                            # data
                              np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_input
                              ctypes.POINTER(ctypes.c_int),                                              # ngpus
                              ctypes.c_int,
                              ctypes.c_int,
                              SSC_PWCDI_PARAMS,
                              ctypes.POINTER(SSC_PWCDI_METHOD)]  
  libssccdi.pwcdi.restype  = None
    
except:
  print ('ssc-cdi: No plane-wave CDI 3D Functions compiled!!')
  pass

