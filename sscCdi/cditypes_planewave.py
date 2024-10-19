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
  _fields_ = [("timing", ctypes.c_int ),
              ("N", ctypes.c_int ),
              ("sthreads", ctypes.c_int ),
              ("pnorm", ctypes.c_int ),
              ("radius", ctypes.c_float ),
              ("sup_data", ctypes.POINTER(ctypes.c_ubyte)),
              ("eps_zeroamp", ctypes.c_float),
              ("err_type", ctypes.c_int),
              ("err_subiter", ctypes.c_int),
              ("map_d_signal", ctypes.c_bool),
              ("map_d_support", ctypes.c_bool),
              ("swap_d_x", ctypes.c_bool)]

class SSC_PWCDI_METHOD(ctypes.Structure):
  """A ssc_pwcdi_method C structure:"""
  _fields_ = [("name", ctypes.c_char_p),
              ("iteration", ctypes.c_int),
              ("shrinkwrap_subiter", ctypes.c_int),
              ("initial_shrinkwrap_subiter", ctypes.c_int),
              ("extra_constraint", ctypes.c_int),
              ("extra_constraint_subiter", ctypes.c_int),
              ("initial_extra_constraint_subiter", ctypes.c_int),
              ("shrinkwrap_threshold", ctypes.c_float),
              ("shrinkwrap_iter_filter", ctypes.c_int),
              ("shrinkwrap_mask_multiply", ctypes.c_int),
              ("shrinkwrap_fftshift_gaussian", ctypes.c_bool),
              ("sigma", ctypes.c_float),
              ("sigma_mult", ctypes.c_float),
              ("beta", ctypes.c_float),
              ("beta_update", ctypes.c_float),
              ("beta_beta_reset_subiter", ctypes.c_int)]

# Define the make_SSC_PWCDI_PARAMS function
def make_SSC_PWCDI_PARAMS(tupla):
  """Make a ssc_pwcdi_params from a Python tuple"""
  return SSC_PWCDI_PARAMS(tupla[0],                      # timing
                          tupla[1],                      # N
                          tupla[2],                      # sthreads
                          tupla[3],                      # pnorm
                          tupla[4],                      # radius
                          tupla[5],                      # sup_data
                          tupla[6],                      # eps_zeroamp
                          tupla[7],                      # err_type
                          tupla[8],                      # err_subiter
                          tupla[9],                      # map_d_signal
                          tupla[10],                     # map_d_support
                          tupla[11])                     # swap_d_x


def make_SSC_PWCDI_METHOD(tupla):
  """Make a ssc_pwcdi_method from a Python tuple """

  return SSC_PWCDI_METHOD(ctypes.c_char_p(tupla[0].encode('utf-8')),   # name 
                          ctypes.c_int(tupla[1]),                      # niter
                          ctypes.c_int(tupla[2]),                      # shrinkwrap_subiter
                          ctypes.c_int(tupla[3]),                      # initial_shrinkwrap_subiter
                          ctypes.c_int(tupla[4]),                      # extra_constraint
                          ctypes.c_int(tupla[5]),                      # extra_constraint_subiter
                          ctypes.c_int(tupla[6]),                      # initial_extra_constraint_subiter 
                          ctypes.c_float(tupla[7]),                    # shrinkwrap_threshold                 # here 
                          ctypes.c_int(tupla[8]),                      # shrinkwrap_iter_filter
                          ctypes.c_int(tupla[9]),                      # shrinkwrap_mask_multiply
                          ctypes.c_bool(tupla[10]),                    # shrinkwrap_fftshift_gaussian 
                          ctypes.c_float(tupla[11]),                   # sigma 
                          ctypes.c_float(tupla[12]),                   # sigma_mult
                          ctypes.c_float(tupla[13]),                   # beta  
                          ctypes.c_float(tupla[14]),                   # beta_update
                          ctypes.c_int(tupla[15]))                     # beta_reset_subiter

 

try:
  libssccdi.pwcdi.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_output
                              np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),      # finsup_output
                              ctypes.POINTER(ctypes.c_float),                                            # data
                              ctypes.c_void_p, # np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_input, 
                              ctypes.POINTER(ctypes.c_int),                                              # ngpus
                              ctypes.c_int,
                              ctypes.c_int,
                              SSC_PWCDI_PARAMS,
                              ctypes.POINTER(SSC_PWCDI_METHOD)]  
  libssccdi.pwcdi.restype  = None
    
except:
  print ('ssc-cdi: No plane-wave CDI 3D Functions compiled!!')
  pass

