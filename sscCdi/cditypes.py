#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import os
import sys

nthreads = multiprocessing.cpu_count()

# Load required libraies:

libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
libcufft  = ctypes.CDLL( ctypes.util.find_library( "cufft" ), mode=ctypes.RTLD_GLOBAL )
libcublas  = ctypes.CDLL( ctypes.util.find_library( "cublas" ), mode=ctypes.RTLD_GLOBAL )

_lib = "lib/libssccdi"

if sys.version_info[0] >= 3:
    import sysconfig
    ext = sysconfig.get_config_var('SO')
else:
    ext = '.so'

_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + _lib + ext
libradon  = ctypes.CDLL(_path)


if __name__ == "__main__":
   pass

