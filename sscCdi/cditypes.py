# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
# will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################
# encoding: utf8

# Ctypes =========
import ctypes
from ctypes import *
import ctypes.util

# General =========
import multiprocessing
import os
import sys
from typing import Optional, Tuple
import numpy as np
from time import time
import glob
# ======================

from sscCdi import __version__

'''----------------------------------------------'''
import logging

console_log_level = 10

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
console_handler.setLevel(console_log_level)

DEBUG = 10
'''----------------------------------------------'''

nthreads = multiprocessing.cpu_count()

############# Load required libraries ##############

libstdcpp = ctypes.CDLL(ctypes.util.find_library("stdc++"), mode=ctypes.RTLD_GLOBAL)
libcufft  = ctypes.CDLL(ctypes.util.find_library("cufft") , mode=ctypes.RTLD_GLOBAL)
libcublas = ctypes.CDLL(ctypes.util.find_library("cublas"),mode=ctypes.RTLD_GLOBAL)

############# Load library C/C++/CUDA ##############

_lib = "lib/libssccdi"
ext = '.so'

def load_library(lib, ext):
    _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
    #print("ssc-cdi: Trying to load library:", _path)
    try:
        lib = ctypes.CDLL(_path)
        return lib
    except Exception as e:
        print(f"ssc-cdi: Failed to load library {_path}: {e}")
    return None


libcdi = load_library(_lib, ext)

#########################

#########################
#|       ssc-cdi       |#
#| Function prototypes |#
#########################

############# Ptycho-Engines ##############
try:
    libcdi.ap_call.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, c_int,
        c_float, c_float, c_float,
        c_float, c_float, c_float, c_float
    ]
    libcdi.ap_call.restype = None
    libcdi.raarcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, c_int,
        c_float, c_float, c_float, c_float,
        c_float, c_float, c_float, c_float
    ]
    libcdi.raarcall.restype = None
    libcdi.piecall.argtypes = [
        ctypes.c_void_p, c_int, c_int, ctypes.c_void_p, c_int, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int,
        c_float, c_float, c_float, c_float,
        c_float, c_float, c_float
    ]

    libcdi.piecall.restype = None

except Exception as e:
    print('>>>', e)

############# Plane-Wave ##############

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
                            ctypes.c_float(tupla[7]),                    # shrinkwrap_threshold                  
                            ctypes.c_int(tupla[8]),                      # shrinkwrap_iter_filter
                            ctypes.c_int(tupla[9]),                      # shrinkwrap_mask_multiply
                            ctypes.c_bool(tupla[10]),                    # shrinkwrap_fftshift_gaussian 
                            ctypes.c_float(tupla[11]),                   # sigma 
                            ctypes.c_float(tupla[12]),                   # sigma_mult
                            ctypes.c_float(tupla[13]),                   # beta  
                            ctypes.c_float(tupla[14]),                   # beta_update
                            ctypes.c_int(tupla[15]))                     # beta_reset_subiter

 

try:
    libcdi.pwcdi.argtypes = [
                            np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_output
                            np.ctypeslib.ndpointer(dtype=np.uint8    , ndim=3, flags='C_CONTIGUOUS'),  # finsup_output
                            ctypes.POINTER(ctypes.c_float),                                            # data
                            ctypes.c_void_p, # np.ctypeslib.ndpointer(dtype=np.complex64, ndim=3, flags='C_CONTIGUOUS'),  # obj_input, 
                            ctypes.POINTER(ctypes.c_int),                                              # ngpus
                            ctypes.c_int,
                            ctypes.c_int,
                            SSC_PWCDI_PARAMS,
                            ctypes.POINTER(SSC_PWCDI_METHOD)
                            ]  
    libcdi.pwcdi.restype  = None
    
except Exception as e:
    print ('Cannot find C/CUDA library: -.SSC-CDI_PLANE_WAVE-', e)

########################
#|      ssc-cdi       |#
#|     Functions      |#
########################

def CNICE(darray,dtype=np.float32):
    if darray.dtype != dtype:
        return np.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return np.ascontiguousarray(darray)
    else:
        return darray
        
def ctypes_array(c: np.ndarray) -> Tuple[np.ndarray, c_void_p, list[c_int]]:
    contiguous_array = np.ascontiguousarray(c)
    cptr = contiguous_array.ctypes.data_as(c_void_p)
    cshape = [c_int(d) for d in c.shape]
    return contiguous_array, cptr, cshape


def ctypes_opt_array(c: Optional[np.ndarray]) -> Tuple[np.ndarray, c_void_p, list[c_int]]:
    if c is None:
        return np.empty(0, dtype='float32'), c_void_p(0), [c_int(0)]
    return ctypes_array(c)

def sanitize_rois(rois, obj, difpads, probe) -> np.ndarray:

    rois = rois.astype('float32')

    if rois.shape[0] != difpads.shape[0]:
        raise ArgumentError("Error:", difpads.shape[0], "difpads  v ", rois.shape[0],
              " rois mismatch")
    if rois.shape[-1] != 2:
        raise ArgumentError("Incorrect positions specification: shape=", rois.shape,
              ' rois has 2 attribs: x,y')

    if probe.shape[-1] % difpads.shape[-1] != 0 or probe.shape[
            -2] % difpads.shape[-2] != 0:
        raise ArgumentError("Error:", probe.shape, "probe  v ", difpads.shape,
              " difpads shape mismatch")

    for roi in rois:
        if probe.shape[-1] + roi[0] > obj.shape[-1] or probe.shape[-2] + roi[1] > obj.shape[-2]:
            raise ArgumentError("Error: Roi", roi, "is outside object bounds:", roi, "+",
                  probe.shape, "=", obj.shape)
        elif roi[0] < 0 or roi[1] < 0:
            raise ArgumentError("Error: Roi", roi, "has negative indexing.")
    return rois


def PIE(obj: np.ndarray,
        probe: np.ndarray,
        difpads: np.ndarray,
        rois: np.ndarray,
        iterations: int,
        probesupp: Optional[np.ndarray] = None,
        poscorr_iter: int = 0,
        step_obj: float = 0.5,
        step_probe: float = 0.5,
        reg_obj: float = 1e-3,
        reg_probe: float = 1e-3,
        wavelength_m: float = 0.0,
        pixelsize_m: float = 0.0,
        distance_m: float = 0.0,
        params: dict = {}):
    """ Ptychography PIE algorithm.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            beta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            wavelength_m (float, optional): The wavelength of the light used in the propagation, in meters. Defaults to 0.
            pixelsize_m (float, optional): The detector pixel size in meters. Defaults to 0.
            distance_m (float, optional): The distance to the detector in meters. Defaults to 0.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``param`` input:

                * ``param['device']`` (int list): The list of GPUs. Defaults to [0].

                * ``param['probecycles']`` (int list): Probe cycles. Defaults to 3.


        Dictionary ``mydict`` output:

                * ``mydict['obj']`` (complex ndarray): The complex 2D object sample retrieved (obj = Attenuation *  e^(-phase)).

                * ``mydict['probe']`` (complex ndarray): The complex 2D probe retrieved.

                * ``mydict['difpads']`` (ndarray): The 3D stack of diffraction data measured.

                * ``mydict['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1].

                * ``mydict['W']`` (ndarray): W-function for CARNAUBA.

                * ``mydict['error']`` (ndarray): Error of iterations.

                * ``mydict['objsupp']`` (int ndarray): Object support.

                * ``mydict['probesupp']`` (int ndarray): Probe support.

                * ``mydict['bkg']`` (ndarray): The 2D background retrieved.

        """

    obj, objptr, (osizey, osizex) = ctypes_array(obj)
    probe, probeptr, (psizez, _, psizex) = ctypes_array(probe)

    difpads, difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois, roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices, devicesptr, (ndevices, ) = ctypes_array(devices)

    # allocate memory for errors 
    error_rfactor = np.zeros(iterations, dtype=np.float32)
    error_rfactor, error_rfactorptr, _ = ctypes_array(error_rfactor)

    error_llk = np.zeros(iterations, dtype=np.float32)
    error_llk, error_llkptr, _ = ctypes_array(error_llk)

    error_mse = np.zeros(iterations, dtype=np.float32)
    error_mse, error_mseptr, _ = ctypes_array(error_mse)

    if probesupp is not None:
        probesupp = probesupp.astype('float32')
    probesupp, probesuppptr, _ = ctypes_opt_array(probesupp)

    time0 = time()

    libcdi.piecall(objptr, osizex, osizey, probeptr, psizex, psizez,
                   difpadsptr, dsizex, roisptr, numrois,
                   c_int(iterations), devicesptr, ndevices, error_rfactorptr, error_llkptr, error_mseptr, probesuppptr,
                   c_int(poscorr_iter),
                   c_float(step_obj), c_float(step_probe),
                   c_float(reg_obj), c_float(reg_probe),
                   c_float(wavelength_m), c_float(pixelsize_m), c_float(distance_m))

    print(f"\tDone in: {time()-time0:.2f} seconds")

    return obj, probe, error_rfactor, error_llk, error_mse, rois

def RAAR(obj: np.ndarray,
         probe: np.ndarray,
         difpads: np.ndarray,
         rois: np.ndarray,
         iterations: int,
         objbeta: float,
         probebeta: float,
         beta: float = 0.95,
         batch: int = 16,
         objsupp: Optional[np.ndarray] = None,
         probesupp: Optional[np.ndarray] = None,
         step_obj: float = 0.5,
         step_probe: float = 0.5,
         reg_obj: float = 1e-3,
         reg_probe: float = 1e-3,
         poscorr_iter: int = 0,
         wavelength_m: float = 0.0,
         pixelsize_m: float = 0.0,
         distance_m: float = 0.0,
         params: dict = {}):
    """ Ptychography RAAR algorithm.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            beta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            batch (int, optional): Size . Defaults to 16.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            wavelength_m (float, optional): The wavelength of the light used in the propagation, in meters. Defaults to 0.
            pixelsize_m (float, optional): The detector pixel size in meters. Defaults to 0.
            distance_m (float, optional): The distance to the detector in meters. Defaults to 0.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``param`` input:

                * ``param['device']`` (int list): The list of GPUs. Defaults to [0].


        Dictionary ``mydict`` output:

                * ``mydict['obj']`` (complex ndarray): The complex 2D object sample retrieved (obj = Attenuation *  e^(-phase)).

                * ``mydict['probe']`` (complex ndarray): The complex 2D probe retrieved.

                * ``mydict['difpads']`` (ndarray): The 3D stack of diffraction data measured.

                * ``mydict['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1].

                * ``mydict['W']`` (ndarray): W-function for CARNAUBA.

                * ``mydict['error']`` (ndarray): Error of iterations.

                * ``mydict['objsupp']`` (int ndarray): Object support.

                * ``mydict['probesupp']`` (int ndarray): Probe support.

                * ``mydict['bkg']`` (ndarray): The 2D background retrieved.

        """

    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe, probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads, difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois, roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    # allocate memory for errors 
    error_rfactor = np.zeros(iterations, dtype=np.float32)
    error_rfactor, error_rfactorptr, _ = ctypes_array(error_rfactor)

    error_llk = np.zeros(iterations, dtype=np.float32)
    error_llk, error_llkptr, _ = ctypes_array(error_llk)

    error_mse = np.zeros(iterations, dtype=np.float32)
    error_mse, error_mseptr, _ = ctypes_array(error_mse)

    nummodes = psizez

    assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport,) = ctypes_opt_array(objsupp)

    # if objsupp is not None:
    # assert (objsupp.size >= obj.size)
    # objsupp = np.ascontiguousarray(objsupp.astype(np.float32))
    # objsuppptr = objsupp.ctypes.data_as(c_void_p)
    # if len(objsupp.shape) >= 3:
    # numobjsupport = c_int(objsupp.shape[0])
    # else:
    # numobjsupport = c_int(1)
    # else:
    # objsuppptr = c_void_p(0)
    # numobjsupport = c_int(0)

    libcdi.raarcall(objptr, probeptr, difpadsptr, psizex, osizex,
                    osizey, dsizex, roisptr, numrois, c_int(batch),
                    c_int(iterations), ndevices, devicesptr, error_rfactorptr, error_llkptr, error_mseptr,
                    c_float(objbeta), c_float(probebeta), nummodes,
                    objsuppptr, probesuppptr, numobjsupport, c_int(poscorr_iter),
                    c_float(step_obj), c_float(step_probe),
                    c_float(reg_obj), c_float(reg_probe),
                    c_float(wavelength_m), c_float(pixelsize_m), c_float(distance_m),
                    c_float(beta))

    return obj, probe, error_rfactor, error_llk, error_mse, rois



def AP(obj: np.ndarray,
       probe: np.ndarray,
       difpads: np.ndarray,
       rois: np.ndarray,
       iterations: int,
       objbeta: float,
       probebeta: float,
       batch: int = 16,
       objsupp: Optional[np.ndarray] = None,
       probesupp: Optional[np.ndarray] = None,
       step_obj: float = 0.5,
       step_probe: float = 0.5,
       reg_obj: float = 1e-3,
       reg_probe: float = 1e-3,
       poscorr_iter: int = 0,
       wavelength_m: float = 0.0,
       pixelsize_m: float = 0.0,
       distance_m: float = 0.0,
       params: dict = {}):
    """ Ptychography Alternate Projections algorithm.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            objbeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            probebeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.
            batch (int, optional): Size . Defaults to 16.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            wavelength_m (float, optional): The wavelength of the light used in the propagation, in meters. Defaults to 0.
            pixelsize_m (float, optional): The detector pixel size in meters. Defaults to 0.
            distance_m (float, optional): The distance to the detector in meters. Defaults to 0.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``param`` input:

                *``param['device']`` (int list): The list of GPUs. Defaults to [0].

        Dictionary ``mydict`` output:

                *``mydict['obj']`` (complex ndarray): The complex 2D object sample retrieved (obj = Attenuation *  e^(-phase)).

                *``mydict['probe']`` (complex ndarray): The complex 2D probe retrieved.

                *``mydict['difpads']`` (ndarray): The 3D stack of diffraction data measured.

                *``mydict['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1].

                *``mydict['W']`` (ndarray): W-function for CARNAUBA.

                *``mydict['error']`` (ndarray): Error of iterations.

                *``mydict['objsupp']`` (int ndarray): Object support.

                *``mydict['probesupp']`` (int ndarray): Probe support.

                *``mydict['bkg']`` (ndarray): The 2D background retrieved.

        """
    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe,probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads,difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois,roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    # allocate memory for errors 
    error_rfactor = np.zeros(iterations, dtype=np.float32)
    error_rfactor, error_rfactorptr, _ = ctypes_array(error_rfactor)

    error_llk = np.zeros(iterations, dtype=np.float32)
    error_llk, error_llkptr, _ = ctypes_array(error_llk)

    error_mse = np.zeros(iterations, dtype=np.float32)
    error_mse, error_mseptr, _ = ctypes_array(error_mse)

    nummodes = psizez

    assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport,) = ctypes_opt_array(objsupp)

    libcdi.ap_call(objptr, probeptr, difpadsptr, psizex, osizex,
                  osizey, dsizex, roisptr, numrois, c_int(batch),
                  c_int(iterations), ndevices, devicesptr, error_rfactorptr, error_llkptr, error_mseptr,
                  c_float(objbeta), c_float(probebeta), nummodes, objsuppptr,
                  probesuppptr, numobjsupport,
                  c_int(poscorr_iter),
                  c_float(step_obj), c_float(step_probe),
                  c_float(reg_obj), c_float(reg_probe),
                  c_float(wavelength_m), c_float(pixelsize_m), c_float(distance_m))

    return obj, probe, error_rfactor, error_llk, error_mse, rois


def PosCorrection(obj: np.ndarray,
                  probe: np.ndarray,
                  difpads: np.ndarray,
                  rois: np.ndarray,
                  iterations: int,
                  objbeta: float,
                  probebeta: float,
                  batch: int,
                  objsupp: Optional[np.ndarray] = None,
                  probesupp: Optional[np.ndarray] = None,
                  step_obj: float = 0.5,
                  step_probe: float = 0.5,
                  reg_obj: float = 1e-3,
                  reg_probe: float = 1e-3,
                  wavelength_m: float = 0.0,
                  pixelsize_m: float = 0.0,
                  distance_m: float = 0.0,
                  params: dict = {}):
    """ Ptychography algorithm for positions correction.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            objbeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            probebeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.
            batch (int, optional): Size . Defaults to 16.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            wavelength_m (float, optional): The wavelength of the light used in the propagation, in meters. Defaults to 0.
            pixelsize_m (float, optional): The detector pixel size in meters. Defaults to 0.
            distance_m (float, optional): The distance to the detector in meters. Defaults to 0.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``param`` input:

                *``param['device']`` (int list): The list of GPUs. Defaults to [0].

        Dictionary ``mydict`` output:

                *``mydict['obj']`` (complex ndarray): The complex 2D object sample retrieved (obj = Attenuation *  e^(-phase)).

                *``mydict['probe']`` (complex ndarray): The complex 2D probe retrieved.

                *``mydict['difpads']`` (ndarray): The 3D stack of diffraction data measured.

                *``mydict['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1].

                *``mydict['W']`` (ndarray): W-function for CARNAUBA.

                *``mydict['error']`` (ndarray): Error of iterations.

                *``mydict['objsupp']`` (int ndarray): Object support.

                *``mydict['probesupp']`` (int ndarray): Probe support.

                *``mydict['bkg']`` (ndarray): The 2D background retrieved.

        """
    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe,probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads,difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois,roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    error_rfactor = np.zeros(iterations, dtype=np.float32)
    error_rfactor,error_rfactorptr, _ = ctypes_array(error_rfactor)

    nummodes = psizez

    assert (probesupp.shape[-1] == probe.shape[-1] and
            probesupp.shape[-2] == probe.shape[-2] and
            probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport,) = ctypes_opt_array(objsupp)

    libcdi.poscorrcall(objptr, probeptr, difpadsptr, psizex, osizex,
                       osizey, dsizex, roisptr, numrois, c_int(batch),
                       c_int(iterations), ndevices, devicesptr, error_rfactorptr,
                       c_float(objbeta), c_float(probebeta), nummodes,
                       objsuppptr, probesuppptr, numobjsupport,
                       c_float(step_obj), c_float(step_probe),
                       c_float(reg_obj),c_float(reg_probe),
                       c_float(wavelength_m), c_float(pixelsize_m), c_float(distance_m))

    return obj, probe, error_rfactor, rois

def log_start(level="error"):
    libcdi.ssc_log_start(ctypes.c_char_p(level.encode('UTF-8')))

def log_stop():
    libcdi.ssc_log_stop()
