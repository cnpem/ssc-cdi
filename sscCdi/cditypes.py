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
from ctypes import c_int, c_void_p, c_float
import ctypes.util
import multiprocessing
import os
from typing import Optional, Tuple
import numpy as np
from time import time


nthreads = multiprocessing.cpu_count()

# Load required libraries:

libstdcpp = ctypes.CDLL(ctypes.util.find_library("stdc++"),
                        mode=ctypes.RTLD_GLOBAL)
libcufft = ctypes.CDLL(ctypes.util.find_library("cufft"),
                       mode=ctypes.RTLD_GLOBAL)
libcublas = ctypes.CDLL(ctypes.util.find_library("cublas"),
                        mode=ctypes.RTLD_GLOBAL)

_lib = "lib/libssccdi"
ext = '.so'


def load_library(lib, ext):
    _path = os.path.dirname(
        os.path.abspath(__file__)) + os.path.sep + lib + ext
    try:
        lib = ctypes.CDLL(_path)
        return lib
    except:
        pass
    return None


libcdi = load_library(_lib, ext)

try:

    libcdi.glcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, c_int,
        c_float, c_float, c_float, c_float,
        c_float
    ]
    libcdi.glcall.restype = None
    libcdi.raarcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, c_int,
        c_float, c_float, c_float, c_float,
        c_float, c_float
    ]
    libcdi.raarcall.restype = None
    libcdi.poscorrcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, c_int,
        c_float, c_float, c_float, c_float,
        c_float
    ]
    libcdi.poscorrcall.restype = None
    libcdi.piecall.argtypes = [
        ctypes.c_void_p, c_int, c_int, ctypes.c_void_p, c_int, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_float, c_float, c_float,
        c_float
    ]

    libcdi.piecall.restype = None

except Exception as e:
    print('>>>', e)
    # pass


def ctypes_array(c: np.ndarray) -> Tuple[np.ndarray, c_void_p, list[c_int]]:
    contiguous_array = np.ascontiguousarray(c)
    cptr = contiguous_array.ctypes.data_as(c_void_p)
    cshape = [c_int(d) for d in c.shape]
    return contiguous_array, cptr, cshape


def ctypes_opt_array(c: Optional[np.ndarray]) -> Tuple[c_void_p, list[c_int]]:
    if c is None:
        return c_void_p(0), c_void_p(0), [c_int(0)]
    return ctypes_array(c)

def append_ones(probe_positions):
    """ Adjust shape and column order of positions array to be accepted by Giovanni's code

    Args:
        probe_positions (array): initial positions array in (PY,PX) shape

    Returns:
        probe_positions2 (array): rearranged probe positions array
    """
    zeros = np.zeros((probe_positions.shape[0],1))
    probe_positions = np.concatenate((probe_positions,zeros),axis=1)
    probe_positions = np.concatenate((probe_positions,zeros),axis=1) # concatenate columns to use Giovanni's ptychography code

    return probe_positions

def sanitize_rois(rois, obj, difpads, probe) -> np.ndarray:

    rois = rois.astype('float32')

    if rois.shape[0] != difpads.shape[0]:
        print("Error:", difpads.shape[0], "difpads  v ", rois.shape[0],
              " rois mismatch")
        quit()
    if rois.shape[-1] != 4:
        print("Incorrect roi specification: shape=", rois.shape,
              ' rois has 4 attribs: x,y,exptime,I0')
        quit()

    if probe.shape[-1] % difpads.shape[-1] != 0 or probe.shape[
            -2] % difpads.shape[-2] != 0:
        print("Error:", probe.shape, "probe  v ", difpads.shape,
              " difpads shape mismatch")
        quit()
    if len(rois.shape) == 2:
        rois = rois[:, None]

    for acqui in rois:
        for roi in acqui:
            if probe.shape[-1] + roi[0] > obj.shape[
                    -1] or probe.shape[-2] + roi[1] > obj.shape[-2]:
                print("Error: Roi", roi, "is outside object bounds:", roi, "+",
                      probe.shape, "=", obj.shape)
                quit()
            elif roi[0] < 0 or roi[1] < 0:
                print("Error: Roi", roi, "has negative indexing.")
                quit()
    return rois


def PIE(obj: np.ndarray,
        probe: np.ndarray,
        difpads: np.ndarray,
        rois: np.ndarray,
        iterations: int,
        step_obj: float,
        step_probe: float,
        reg_obj: float,
        reg_probe: float,
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
            probef1 (float, optional): Fresnel number F1. Defaults to None.
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

    rois = append_ones(rois)

    obj, objptr, (osizey, osizex) = ctypes_array(obj)
    probe, probeptr, (psizez, _, psizex) = ctypes_array(probe)

    difpads, difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois, roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices, devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactor, rfactorptr, _ = ctypes_array(rfactor)

    time0 = time()

    libcdi.piecall(objptr, osizex, osizey, probeptr, psizex, psizez,
                   difpadsptr, dsizex, roisptr, numrois,
                   c_int(iterations), devicesptr, ndevices, rfactorptr,
                   c_float(step_obj), c_float(step_probe),
                   c_float(reg_obj), c_float(reg_probe))

    print(f"\tDone in: {time()-time0:.2f} seconds")

    return obj, probe, rfactor, rois[:,0,0:2]

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
         probef1: float = 0.0,
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
            probef1 (float, optional): Fresnel number F1. Defaults to None.
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

    rois = append_ones(rois)

    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe, probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads, difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois,roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactor,rfactorptr, _ = ctypes_array(rfactor)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)

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
                    c_int(iterations), ndevices, devicesptr, rfactorptr,
                    c_float(objbeta), c_float(probebeta), nummodes,
                    objsuppptr, probesuppptr, numobjsupport,
                    c_int(flyscansteps),
                    c_float(step_obj), c_float(step_probe),
                    c_float(reg_obj), c_float(reg_probe),
                    c_float(probef1), c_float(beta))

    return obj, probe, rfactor, rois[:,0,0:2]



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
       probef1: float = 0.0,
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
            probef1 (float, optional): Fresnel number F1. Defaults to None.
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
    rois = append_ones(rois)

    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe,probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads,difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois,roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactor,rfactorptr, _ = ctypes_array(rfactor)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)

    libcdi.glcall(objptr, probeptr, difpadsptr, psizex, osizex,
                  osizey, dsizex, roisptr, numrois, c_int(batch),
                  c_int(iterations), ndevices, devicesptr, rfactorptr,
                  c_float(objbeta), c_float(probebeta), nummodes, objsuppptr,
                  probesuppptr, numobjsupport, c_int(flyscansteps),
                  c_float(step_obj), c_float(step_probe),
                  c_float(reg_obj), c_float(reg_probe),
                  c_float(probef1))

    return obj, probe, rfactor, rois[:,0,0:2]


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
                  probef1: float = 0.0,
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
            probef1 (float, optional): Fresnel number F1. Defaults to None.
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
    
    rois = append_ones(rois)
    obj,objptr, (osizey, osizex) = ctypes_array(obj)
    probe,probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpads,difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    rois,roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devices,devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactor,rfactorptr, _ = ctypes_array(rfactor)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
    probesupp,probesuppptr, _ = ctypes_array(probesupp.astype(np.float32))
    objsupp,objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)

    libcdi.poscorrcall(objptr, probeptr, difpadsptr, psizex, osizex,
                       osizey, dsizex, roisptr, numrois, c_int(batch),
                       c_int(iterations), ndevices, devicesptr, rfactorptr,
                       c_float(objbeta), c_float(probebeta), nummodes,
                       objsuppptr, probesuppptr, numobjsupport,
                       c_int(flyscansteps),
                       c_float(step_obj), c_float(step_probe),
                       c_float(reg_obj),c_float(reg_probe),
                       c_float(probef1))

    return obj, probe, rfactor, rois[:,0,0:2]

def log_start(level="error"):
    libcdi.ssc_log_start(ctypes.c_char_p(level.encode('UTF-8')))

def log_stop():
    libcdi.ssc_log_stop()
