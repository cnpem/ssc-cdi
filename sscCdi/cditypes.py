#We use the one encoding: utf8
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
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int,
        c_float, c_float, c_float, c_float,
        ctypes.c_void_p, c_float
    ]
    libcdi.glcall.restype = None
    libcdi.raarcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int,
        c_float, c_float, c_float, c_float,
        ctypes.c_void_p, c_float, c_float
    ]
    libcdi.raarcall.restype = None
    libcdi.poscorrcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int,
        c_float, c_float, c_float, c_float,
        ctypes.c_void_p, c_float
    ]
    libcdi.poscorrcall.restype = None
    libcdi.piecall.argtypes = [
        ctypes.c_void_p, c_int, c_int, ctypes.c_void_p, c_int, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, ctypes.c_void_p, c_int,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_float, c_float, c_float,
        c_float
    ]

    libcdi.piecall.restype = None

except Exception as e:
    print('>>>', e)
    # pass


def ctypes_array(c: np.ndarray) -> Tuple[c_void_p, list[c_int]]:
    cptr = np.ascontiguousarray(c).ctypes.data_as(c_void_p)
    cshape = [c_int(d) for d in c.shape]
    return cptr, cshape


def ctypes_opt_array(c: Optional[np.ndarray]) -> Tuple[c_void_p, list[c_int]]:
    if c is None:
        return c_void_p(0), [c_int(0)]
    return ctypes_array(c)


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
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
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

    objptr, (osizey, osizex) = ctypes_array(obj)
    probeptr, (psizez, _, psizex) = ctypes_array(probe)

    difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactorptr, _ = ctypes_array(rfactor)

    sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr, _ = ctypes_array(sigmask)

    time0 = time()

    libcdi.piecall(objptr, osizex, osizey, probeptr, psizex, psizez,
                   difpadsptr, dsizex, roisptr, numrois, sigmaskptr,
                   c_int(iterations), devicesptr, ndevices, rfactorptr,
                   c_float(step_obj), c_float(step_probe),
                   c_float(reg_obj), c_float(reg_probe))

    print(f"\tDone in: {time()-time0:.2f} seconds")

    return {
        'obj': obj,
        'probe': probe,
        'error': rfactor,
        'bkg': None,
        'rois': rois,
        'difpads': difpads,
        'probesupp': None,
        'objsupp': None
    }


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
         sigmask: Optional[np.ndarray] = None,
         step_obj: float = 0.5,
         step_probe: float = 0.5,
         reg_obj: float = 1e-3,
         reg_probe: float = 1e-3,
         bkg: Optional[np.ndarray] = None,
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
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
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

    objptr, (osizey, osizex) = ctypes_array(obj)
    probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactorptr, _ = ctypes_array(rfactor)

    sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr, _ = ctypes_array(sigmask)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    probesuppptr, _ = ctypes_opt_array(probesupp)
    objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)
    bkgptr, _ = ctypes_opt_array(bkg)

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
                    objsuppptr, probesuppptr, numobjsupport, sigmaskptr,
                    c_int(flyscansteps),
                    c_float(step_obj), c_float(step_probe),
                    c_float(reg_obj), c_float(reg_probe),
                    bkgptr, c_float(probef1), c_float(beta))

    return {
        'obj': obj,
        'probe': probe,
        'error': rfactor,
        'bkg': bkg,
        'rois': rois,
        'difpads': difpads,
        'probesupp': probesupp,
        'objsupp': objsupp
    }


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
       sigmask: Optional[np.ndarray] = None,
       step_obj: float = 0.5,
       step_probe: float = 0.5,
       reg_obj: float = 1e-3,
       reg_probe: float = 1e-3,
       bkg: Optional[np.ndarray] = None,
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
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
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
    objptr, (osizey, osizex) = ctypes_array(obj)
    probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactorptr, _ = ctypes_array(rfactor)

    sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr, _ = ctypes_array(sigmask)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    probesuppptr, _ = ctypes_opt_array(probesupp)
    objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)
    bkgptr, _ = ctypes_opt_array(bkg)

    libcdi.glcall(objptr, probeptr, difpadsptr, psizex, osizex,
                  osizey, dsizex, roisptr, numrois, c_int(batch),
                  c_int(iterations), ndevices, devicesptr, rfactorptr,
                  c_float(objbeta), c_float(probebeta), nummodes, objsuppptr,
                  probesuppptr, numobjsupport, sigmaskptr, c_int(flyscansteps),
                  c_float(step_obj), c_float(step_probe),
                  c_float(reg_obj), c_float(reg_probe), bkgptr,
                  c_float(probef1))

    return {
        'obj': obj,
        'probe': probe,
        'error': rfactor,
        'bkg': bkg,
        'rois': rois,
        'difpads': difpads,
        'probesupp': probesupp,
        'objsupp': objsupp
    }


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
                  sigmask: Optional[np.ndarray] = None,
                  step_obj: float = 0.5,
                  step_probe: float = 0.5,
                  reg_obj: float = 1e-3,
                  reg_probe: float = 1e-3,
                  bkg: Optional[np.ndarray] = None,
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
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
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
    objptr, (osizey, osizex) = ctypes_array(obj)
    probeptr, (psizez, _, psizex) = ctypes_array(probe)
    difpadsptr, (*_, dsizex) = ctypes_array(difpads)

    rois = sanitize_rois(rois, obj, difpads, probe)
    roisptr, (numrois, *_) = ctypes_array(rois)

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr, (ndevices, ) = ctypes_array(devices)

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactorptr, _ = ctypes_array(rfactor)

    sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr, _ = ctypes_array(sigmask)

    nummodes = psizez

    flyscansteps = int(rois.shape[1])

    probesuppptr, _ = ctypes_opt_array(probesupp)
    objsuppptr, (numobjsupport, ) = ctypes_opt_array(objsupp)
    bkgptr, _ = ctypes_opt_array(bkg)

    libcdi.poscorrcall(objptr, probeptr, difpadsptr, psizex, osizex,
                       osizey, dsizex, roisptr, numrois, c_int(batch),
                       c_int(iterations), ndevices, devicesptr, rfactorptr,
                       c_float(objbeta), c_float(probebeta), nummodes,
                       objsuppptr, probesuppptr, numobjsupport, sigmaskptr,
                       c_int(flyscansteps),
                       c_float(step_obj), c_float(step_probe),
                       c_float(reg_obj),c_float(reg_probe),
                       bkgptr, c_float(probef1))

    return {
        'obj': obj,
        'probe': probe,
        'error': rfactor,
        'bkg': bkg,
        'rois': rois,
        'difpads': difpads,
        'probesupp': probesupp,
        'objsupp': objsupp
    }
