#We use the one encoding: utf8
import ctypes
from ctypes import c_int, c_void_p, c_float
import ctypes.util
import multiprocessing
import os
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
    import pdb
    pdb.set_trace()
    _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
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
        ctypes.c_void_p, c_float, c_float, c_int, c_float, ctypes.c_void_p,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, c_float,
        ctypes.c_void_p, c_float
    ]
    libcdi.glcall.restype = None
    libcdi.raarcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, c_float, ctypes.c_void_p,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, c_float,
        ctypes.c_void_p, c_float
    ]
    libcdi.raarcall.restype = None
    libcdi.poscorrcall.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int,
        c_int, ctypes.c_void_p, c_int, c_int, c_int, c_int, ctypes.c_void_p,
        ctypes.c_void_p, c_float, c_float, c_int, c_float, ctypes.c_void_p,
        ctypes.c_void_p, c_int, ctypes.c_void_p, c_int, c_float,
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


def MAKENICE(obj, probe, difpads, rois, iter, batch, objbeta, probebeta, _funcc,
             regularization, objsupp, probesupp, sigmask, epsilon, bkg, probef1,
             data, params):

    if obj is None:
        obj = data['obj']
    if probe is None:
        probe = data['probe']
    if difpads is None:
        difpads = data['difpads']
    if rois is None:
        rois = data['rois']
    if objsupp is None:
        try:
            objsupp = data['objsupp']
        except:
            pass
    if probesupp is None:
        try:
            probesupp = data['probesupp']
        except:
            pass
    if sigmask is None:
        try:
            sigmask = data['sigmask']
        except:
            pass
    if bkg is None:
        try:
            bkg = data['bkg']
        except:
            pass

    if iter is None:
        try:
            iter = data['iter']
        except:
            iter = 100

    if batch is None:
        try:
            batch = data['batch']
        except:
            batch = 16
    if objbeta is None:
        try:
            objbeta = data['objbeta']
        except:
            objbeta = 0.95
    if probebeta is None:
        try:
            probebeta = data['probebeta']
        except:
            probebeta = 0.9
    if regularization is None:
        try:
            regularization = data['regularization']
        except:
            regularization = -1
    if epsilon is None:
        try:
            epsilon = data['epsilon']
        except:
            epsilon = 1E-3
    if probef1 is None:
        try:
            probef1 = data['probef1']
        except:
            probef1 = 0

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

    if probesupp is not None:
        assert (probesupp.shape[-1] == probe.shape[-1] and
                probesupp.shape[-2] == probe.shape[-2] and
                probesupp.size == probe.size)
        probesupp = np.ascontiguousarray(probesupp.astype(np.float32))
        probesuppptr = probesupp.ctypes.data_as(c_void_p)
    else:
        probesuppptr = c_void_p(0)

    if objsupp is not None:
        assert (objsupp.size >= obj.size)
        objsupp = np.ascontiguousarray(objsupp.astype(np.float32))
        objsuppptr = objsupp.ctypes.data_as(c_void_p)
        if len(objsupp.shape) >= 3:
            numobjsupport = c_int(objsupp.shape[0])
        else:
            numobjsupport = c_int(1)
    else:
        objsuppptr = c_void_p(0)
        numobjsupport = c_int(0)

    if bkg is not None:
        assert (bkg.shape[-1] == difpads.shape[-1] & bkg.shape[-2] ==
                difpads.shape[-2])
        bkg = np.ascontiguousarray(bkg.astype(np.float32))
        bkgptr = bkg.ctypes.data_as(c_void_p)
    else:
        bkgptr = c_void_p(0)

    obj = np.ascontiguousarray(obj.astype(np.complex64))
    objptr = obj.ctypes.data_as(c_void_p)

    probe = np.ascontiguousarray(probe.astype(np.complex64))
    probeptr = probe.ctypes.data_as(c_void_p)

    difpads = np.ascontiguousarray(difpads.astype(np.float32))
    difpadsptr = difpads.ctypes.data_as(c_void_p)

    if sigmask is not None:
        assert (sigmask.shape[-1] == difpads.shape[-1] & sigmask.shape[-2] ==
                difpads.shape[-2])
        sigmask = np.ascontiguousarray(sigmask.astype(np.float32))
    else:
        sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr = sigmask.ctypes.data_as(c_void_p)

    rois = np.ascontiguousarray(rois.astype(np.float32))
    roisptr = rois.ctypes.data_as(c_void_p)

    wfun = np.zeros((rois.shape[1]), dtype=np.float32)
    wfunptr = wfun.ctypes.data_as(c_void_p)

    rfactor = np.zeros((iter), dtype=np.float32)
    rfactorptr = rfactor.ctypes.data_as(c_void_p)

    numdev = c_int(len(params['device']))
    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr = devices.ctypes.data_as(c_void_p)

    psizex = c_int(probe.shape[-1])
    osizex = c_int(obj.shape[-1])
    osizey = c_int(obj.shape[-2])
    dsizex = c_int(difpads.shape[-1])
    iter = c_int(iter)
    batch = c_int(batch)
    objbeta = c_float(objbeta)
    probebeta = c_float(probebeta)
    numrois = c_int(rois.shape[0])
    flyscansteps = c_int(rois.shape[1])
    epsilon = c_float(epsilon)
    probef1 = c_float(probef1)

    nummodes = c_int(probe.shape[0])
    if len(probe.shape) < 3:
        nummodes = c_int(1)

    time0 = time()

    _funcc(objptr, probeptr, difpadsptr, psizex, osizex, osizey, dsizex,
           roisptr, numrois, batch, iter, numdev,
           devicesptr, rfactorptr, objbeta, probebeta, nummodes,
           c_float(regularization), objsuppptr, probesuppptr, numobjsupport,
           sigmaskptr, flyscansteps, epsilon, bkgptr, probef1)

    print(f"\tDone in: {time()-time0:.2f} seconds")

    mydict = {}
    mydict['obj'] = obj
    mydict['probe'] = probe
    mydict['error'] = rfactor
    mydict['bkg'] = bkg
    mydict['rois'] = rois
    mydict['difpads'] = difpads
    mydict['probesupp'] = probesupp
    mydict['objsupp'] = objsupp
    mydict['W'] = wfun
    return mydict


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
            data (dic, optional): Dictionary containing all parameters above. Defaults to None.
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

    objptr = obj.ctypes.data_as(c_void_p)
    osizey, osizex = [c_int(d) for d in obj.shape]

    probeptr = probe.ctypes.data_as(c_void_p)
    psizez, _, psizex = [c_int(d) for d in probe.shape]

    difpadsptr = difpads.ctypes.data_as(c_void_p)
    *_, dsizex = [c_int(d) for d in difpads.shape]

    rois = np.ascontiguousarray(rois, dtype='float32')
    roisptr = rois.ctypes.data_as(c_void_p)
    numrois = c_int(len(rois))

    devices = np.ascontiguousarray(
        np.asarray(params['device']).astype(np.int32))
    devicesptr = devices.ctypes.data_as(c_void_p)
    ndevices = c_int(len(params['device']))

    rfactor = np.zeros(iterations, dtype=np.float32)
    rfactorptr = rfactor.ctypes.data_as(c_void_p)

    sigmask = np.ones(difpads.shape[-2:], dtype=np.float32)
    sigmaskptr = sigmask.ctypes.data_as(c_void_p)

    libcdi.piecall(objptr, osizex, osizey, probeptr, psizex, psizez, difpadsptr,
                   dsizex, roisptr, numrois, sigmaskptr, iterations, devicesptr,
                   ndevices, rfactorptr, step_obj, step_probe, reg_obj,
                   reg_probe)

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


def RAAR(obj=None,
         probe=None,
         difpads=None,
         rois=None,
         iter=None,
         beta=None,
         probecycles=None,
         batch=None,
         tvmu=None,
         objsupp=None,
         probesupp=None,
         sigmask=None,
         epsilon=None,
         bkg=None,
         probef1=None,
         data=None,
         params=None):
    """ Ptychography RAAR algorithm.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            beta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            batch (int, optional): Size . Defaults to 16.
            tvmu (float, optional): Regularization parameter for total variation. Defaults to None.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
            probef1 (float, optional): Fresnel number F1. Defaults to None.
            data (dic, optional): Dictionary containing all parameters above. Defaults to None.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``data`` input:

                * ``data['obj']`` (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.

                * ``data['probe']`` (complex ndarray, optional): The 2D probe inital guess. Defaults to None.

                * ``data['difpads']`` (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.

                * ``data['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1] . Defaults to None.

                * ``data['iter']`` (int, optional): Number of iterations. Defaults to 100.

                * ``data['objbeta']`` (float, optional): Relaxation parameter for object (same as beta above), 0 < objbeta < 1. Defaults to 0.95.

                * ``data['batch']`` (int, optional): Size . Defaults to 16.

                * ``data['tvmu']`` (float, optional): Regularization parameter for total variation. Defaults to None.

                * ``data['objsupp']`` (int ndarray, optional): Object support. Defaults to None.

                * ``data['probesupp']`` (int ndarray, optional): Probe support. Defaults to None.

                * ``data['sigmask']`` (int ndarray, optional): Mask for invalid pixels . Defaults to None.

                * ``data['epsilon']`` (float, optional): Regularization parameter. Defaults to 1E-3.

                * ``data['bkg']`` (ndarray, optional): The 2D background inital guess. Defaults to None.

                * ``data['probef1']`` (float, optional): Fresnel number F1. Defaults to None.


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

    if probecycles is None and params is not None:
        try:
            params['probecycles']
        except:
            params['probecycles'] = 3
    return MAKENICE(obj, probe, difpads, rois, iter, batch, beta, probecycles,
                    libcdi.raarcall, tvmu, objsupp, probesupp, sigmask, epsilon,
                    bkg, probef1, data, params)
    #return MAKENICE(obj,probe,difpads,rois,iter,batch,objbeta,probebeta,psiccdll.Raar,-1)


def GL(obj=None,
       probe=None,
       difpads=None,
       rois=None,
       iter=None,
       objbeta=None,
       probebeta=None,
       batch=None,
       tvmu=None,
       objsupp=None,
       probesupp=None,
       sigmask=None,
       epsilon=None,
       bkg=None,
       probef1=None,
       data=None,
       params=None):
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
            tvmu (float, optional): Regularization parameter for total variation. Defaults to None.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
            probef1 (float, optional): Fresnel number F1. Defaults to None.
            data (dic, optional): Dictionary containing all parameters above. Defaults to None.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``data`` input:

                *``data['obj']`` (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.

                *``data['probe']`` (complex ndarray, optional): The 2D probe inital guess. Defaults to None.

                *``data['difpads']`` (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.

                *``data['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1] . Defaults to None.

                *``data['iter']`` (int, optional): Number of iterations. Defaults to 100.

                *``data['objbeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.

                *``data['probebeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.

                *``data['batch']`` (int, optional): Size . Defaults to 16.

                *``data['tvmu']`` (float, optional): Regularization parameter for total variation. Defaults to None.

                *``data['objsupp']`` (int ndarray, optional): Object support. Defaults to None.

                *``data['probesupp']`` (int ndarray, optional): Probe support. Defaults to None.

                *``data['sigmask']`` (int ndarray, optional): Mask for invalid pixels . Defaults to None.

                *``data['epsilon']`` (float, optional): Regularization parameter. Defaults to 1E-3.

                *``data['bkg']`` (ndarray, optional): The 2D background inital guess. Defaults to None.

                *``data['probef1']`` (float, optional): Fresnel number F1. Defaults to None.

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

    return MAKENICE(obj, probe, difpads, rois, iter, batch, objbeta, probebeta,
                    libcdi.glcall, tvmu, objsupp, probesupp, sigmask, epsilon,
                    bkg, probef1, data, params)


def GlobalPSF(obj=None,
              probe=None,
              difpads=None,
              rois=None,
              iter=None,
              objbeta=None,
              probebeta=None,
              batch=None,
              tvmu=None,
              objsupp=None,
              probesupp=None,
              sigmask=None,
              epsilon=None,
              bkg=None,
              probef1=None,
              data=None,
              params=None):
    """ Ptychography algorithm to retrieve W-function.

        Args:
            obj (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.
            probe (complex ndarray, optional): The 2D probe inital guess. Defaults to None.
            difpads (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.
            rois (ndarray, optional): The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1]. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 100.
            objbeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.
            probebeta (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.
            batch (int, optional): Size . Defaults to 16.
            tvmu (float, optional): Regularization parameter for total variation. Defaults to None.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
            probef1 (float, optional): Fresnel number F1. Defaults to None.
            data (dic, optional): Dictionary containing all parameters above. Defaults to None.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``data`` input:

                *``data['obj']`` (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.

                *``data['probe']`` (complex ndarray, optional): The 2D probe inital guess. Defaults to None.

                *``data['difpads']`` (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.

                *``data['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1] . Defaults to None.

                *``data['iter']`` (int, optional): Number of iterations. Defaults to 100.

                *``data['objbeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.

                *``data['probebeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.

                *``data['batch']`` (int, optional): Size . Defaults to 16.

                *``data['tvmu']`` (float, optional): Regularization parameter for total variation. Defaults to None.

                *``data['objsupp']`` (int ndarray, optional): Object support. Defaults to None.

                *``data['probesupp']`` (int ndarray, optional): Probe support. Defaults to None.

                *``data['sigmask']`` (int ndarray, optional): Mask for invalid pixels . Defaults to None.

                *``data['epsilon']`` (float, optional): Regularization parameter. Defaults to 1E-3.

                *``data['bkg']`` (ndarray, optional): The 2D background inital guess. Defaults to None.

                *``data['probef1']`` (float, optional): Fresnel number F1. Defaults to None.

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
    return MAKENICE(obj, probe, difpads, rois, iter, batch, objbeta, probebeta,
                    libcdi.psfcall, tvmu, objsupp, probesupp, sigmask, epsilon,
                    bkg, probef1, data, params)


def PosCorrection(obj=None,
                  probe=None,
                  difpads=None,
                  rois=None,
                  iter=None,
                  objbeta=None,
                  probebeta=None,
                  batch=None,
                  tvmu=None,
                  objsupp=None,
                  probesupp=None,
                  sigmask=None,
                  epsilon=None,
                  bkg=None,
                  probef1=None,
                  data=None,
                  params=None):
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
            tvmu (float, optional): Regularization parameter for total variation. Defaults to None.
            objsupp (int ndarray, optional): Object support. Defaults to None.
            probesupp (int ndarray, optional): Probe support. Defaults to None.
            sigmask (int ndarray, optional): Mask for invalid pixels . Defaults to None.
            epsilon (float, optional): Regularization parameter. Defaults to 1E-3.
            bkg (ndarray, optional): The 2D background inital guess. Defaults to None.
            probef1 (float, optional): Fresnel number F1. Defaults to None.
            data (dic, optional): Dictionary containing all parameters above. Defaults to None.
            params (dic, optional): Dictionary containing aditional parameters. Defaults to None.

        Returns:
            mydict (dic): Dictionary containing results

        Dictionary ``data`` input:

                *``data['obj']`` (complex ndarray, optional): The 2D object sample inital guess. Defaults to None.

                *``data['probe']`` (complex ndarray, optional): The 2D probe inital guess. Defaults to None.

                *``data['difpads']`` (ndarray, optional): The 3D stack of diffraction data measured. Defaults to None.

                *``data['rois']`` The 2D probe npoints positions. Size [npoints,2] where x = [npoints,0] and y = [npoints,1] . Defaults to None.

                *``data['iter']`` (int, optional): Number of iterations. Defaults to 100.

                *``data['objbeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.95.

                *``data['probebeta']`` (float, optional): Relaxation parameter for object, 0 < objbeta < 1. Defaults to 0.9.

                *``data['batch']`` (int, optional): Size . Defaults to 16.

                *``data['tvmu']`` (float, optional): Regularization parameter for total variation. Defaults to None.

                *``data['objsupp']`` (int ndarray, optional): Object support. Defaults to None.

                *``data['probesupp']`` (int ndarray, optional): Probe support. Defaults to None.

                *``data['sigmask']`` (int ndarray, optional): Mask for invalid pixels . Defaults to None.

                *``data['epsilon']`` (float, optional): Regularization parameter. Defaults to 1E-3.

                *``data['bkg']`` (ndarray, optional): The 2D background inital guess. Defaults to None.

                *``data['probef1']`` (float, optional): Fresnel number F1. Defaults to None.

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
    return MAKENICE(obj, probe, difpads, rois, iter, batch, objbeta, probebeta,
                    libcdi.poscorrcall, tvmu, objsupp, probesupp, sigmask,
                    epsilon, bkg, probef1, data, params)
