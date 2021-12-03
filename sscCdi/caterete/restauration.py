import numpy
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
from scipy import signal
import time
from sscPimega import misc
from sscPimega import pi540D
from sscIO import io

#################

def cat_preproc_ptycho_measurement( data, args ):
    def _get_center(dbeam, project):
        aimg = pi540D._worker_annotation_image ( numpy.clip( dbeam, 0, 100) )
        aimg = ndimage.gaussian_filter( aimg, sigma=0.95, order=0 )
        aimg = aimg/aimg.max()
        aimg = 1.0 * ( aimg > 0.98 )    
        u = numpy.array(range(3072))
        xx,yy = numpy.meshgrid(u,u)
        xc = ((aimg * xx).sum() / aimg.sum() ).astype(int)
        yc = ((aimg * yy).sum() / aimg.sum() ).astype(int)
        annotation = numpy.array([ [xc, yc] ])
        tracking = pi540D.annotation_points_standard ( annotation )
        tracking = pi540D.tracking540D_vec_standard ( project, tracking ) 
        xc = int( tracking[0][2] )
        yc = int( tracking[0][3] ) 
        return xc, yc
    def _operator_T(u):
        d   = 1.0
        uxx = (numpy.roll(u,1,1) - 2 * u + numpy.roll(u,-1,1) ) / (d**2)
        uyy = (numpy.roll(u,1,0) - 2 * u + numpy.roll(u,-1,0) ) / (d**2)
        uyx = (numpy.roll(numpy.roll(u,1,1),1,1) - numpy.roll(numpy.roll(u,1,1),-1,0) - numpy.roll(numpy.roll(u,1,0),-1,1) + numpy.roll(numpy.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
        uxy = (numpy.roll(numpy.roll(u,1,1),1,1) - numpy.roll(numpy.roll(u,-1,1),1,0) - numpy.roll(numpy.roll(u,-1,0),1,1) + numpy.roll(numpy.roll(u,-1,1),-1,0)   )/ (2 * d**2)
        delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
        z = numpy.sqrt( delta )
        return z
    def _get_roi( img, roi, center,binning):
        xc, yc = center
        X = img[yc-roi:yc+roi:binning,xc-roi:xc+roi:binning]
        return X
    def set_binning(data, binning):
        if binning>0:
            # Define kernel for convolution                                         
            kernel = numpy.ones([binning,binning]) 
            # Perform 2D convolution with input data and kernel 
            X = signal.convolve2d(data, kernel, mode='same')/kernel.sum()
        else:
            X = data
        return X

    ######
    dic   = args[0]
    geo   = args[1]
    proj  = args[2]
    empty = args[3]
    flat  = args[4]
    xc, yc = _get_center(data, proj)

    #operation on the detector domain:
    flat[flat == 0] = 1 
    flat[numpy.isnan(flat)] = 1
    data = data*flat # Flatfield application. Convention with DET group is a product between data and flat!
    
    if 0:
        data[empty > 0] = _operator_T(data).real[ empty > 0] # remove bad datapoints with Miquele's operator
    else:
        data[empty > 0] = -1

    back  = pi540D.backward540D ( data , geo)
    where = (back == -1)
    back = set_binning ( back, dic['binning'] )
    back[where] = -1
    backroi  = _get_roi( back, dic['roi'], [xc, yc],dic['binning'])
    
    where = _get_roi( where, dic['roi'], [xc, yc],dic['binning'])
    backroi[where] = -1

    return backroi
##############
def cat_preproc_ptycho_projections( dic ):
    #-----------------------
    #read data using ssc-io:
    empty     = h5py.File(dic['empty'], 'r')['entry/data/data/'][0,0,:,:]
    measure,_ = io.read_volume(dic['data'],'numpy', use_MPI=True, nprocs=32)
    flat      = numpy.load(dic['flat']) # flatfield file needs to be a numpy!
    #------------------------------------
    # computing ssc-pimega 540D geometry:
    xdet     = pi540D.get_project_values_geometry()
    project  = pi540D.get_detector_dictionary( xdet, dic['distance'] )
    project['s'] = [dic['susp'],dic['susp']]
    geometry = pi540D.geometry540D( project )
    #-------------------------------------------------------------------------------
    # applying the input dic['function'] to the ptychographic sequence measured (2d)
    # -> function must include restoration + all preprocessing details
    params = (dic, geometry, project, empty, flat )
    start = time.time()
    output,_ = pi540D.backward540D_batch( measure[1:], dic['distance'], dic['nproc'], [2*dic['roi']//dic['binning'], 2*dic['roi']//dic['binning']], dic['function'], params, dic['order'] )
    elapsed = time.time() - start
    return output, elapsed
#################
