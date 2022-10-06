#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:52:47 2020

@author: francesco
"""

import warnings
warnings.filterwarnings("ignore")

datapath = 'synthdata/diatom_synth.tif'
pospath = 'synthdata/synthpos.npy'
DIMSAMPLE = 1024
wavelength = 4.892e-10; #wavelength
fd = 0.8932; #focal to detector
fs = 2.5e-3; #focal to sample
ps = 13.5e-6; #pixel size

if __name__ == '__main__':
    import argparse
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=f'SciComPty 2022 - Minimal running example')
    parser.add_argument('--enablegui', help='Enable or not the running result update [yes|no]', type=str, default='yes')
    args = parser.parse_args()

    print('pytorch', torch.__version__)
    print('gpu',torch.cuda.get_device_name(torch.cuda.current_device()))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    import tifffile as tf
    import matplotlib.pyplot as plt
    from skimage.filters import gaussian
    from skimage.transform import resize
    import time
    import cv2

    # SciComPty libs
    from SciComPty_libs.ptyobj_tools import PtyRecon, propagator, blur_mask
    from SciComPty_libs.generic_tools import composeAgainImage, contraststretch, load_positions, genCircleMask, plot_curr
    #from SciComPty_libs.fft_tools import propagator
    
    MagFac = fd/fs
    delta1 = ps/MagFac

    data = tf.imread(datapath)**0.5
    data = data/np.amax(data)

    data = data/np.amax(data)

    white = np.mean(data, axis=0)
    mask = white > 0.5*np.amax(white)
    pixpos = load_positions(pospath)/delta1

    localp = propagator(fd, fs, ps, DIMSAMPLE, wavelength)

    # init guess
    probeguess = data.mean(axis=0)
    probeguess = gaussian(probeguess, sigma=5)
    probeguess = torch.from_numpy(probeguess).to(device)
    probeguess = localp.tosampleplane(probeguess).unsqueeze(0).cpu().numpy()

##    plt.figure()
##    plt.subplot(121); plt.imshow(np.abs(probeguess[0]), cmap='gray')
##    plt.subplot(122); plt.imshow(np.angle(probeguess[0]), cmap='gray')
##    plt.show()

    # init full obj
    canvas, c,s, xpos, ypos, data = composeAgainImage(data, pixpos, flipIt=False, twinMicexp=False, weightFun=white)

    # rscanvas for pos
    rscanvas = s**0.125
    rscanvas -= rscanvas.min()
    rscanvas = rscanvas/rscanvas.max()# + 1
    
    rscanvas = resize(rscanvas, (len(np.unique(pixpos[:,0])), len(np.unique(pixpos[:,0]))))
    rscanvas /= rscanvas.max()
   
    #plt.figure(); plt.imshow(contraststretch(np.abs(canvas)), cmap='gray'); plt.title('Init canvas'); plt.show()

    # init illumination
    mask = genCircleMask(data.shape[1], data.shape[1] *0.5, 1, 1)

    # set init
    objguess = 0.5*np.ones_like(canvas).astype(np.complex)
    
    # init pty obj
    Ptyobj = PtyRecon(data, xpos, ypos,
                      fd, fs, ps, wavelength,
                      probeguess=probeguess.astype(np.complex64), objguess=objguess, canvasmask=rscanvas, posupalpha=10.,
                      detectormask=None)

    losshist, gtposerrhist, estposerrhist = [], [], []

    starttime = time.time()
    for ep in range(13):
        t0 = time.time()
        rloss, posloss = Ptyobj.updateRpieMulti(upobj=.75, upill=.75, uppos=0, beta=.5)#ep>5)
        t1 = time.time()
        print('It: {} Err: {} PosErr: {} Time: {}'.format(ep, np.round(rloss,decimals=4), posloss, np.round(t1-t0,decimals=3)))
        # show update
        if ep%3 == 0 and args.enablegui == 'yes':
            obj, ccprobe = Ptyobj.objguess.cpu().numpy()[0], Ptyobj.probeguess.cpu().numpy()
            plot_curr(obj, ccprobe.sum(axis=0), ep, rloss,frac=1/8)
        # save hist
        losshist.append(rloss)

    endtime = time.time()

    print(f'All done in {endtime-starttime:.2f} s')
    cv2.destroyAllWindows()
    print ('end')

