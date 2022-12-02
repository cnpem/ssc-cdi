#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:03:31 2022

@author: feynman
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def func(x,alpha,cut=8):
    maximum = np.max(np.where(np.abs(x)<cut,0,1 - np.exp(alpha*x)))
    # func += np.where(np.abs(x) < cut,maximum,1 - np.exp(alpha*x))
    func = np.where(np.abs(x) < cut ,1 - np.exp(alpha*x),0)
    func = np.where(np.abs(x) < cut, 1 + func / np.max(np.abs(func)),0)
    return func

def get_best_plane_fit_inside_mask(mask2,frame ):
    new   = np.zeros(frame.shape)
    row   = new.shape[0]
    col   = new.shape[1]
    XX,YY = np.meshgrid(np.arange(col),np.arange(row))

    a = b = c = 1e9
    counter = 0
    while np.abs(a) > 1e-8 or np.abs(b) > 1e-8 or counter > 5:
        grad_removed, a,b,c = RemoveGrad_new(frame,mask2)
        plane_fit = plane((XX,YY),a,b,c).reshape(XX.shape)
        frame = frame - plane_fit
        counter += 1
    return frame

def border_filter_frames(loadpath,savepath,fit_region=(500,650,850,1100),cutoff=40, decay = 1, null_size=50, save=True, preview_filter=True):

    data = np.load(loadpath)

    # crop = 150
    # data = data[:,crop:-crop,crop:-crop]

    top, bottom, left, right = fit_region

    mask = np.zeros_like(data[0],dtype=bool) # mask indicating where to fit plane
    
    mask[top:bottom,left:right] = True

    get_best_plane_fit_inside_mask_partial = partial(get_best_plane_fit_inside_mask,mask2)
    
    """ Get filter """
    frame = data[0]
    if 1: # exponential decay at border
        N = null_size
        x = np.linspace(-N,N,frame.shape[1])
        y = np.linspace(-N,N,frame.shape[0])
        func_x = np.where(x>=0,func(x,decay,cut=cutoff) , func(x,-decay,cut=cutoff))
        func_y = np.where(y>=0,func(y,decay,cut=cutoff) , func(y,-decay,cut=cutoff))
        meshY, meshX = np.meshgrid(func_x,func_y)
        border_attenuation_matrix = meshY*meshX
    elif 0: #. gaussian filter
        sigma = 2
        x = np.linspace(-5, 5,frame.shape[1])
        y = np.linspace(-5, 5,frame.shape[0])
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        border_attenuation_matrix = gaus2d(x, y,sx=sigma,sy=sigma)
    
    frames = [data[i] for i in range(data.shape[0])]

    """ Remove gradient from bkg """
    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(tqdm(executor.map(get_best_plane_fit_inside_mask_partial,frames),total=data.shape[0]))
        for i, result in enumerate(results):
            data[i] = result - np.min(result)
            
    """ Apply border filter """
    data[:] = data[:]*border_attenuation_matrix

    if save: np.save(savepath,data)
                            
    return data, border_attenuation_matrix

frame = np.random.rand((100,100))


N = 50
alpha = 0.2
a = 40
b = a
x = np.linspace(-N,N,frame.shape[1])
y = np.linspace(-N,N,frame.shape[0])
func_x = np.where(x>=0,func(x,alpha,cut=a) , func(x,-alpha,cut=a))
func_y = np.where(y>=0,func(y,alpha,cut=b) , func(y,-alpha,cut=b))
meshY, meshX = np.meshgrid(func_x,func_y)
border_attenuation_matrix = meshY*meshX


