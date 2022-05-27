#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:55:18 2022

@author: yurirt
"""
from sys import path_importer_cache
from turtle import st
import matplotlib.pyplot as plt
import numpy as np
 
# importing movie py libraries
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
def plotshow(imgs, f1,file, legending_f_value=[],cmap='jet',nlines=1, bLog = False, interpolation='bilinear'): # legending_f_value = plot titles
        num = len(imgs)

        for j in range(num):
                if type(cmap) == str:
                        colormap = cmap
                elif len(cmap) == len(imgs):
                        colormap = cmap[j]
                else:
                        colormap = cmap[j//(len(imgs)//nlines)]

                sb = plt.subplot((num+nlines-1)//nlines,nlines,j+1)

                if type(imgs[j][0,0]) == np.complex64 or type(imgs[j][0,0]) == np.complex128:
                        sb.imshow(CMakeRGB(imgs[j]),cmap='hsv',interpolation=interpolation)
                elif bLog:
                        sb.imshow(np.log(1+np.maximum(imgs[j],-0.1))/np.log(10),cmap=colormap,interpolation=interpolation)
                else:
                        sb.imshow(imgs[j],cmap=colormap,interpolation=interpolation)

                sb.set_xticks([])
                sb.set_yticks([])
                sb.set_title(f'{f1[j]:.2e}',fontsize=10)

                if len(legending_f_value)>j:
                        sb.set_title(legending_f_value[j])

        plt.tight_layout()
        # plt.subplots_adjust(left=0.4, bottom=0, right=0.6, top=1, wspace=0, hspace=0.2)
        plt.savefig(file + '.png', format='png', dpi=300)
        plt.clf()
        plt.close()


def Prop(img,f1):
    hs = img.shape[-1]//2
    ar = np.arange(-hs,hs) / float(2*hs)
    xx,yy = np.meshgrid(ar,ar)
    g = np.exp(-1j*np.pi/f1 * (xx**2+yy**2))
    return np.fft.ifft2(np.fft.fft2(img)*np.fft.fftshift(g))#[64:-64,64:-64]#[160:-160,160:-160]


def create_propagation_video(path_to_probefile,
                             starting_f_value=1e-3,
                             ending_f_value=9e-4,
                             number_of_frames=100,
                             frame_rate=10,
                             mp4=False, 
                             gif=False,
                             jupyter=False):
    from tqdm import tqdm
    probe = np.load(path_to_probefile)[0] # load probe
    
    # delta = -1e-4
    # f1 = [starting_f_value + delta*i for i in range(0,number_of_frames)]
    
    f1 = np.linspace(starting_f_value,ending_f_value,number_of_frames)
    
    # Create list of propagated probes
    b =  [np.sqrt(np.sum([abs(Prop(a,f1[0]))**2 for a in probe],0))]
    for i in range(1,number_of_frames):
            b += [np.sqrt(np.sum([abs(Prop(a,f1[i]))**2 for a in probe],0))]
    

    image_list = []
    for j, probe in enumerate(tqdm(b)):
            if jupyter == False:
                animation_fig, subplot = plt.subplots(dpi=300)
                img = subplot.imshow(probe,cmap='jet')#,animated=True)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.set_title(f'f#={f1[j]:.3e}')
            if jupyter == False:
                image_list.append(mplfig_to_npimage(animation_fig))
            else:    
                image_list.append(probe)
            if jupyter == False: plt.close()

    if mp4 or gif:  
        clip = ImageSequenceClip(image_list, fps=frame_rate)
        if mp4:
            clip.write_videofile("propagation.mp4",fps=frame_rate)
        if gif:
            clip.write_gif('propagation.gif', fps=frame_rate)

    return image_list, f1


if __name__ == '__main__':
    
    from sys import argv

    # path_to_probefile = '/ibira/lnls/labs/tepui/proposals/20210062/yuri/Caterete/yuri-ssc-cdi/outputs/reconstruction/probe_mfi_4keV_01_061121.npy'
    path_to_probefile = argv[1]

    _, _ = create_propagation_video(path_to_probefile,
                            starting_f_value=-1.0e-4,
                            ending_f_value=-9.0e-4,
                            number_of_frames=100,
                            frame_rate=10,
                            mp4=True, 
                            gif=False)
    
