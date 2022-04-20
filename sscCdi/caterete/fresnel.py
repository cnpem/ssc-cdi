#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:55:18 2022

@author: yurirt
"""

# importing matplot lib
from sys import path_importer_cache
from turtle import st
import matplotlib.pyplot as plt
import numpy as np
 
# importing movie py libraries
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
import ipywidgets as widgets 
from ipywidgets import fixed

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





def update_values(n_frames,start_f,end_f,power):
    global starting_f_value
    global ending_f_value
    global number_of_frames
    starting_f_value=-start_f*10**power
    ending_f_value=-end_f*10**power
    number_of_frames=int(n_frames)


def update_imshow(fig,ax1,image_list,f1_list,index):
    print('Plotting new probe propagation...')
    ax1.clear()
    ax1.set_title(f'f1 = {f1_list[index]:.2e}')
    ax1.imshow(image_list[index],cmap='jet')
    fig.canvas.draw_idle()
    

def update_dic(output_dictionary,key,boxvalue):
    output_dictionary[key]  = boxvalue 

def on_click_propagate(dummy,args=()):
 
    path_to_probefile,starting_f_value,ending_f_value,number_of_frames,play,slider,fig, ax1 = args

    image_list, f1_list = create_propagation_video(path_to_probefile,
                            starting_f_value=starting_f_value,
                            ending_f_value=ending_f_value,
                            number_of_frames=number_of_frames,
                            jupyter=True)

    play.max = len(image_list)-1
    slider.max = len(image_list)-1

    out = widgets.interactive_output(update_imshow,{'fig':fixed(fig),'ax1':fixed(ax1),'image_list':fixed(image_list),'f1_list':fixed(f1_list),'index':play})
    display(out)


def deploy_interface_fresnel(path_to_probefile,output_dictionary):

    from functools import partial
    output = widgets.Output()


    centered_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',width='100%')
    centered_button_layout = widgets.Layout(flex_flow='column',align_items='flex-end',width='100%',height='50px')

    power   = widgets.BoundedFloatText(value=-4,min=-10,max=0, description='10^n', readout_format='d',disabled=False,layout=centered_box_layout)
    start_f = widgets.FloatText(value=1, description='Start F1', readout_format='d',disabled=False,layout=centered_box_layout)
    end_f   = widgets.FloatText(value=9, description='End F', readout_format='d',disabled=False,layout=centered_box_layout)
    n_frames= widgets.FloatText(value=20, description='#Frames', readout_format='d',disabled=False,layout=centered_box_layout)

    out2 = widgets.interactive_output(update_values,{'n_frames':n_frames,'start_f':start_f,'end_f':end_f,'power':power})
    box1 = widgets.VBox([n_frames, power, start_f,end_f])
    start_ptycho_button = widgets.Button(description=('Propagate Probe'),layout=centered_button_layout)

    """ Fresnel Number box """
    fresnel_box = widgets.FloatText(value=-5e-3, description='Chosen F (float)', readout_format='.3e',disabled=False,layout=centered_box_layout)
    widgets.interactive_output(update_dic,{'output_dictionary':fixed(output_dictionary),'key':fixed('f1'),'boxvalue':fresnel_box})
    box4 = widgets.VBox([start_ptycho_button,fresnel_box])
    box2 = widgets.VBox([box1,box4])

    image_list, f1_list = [np.ones((100,100))], [0]

    play = widgets.Play(
        value=0,
        min=0,
        max=len(image_list)-1,
        step=1,
        interval=300,
        description="Press play",
        disabled=False
    )

    slider = widgets.IntSlider(value=0, min=0, max=len(image_list)-1)
    play_box = widgets.HBox([play, slider])
    widgets.jslink((play, 'value'), (slider, 'value'))
    

    with output:
        fig, ax1 = plt.subplots(figsize=(5,5))
        ax1.imshow(image_list[0],cmap='jet') # initialize
    
    args = (path_to_probefile,starting_f_value,ending_f_value,number_of_frames,play,slider,fig,ax1)
    start_ptycho_button.on_click(partial(on_click_propagate,args=args))

    box3 = widgets.VBox([play_box,output])
    saida = widgets.HBox([box2,box3])

    return fig, ax1, saida

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
    
