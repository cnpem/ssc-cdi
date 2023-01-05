import ipywidgets as widgets
from ipywidgets import fixed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm
import json
import os
import h5py
import sys

from sscIO import io
from sscPimega import pi540D

from .jupyter import slide_and_play
from .ptycho_restauration import restauration_processing_binning, Geometry, Restaurate

def restoration_via_interface(data_path,inputs,apply_flat=True,apply_empty=True,apply_mask=True, save_data=True, preview=True,hdf5_datapath='/entry/data/data'):
    
    n_of_threads, centerx, centery, distance, apply_crop, apply_binning = inputs

    metadata = json.load(open(os.path.join(data_path.rsplit('/',2)[0],'mdata.json')))
    empty_path = os.path.join(data_path.rsplit('/',2)[0],'images','empty.hdf5')
    flat_path = os.path.join(data_path.rsplit('/',2)[0],'images','flat.hdf5')
    mask_path = os.path.join(data_path.rsplit('/',2)[0],'images','mask.hdf5')    
    if 0: # not yet automatic for all techniques; use manual input for now
        distance = float(metadata['/entry/beamline/experiment']["distance"])
    else: 
        distance = distance*1000 # convert from m to mm

    """ Get detector geometry from distance """
    geometry = Geometry(distance)
    


    os.system(f"h5clear -s {data_path}") # gambiarra because file is not closed at the backend!
    os.system(f"h5clear -s {empty_path}") # gambiarra because file is not closed at the backend!

    raw_difpads,_ = io.read_volume(data_path, 'numpy', use_MPI=True, nprocs=n_of_threads)
    print("\tRaw data shape: ", raw_difpads.shape)
    
    """ SIMPLE RESTORATION PRIOR TO CENTER SELECTION. USED TO FIND CENTER """
    restored_full_DP = Restaurate(raw_difpads[0,:,:].astype(np.float32), geometry) # restaurate
    restored_full_DP = restored_full_DP.astype(np.int32)

    """ How the corrections are made prior to restoration:
    
        img[empty > 1] = -1 # Apply empty 
        img = img * np.squeeze(flat) # Apply flatfield
        img[np.abs(mask) ==1] = -1   # Apply Mask
        img = img[cy - hsize:cy + hsize, cx - hsize:cx + hsize] # Center data

    """
    if apply_empty:
        empty = np.asarray(h5py.File(empty_path, 'r')[hdf5_datapath]).squeeze().astype(np.float32)
    else:
        empty = np.zeros_like(raw_difpads[0])

    if apply_flat:
        flat = np.array(h5py.File(flat_path, 'r')[hdf5_datapath][()][0, 0, :, :])
        flat[np.isnan(flat)] = -1
        flat[flat == 0] = 1
    else:
        flat = np.ones_like(raw_difpads[0])
    
    if apply_mask:
        mask = h5py.File(mask_path, 'r')[hdf5_datapath][()][0, 0, :, :]
    else:
        mask  = np.zeros_like(raw_difpads[0])

    
    if preview:
        img = np.ones_like(mask)
        plot1, plot2, plot3 = empty, flat, mask
        empty = np.asarray(h5py.File(empty_path, 'r')[hdf5_datapath]).squeeze().astype(np.float32)
        flat = np.array(h5py.File(flat_path, 'r')[hdf5_datapath][()][0, 0, :, :])
        flat[np.isnan(flat)] = -1
        flat[flat == 0] = 1
        mask = h5py.File(mask_path, 'r')[hdf5_datapath][()][0, 0, :, :]

        img[empty > 1] = -1 # Apply empty 
        img = img * np.squeeze(flat) # Apply flatfield
        img[np.abs(mask) ==1] = -1   # Apply Mask

        fig, ax = plt.subplots(1,4,figsize=(15,5))
        ax[0].imshow(plot1), ax[0].set_title('empty')
        ax[1].imshow(plot2), ax[1].set_title('flat')
        ax[2].imshow(plot3), ax[2].set_title('mask')
        ax[3].imshow(img),   ax[3].set_title('all')        
        plt.show()    
        
    Binning = 4 # standard is 4 for now

    jason = {} # dummy dictionary with dummy values to be used within restoration function 
    jason["DetectorExposure"] = [False,0.15]
    jason["CentralMask"] = [False,5]
    jason["DifpadCenter"] = [centery, centerx]
    

    if apply_crop:
        L = 3072 # PIMEGA540D size
        half_square_side = min(min(centerx,L-centerx),min(centery,L-centery)) # get the biggest size possible such that the restored difpad is still squared
    else:
        half_square_side = 3072*2

    """ Call corrections and restoration """
    print("Correcting and restoring diffraction patterns... ")
    r_params = (Binning, empty, flat, centerx, centery, half_square_side, geometry, mask, jason, apply_crop, apply_binning)
    output, _ = pi540D.backward540D_nonplanar_batch(raw_difpads, distance, n_of_threads, [ half_square_side//2 , half_square_side//2 ], restauration_processing_binning,  r_params, 'only') # Apply empty, flatfield, mask and restore!
    output = output.astype(np.int32)
    print("\tRestored data shape: ", output.shape)



    if save_data:
        savepath = os.path.join(data_path.rsplit('/',5)[0],'proc','recons',data_path.rsplit('/',3)[-3],'restoration',data_path.rsplit('/',2)[-1])
        if not os.path.exists(savepath.rsplit('/',1)[0]):
            os.makedirs(savepath.rsplit('/',1)[0])
        print("Saving data at: ",savepath)
        h5f = h5py.File(savepath, 'w')
        h5f.create_dataset(data_path.rsplit('/',2)[-1][:-5], data=output)
        h5f.close()

    print("Done!")
    print(f"Output data shape {output.shape}. Type: {output.dtype}")
    print(f"Dataset size: {sys.getsizeof(output)/(1e6):.2f} MBs = {sys.getsizeof(output)/(1e9):.2f} GBs")
    return output, restored_full_DP


def plot_DPs_with_slider(data,axis=0):

    colornorm=colors.Normalize(vmin=data.min(), vmax=data.max())
    cmap = 'viridis'
    
    def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title=False,clear_axis=False,cmap=cmap,norm=colors.LogNorm()):
        subplot.clear()
        if bottom == None or right == None:
            if axis == 0:
                subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap=cmap,norm=norm)
            elif axis == 1:
                subplot.imshow(sinogram[top:bottom,frame_number,left:right],cmap=cmap,norm=norm)
            elif axis == 2:
                subplot.imshow(sinogram[top:bottom,left:right,frame_number],cmap=cmap,norm=norm)
        else:
            if axis == 0:
                subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap=cmap,norm=norm)
            elif axis == 1:
                subplot.imshow(sinogram[top:-bottom,frame_number,left:-right],cmap=cmap,norm=norm)
            elif axis == 2:
                subplot.imshow(sinogram[top:-bottom,left:-right,frame_number],cmap=cmap,norm=norm)
        if title == True:
            subplot.set_title(f'#{frame_number}')
        if clear_axis == True:
            subplot.set_xticks([])
            subplot.set_yticks([])    
        figure.canvas.draw_idle()
    
    output = widgets.Output()
    
    with output:
        figure, ax = plt.subplots(dpi=150)
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False
        figure.colorbar(matplotlib.cm.ScalarMappable(norm=colornorm, cmap=cmap))
        plt.show()   

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)

    selection_slider.widget.max, selection_slider.widget.value = data.shape[0] - 1, data.shape[0]//2
    play_control.widget.max =  selection_slider.widget.max
    widgets.interactive_output(update_imshow, {'sinogram':fixed(data),'figure':fixed(figure),'title':fixed(True),'subplot':fixed(ax),'axis':fixed(axis), 'norm':fixed(colors.LogNorm()),'frame_number': selection_slider.widget})    
    box = widgets.VBox([play_box,output])
    return box

def plot_flipped_full_DP(restored_full_DP):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(restored_full_DP,norm=LogNorm()), ax.set_title("Average of DPs")