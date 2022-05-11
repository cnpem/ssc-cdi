import ipywidgets as widgets
from ipywidgets import fixed
import ast 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import json
from tqdm import tqdm

from sscRadon import radon


from .jupyter import call_cmd_terminal, monitor_job_execution, call_and_read_terminal
from .unwrap import unwrap_in_parallel
from .misc import list_files_in_folder

sinogram = np.random.random((2,2,2))

global_dict = {"ibira_data_path": "/ibira/lnls/beamlines/caterete/proposals/20210177/data/ptycho3d/",
               "folders_list": ["microagg_P2_01"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/00000000/proc/recons/microagg_P2_01/object_microagg_P2_01.npy",
               "jupyter_folder":"/ibira/lnls/beamlines/caterete/apps/jupyter-dev/"  , # FIXED PATH FOR BEAMLINE
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_list": [],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,
               "bad_frames_list2": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               
               "wiggle_reference_frame": 0,
               "wiggle_cpus": 32,
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "tomo_n_of_gpus": [0],
               "tomo_threshold" : float(0.0), # max value to be left in reconstructed absorption
               "run_all_tomo_steps":False
}

output_folder = global_dict["sinogram_path"].rsplit('/',1)[0]
print('Output folder: ', output_folder) 

############################################ PROCESSING FUNCTIONS ###########################################################################

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from skimage.morphology import square, erosion, opening, convex_hull_image, dilation
def _operator_T(u):
    d   = 1.0
    uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
    uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
    uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
    uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
    delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
    z = np.sqrt( delta )
    return z

def do_chull(sinogram,invert,tolerance,opening_param,erosion_param,chull_param,frame):
    img = sinogram[frame,:,:] 
    where = _operator_T(img).real
    new = np.copy(img)
    if invert:
        new[ new > 0] = _operator_T(new).real[ img > 0]
    else:
        new[ new < 0] = _operator_T(new).real[ img < 0]

    mask = (np.abs( new - img) < tolerance) * 1.0
    mask2 = opening(mask, square(opening_param))
    mask3 = erosion(mask2, square(erosion_param))
    chull = dilation( convex_hull_image(mask3), square(chull_param) ) # EXPAND CASCA DA MASCARA
    img_masked = np.copy(img * chull)  #nova imagem apenas com o suporte
    # sinogram[frame,:,:] = img_masked
    return new,mask,mask2,mask3,chull,img_masked

def apply_chull_parallel(sinogram,invert=True,tolerance=1e-5,opening_param=10,erosion_param=30,chull_param=50):
    if sinogram.ndim == 2:
        sinogram = np.expand_dims(sinogram, axis=0) # add dummy dimension to get 3d array
    chull_sinogram = np.empty_like(sinogram)
    do_chull_partial = partial(do_chull,sinogram,invert,tolerance,opening_param,erosion_param,chull_param)
    frames = [f for f in range(sinogram.shape[0])]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(do_chull_partial,frames),total=sinogram.shape[0]))
        for counter, result in enumerate(results):
            new,mask,mask2,mask3,chull,img_masked = result
            chull_sinogram[counter,:,:] = img_masked
    return [new,mask,mask2,mask3,chull,img_masked,chull_sinogram]

def sort_frames_by_angle(ibira_path,foldernames):
    rois = []
    counter = -1 
    for folder in foldernames:

        print(f"Sorting data for {folder} folder")

        filepaths, filenames = list_files_in_folder(os.path.join(ibira_path, folder,'positions'), look_for_extension=".txt")
        # filepaths, filenames = list_files_in_folder(os.path.join(ibira_path, folder), look_for_extension="1_001.txt") # use for old file standard

        print('\t # of files in folder:',len(filenames))

        for filepath in filepaths:
            roisname = filepath  
            if roisname == os.path.join(ibira_path,folder, 'positions', folder + '_Ry_positions.txt'): # ignore this file, to use only the positions file inside /positions/ folder
            # if roisname == os.path.join(ibira_path, folder) + '/Ry_positions.txt': # use for old file standard
                continue
            else:
                counter += 1 
                posfile = open(roisname)
                a = 0
                for line in posfile:
                    line = str(line)
                    if a < 1: # get value from first line of the file only
                        angle = line.split(':')[1].split('\t')[0]
                        rois.append([int(counter),float(angle)])
                    a += 1

    rois = np.asarray(rois)
    rois = rois[rois[:,1].argsort(axis=0)]
    return rois 

def reorder_slices_low_to_high_angle(object, rois):
    object_temporary = np.zeros_like(object)

    for k in range(object.shape[0]): # reorder slices from lowest to highest angle
            # print(f'New index: {k}. Old index: {int(rois[k,0])}')
            object_temporary[k,:,:] = object[int(rois[k,0]),:,:] 

    return object_temporary

def regularization(sino, L):
    a = 1
    R = sino.shape[1]
    V = sino.shape[0]
    th = np.linspace(0, np.pi, V, endpoint=False)
    t  = np.linspace(-a, a, R)
    dt = (2*a)/float((R-1))
    wc = 1.0/(2*dt)
    w = np.linspace(-wc, wc, R)
    if 1: # two options
        h = np.abs(w) / (1 + 4 * np.pi * L * (w**2) )
    else:
        h = 1 / (1 + 4 * np.pi * L * (w**2) )
    G = np.fft.fftshift(np.transpose(np.kron(np.ones((V, 1)), h))).T
    B = np.fft.fft(sino, axis=1)
    D = np.fft.ifft(B * G, axis=1).real
    return D

############################################ INTERFACE / GUI ###########################################################################

def write_to_file(tomo_script_path,jsonFile_path,output_path="",slurmFile = 'tomoJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    # Create slurm file
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o ./slurm.out        # Select output path of slurm file

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

python3 {tomo_script_path} {jsonFile_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
"""
    
    with open(slurmFile,'w') as the_file:
        the_file.write(string)
    
    return slurmFile

def call_cmd_terminal(filename,mafalda,remove=False):
    cmd = f'sbatch {filename}'
    terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
    given_jobID = terminal_output.rsplit("\n",1)[0].rsplit(" ",1)[1]
    if remove: # Remove file after call
        cmd = f'rm {filename}'
        subprocess.call(cmd, shell=True)
        
    return given_jobID

def run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32,run_all_steps=False):
    slurm_file = write_to_file(tomo_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

class VideoControl:
    
    def __init__ (self,slider,value,minimum,maximum,step,interval,description):
    
        self.widget = widgets.Play(value=value,
                            min=minimum,
                            max=maximum,
                            step=step,
                            interval=interval,
                            description=description,
                            disabled=False )

        widgets.jslink((self.widget, 'value'), (slider, 'value'))

class Button:

    def __init__(self,description="DESCRIPTION",width="50%",height="50px",icon=""):

        self.button_layout = widgets.Layout(width=width, height=height)
        self.widget = widgets.Button(description=description,layout=self.button_layout,icon=icon)

    def trigger(self,func):
        self.widget.on_click(func)

class Input(object):

    def __init__(self,dictionary,key,description="",layout=None,bounded=(),slider=False):
        
        self.dictionary = dictionary
        self.key = key
        
        if layout == None:
            field_layout = widgets.Layout(align_items='flex-start',width='50%')
        else:
            field_layout = layout
        field_style = {'description_width': 'initial'}
        

        if description == "":
            field_description = f'{key}{str(type(self.dictionary[self.key]))}'
        else:
            field_description = description

        if isinstance(self.dictionary[self.key],bool):
            self.widget = widgets.Checkbox(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],int):
            if bounded == ():
                self.widget = widgets.IntText( description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
            else:
                if slider:
                    self.widget = widgets.IntSlider(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key])
                else:
                    self.widget = widgets.BoundedIntText(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],float):
            if bounded == ():
                self.widget = widgets.FloatText(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
            else:
                self.widget = widgets.BoundedFloatText(min=bounded[0],max=bounded[1],step=bounded[2],description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],list):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],str):
            self.widget = widgets.Text(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],dict):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        
        widgets.interactive_output(self.update_dict_value,{'value':self.widget})

    def update_dict_value(self,value):
        if isinstance(self.dictionary[self.key],list):
            self.dictionary[self.key] = ast.literal_eval(value)
        elif isinstance(self.dictionary[self.key],dict):
            self.dictionary[self.key] = ast.literal_eval(value)
        else:
            self.dictionary[self.key] = value            

def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0):
    subplot.clear()
    if bottom == None or right == None:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap='gray')
        elif axis == 1:
            subplot.imshow(sinogram[top:bottom,frame_number,left:right],cmap='gray')
        elif axis == 2:
            subplot.imshow(sinogram[top:bottom,left:right,frame_number],cmap='gray')
    else:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap='gray')
        elif axis == 1:
            subplot.imshow(sinogram[top:-bottom,frame_number,left:-right],cmap='gray')
        elif axis == 2:
            subplot.imshow(sinogram[top:-bottom,left:-right,frame_number],cmap='gray')    
    figure.canvas.draw_idle()

            
import asyncio
class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator
            
def folders_tab():

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(np.random.random((4,4)),cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()

    def update_fields(ibira_data_path,folders_list,sinogram_path):

        global_dict["ibira_data_path"] = ibira_data_path
        global_dict["folders_list"]    = folders_list
        global_dict["sinogram_path"]   = sinogram_path
    
    def sort_frames(dummy,ibira_path='',foldernames=[''],sinogram_path='',args=()):
        global object

        save_path = sinogram_path.rsplit('/',1)[0]
        print(f'Saving sorted frames to: {save_path}')

        selection_slider,play_control = args
        foldernames = ast.literal_eval(foldernames)

        complex_object_file  = os.path.join(global_dict["sinogram_path"].rsplit('/',1)[0], 'object_' + foldernames[0] + '.npy') #hard coded path
        
        print('Loading sinogram...')
        object = np.load(complex_object_file)
        print('\t Loaded!')

        angles_filename = foldernames[0] + '_ordered_angles.npy'
        rois = sort_frames_by_angle(ibira_path,foldernames)

        object_filename = foldernames[0]  + '_ordered_object.npy'

        object = reorder_slices_low_to_high_angle(object, rois)

        print(f'Extracting sinogram {data_selection.value}...')
        if data_selection.value == 'Magnitude':
            object = np.abs(object)
        elif data_selection.value == "Phase":
            object = np.angle(object)
        print('\t Extraction done!')

        print('Saving angles file...')
        np.save(os.path.join(save_path,angles_filename),rois)
        print('Saving ordered sinogram...')
        np.save(os.path.join(save_path,object_filename), object) 
        print('\tSaved! Sinogram shape: ',object.shape)
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = object.shape[0] - 1, object.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max

        widgets.interactive_output(update_imshow, {'sinogram':fixed(object),'figure':fixed(figure),'subplot':fixed(subplot), 'frame_number': selection_slider.widget})  


    field_layout = widgets.Layout(width='100%')
    ibira_data_path = Input(global_dict,"ibira_data_path",layout=field_layout)
    folders_list    = Input(global_dict,"folders_list",layout=field_layout)
    sinogram_path   = Input(global_dict,"sinogram_path",layout=field_layout)
    widgets.interactive_output(update_fields, {'ibira_data_path':ibira_data_path.widget,'folders_list':folders_list.widget,'sinogram_path':sinogram_path.widget})

    selection_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,10,1),slider=True)
    play_control = VideoControl(selection_slider.widget,selection_slider.widget.value,selection_slider.widget.min,selection_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([selection_slider.widget,play_control.widget])

    sort_button = Button(description="Sort frames",icon="fa-sort-numeric-asc")
    sort_button.trigger(partial(sort_frames,ibira_path=ibira_data_path.widget.value,foldernames=folders_list.widget.value,sinogram_path=sinogram_path.widget.value,args=(selection_slider,play_control)))    

    controls_box = widgets.VBox([ibira_data_path.widget,folders_list.widget,sinogram_path.widget,sort_button.widget,play_box])

    box = widgets.HBox([output,controls_box])

    return box


def crop_tab():

    initial_image = np.ones((100,100)) # dummy
    vertical_max, horizontal_max = initial_image.shape[0]//2, initial_image.shape[1]//2

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()
    
    def load_frames(dummy, args = ()):
        global sinogram
        top_crop, bottom_crop, left_crop, right_crop, select_slider, play_control = args
        print("Loading sinogram")
        sinogram = np.load( os.path.join(global_dict["sinogram_path"].rsplit('/',1)[0],ast.literal_eval(global_dict['folders_list'])[0] + '_ordered_object.npy')) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        select_slider.widget.max, select_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = select_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
      
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': select_slider.widget})


    def save_cropped_sinogram(dummy,args=()):
        top,bottom,left,right = args
        cropped_sinogram = sinogram[:,top.value:-bottom.value,left.value:-right.value]
        print('Saving cropped frames...')
        np.save(os.path.join(output_folder,'cropped_sinogram.npy'),cropped_sinogram)
        print('\t Saved!')

    top_crop      = Input(global_dict,"top_crop"   ,description="Top",   bounded=(0,vertical_max,1),  slider=True)
    bottom_crop   = Input(global_dict,"bottom_crop",description="Bottom",bounded=(1,vertical_max,1),  slider=True)
    left_crop     = Input(global_dict,"left_crop"  ,description="Left",  bounded=(0,horizontal_max,1),slider=True)
    right_crop    = Input(global_dict,"right_crop" ,description="Right", bounded=(1,horizontal_max,1),slider=True)
    select_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,100,1),slider=True)

    play_control = VideoControl(select_slider.widget,select_slider.widget.value,select_slider.widget.min,select_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([select_slider.widget,play_control.widget])
    
    load_frames_button  = Button(description="Load Frames",width='50%', height='50px',icon='folder-open-o')
    args = (top_crop, bottom_crop, left_crop, right_crop, select_slider, play_control)
    load_frames_button.trigger(partial(load_frames,args=args))

    save_cropped_frames_button = Button(description="Save cropped frames",width='70%', height='50px',icon='fa-floppy-o') 
    args2 = (top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget)
    save_cropped_frames_button.trigger(partial(save_cropped_sinogram,args=args2))
    
    sliders_box = widgets.VBox([load_frames_button.widget,play_box,top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget,save_cropped_frames_button.widget])
    box = widgets.HBox([sliders_box,output])
    return box


def unwrap_tab():
    
    global unwrapped_sinogram
    
    output = widgets.Output()
    with output:
        figure_unwrap, subplot_unwrap = plt.subplots(1,2)
        subplot_unwrap[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap[1].imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap[0].set_title('Cropped image')
        subplot_unwrap[1].set_title('Unwrapped image')
        figure_unwrap.canvas.draw_idle()
        figure_unwrap.canvas.header_visible = False 
        plt.show()
    
    def phase_unwrap(dummy):
        global unwrapped_sinogram
        print('Performing phase unwrap...')
        unwrapped_sinogram = unwrap_in_parallel(cropped_sinogram,iterations_slider.widget.value,non_negativity=non_negativity_checkbox.widget.value,remove_gradient = gradient_checkbox.widget.value)
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap[1]), 'frame_number': selection_slider.widget})    
        

    def format_chull_plot(figure,subplots,frame_number):
        subplots[0].set_title(f'Frame #{frame_number}')
        subplots[1].set_title('Unwrapped')

        for subplot in subplots.reshape(-1):
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 

    def load_cropped_frames(dummy,args=()):
        global cropped_sinogram
        selection_slider, play_control = args
        print('Loading cropped sinogram...')
        cropped_sinogram = np.load(os.path.join(output_folder,'cropped_sinogram.npy'))
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = cropped_sinogram.shape[0] - 1, cropped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(cropped_sinogram),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap[0]), 'frame_number': selection_slider.widget})    
        widgets.interactive_output(format_chull_plot, {'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap), 'frame_number': selection_slider.widget})    

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames)
        global unwrapped_sinogram
        unwrapped_sinogram = np.empty_like(cropped_sinogram)
        cropped_sinogram[bad_frames,:,:]   = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        unwrapped_sinogram[bad_frames,:,:] = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        print('\t Done!')

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_list1,bad_frames_list2):
        bad_frames_listA = ast.literal_eval(bad_frames_list1)
        bad_frames_listB = ast.literal_eval(bad_frames_list2)
        global bad_frames
        bad_frames = bad_frames_listA + bad_frames_listB # concatenate lists
   
    def save_sinogram(dummy):
        print('Saving unwrapped sinogram...')
        np.save(os.path.join(output_folder,'unwrapped_sinogram.npy'),unwrapped_sinogram)
        print('\tSaved sinogram at: ',os.path.join(output_folder,'unwrapped_sinogram.npy'))

    def update_frame_time(play_control,time_per_frame):
        play_control.widget.interval = time_per_frame

    load_cropped_frames_button = Button(description="Load cropped frames",width='50%', height='50px',icon='folder-open-o')

    bad_frames_list = Input(global_dict,"bad_frames_list", description = 'Bad frames',layout=widgets.Layout(align_items='flex-start',width='80%'))
    bad_frames_list2 = Input(global_dict,"bad_frames_list2",description='Bad Frames after Unwrap',layout=widgets.Layout(align_items='flex-start',width='80%'))
    widgets.interactive_output(update_lists,{ "bad_frames_list1":bad_frames_list.widget,"bad_frames_list2":bad_frames_list2.widget})
    
    iterations_slider = Input(global_dict,"unwrap_iterations",bounded=(0,10,1),slider=True, description='Iterations')
    non_negativity_checkbox = Input(global_dict,"unwrap_non_negativity",layout=widgets.Layout(align_items='flex-start',width='40%'),description='Non-negativity')
    gradient_checkbox = Input(global_dict,"unwrap_gradient_removal",layout=widgets.Layout(align_items='flex-start',width='40%'),description='Gradient')
    preview_unwrap_button = Button(description="Preview unwrap",width='50%', height='50px',icon='play')
    preview_unwrap_button.trigger(phase_unwrap)
    
    selection_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,10,1),slider=True)
    play_control = VideoControl(selection_slider.widget,selection_slider.widget.value,selection_slider.widget.min,selection_slider.widget.max,1,300,"Play Button")
    frame_time = Input({"dummy_key":300},"dummy_key",description="Time per frame [ms]",layout=widgets.Layout(width='60%'))
    widgets.interactive_output(update_frame_time, {'play_control':fixed(play_control),'time_per_frame':frame_time.widget})
    play_box = widgets.HBox([selection_slider.widget,widgets.VBox([play_control.widget,frame_time.widget])])
    
    args = (selection_slider,play_control)
    load_cropped_frames_button.trigger(partial(load_cropped_frames,args=args))

    correct_bad_frames_button = Button(description='Correct Bad Frames',icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    save_unwrapped_button = Button(description="Save unwrapped frames",icon='fa-floppy-o') 
    save_unwrapped_button.trigger(save_sinogram)
    
    unwrap_params_box = widgets.VBox([iterations_slider.widget,non_negativity_checkbox.widget,gradient_checkbox.widget])
    controls_box = widgets.VBox([load_cropped_frames_button.widget,correct_bad_frames_button.widget,preview_unwrap_button.widget,save_unwrapped_button.widget,play_box, unwrap_params_box,bad_frames_list.widget,bad_frames_list2.widget])
    plot_box = widgets.VBox([output])
        
    box = widgets.HBox([controls_box,plot_box])
    
    return box


def chull_tab():
    
    def format_chull_plot(figure,subplots):
        subplots[0,0].set_title('Original')
        subplots[0,1].set_title('Threshold')
        subplots[0,2].set_title('Opening')
        subplots[1,0].set_title('Erosion')
        subplots[1,1].set_title('Convex Hull')
        subplots[1,2].set_title('Masked Image')

        for subplot in subplots.reshape(-1):
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
    
    output = widgets.Output()
    
    with output:
        figure, subplots = plt.subplots(2,3)
        subplots[0,0].imshow(np.random.random((4,4)),cmap='gray')
        subplots[0,1].imshow(np.random.random((4,4)),cmap='gray')
        subplots[0,2].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,0].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,1].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,2].imshow(np.random.random((4,4)),cmap='gray')
        format_chull_plot(figure,subplots)
        plt.show()
    
    def load_unwrapped_sinogram(dummy,args=()):
        global unwrapped_sinogram
        print('Loading unwrapped sinogram...')
        unwrapped_sinogram = np.load(os.path.join(output_folder,'unwrapped_sinogram.npy'))
        print('\t Loaded!')
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = unwrapped_sinogram.shape[0] - 1, unwrapped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplots[0,0]), 'frame_number': selection_slider.widget})    
        format_chull_plot(figure,subplots)

    def preview_cHull(dummy,args=()):
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram[selection_slider.widget.value,:,:],invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        cHull_sinogram = output_list[-1]
        selection_slider.widget.max, selection_slider.widget.value = cHull_sinogram.shape[0] - 1, cHull_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        for subplot, image in zip(subplots.reshape(-1),output_list[0::]):
            image = np.expand_dims(image, axis=0)
            widgets.interactive_output(update_imshow, {'sinogram':fixed(image),'figure':fixed(figure),'subplot':fixed(subplot), 'frame_number': selection_slider.widget})    
        format_chull_plot(figure,subplots)
        print('\tDone with convex hull...')

    def complete_cHull(dummy,args=()):
        print('Applying complete Convex Hull...')
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram,invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        cHull_sinogram = output_list[-1]
        print('Saving cHull sinogram...')
        np.save(os.path.join(output_folder,'chull_sinogram.npy'),cHull_sinogram)
        print('\tSaved!')
        

    selection_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,10,1),slider=True)
    play_control = VideoControl(selection_slider.widget,selection_slider.widget.value,selection_slider.widget.min,selection_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([selection_slider.widget,play_control.widget])
    
    load_button = Button(description="Load unwrapped sinogram",icon='folder-open-o')
    load_button.trigger(partial(load_unwrapped_sinogram,args=(selection_slider,play_control)))

    invert_checkbox = Input(global_dict,"chull_invert",    description='Invert')
    tolerance       = Input(global_dict,"chull_tolerance", description='Threshold')
    opening_slider  = Input(global_dict,"chull_opening",   description="Opening",     bounded=(1,100,1),slider=True)
    erosion_slider  = Input(global_dict,"chull_erosion",   description="Erosion",     bounded=(1,100,1),slider=True)
    param_slider    = Input(global_dict,"chull_param",     description="Convex Hull", bounded=(1,200,1),slider=True)
    
    preview_button = Button(description="Convex Hull Preview",icon='play')
    preview_button.trigger(partial(preview_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    start_button = Button(description="Do complete Convex Hull",icon='play')
    start_button.trigger(partial(complete_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    controls0 = widgets.VBox([invert_checkbox.widget,tolerance.widget,opening_slider.widget,erosion_slider.widget,param_slider.widget])
    controls_box = widgets.VBox([load_button.widget,preview_button.widget,start_button.widget,play_box,controls0])
    
    box = widgets.HBox([controls_box,output])
    
    return box


def wiggle_tab():
    
    def format_wiggle_plot(figure,subplots):
        subplots[0,0].set_title('Pre-wiggle')
        subplots[0,1].set_title('Post_wiggle')
        subplots[0,0].set_ylabel('XY')
        subplots[1,0].set_ylabel('XZ')

        for subplot in subplots.reshape(-1):
            subplot.set_aspect('auto')
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
        figure.tight_layout()
    
    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5))
        subplot.imshow(np.random.random((4,4)),cmap='gray')
        subplot.set_title('Reference Frame')
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(2,2)
        subplot2[0,0].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[0,1].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[1,0].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[1,1].imshow(np.random.random((4,4)),cmap='gray')
        format_wiggle_plot(figure2,subplot2)
        plt.show()
    
    def load_sinogram(dummy,args=()):
        selection_slider, play_control,sinogram_selection = args
        
        if sinogram_selection.value == "unwrapped":
            file = 'unwrapped_sinogram.npy'
        elif sinogram_selection.value == "convexHull":
            file = 'chull_sinogram.npy'

        global sinogram
        print('Loading ...: ',file)
        sinogram = np.load(os.path.join(output_folder,file))
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'frame_number': selection_slider.widget})    
    
    def update_imshow_with_format(sinogram,figure,subplot,frame_number,axis):
        update_imshow(sinogram,figure,subplot,frame_number,axis=axis)
        format_wiggle_plot(figure2,subplot2)

    global wiggled_sinogram
    def start_wiggle(dummy,args=()):
        sinogram_selection,sinogram_slider1,sinogram_slider2,cpus_slider,ref_frame_slider = args
        
        print("Starting wiggle...")
        wiggled_sinogram = radon.get_wiggle( sinogram,  'vertical', cpus_slider.widget.value, ref_frame_slider.widget.value)
        wiggled_sinogram = radon.get_wiggle( wiggled_sinogram, 'horizontal', cpus_slider.widget.value, ref_frame_slider.widget.value)
        print("\t Wiggle done!")
        sinogram_slider1.widget.max, sinogram_slider1.widget.value = wiggled_sinogram.shape[1] - 1, wiggled_sinogram.shape[1]//2
        sinogram_slider2.widget.max, sinogram_slider2.widget.value = wiggled_sinogram.shape[2] - 1, wiggled_sinogram.shape[2]//2
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[0,0]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[1,0]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0,1]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[1,1]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        print("Saving wiggle sinogram...")
        np.save(os.path.join(output_folder,'wiggle_sinogram.npy'),wiggled_sinogram)
        print("\t Saved!")

    ref_frame_slider = Input(global_dict,"wiggle_reference_frame", description="Reference Frame", bounded=(0,sinogram.shape[0],1),slider=True)
    cpus_slider      = Input(global_dict,"wiggle_cpus", description="# of CPUs", bounded=(1,128,1),slider=True)

    wiggle_button = Button(description='Perform Wiggle',icon='play')
    
    play_control = VideoControl(ref_frame_slider.widget,ref_frame_slider.widget.value,ref_frame_slider.widget.min,ref_frame_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([ref_frame_slider.widget,play_control.widget])
    
    sinogram_selection = widgets.RadioButtons(options=['unwrapped', 'convexHull'], value='unwrapped', layout={'width': 'max-content'},description='Sinogram to import:',disabled=False)
    sinogram_slider1   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Y", bounded=(1,10,1),slider=True)
    sinogram_slider2   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Z", bounded=(1,10,1),slider=True)
    
    args = (ref_frame_slider, play_control,sinogram_selection)
    load_button = Button(description="Load sinogram",icon='folder-open-o')
    load_button.trigger(partial(load_sinogram,args=(ref_frame_slider,play_control,sinogram_selection)))
    
    args2 = (sinogram_selection,sinogram_slider1,sinogram_slider2,cpus_slider,ref_frame_slider)
    wiggle_button.trigger(partial(start_wiggle,args=args2))

    controls = widgets.VBox([sinogram_selection,load_button.widget,play_box,cpus_slider.widget,wiggle_button.widget,sinogram_slider1.widget,sinogram_slider2.widget])
    box = widgets.HBox([controls,output,output2])
    
    return box


def tomo_tab():
    
    def format_tomo_plot(figure,subplots):
        subplots[0].set_title('YZ')
        subplots[1].set_title('XZ')
        subplots[2].set_title('XY')

        for subplot in subplots.reshape(-1):
            subplot.set_aspect('equal')
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
        figure.tight_layout()
        
    def run_tomo(dummy,args=()):
        algo_dropdown,iter_slider,gpus_field,filename_field, cpus_field,jobname_field,queue_field, tomo_selection = args

        if tomo_selection.value == 'Full Recon':
            global_dict["run_all_tomo_steps"] = True
        elif tomo_selection.value == 'Only Tomo': 
            global_dict["run_all_tomo_steps"] = False

        tomo_script_path = '~/ssc-cdi/bin/sscptycho_raft.py' # NEED TO CHANGE FOR EACH USER? 
        output_path = global_dict["jupyter_folder"] 
        
        slurm_filepath = os.path.join(output_path,'tomo_job.srm')

        jsonFile_path = os.path.join(output_path,'user_input_tomo.json')

        n_gpus = len(ast.literal_eval(gpus_field.widget.value))
        run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path=output_path,slurmFile = slurm_filepath,  jobName=jobname_field.widget.value,queue=queue_field.widget.value,gpus=n_gpus,cpus=cpus_field.widget.value,run_all_steps=global_dict["run_all_tomo_steps"])

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(1,3)
        subplot[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot[1].imshow(np.random.random((4,4)),cmap='gray')
        subplot[2].imshow(np.random.random((4,4)),cmap='gray')
        format_tomo_plot(figure,subplot)
        plt.show()
    
    
    field_layout    = widgets.Layout(align_items='flex-start',width='90%')
    reg_checkbox    = Input(global_dict,"tomo_regularization",description = "Apply Regularization")
    reg_param       = Input(global_dict,"tomo_regularization_param",description = "Regularization Parameter",layout=field_layout)
    iter_slider     = Input(global_dict,"tomo_iterations",description = "Iterations", bounded=(1,200,2),slider=True)
    cpus_field      = Input(global_dict,"wiggle_cpus",description = "# of CPUs",layout=field_layout)
    gpus_field      = Input(global_dict,"tomo_n_of_gpus",description = "GPUs list",layout=field_layout)
    queue_field     = Input({"dummy_str":'cat-proc'},"dummy_str",description = "Machine Queue",layout=field_layout)
    jobname_field   = Input({"dummy_str":'myJobName'},"dummy_str",description = "Slurm Job Name",layout=field_layout)
    filename_field  = Input({"dummy_str":'reconstruction3Dphase'},"dummy_str",description = "Output Filename",layout=field_layout)
    tomo_threshold  = Input(global_dict,"tomo_threshold",description = "Value threshold for recon",layout=field_layout)
    tomo_sliceX     = Input({"dummy_key":1},"dummy_key", description="Slice X", bounded=(1,10,1),slider=True)
    tomo_sliceY     = Input({"dummy_key":1},"dummy_key", description="Slice Y", bounded=(1,10,1),slider=True)
    tomo_sliceZ     = Input({"dummy_key":1},"dummy_key", description="Slice Z", bounded=(1,10,1),slider=True)

    algo_dropdown = widgets.Dropdown(options=[('EEM', 1), ('EM', 2), ('ART', 3),('FBP', 3)], value=1,description='Algorithm:')

    tomo_selection = widgets.RadioButtons(options=['Only Tomo', 'Full Recon'], value='Only Tomo', layout={'width': 'max-content'},description='Recon type:',disabled=False)

    start_tomo = Button(description="Start",icon='play')
    args = algo_dropdown,iter_slider,gpus_field,filename_field,cpus_field,jobname_field,queue_field, tomo_selection
    start_tomo.trigger(partial(run_tomo,args=args))
    save_thresholded_tomo = Button(description="Save thresholded tomo",icon='play')
   
    def save_on_click(dummy):
        json_filepath = os.path.join(global_dict["jupyter_folder"],'user_input_tomo.json') #INPUT
        with open(json_filepath, 'w') as file:
            json.dump(global_dict, file)
    save_dict_button  = Button(description="Save JSON",width='90%', height='50px',icon='fa-floppy-o')
    save_dict_button.trigger(save_on_click)    
    

    start_box = widgets.HBox([start_tomo.widget,tomo_selection])
    slurm_box = widgets.VBox([cpus_field.widget,gpus_field.widget,queue_field.widget,jobname_field.widget])
    controls = widgets.VBox([algo_dropdown,reg_checkbox.widget,reg_param.widget,iter_slider.widget,slurm_box,save_dict_button.widget,start_box,tomo_sliceX.widget,tomo_sliceY.widget,tomo_sliceZ.widget,tomo_threshold.widget,save_thresholded_tomo.widget])
    box = widgets.HBox([controls,output])
    
    return box 


def deploy_tabs(mafalda_session,tab1=folders_tab(),tab2=crop_tab(),tab3=unwrap_tab(),tab4=chull_tab(),tab5=wiggle_tab(),tab6=tomo_tab()):
    
    global mafalda
    mafalda = mafalda_session
    
    children_dict = {
    "Select Folders" : tab1,
    "Cropping"       : tab2,
    "Phase Unwrap"   : tab3,
    "Convex Hull"    : tab4,
    "Wiggle"         : tab5,
    "Tomography"     : tab6}
    
    def load_json(dummy,dictionary={}):
        template_dict = {"ibira_data_path": "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/00000000/data/ptycho2d/",
               "folders_list": ["SS61"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/00000000/proc/recons/SS61/phase_microagg_P2_01.npy",
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_list": [],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,
               "bad_frames_list2": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               
               "wiggle_reference_frame": 0,
               "wiggle_cpus": 32,
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "tomo_n_of_gpus": [0],
               "tomo_threshold" : float(0.0), # max value to be left in reconstructed absorption
               "run_all_tomo_steps":False}
    
        for key in template_dict:
            dictionary[key] = template_dict[key]
    
    

    load_json_button  = Button(description="Load JSON template",width='50%', height='50px',icon='folder-open-o')
    load_json_button.trigger(partial(load_json,dictionary=global_dict))
    
    
    global data_selection
    data_selection = widgets.RadioButtons(options=['Magnitude', 'Phase'], value='Phase', layout={'width': '30%'},description='Visualize',disabled=False)

    box = widgets.HBox([data_selection,load_json_button.widget])
    display(box)
    
    
    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return tab, global_dict  

if __name__ == "__main__":

    pass