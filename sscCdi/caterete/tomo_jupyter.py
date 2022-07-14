import ipywidgets as widgets
from ipywidgets import fixed
import ast 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os, time
import json
from skimage.io import imsave
import asyncio
from functools import partial
import subprocess

from sscRadon import radon
from .unwrap import unwrap_in_parallel
from .tomo_processing import angle_mesh_organize, tomography, apply_chull_parallel, sort_frames_by_angle, reorder_slices_low_to_high_angle, equalize_frames_parallel
from .jupyter import call_and_read_terminal, monitor_job_execution, call_cmd_terminal, VideoControl, Button, Input, update_imshow

global sinogram
sinogram = np.random.random((2,2,2)) # dummy sinogram

""" Standard folders definitions"""
if 1: # paths for beamline use
    tomo_script_path    = '/ibira/lnls/beamlines/caterete/apps/ssc-cdi/bin/sscptycho_raft.py' # path with python script to run
else: # paths for GCC tests       
    tomo_script_path = '~/ssc-cdi/bin/sscptycho_raft.py' # NEED TO CHANGE FOR EACH USER? 

""" Standard dictionary definition """
global_dict = {"jupyter_folder":"/ibira/lnls/beamlines/caterete/apps/jupyter/", # FIXED PATH FOR BEAMLINE

               "ibira_data_path": "/ibira/lnls/beamlines/caterete/proposals/20210177/data/ptycho3d/",
               "folders_list": ["microagg_P2_01"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/proc/recons/microagg_P2_01/object_microagg_P2_01.npy",

               "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "ConvexHull":1, "Wiggle":1, "Tomo":1 }, # select steps when performing full recon
               "contrast_type": "Phase", # Phase or Absolute

               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,

               "bad_frames_before_unwrap": [7,20,36,65,94,123,152,181,210,239,268,296,324],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,

                "equalize_invert":False,
                "equalize_gradient":1,
                "equalize_outliers":1,
                "equalize_global_offset":False,
                "equalize_local_offset":[0,slice(0,None),slice(0,None)],

               "bad_frames_before_cHull": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               

               "bad_frames_before_wiggle": [],
               "wiggle_reference_frame": 0,
               "CPUs": 32,
              
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "GPUs": [0],
               "tomo_threshold" : float(100.0), # max value to be left in reconstructed matrix
}


""" Standard styling definitions """
standard_border='1px none black'
vbar = widgets.HTML(value="""<div style="border-left:2px solid #000;height:500px"></div>""")
vbar2 = widgets.HTML(value="""<div style="border-left:2px solid #000;height:1000px"></div>""")
hbar = widgets.HTML(value="""<hr class="solid" 2px #000>""")
hbar2 = widgets.HTML(value="""<hr class="solid" 2px #000>""")
slider_layout = widgets.Layout(width='90%')
items_layout = widgets.Layout( width='90%',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
checkbox_layout = widgets.Layout( width='150px',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
buttons_layout = widgets.Layout( width='90%',height="40px")     # override the default width of the button to 'auto' to let the button grow
center_all_layout = widgets.Layout(align_items='center',width='100%',border=standard_border) #align_content='center',justify_content='center'
box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
sliders_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
style = {'description_width': 'initial'}

def get_box_layout(width,flex_flow='column',align_items='center',border=standard_border):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)

############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################


def update_paths(global_dict,dummy1,dummy2):
    # dummy variable is used to trigger update
    global_dict["output_folder"] = global_dict["sinogram_path"].rsplit('/',1)[0]
    # global_dict["contrast_type"] = data_selection.value
    
    if type(global_dict["folders_list"]) == type([1,2]): # correct data type of this input
        pass # if list
    else: # if string
        global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"])

    global_dict["complex_object_filepath"]             = os.path.join(global_dict["output_folder"],'object_' + global_dict["folders_list"][0] + '.npy')
    global_dict["ordered_angles_filepath"]             = os.path.join(global_dict["output_folder"],global_dict["folders_list"][0] + '_ordered_angles.npy')
    global_dict["ordered_object_filepath"]             = os.path.join(global_dict["output_folder"],global_dict["folders_list"][0] + '_ordered_object.npy')
    global_dict["reconstruction_thresholded_filepath"] = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_' + global_dict["folders_list"][0] + '_reconstruction3D_' + global_dict["tomo_algorithm"] + '_thresholded.npy')
    global_dict["reconstruction_filepath"]             = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_' + global_dict["folders_list"][0] + '_reconstruction3D_' + global_dict["tomo_algorithm"] + '.npy')
    global_dict["cropped_sinogram_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_cropped_sinogram.npy')
    global_dict["unwrapped_sinogram_filepath"]         = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_unwrapped_sinogram.npy')
    global_dict["equalized_sinogram_filepath"]         = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_equalized_sinogram.npy')
    global_dict["chull_sinogram_filepath"]             = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_chull_sinogram.npy')
    global_dict["wiggle_sinogram_filepath"]            = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_wiggle_sinogram.npy')
    global_dict["projected_angles_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["ordered_angles_filepath"][:-4]+'_projected.npy')
    return global_dict

def write_slurm_file(tomo_script_path,jsonFile_path,output_path="",slurmFile = 'tomoJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
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

def run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    slurm_file = write_slurm_file(tomo_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

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

def slide_and_play(slider_layout=slider_layout,label="",description="",frame_time_milisec = 0):

    def update_frame_time(play_control,time_per_frame):
        play_control.widget.interval = time_per_frame

    selection_slider = Input({"dummy_key":1},"dummy_key",description=description, bounded=(0,100,1),slider=True,layout=widgets.Layout(width='max-width'))
    play_control = VideoControl(selection_slider,1,100,"Play Button")

    pbox = widgets.Box([play_control.widget],layout=get_box_layout('max-width'))

    if frame_time_milisec != 0:
        frame_time = Input({"dummy_key":frame_time_milisec},"dummy_key",description="Time/frame [ms]",layout=widgets.Layout(width='160px'))
        widgets.interactive_output(update_frame_time, {'play_control':fixed(play_control),'time_per_frame':frame_time.widget})
        play_box = widgets.HBox([selection_slider.widget,widgets.Box([pbox,frame_time.widget],layout=get_box_layout('max-width'))])
    else:
        play_box = widgets.HBox([selection_slider.widget, play_control.widget])

    if label != "":
        play_label = widgets.HTML(f'<b><font size=4.9px>{label}</b>' )
        play_box = widgets.VBox([play_label,play_box])

    return play_box, selection_slider,play_control

def update_gpu_limits(machine_selection):

    if machine_selection == 'Cluster':
        gpus_slider.widget.value = 0
        gpus_slider.widget.max = 4
    elif machine_selection == 'Local':
        gpus_slider.widget.value = 0
        gpus_slider.widget.max = 1

def update_cpus_gpus(cpus,gpus):
    global_dict["CPUs"] = cpus

    if machine_selection.value == 'Cluster':
        if gpus == 0:
            global_dict["GPUs"] = []
        elif gpus == 1:
            global_dict["GPUs"] = [0] 
        elif gpus == 2:
            global_dict["GPUs"] = [0,1]
        elif gpus == 3:
            global_dict["GPUs"] = [0,1,2]
        elif gpus == 4:
            global_dict["GPUs"] = [0,1,2,3]
    elif machine_selection.value == 'Local':
        if gpus == 0:
            global_dict["GPUs"] = []
        elif gpus == 1:
            global_dict["GPUs"] = [5] 
    else:
        print('You can only use 1 GPU to run in the local machine!')
############################################ INTERFACE / GUI : TABS ###########################################################################
            
def folders_tab():

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5))
        subplot.imshow(np.random.random((4,4)),cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(figsize=(5,5))
        subplot2.imshow(np.random.random((4,4)),cmap='gray')
        figure2.canvas.header_visible = False 
        plt.show()


    def update_fields(ibira_data_path,folders_list,sinogram_path):
        global_dict["ibira_data_path"] = ibira_data_path
        global_dict["folders_list"]    = folders_list
        global_dict["sinogram_path"]   = sinogram_path


    def load_sinogram(dummy):
        global object

        print('Loading sinogram: ',global_dict["complex_object_filepath"])
        object = np.load(global_dict["complex_object_filepath"])
        print('\t Loaded!')

        print(f'Extracting sinogram {data_selection.value}...')
        global_dict["contrast_type"] = data_selection.value
        if data_selection.value == 'Magnitude':
            object = np.abs(object)
        elif data_selection.value == "Phase":
            object = np.angle(object)
        print('\t Extraction done!')

        selection_slider.widget.max, selection_slider.widget.value = object.shape[0] - 1, object.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(object),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed(True), 'frame_number': selection_slider2.widget})  


    def sort_frames(dummy):
        global object

        rois = sort_frames_by_angle(ibira_data_path.widget.value,global_dict["folders_list"])

        object = reorder_slices_low_to_high_angle(object, rois)

        print('Saving angles file: ',global_dict["ordered_angles_filepath"])
        np.save(global_dict["ordered_angles_filepath"],rois)
        print('Saving ordered sinogram: ', global_dict["ordered_object_filepath"])
        np.save(global_dict["ordered_object_filepath"], object) 
        print('\tSaved! Sinogram shape: ',object.shape)
        selection_slider.widget.max, selection_slider.widget.value = object.shape[0] - 1, object.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max

        widgets.interactive_output(update_imshow, {'sinogram':fixed(object),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})  


    ibira_data_path = Input(global_dict,"ibira_data_path",layout=items_layout,description='Ibira Datapath (str)')
    folders_list    = Input(global_dict,"folders_list",layout=items_layout,description='Ibira Datafolders (list)')
    sinogram_path   = Input(global_dict,"sinogram_path",layout=items_layout,description='Ptycho sinogram path (str)')
    widgets.interactive_output(update_fields, {'ibira_data_path':ibira_data_path.widget,'folders_list':folders_list.widget,'sinogram_path':sinogram_path.widget})
    widgets.interactive_output(update_paths,{'global_dict':fixed(global_dict),'dummy1':sinogram_path.widget,'dummy2':folders_list.widget})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector (Angle)")
    play_box2, selection_slider2,play_control2 = slide_and_play(label="Frame Selector (Time)")

    load_button = Button(description="Load Data",layout=buttons_layout, icon='folder-open-o')
    load_button.trigger(load_sinogram)

    sort_button = Button(description="Sort frames",layout=buttons_layout, icon="fa-sort-numeric-asc")
    sort_button.trigger(sort_frames)

    controls_box = widgets.Box(children=[load_button.widget,play_box2,sort_button.widget,play_box], layout=get_box_layout('500px',align_items='center'))

    paths_box = widgets.VBox([ibira_data_path.widget,folders_list.widget,sinogram_path.widget])
    box = widgets.HBox([controls_box,vbar,output2,output])
    box = widgets.VBox([paths_box,box])

    return box

def crop_tab():

    initial_image = np.ones((5,5)) # dummy
    vertical_max, horizontal_max = initial_image.shape[0]//2, initial_image.shape[1]//2

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5))
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()

    
    def load_frames(dummy, args = ()):
        global sinogram
        top_crop, bottom_crop, left_crop, right_crop, selection_slider, play_control = args
        
        print("Loading sinogram from: ",global_dict["ordered_object_filepath"] )
        sinogram = np.load(global_dict["ordered_object_filepath"] ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = selection_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
      
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})


    def save_cropped_sinogram(dummy,args=()):
        top,bottom,left,right = args
        cropped_sinogram = sinogram[:,top.value:-bottom.value,left.value:-right.value]
        print('Saving cropped frames...')
        if np.isnan(cropped_sinogram).any():
            print("NaN values were found. Substituting by 0 before save!")
            cropped_sinogram = np.where(np.isnan(cropped_sinogram),0,cropped_sinogram)
        np.save(global_dict['cropped_sinogram_filepath'],cropped_sinogram)
        print('\t Saved!')

    top_crop      = Input(global_dict,"top_crop"   ,description="Top",   bounded=(0,vertical_max,1),  slider=True,layout=slider_layout)
    bottom_crop   = Input(global_dict,"bottom_crop",description="Bottom",bounded=(1,vertical_max,1),  slider=True,layout=slider_layout)
    left_crop     = Input(global_dict,"left_crop"  ,description="Left",  bounded=(0,horizontal_max,1),slider=True,layout=slider_layout)
    right_crop    = Input(global_dict,"right_crop" ,description="Right", bounded=(1,horizontal_max,1),slider=True,layout=slider_layout)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    args = (top_crop, bottom_crop, left_crop, right_crop, selection_slider, play_control)
    load_frames_button.trigger(partial(load_frames,args=args))

    save_cropped_frames_button = Button(description="Save cropped frames",layout=buttons_layout,icon='fa-floppy-o') 
    args2 = (top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget)
    save_cropped_frames_button.trigger(partial(save_cropped_sinogram,args=args2))
    
    buttons_box = widgets.Box([load_frames_button.widget,save_cropped_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar,output])
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
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure_unwrap),'title':fixed(True),'subplot':fixed(subplot_unwrap[1]), 'frame_number': selection_slider.widget})    
        

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
        cropped_sinogram = np.load(global_dict["cropped_sinogram_filepath"])
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = cropped_sinogram.shape[0] - 1, cropped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(cropped_sinogram),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap[0]),'title':fixed(True), 'frame_number': selection_slider.widget})    
        widgets.interactive_output(format_chull_plot, {'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap), 'frame_number': selection_slider.widget})    

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames)
        global unwrapped_sinogram
        unwrapped_sinogram = np.empty_like(cropped_sinogram)
        cropped_sinogram[bad_frames,:,:]   = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        unwrapped_sinogram[bad_frames,:,:] = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        print('\t Done!')

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_list1):
        global bad_frames
        bad_frames = ast.literal_eval(bad_frames_list1)

    def save_sinogram(dummy):
        global unwrapped_sinogram
        if np.isnan(unwrapped_sinogram).any() == True:
            print('Removing NaN values from unwrapped sinogram...')
            unwrapped_sinogram = np.where(np.isnan(unwrapped_sinogram),0,unwrapped_sinogram)

        print('Saving unwrapped sinogram...')
        np.save(global_dict["unwrapped_sinogram_filepath"] ,unwrapped_sinogram)
        print('\tSaved sinogram at: ',global_dict["unwrapped_sinogram_filepath"] )


    load_cropped_frames_button = Button(description="Load cropped frames",layout=buttons_layout,icon='folder-open-o')

    bad_frames_before_unwrap  = Input(global_dict,"bad_frames_before_unwrap", description = 'Bad frames',layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_list1":bad_frames_before_unwrap.widget})
    
    iterations_slider = Input(global_dict,"unwrap_iterations",bounded=(0,10,1),slider=True, description='Unwrap Iterations',layout=slider_layout)
    non_negativity_checkbox = Input(global_dict,"unwrap_non_negativity",layout=items_layout,description='Non-negativity')
    gradient_checkbox = Input(global_dict,"unwrap_gradient_removal",layout=items_layout,description='Gradient')
    preview_unwrap_button = Button(description="Perform Unwrap",layout=buttons_layout,icon='play')
    preview_unwrap_button.trigger(phase_unwrap)
    
    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)

    args = (selection_slider,play_control)
    load_cropped_frames_button.trigger(partial(load_cropped_frames,args=args))

    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    save_unwrapped_button = Button(description="Save unwrapped frames",layout=buttons_layout,icon='fa-floppy-o') 
    save_unwrapped_button.trigger(save_sinogram)
    
    unwrap_params_box = widgets.Box([iterations_slider.widget,non_negativity_checkbox.widget,gradient_checkbox.widget],layout=get_box_layout('100%'))
    controls_box = widgets.Box([load_cropped_frames_button.widget,correct_bad_frames_button.widget,preview_unwrap_button.widget,save_unwrapped_button.widget,play_box, unwrap_params_box,bad_frames_before_unwrap.widget],layout=get_box_layout('500px'))
    plot_box = widgets.VBox([output])
        
    box = widgets.HBox([controls_box,vbar,plot_box])
    
    return box

def equalizer_tab():


    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5))
        subplot.imshow(np.random.random((3,3)),cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()

    def plot_hist(data):
        plt.figure(dpi=150,figsize=(3,3))
        n, bins, patches = plt.hist(data.flatten(), 300, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('Pixel values')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.show()

    def load_unwrapped_sinogram(dummy,args=()):
        global unwrapped_sinogram
        print('Loading unwrapped sinogram: ',global_dict["unwrapped_sinogram_filepath"] )
        unwrapped_sinogram = np.load(global_dict["unwrapped_sinogram_filepath"] )
        print('\t Loaded!')
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = unwrapped_sinogram.shape[0] - 1, unwrapped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    
    
    def start_equalization(dummy):
        print("Starting equalization...")
        global equalized_sinogram
        equalized_sinogram = equalize_frames_parallel(sinogram,invert_checkbox.widget.value,remove_gradient_slider.widget.value, remove_outliers_slider.widget.value, remove_global_offset_checkbox.widget.value, remove_local_offset_field.widget.value)
        widgets.interactive_output(update_imshow, {'sinogram':fixed(equalized_sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    

    def save_sinogram(dummy):
        print('Saving equalized sinogram...')
        np.save(global_dict["unwrapped_sinogram_filepath"] ,unwrapped_sinogram)
        print('\tSaved sinogram at: ',global_dict["unwrapped_sinogram_filepath"] )

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_button = Button(description="Load unwrapped sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(partial(load_unwrapped_sinogram,args=(selection_slider,play_control)))

    start_button = Button(description="Start equalization",layout=buttons_layout,icon='play')
    start_button.trigger(partial(start_equalization,args=(selection_slider,play_control)))

    save_equalized_button = Button(description="Save equalized frames",layout=buttons_layout,icon='fa-floppy-o') 
    save_equalized_button.trigger(save_sinogram)

    invert_checkbox               = Input(global_dict,"equalize_invert",        description='Invert',layout=items_layout)
    remove_gradient_slider        = Input(global_dict,"equalize_gradient",      description="Remove Gradient", bounded=(1,10,1), slider=True,layout=slider_layout)
    remove_outliers_slider        = Input(global_dict,"equalize_outliers",      description="Remove Outliers", bounded=(1,10,1), slider=True,layout=slider_layout)
    remove_global_offset_checkbox = Input(global_dict,"equalize_global_offset", description='Remove Global Offset',layout=items_layout)
    remove_local_offset_field     = Input(global_dict,"equalize_local_offset",  description='Remove Local Offset',layout=items_layout)

    controls_box = widgets.Box([load_button.widget,play_box, invert_checkbox.widget,remove_gradient_slider.widget,remove_outliers_slider.widget,remove_global_offset_checkbox.widget,remove_local_offset_field.widget,start_button.widget,save_equalized_button.widget],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar, output]) 

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
        for subplot in subplots.reshape(-1):
            subplot.imshow(np.random.random((3,3)),cmap='gray')
        format_chull_plot(figure,subplots)
        plt.show()

    
    def load_unwrapped_sinogram(dummy,args=()):
        global unwrapped_sinogram
        print('Loading unwrapped sinogram: ',global_dict["unwrapped_sinogram_filepath"] )
        unwrapped_sinogram = np.load(global_dict["unwrapped_sinogram_filepath"] )
        print('\t Loaded!')
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = unwrapped_sinogram.shape[0] - 1, unwrapped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplots[0,0]), 'title':fixed(True),'frame_number': selection_slider.widget})    
        format_chull_plot(figure,subplots)

    def preview_cHull(dummy,args=()):
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram[selection_slider.widget.value,:,:],invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        for subplot, image in zip(subplots.reshape(-1),output_list[0::]):
            image = np.expand_dims(image, axis=0)
            widgets.interactive_output(update_imshow, {'sinogram':fixed(image),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': fixed(0)})    
        format_chull_plot(figure,subplots)
        print('\tDone with convex hull...')

    def complete_cHull(dummy,args=()):
        print('Applying complete Convex Hull...')
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram,invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        cHull_sinogram = output_list[-1]
        print('Saving cHull sinogram...',global_dict["chull_sinogram_filepath"])
        np.save(global_dict["chull_sinogram_filepath"],cHull_sinogram)
        print('\tSaved!')
    
    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames2)
        global unwrapped_sinogram
        unwrapped_sinogram[bad_frames2,:,:]  = np.zeros((unwrapped_sinogram.shape[1],unwrapped_sinogram.shape[2]))
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplots[0,0]), 'title':fixed(True),'frame_number': selection_slider.widget})    

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_before_cHull):
        global bad_frames2
        bad_frames2 = ast.literal_eval(bad_frames_before_cHull)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_button = Button(description="Load unwrapped sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(partial(load_unwrapped_sinogram,args=(selection_slider,play_control)))

    invert_checkbox  = Input(global_dict,"chull_invert",    description='Invert')
    tolerance        = Input(global_dict,"chull_tolerance", description='Threshold')
    opening_slider   = Input(global_dict,"chull_opening",   description="Opening",     bounded=(1,100,1),slider=True)
    erosion_slider   = Input(global_dict,"chull_erosion",   description="Erosion",     bounded=(1,100,1),slider=True)
    param_slider     = Input(global_dict,"chull_param",     description="Convex Hull", bounded=(1,200,1),slider=True)
    bad_frames_before_cHull = Input(global_dict,"bad_frames_before_cHull",description='Bad Frames',  layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_before_cHull":bad_frames_before_cHull.widget})

    preview_button = Button(description="Convex Hull Preview",layout=buttons_layout,icon='play')
    preview_button.trigger(partial(preview_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    start_button = Button(description="Do complete Convex Hull",layout=buttons_layout,icon='play')
    start_button.trigger(partial(complete_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    controls0 = widgets.Box([invert_checkbox.widget,tolerance.widget,opening_slider.widget,erosion_slider.widget,param_slider.widget],layout=box_layout)
    controls_box = widgets.Box([load_button.widget,bad_frames_before_cHull.widget,correct_bad_frames_button.widget,preview_button.widget,start_button.widget,play_box,controls0],layout=get_box_layout('500px'))

    box = widgets.HBox([controls_box,vbar, output])
    
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
        figure, subplot = plt.subplots(figsize=(3,3))
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

    
    def load_sinogram(dummy):
        
        if sinogram_selection.value == "unwrapped":
            filepath = global_dict["unwrapped_sinogram_filepath"]
        elif sinogram_selection.value == "convexHull":
            filepath = global_dict["chull_sinogram_filepath"]

        global sinogram
        print('Loading sinogram',filepath)
        sinogram = np.load(filepath)
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})    
    
    def update_imshow_with_format(sinogram,figure,subplot,frame_number,axis):
        update_imshow(sinogram,figure,subplot,frame_number,axis=axis)
        format_wiggle_plot(figure2,subplot2)

    global wiggled_sinogram
  
    def preview_angle_projection(dummy):
        print("Simulating projection of angles to regular grid...")
        angles  = np.load(global_dict["ordered_angles_filepath"])
        angles = (np.pi/180.) * angles
        _, selected_indices, _, projected_angles = angle_mesh_organize(sinogram, angles,percentage=angle_step_slider.widget.value)
        print(f' Sinogram shape {sinogram.shape} \n Number of Original Angles: {angles.shape} \n Number of Projected Angles: {projected_angles.shape}')
        selected_indices = [ i for i in selected_indices if i > 0]
        number_of_repeated_indices = len(selected_indices) - len(set(selected_indices))
        print(f"{number_of_repeated_indices} frames are being repeated!")

    def project_angles_to_regular_mesh(dummy):

        global sinogram 
        print('Projecting angles to regular mesh...')
        angles  = np.load(global_dict["ordered_angles_filepath"])
        angles = (np.pi/180.) * angles
        sinogram, _, _, projected_angles = angle_mesh_organize(sinogram, angles,percentage=angle_step_slider.widget.value)
        print(f'Sinogram max = {np.max(sinogram)} \t Sinogram min = {np.min(sinogram)}')
        print(f' Sinogram shape {sinogram.shape} \n Number of Original Angles: {angles.shape} \n Number of Projected Angles: {projected_angles.shape}')
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})    

        global_dict['NumberOriginalAngles'] = angles.shape # save to output log
        global_dict['NumberUsedAngles']     = projected_angles.shape 
        np.save(global_dict["projected_angles_filepath"],projected_angles)
        print('\tDone!')


    def start_wiggle(dummy,args=()):

        global sinogram

        _,_,_,cpus_slider,selection_slider = args

        print("Starting wiggle...")
        global wiggled_sinogram
        temporary_sinogram = radon.get_wiggle( sinogram,  'vertical', cpus_slider.value, selection_slider.widget.value)[0]
        print('Finished vertical wiggle. Starting horizontal wiggle...')
        wiggled_sinogram = radon.get_wiggle( temporary_sinogram, 'horizontal', cpus_slider.value, selection_slider.widget.value)[0]
        print("\t Wiggle done!")
        
        print("Saving wiggle sinogram to: ", global_dict["wiggle_sinogram_filepath"] )
        np.save(global_dict["wiggle_sinogram_filepath"] ,wiggled_sinogram)
        print("\t Saved!")

    def load_wiggle(dummy):
        global wiggled_sinogram
        print('Loading wiggled frames from:',global_dict["wiggle_sinogram_filepath"])
        wiggled_sinogram = np.load(global_dict["wiggle_sinogram_filepath"])
        sinogram_slider1.widget.max, sinogram_slider1.widget.value = wiggled_sinogram.shape[1] - 1, wiggled_sinogram.shape[1]//2
        sinogram_slider2.widget.max, sinogram_slider2.widget.value = wiggled_sinogram.shape[2] - 1, wiggled_sinogram.shape[2]//2
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[0,0]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[1,0]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0,1]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[1,1]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        print('\tLoaded!')

    def save_inverted_sinogram(dummy):
        print('Multiplying sinogram by -1 and saving at:',global_dict["wiggle_sinogram_filepath"])
        global wiggled_sinogram
        wiggled_sinogram = -1*wiggled_sinogram
        np.save(global_dict["wiggle_sinogram_filepath"],wiggled_sinogram)
        print(f'\t Saved! New max = {np.max(wiggled_sinogram)}. New min = {np.min(wiggled_sinogram)}')

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames3)
        global sinogram
        sinogram[bad_frames3,:,:]  = np.zeros((sinogram.shape[1],sinogram.shape[2]))
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_before_wiggle):
        global bad_frames3
        bad_frames3 = ast.literal_eval(bad_frames_before_wiggle)

    play_box, selection_slider,play_control = slide_and_play(label="Reference Frame")

    simulation_button = Button(description='Simulate Projection',icon='play',layout=buttons_layout)
    simulation_button.trigger(preview_angle_projection)
    projection_button = Button(description='Project Angles',icon='play',layout=buttons_layout)
    projection_button.trigger(project_angles_to_regular_mesh)
    angle_step_slider   = Input({"dummy_key":100},"dummy_key", description="Angle Step", bounded=(0,100,1),slider=True,layout=slider_layout)
    projection_box = widgets.VBox([angle_step_slider.widget,simulation_button.widget,projection_button.widget,play_box])

    wiggle_button = Button(description='Perform Wiggle',icon='play',layout=buttons_layout)
    load_wiggle_button   = Button(description='Load Wiggle',icon='folder-open-o',layout=buttons_layout)

    bad_frames_before_wiggle = Input(global_dict,"bad_frames_before_wiggle",description='Bad Frames',  layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_before_wiggle":bad_frames_before_wiggle.widget})

    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)


    sinogram_selection = widgets.RadioButtons(options=['unwrapped', 'convexHull'], value='unwrapped', style=style,layout=items_layout,description='Sinogram to import:',disabled=False)
    sinogram_slider1   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Y", bounded=(1,10,1),slider=True,layout=slider_layout)
    sinogram_slider2   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Z", bounded=(1,10,1),slider=True,layout = slider_layout)

    load_button = Button(description="Load sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(load_sinogram)
    
    global cpus_slider, gpus_slider
    gpus_slider = Input({'dummy_key':1}, 'dummy_key',bounded=(0,4,1),  slider=True,description="# of GPUs:")
    cpus_slider = Input({'dummy_key':32},'dummy_key',bounded=(1,128,1),slider=True,description="# of CPUs:")
    widgets.interactive_output(update_cpus_gpus,{"cpus":cpus_slider.widget,"gpus":gpus_slider.widget})

    args2 = (sinogram_selection,sinogram_slider1,sinogram_slider2,cpus_slider,selection_slider)
    wiggle_button.trigger(partial(start_wiggle,args=args2))
    load_wiggle_button.trigger(load_wiggle)

    invert_sinogram_buttom = Button(description='Invert Sinogram',icon='undo',layout=buttons_layout)
    invert_sinogram_buttom.trigger(save_inverted_sinogram)


    controls = widgets.VBox([sinogram_selection,load_button.widget,correct_bad_frames_button.widget,bad_frames_before_wiggle.widget,hbar2,projection_box,hbar2,cpus_slider.widget,wiggle_button.widget,load_wiggle_button.widget,sinogram_slider1.widget,sinogram_slider2.widget,invert_sinogram_buttom.widget])
    output = widgets.Box([output],layout=widgets.Layout(align_content='center'))#,align_items='center',justify_content='center'))
    box = widgets.HBox([controls,vbar,output,vbar,output2])
    
    return box

def tomo_tab():

    def format_tomo_plot(figure,subplots):
        for subplot in subplots.reshape(-1):
            subplot.set_aspect('equal')
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
        figure.tight_layout()


    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(1,3)
        subplot[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot[1].imshow(np.random.random((4,4)),cmap='gray')
        subplot[2].imshow(np.random.random((4,4)),cmap='gray')
        format_tomo_plot(figure,subplot)
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, axs = plt.subplots(2,2,figsize=(10,5))
        axs[0,0].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[0,1].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[1,0].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[1,1].hist(np.random.random((10,10)).flatten(),bins=100)
        figure2.canvas.header_visible = False 
        figure2.tight_layout()
        plt.show()


    def update_imshow_with_format(sinogram,figure1,subplot1,frame_number,axis):
        update_imshow(sinogram,figure1,subplot1,frame_number,axis=axis,title=True)
        format_tomo_plot(figure,subplot)

    def run_tomo(dummy,args=()):
        iter_slider,gpus_slider,filename_field, cpus_slider,jobname_field,queue_field, checkboxes = args

        global_dict["processing_steps"] = { "Sort":checkboxes[0].value , "Crop":checkboxes[1].value , "Unwrap":checkboxes[2].value, "ConvexHull":checkboxes[3].value, "Wiggle":checkboxes[4].value, "Tomo":checkboxes[5].value } # select steps when performing full recon

        output_path = global_dict["jupyter_folder"] 
        
        slurm_filepath = os.path.join(output_path,'tomo_job.srm')

        jsonFile_path = os.path.join(output_path,'user_input_tomo.json')


        global machine_selection
        print(f'Running tomo with {machine_selection.value}...')
        if machine_selection.value == 'Local':               
            reconstruction3D = tomography(global_dict)
            print('\t Done! Please, load the reconstruction with the button...')
            reconstruction3D = reconstruction3D.astype(np.float32)
            print('Saving 3D recon...')
            if type(global_dict["folders_list"]) == type('a'):
                global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list
            np.save(global_dict["reconstruction_filepath"],reconstruction3D)
            imsave(global_dict["reconstruction_filepath"][:-4] + '.tif',reconstruction3D)
            print('\t Saved!')

        elif machine_selection.value == "Cluster": 
            n_gpus = len(ast.literal_eval(gpus_slider.widget.value))
            run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path=output_path,slurmFile = slurm_filepath,  jobName=jobname_field.widget.value,queue=queue_field.widget.value,gpus=n_gpus,cpus=cpus_slider.value)

    def load_recon(dummy):

        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list

        if load_selection.value == "Original":
            savepath = global_dict["reconstruction_filepath"]
        elif load_selection.value == "Threshold":
            savepath = global_dict["reconstruction_thresholded_filepath"]
        
        print('Loading 3D recon from: ',savepath)
        time.sleep(0.5)
        global reconstruction
        reconstruction = np.load(savepath)
        print('\t Loaded!')
        print(f'Max = {np.max(reconstruction)}, Min = {np.min(reconstruction)}, Mean = {np.mean(reconstruction)}')
        tomo_sliceX.widget.max = reconstruction.shape[0]
        tomo_sliceY.widget.max = reconstruction.shape[1]
        tomo_sliceZ.widget.max = reconstruction.shape[2]
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[0]), 'axis':fixed(0), 'frame_number': tomo_sliceX.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[1]), 'axis':fixed(1), 'frame_number': tomo_sliceY.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[2]), 'axis':fixed(2), 'frame_number': tomo_sliceZ.widget})    

    def save_thresholded_tomo(dummy):
        print(f'Applying threshold value of {tomo_threshold.widget.value} to reconstruction')
        thresholded_recon = np.where( np.abs(reconstruction) > tomo_threshold.widget.value,0,reconstruction)
        print('\t Done!')
        print('Saving thresholded reconstruction...')
        np.save(global_dict["reconstruction_thresholded_filepath"],thresholded_recon)
        imsave(global_dict["reconstruction_thresholded_filepath"][:-4] + '.tif',thresholded_recon)
        print('\tSaved reconstruction at: ',global_dict["reconstruction_thresholded_filepath"])

    def plot_histograms(dummy):

        n_bins=100
        threshold = tomo_threshold.widget.value
        
        raw_data = np.load(global_dict["reconstruction_thresholded_filepath"])

        raw_data = raw_data.flatten()
        data = raw_data
        data = np.where(np.abs(data)>threshold,0,data)

        print('Computing statistics...')
        statistics_raw         = (np.max(raw_data),np.min(raw_data),np.mean(raw_data),np.std(raw_data))
        label_raw = f'\n\tMax = {statistics_raw[0]:.2e}\n\t Min = {statistics_raw[1]:.2e}\n\t Mean = {statistics_raw[2]:.2e}\n\t StdDev = {statistics_raw[3]:.2e}'
        statistics_thresholded = (np.max(data),np.min(data),np.mean(data),np.std(data))
        label_thresh = f'\n\tMax = {statistics_thresholded[0]:.2e}\n\t Min = {statistics_thresholded[1]:.2e}\n\t Mean = {statistics_thresholded[2]:.2e}\n\t StdDev = {statistics_thresholded[3]:.2e}'
        print('Raw data statistics: ', label_raw)
        print('Thresholded data statistics: ',label_thresh)


        print('Plotting histograms...')
        with output2:
            for ax in axs.reshape(-1):
                ax.clear()
            try:
                axs[0,0].hist(raw_data,bins=n_bins)
                axs[0,1].hist(raw_data,bins=n_bins)
            except:
                print('Problem found when plotting raw data! Check values!')
            axs[1,0].hist(data,bins=n_bins)
            axs[1,1].hist(data,bins=n_bins)
            axs[0,0].set_title('Raw histogram')
            axs[1,0].set_title('Threshold')
            axs[0,1].set_title('Log(Raw)')
            axs[1,1].set_title('Log(Threshold)')
            axs[0,1].set_yscale('log')
            axs[1,1].set_yscale('log')
        print('\t Done!')


    reg_checkbox    = Input(global_dict,"tomo_regularization",description = "Apply Regularization")
    reg_param       = Input(global_dict,"tomo_regularization_param",description = "Regularization Parameter",layout=items_layout)
    iter_slider     = Input(global_dict,"tomo_iterations",description = "Iterations", bounded=(1,200,2),slider=True,layout=slider_layout)
    widgets.interactive_output(update_cpus_gpus,{"cpus":cpus_slider.widget,"gpus":gpus_slider.widget})
    queue_field     = Input({"dummy_str":'cat-proc'},"dummy_str",description = "Machine Queue",layout=items_layout)
    jobname_field   = Input({"dummy_str":'myTomography'},"dummy_str",description = "Slurm Job Name",layout=items_layout)
    filename_field  = Input({"dummy_str":'reconstruction3Dphase'},"dummy_str",description = "Output Filename",layout=items_layout)
    tomo_threshold  = Input(global_dict,"tomo_threshold",description = "Value threshold for recon",layout=items_layout)
    tomo_sliceX     = Input({"dummy_key":1},"dummy_key", description="Slice X", bounded=(1,10,1),slider=True,layout=slider_layout)
    tomo_sliceY     = Input({"dummy_key":1},"dummy_key", description="Slice Y", bounded=(1,10,1),slider=True,layout=slider_layout)
    tomo_sliceZ     = Input({"dummy_key":1},"dummy_key", description="Slice Z", bounded=(1,10,1),slider=True,layout=slider_layout)
    algo_dropdown   = widgets.Dropdown(options=['EEM','EM', 'ART','FBP'], value='EEM',description='Algorithm:',layout=items_layout)
    load_selection  = widgets.RadioButtons(options=['Original', 'Threshold'], value='Original',style=style, layout=items_layout,description='Load:',disabled=False)
    checkboxes      = [widgets.Checkbox(value=False, description=label,layout=checkbox_layout, style=style) for label in ["Sort", "Crop", "Unwrap", "ConvexHull", "Wiggle", "Tomo"]]
    checkboxes_box  = widgets.VBox(children=checkboxes)

    widgets.interactive_output(update_paths,{'global_dict':fixed(global_dict),'dummy1':algo_dropdown,'dummy2':fixed(algo_dropdown)})


    def update_processing_steps(dictionary,sort_checkbox,crop_checkbox,unwrap_checkbox,chull_checkbox,wiggle_checkbox,tomo_checkbox):
        # "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "ConvexHull":1, "Wiggle":1, "Tomo":1 } # select steps when performing full recon
        dictionary["processing_steps"]["Sort"]       = sort_checkbox 
        dictionary["processing_steps"]["Crop"]       = crop_checkbox 
        dictionary["processing_steps"]["Unwrap"]     = unwrap_checkbox 
        dictionary["processing_steps"]["ConvexHull"] = chull_checkbox 
        dictionary["processing_steps"]["Wiggle"]     = wiggle_checkbox 
        dictionary["processing_steps"]["Tomo"]       = tomo_checkbox 
    widgets.interactive_output(update_processing_steps,{'dictionary':fixed(global_dict),'sort_checkbox':checkboxes[0],'crop_checkbox':checkboxes[1],'unwrap_checkbox':checkboxes[2],'chull_checkbox':checkboxes[3],'wiggle_checkbox':checkboxes[4],'tomo_checkbox':checkboxes[5]})

    start_tomo = Button(description="Start",layout=buttons_layout,icon='play')
    args = iter_slider,gpus_slider,filename_field,cpus_slider,jobname_field,queue_field, checkboxes
    start_tomo.trigger(partial(run_tomo,args=args))
    start_tomo_box = widgets.Box([start_tomo.widget],layout=center_all_layout)

    load_recon_button = Button(description="Load recon slices",layout=buttons_layout,icon='play')
    load_recon_button.trigger(load_recon)

    save_thresholded_tomo_button = Button(description="Save thresholded tomo",layout=buttons_layout,icon='play')
    save_thresholded_tomo_button.trigger(save_thresholded_tomo)

    plot_histogram_button = Button(description="Plot Histograms",layout=buttons_layout,icon='play')
    plot_histogram_button.trigger(plot_histograms)

    def save_on_click(dummy):
        print('Saving JSON input file...')
        global_dict["contrast_type"]  = data_selection.value
        global_dict["tomo_algorithm"] = algo_dropdown.value
        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list
        json_filepath = os.path.join(global_dict["jupyter_folder"],'user_input_tomo.json') #INPUT
        with open(json_filepath, 'w') as file:
            json.dump(global_dict, file)
        
        from pprint import pprint
        pprint(global_dict)
        print('\t Saved!')

    save_dict_button  = Button(description="Save JSON",layout=buttons_layout,icon='fa-floppy-o')
    save_dict_button.trigger(save_on_click)    
    
    load_box = widgets.HBox([load_recon_button.widget,load_selection])
    start_box = widgets.HBox([checkboxes_box,start_tomo_box])#,layout=widgets.Layout(flex_flow='row',width='100%',border=standard_border))
    threshold_box = widgets.HBox([save_thresholded_tomo_button.widget])#, plot_histogram_button.widget])
    slurm_box = widgets.VBox([cpus_slider.widget,gpus_slider.widget,queue_field.widget,jobname_field.widget])
    controls = widgets.VBox([algo_dropdown,reg_checkbox.widget,reg_param.widget,iter_slider.widget,slurm_box,hbar2,save_dict_button.widget,start_box,hbar2,load_box,tomo_sliceX.widget,tomo_sliceY.widget,tomo_sliceZ.widget,hbar2,tomo_threshold.widget,threshold_box])
    box = widgets.HBox([controls,vbar2,output])#widgets.VBox([output,hbar,output2])])
    
    return box 

def deploy_tabs(mafalda_session,tab1=folders_tab(),tab2=crop_tab(),tab3=unwrap_tab(),tab4=chull_tab(),tab5=wiggle_tab(),tab6=tomo_tab(),tab7=equalizer_tab()):
    
    children_dict = {
    "Select and Sort"       : tab1,
    "Cropping"              : tab2,
    "Phase Unwrap"          : tab3,
    "Frame Equalizer"       : tab7,
    "Convex Hull"           : tab4,
    "Wiggle"                : tab5,
    "Tomography"            : tab6}
    
    def load_json(dummy,dictionary={}):
        template_dict = {"ibira_data_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/data/ptycho2d/",
               "folders_list": ["SS61"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/proc/recons/SS61/phase_microagg_P2_01.npy",
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_before_unwrap": [7,20,36,65,94,123,152,181,210,239,268,296,324],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,
               "bad_frames_before_cHull": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               
               "wiggle_reference_frame": 0,
               "CPUs": 32,
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "GPUs": [0],
               "tomo_threshold" : float(0.0), # max value to be left in reconstructed absorption
               "run_all_tomo_steps":False}
    
        for key in template_dict:
            dictionary[key] = template_dict[key]
    
    global mafalda
    mafalda = mafalda_session

    load_json_button  = Button(description="Reset JSON",layout=buttons_layout,icon='folder-open-o')
    load_json_button.trigger(partial(load_json,dictionary=global_dict))
    
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Local', layout={'width': '30%'},description='Machine',disabled=False)
    widgets.interactive_output(update_gpu_limits,{"machine_selection":machine_selection})

    global data_selection
    data_selection = widgets.RadioButtons(options=['Magnitude', 'Phase'], value='Phase', layout={'width': '30%'},description='Visualize',disabled=False)
    widgets.interactive_output(update_paths,{'global_dict':fixed(global_dict),'dummy1':data_selection,'dummy2':fixed(data_selection)})



    def delete_files(dummy):
        sinogram_path = global_dict["sinogram_path"].rsplit('/',1)[0]

        filepaths_to_remove = [ global_dict["ordered_angles_filepath"],  
                                global_dict["ordered_object_filepath"] , 
                                global_dict["cropped_sinogram_filepath"],
                                global_dict["unwrapped_sinogram_filepath"],
                                global_dict["equalized_sinogram_filepath"],
                                global_dict["chull_sinogram_filepath"],  
                                global_dict["wiggle_sinogram_filepath"],
                                global_dict["projected_angles_filepath"]]

        for filepath in filepaths_to_remove:
            print('Removing file/folder: ', filepath)
            if os.path.exists(filepath) == True:
                os.remove(filepath)
                print(f'Deleted {filepath}\n')
            else:
                print(f'Directory {filepath} does not exists. Skipping deletion...\n')

        folderpaths_to_remove =[os.path.join(global_dict["output_folder"],'00_frames_original'),
                                os.path.join(global_dict["output_folder"],'01_frames_ordered'),
                                os.path.join(global_dict["output_folder"],'02_frames_cropped'),
                                os.path.join(global_dict["output_folder"],'03_frames_unwrapped'),
                                os.path.join(global_dict["output_folder"],'04_frames_convexHull')]
                                
        import shutil
        for folderpath in folderpaths_to_remove:
            print('Removing file/folder: ', folderpath)
            if os.path.isdir(folderpath) == True:
                shutil.rmtree(folderpath)
                print(f'Deleted {folderpath}\n')
            else:
                print(f'Directory {folderpath} does not exists. Skipping deletion...\n')


    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    box = widgets.HBox([machine_selection,data_selection,load_json_button.widget,delete_temporary_files_button.widget])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  


if __name__ == "__main__":
    pass
