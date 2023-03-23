import ipywidgets as widgets
from ipywidgets import fixed
import ast 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from functools import partial
import os, time
import json
from skimage.io import imsave
import asyncio
from functools import partial
import subprocess

import sscCdi, sscPimega, sscRaft, sscRadon, sscResolution

from sscRadon import radon
from ..processing.unwrap import unwrap_in_parallel
from ..tomo.tomo_processing import angle_mesh_organize, tomography, apply_chull_parallel, sort_frames_by_angle, reorder_slices_low_to_high_angle, equalize_frames_parallel
from ..tomo.tomo_processing import equalize_tomogram, save_or_load_wiggle_ctr_mass, get_and_save_downsampled_sinogram, add_plot_suffix_to_file
from ..jupyter import call_and_read_terminal, monitor_job_execution, call_cmd_terminal, VideoControl, Button, Input, update_imshow

global sinogram
sinogram = np.random.random((2,2,2)) # dummy sinogram

import getpass
username = getpass.getuser()

""" Standard folders definitions"""
tomo_script_path    = '/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/ssc-cdi/bin/sscptycho_raft.py' # path with python script to run

""" Standard dictionary definition """
global global_dict
global_dict = {
               "00_versions": f"sscCdi={sscCdi.__version__},sscPimega={sscPimega.__version__},sscResolution={sscResolution.__version__},sscRaft={sscRaft.__version__},sscRadon={sscRadon.__version__}",
               "jupyter_folder":"/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/", # FIXED PATH FOR BEAMLINE

               "ibira_data_path": "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/data/ptycho3d/",
               "folders_list": ["phantom_complex"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/proc/recons/phantom_complex/object_phantom_complex.npy",

               "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "Wiggle":1, "Tomo":1 }, # select steps when performing full recon
               "contrast_type": "Phase", # Phase or Absolute

               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,

               "bad_frames_before_unwrap": [],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,

                "bad_frames_before_equalization": [],
                "equalize_invert":False,
                "equalize_gradient":0,
                "equalize_outliers":0,
                "equalize_global_offset":False,
                "equalize_local_offset":[0,None,0,None],

               "bad_frames_before_cHull": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               

               "wiggle_sinogram_selection":"cropped",
               "bad_frames_before_wiggle": [],
               "wiggle_reference_frame": 0,
               "wiggle_ctr_of_mas": [[],[]],
               "CPUs": 32,
              
               "tomo_regularization": False,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 10,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "GPUs": [0],
               "tomo_threshold" : 0.0, # max value to be left in reconstructed matrix
               "tomo_remove_outliers": 0,
               "tomo_local_offset":[],
               "tomo_mask":[]
}

json_filepath = os.path.join(global_dict["jupyter_folder"],'inputs', f'{username}_tomo_input.json') #INPUT
if os.path.exists(json_filepath):  
    with open(json_filepath) as json_file:
        global_dict = json.load(json_file)

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
buttons_layout_fixed = widgets.Layout( width='400px',height="40px")     # override the default width of the button to 'auto' to let the button grow
center_all_layout = widgets.Layout(align_items='center',width='100%',border=standard_border) #align_content='center',justify_content='center'
box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
sliders_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
style = {'description_width': 'initial'}

def get_box_layout(width,flex_flow='column',align_items='center',border=standard_border):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)

machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Cluster', layout={'width': '10%'},description='Machine',disabled=False)

############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################


def update_paths(global_dict,dummy1,dummy2):
    # dummy variable is used to trigger update
    global_dict["output_folder"] = global_dict["sinogram_path"].rsplit('/',1)[0]
    
    if type(global_dict["folders_list"]) == type([1,2]): # correct data type of this input
        pass # if list
    else: # if string
        global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"])

    global_dict["complex_object_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["folders_list"][0] + '_object.npy')
    global_dict["ordered_angles_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["folders_list"][0] + '_ordered_angles.npy')
    global_dict["ordered_object_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["folders_list"][0] + '_ordered_object.npy')
    global_dict["reconstruction_equalized_filepath"] = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_' + global_dict["folders_list"][0] + '_reconstruction3D_' + global_dict["tomo_algorithm"] + '_equalized.npy')
    global_dict["reconstruction_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_' + global_dict["folders_list"][0] + '_reconstruction3D_' + global_dict["tomo_algorithm"] + '.npy')
    global_dict["cropped_sinogram_filepath"]         = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_cropped_sinogram.npy')
    global_dict["unwrapped_sinogram_filepath"]       = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_unwrapped_sinogram.npy')
    global_dict["equalized_sinogram_filepath"]       = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_equalized_sinogram.npy')
    global_dict["chull_sinogram_filepath"]           = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_chull_sinogram.npy')
    global_dict["wiggle_sinogram_filepath"]          = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_wiggle_sinogram.npy')
    global_dict["projected_angles_filepath"]         = os.path.join(global_dict["output_folder"],global_dict["ordered_angles_filepath"][:-4]+'_projected.npy')
    global_dict["wiggle_ctr_mass_filepath"]          = os.path.join(global_dict["output_folder"],global_dict["contrast_type"] + '_wiggle_ctr_mass.npy')
    return global_dict

def write_slurm_file(tomo_script_path,jsonFile_path,output_path="",slurmFile = 'tomoJob.sh',jobName='jobName',queue='cat',GPUs=1,cpus=32):
    # Create slurm file
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{GPUs}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o {output_path}/logfiles/{username}_slurm.log        # Select output path of slurm file

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

python3 {tomo_script_path} {jsonFile_path} > {os.path.join(output_path,'logfiles',f'{username}_tomo_output.log')} 2> {os.path.join(output_path,'logfiles',f'{username}_tomo_error.log')}
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

def run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat',GPUs=1,cpus=32):
    slurm_file = write_slurm_file(tomo_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,GPUs,cpus)
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
        GPUs_slider.widget.value = 0
        GPUs_slider.widget.max = 5
        cpus_slider.widget.value = 32
        cpus_slider.widget.max = 160
    elif machine_selection == 'Local':
        GPUs_slider.widget.value = 0
        GPUs_slider.widget.max = 6
        cpus_slider.widget.value = 32
        cpus_slider.widget.max = 144

def update_cpus_GPUs(cpus,GPUs,machine_selection):
    global_dict["CPUs"] = cpus

    if machine_selection == 'Cluster':
        if GPUs == 0:
            global_dict["GPUs"] = []
        elif GPUs == 1:
            global_dict["GPUs"] = [0] 
        elif GPUs == 2:
            global_dict["GPUs"] = [0,1]
        elif GPUs == 3:
            global_dict["GPUs"] = [0,1,2]
        elif GPUs == 4:
            global_dict["GPUs"] = [0,1,2,3]
        elif GPUs == 5:
            global_dict["GPUs"] = [0,1,2,3,4]
    elif machine_selection == 'Local':
        if GPUs == 0:
            global_dict["GPUs"] = []
        elif GPUs == 1:
            global_dict["GPUs"] = [0] 
        elif GPUs == 2:
            global_dict["GPUs"] = [0,1]
        elif GPUs == 3:
            global_dict["GPUs"] = [0,1,2]
        elif GPUs == 4:
            global_dict["GPUs"] = [0,1,2,3]
        elif GPUs == 5:
            global_dict["GPUs"] = [0,1,2,3,4]
        elif GPUs == 6:
            global_dict["GPUs"] = [0,1,2,3,4,5]
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

        selection_slider2.widget.max, selection_slider2.widget.value = object.shape[0] - 1, object.shape[0]//2
        play_control2.widget.max =  selection_slider2.widget.max
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

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector (Angle)",frame_time_milisec=300)
    play_box2, selection_slider2,play_control2 = slide_and_play(label="Frame Selector (Time)",frame_time_milisec=300)

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

    top_crop      = Input(global_dict,"top_crop"   ,description="Top", bounded=(0,vertical_max,1),  slider=True,layout=slider_layout)
    bottom_crop   = Input(global_dict,"bottom_crop",description="Bottom", bounded=(1,vertical_max,1),  slider=True,layout=slider_layout)
    left_crop     = Input(global_dict,"left_crop"  ,description="Left", bounded=(0,horizontal_max,1),slider=True,layout=slider_layout)
    right_crop    = Input(global_dict,"right_crop" ,description="Right", bounded=(1,horizontal_max,1),slider=True,layout=slider_layout)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)
    
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
        save_sinogram()

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

    def save_sinogram():
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
    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    iterations_slider = Input(global_dict,"unwrap_iterations",bounded=(0,10,1),slider=True, description='Unwrap Iterations',layout=slider_layout)
    non_negativity_checkbox = Input(global_dict,"unwrap_non_negativity",layout=items_layout,description='Non-negativity')
    gradient_checkbox = Input(global_dict,"unwrap_gradient_removal",layout=items_layout,description='Gradient')
    
    preview_unwrap_button = Button(description="Perform Unwrap",layout=buttons_layout,icon='play')
    preview_unwrap_button.trigger(phase_unwrap)
    
    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)

    args = (selection_slider,play_control)
    load_cropped_frames_button.trigger(partial(load_cropped_frames,args=args))

    unwrap_params_box = widgets.Box([iterations_slider.widget,non_negativity_checkbox.widget,gradient_checkbox.widget],layout=get_box_layout('100%'))
    controls_box = widgets.Box([load_cropped_frames_button.widget,bad_frames_before_unwrap.widget,correct_bad_frames_button.widget,preview_unwrap_button.widget,play_box, unwrap_params_box],layout=get_box_layout('500px'))
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
        equalized_sinogram = equalize_frames_parallel(unwrapped_sinogram,invert_checkbox.widget.value,remove_gradient_slider.widget.value, remove_outliers_slider.widget.value, remove_global_offset_checkbox.widget.value, ast.literal_eval(remove_local_offset_field.widget.value))
        widgets.interactive_output(update_imshow, {'sinogram':fixed(equalized_sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    
        save_sinogram()

    def save_sinogram():
        print('Saving equalized sinogram...')
        np.save(global_dict["equalized_sinogram_filepath"] ,equalized_sinogram)
        print('\tSaved sinogram at: ',global_dict["equalized_sinogram_filepath"])

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames4)
        global unwrapped_sinogram
        unwrapped_sinogram[bad_frames4,:,:] = np.zeros((unwrapped_sinogram.shape[1],unwrapped_sinogram.shape[2]))
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    
    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_before_equalization):
        global bad_frames4
        bad_frames4 = ast.literal_eval(bad_frames_before_equalization)

    bad_frames_before_equalization  = Input(global_dict,"bad_frames_before_equalization", description = 'Bad frames',layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_before_equalization":bad_frames_before_equalization.widget})
    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)
    
    load_button = Button(description="Load unwrapped sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(partial(load_unwrapped_sinogram,args=(selection_slider,play_control)))

    start_button = Button(description="Start equalization",layout=buttons_layout,icon='play')
    start_button.trigger(start_equalization)

    invert_checkbox               = Input(global_dict,"equalize_invert",        description='Invert',layout=items_layout)
    remove_gradient_slider        = Input(global_dict,"equalize_gradient",      description="Remove Gradient", bounded=(0,10,1), slider=True,layout=slider_layout)
    remove_outliers_slider        = Input(global_dict,"equalize_outliers",      description="Remove Outliers", bounded=(0,10,1), slider=True,layout=slider_layout)
    remove_global_offset_checkbox = Input(global_dict,"equalize_global_offset", description='Remove Global Offset',layout=items_layout)
    remove_local_offset_field     = Input(global_dict,"equalize_local_offset",  description='Remove Local Offset',layout=items_layout)

    controls_box = widgets.Box([load_button.widget,bad_frames_before_equalization.widget,correct_bad_frames_button.widget,play_box, invert_checkbox.widget,remove_gradient_slider.widget,remove_outliers_slider.widget,remove_global_offset_checkbox.widget,remove_local_offset_field.widget,start_button.widget],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar, output]) 

    return box

#TODO: delete chull tabs and variables!
def chull_tab():
    
    def format_chull_plot(figure,subplots):
        subplots[0,0].set_title('Original')
        subplots[0,1].set_title('Threshold')
        subplots[0,2].set_title('Opening')
        subplots[1,0].set_title('Erosion')
        subplots[1,1].set_title('Convex Hull')
        subplots[1,2].set_title('masked Image')

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

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)
    
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
        subplots[0].set_title('Pre-wiggle sample')
        subplots[1].set_title('Pre-wiggle sinogram')
        subplots[2].set_title('Wiggled sinogram')
        
        for subplot in subplots.reshape(-1):
            subplot.set_aspect('auto')
            subplot.set_xticks([])
            subplot.set_yticks([])
        subplots[0].set_aspect('equal')

        figure.canvas.header_visible = False 
        figure.tight_layout()
    
    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(1,3,figsize=(10,5))
        subplot2[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[1].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[2].imshow(np.random.random((4,4)),cmap='gray')
        format_wiggle_plot(figure2,subplot2)
        plt.show()

    
    def load_sinogram(dummy):
        
        if sinogram_selection.value == "unwrapped":
            filepath = global_dict["unwrapped_sinogram_filepath"]
        elif sinogram_selection.value == "equalized":
            filepath = global_dict["equalized_sinogram_filepath"]
        elif sinogram_selection.value == "cropped":
            filepath = global_dict["cropped_sinogram_filepath"]

        global sinogram
        print('Loading sinogram',filepath)
        sinogram = np.load(filepath)
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0]),'title':fixed(True), 'frame_number': selection_slider.widget})    
    
    def update_imshow_with_format(sinogram,figure,subplot,frame_number,axis):
        update_imshow(sinogram,figure,subplot,frame_number,axis=axis)
        format_wiggle_plot(figure2,subplot2)

    global wiggled_sinogram
  
    def preview_angle_projection(dummy):
        print("Simulating projection of angles to regular grid...")
        angles  = np.load(global_dict["ordered_angles_filepath"])
        angles = (np.pi/180.) * angles
        total_n_of_angles = angles.shape[0]
        
        _, selected_indices, n_of_padding_frames, projected_angles = angle_mesh_organize(sinogram, angles,percentage=angle_step_slider.widget.value)
        n_of_negative_idxs = len([ i for i in selected_indices if i < 0])
        selected_positive_indices = [ i for i in selected_indices if i >= 0]
        complete_array = [i for i in range(total_n_of_angles)]

        # print('Selected indices: \n',selected_indices)
        print('Before+after frames added:',n_of_padding_frames)
        print('Intermediate null frames :',len([ i for i in selected_indices if i < -1]))
        print('                        + -----')
        print("Total null frames        :", n_of_negative_idxs)
        print("Frames being used        :", len(selected_positive_indices)," of ",len(complete_array))
        print('                        + -----')
        print('Projected Angles         :', projected_angles.shape[0])

        
    def project_angles_to_regular_mesh(dummy):

        global sinogram 
        print('Projecting angles to regular mesh...')
        angles  = np.load(global_dict["ordered_angles_filepath"])
        angles = (np.pi/180.) * angles
        total_n_of_angles = angles.shape[0]
        sinogram, selected_indices, n_of_padding_frames, projected_angles = angle_mesh_organize(sinogram, angles,percentage=angle_step_slider.widget.value)
        print(f'Sinogram shape {sinogram.shape} \n Number of Original Angles: {angles.shape} \n Number of Projected Angles: {projected_angles.shape}')
        n_of_negative_idxs = len([ i for i in selected_indices if i < 0])
        selected_positive_indices = [ i for i in selected_indices if i >= 0]
        complete_array = [i for i in range(total_n_of_angles)]

        # print('Selected indices: \n',selected_indices)
        print('Before+after frames added:',n_of_padding_frames)
        print('Intermediate null frames :',len([ i for i in selected_indices if i < -1]))
        print('                        + -----')
        print("Total null frames        :", n_of_negative_idxs)
        print("Frames being used        :", len(selected_positive_indices)," of ",len(complete_array))
        print('                        + -----')
        print('Projected Angles         :', projected_angles.shape[0])


        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0]),'title':fixed(True), 'frame_number': selection_slider.widget})    

        global_dict['NumberOriginalAngles'] = angles.shape # save to output log
        global_dict['NumberUsedAngles']     = projected_angles.shape 
        np.save(global_dict["projected_angles_filepath"],projected_angles)
        print('\tDone!')


    def start_wiggle(dummy,args=()):

        global sinogram

        _,_,_,selection_slider = args

        global_dict["wiggle_reference_frame"] = selection_slider.widget.value 
        global_dict["wiggle_sinogram_selection"] = sinogram_selection.value

        print("Starting wiggle...")
        global wiggled_sinogram
        temp_tomogram, shiftv = radon.get_wiggle( sinogram, "vertical", global_dict["CPUs"], global_dict["wiggle_reference_frame"] )
        temp_tomogram, shiftv = radon.get_wiggle( temp_tomogram, "vertical", global_dict["CPUs"], global_dict["wiggle_reference_frame"] )
        print('Finished vertical wiggle. Starting horizontal wiggle...')
        wiggled_sinogram, shifth, wiggle_cmas_temp = radon.get_wiggle( temp_tomogram, "horizontal", global_dict["CPUs"], global_dict["wiggle_reference_frame"] )
        wiggle_cmas = [[],[]]
        wiggle_cmas[1], wiggle_cmas[0] =  wiggle_cmas_temp[:,1].tolist(), wiggle_cmas_temp[:,0].tolist()
        global_dict["wiggle_ctr_of_mas"] = wiggle_cmas
        save_or_load_wiggle_ctr_mass(global_dict["wiggle_ctr_mass_filepath"],wiggle_cmas,save=True)
        print("\t Wiggle done!")
        print("Saving wiggle sinogram to: ", global_dict["wiggle_sinogram_filepath"] )
        np.save(global_dict["wiggle_sinogram_filepath"] ,wiggled_sinogram)
        print("\t Saved!")
        load_wiggle()

    def load_wiggle():
        global wiggled_sinogram
        print('Loading wiggled frames from:',global_dict["wiggle_sinogram_filepath"])
        if direction_selector.value == "X":
            axis_direction = 0
        elif direction_selector.value == "Y":
            axis_direction = 1
        elif direction_selector.value == "Z":
            axis_direction = 2
        wiggled_sinogram = np.load(global_dict["wiggle_sinogram_filepath"])
        sinogram_slider1.widget.max, sinogram_slider1.widget.value = wiggled_sinogram.shape[1] - 1, wiggled_sinogram.shape[1]//2
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[1]), 'axis':fixed(axis_direction),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[2]), 'axis':fixed(axis_direction),'frame_number': sinogram_slider1.widget})    
        print('\tLoaded!')

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames3)
        global sinogram
        sinogram[bad_frames3,:,:] = np.zeros((sinogram.shape[1],sinogram.shape[2]))
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0]), 'title':fixed(True),'frame_number': selection_slider.widget})    
    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_before_wiggle):
        global bad_frames3
        bad_frames3 = ast.literal_eval(bad_frames_before_wiggle)

    def update_dict(reference_frame):
        global_dict["wiggle_reference_frame"] = reference_frame

    play_box, selection_slider,play_control = slide_and_play(label="Reference Frame",frame_time_milisec=300)
    widgets.interactive_output(update_dict,{ "reference_frame":selection_slider.widget})

    simulation_button = Button(description='Simulate Projection',icon='play',layout=buttons_layout)
    simulation_button.trigger(preview_angle_projection)
    projection_button = Button(description='Project Angles',icon='play',layout=buttons_layout)
    projection_button.trigger(project_angles_to_regular_mesh)
    angle_step_slider = Input({"dummy_key":100},"dummy_key", description="Angle Step", bounded=(0,100,1),slider=True,layout=slider_layout)
    projection_box = widgets.VBox([angle_step_slider.widget,simulation_button.widget,projection_button.widget,play_box])

    direction_selector = widgets.RadioButtons(options=['X','Y','Z'], value='Y', style=style,layout=items_layout,description='Plot direction:',disabled=False)
    wiggle_button = Button(description='Perform Wiggle',icon='play',layout=buttons_layout)

    bad_frames_before_wiggle = Input(global_dict,"bad_frames_before_wiggle",description='Bad Frames',  layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_before_wiggle":bad_frames_before_wiggle.widget})

    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)


    sinogram_selection = widgets.RadioButtons(options=['cropped','unwrapped','equalized'], value='unwrapped', style=style,layout=items_layout,description='Sinogram to import:',disabled=False)
    sinogram_slider1   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice", bounded=(1,10,1),slider=True,layout=slider_layout)

    global cpus_slider, GPUs_slider, machine_selection
    GPUs_slider = Input({'dummy_key':1}, 'dummy_key',bounded=(1,5,1),  slider=True,description="# of GPUs:")
    cpus_slider = Input({'dummy_key':32},'dummy_key',bounded=(1,160,1),slider=True,description="# of CPUs:")
    widgets.interactive_output(update_cpus_GPUs,{"cpus":cpus_slider.widget,"GPUs":GPUs_slider.widget,"machine_selection":machine_selection})

    args2 = (sinogram_selection,sinogram_slider1,cpus_slider,selection_slider)
    wiggle_button.trigger(partial(start_wiggle,args=args2))

    load_button = Button(description="Load sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(load_sinogram)

    controls = widgets.VBox([sinogram_selection,load_button.widget,correct_bad_frames_button.widget,bad_frames_before_wiggle.widget,hbar2,projection_box,hbar2,cpus_slider.widget,direction_selector, wiggle_button.widget,sinogram_slider1.widget])
    box = widgets.HBox([controls,vbar,output2])
    
    return box

def tomo_tab():

    def format_tomo_plot(figure,subplots):
        for subplot in subplots.reshape(-1):
            subplot.set_aspect('equal')
            subplot.set_xticks([])
            subplot.set_yticks([])
        subplots[0].set_title("Tomo slice")
        subplots[1].set_title("Tomo projection")
        figure.canvas.header_visible = False 
        figure.tight_layout()


    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(1,2,figsize=(10,6))
        subplot[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot[1].imshow(np.random.random((4,4)),cmap='gray')
        format_tomo_plot(figure,subplot)
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, axs = plt.subplots(1,2,figsize=(13,2))
        axs[0].hist(np.random.normal(size=(100,100)).flatten(),bins=50)
        axs[1].hist(np.random.normal(size=(100,100)).flatten(),bins=50,color='green')
        axs[0].set_title("Histogram: Original")
        axs[1].set_title("Histogram: Equalized")
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        figure2.canvas.header_visible = False 
        figure2.tight_layout()
        plt.show()


    def update_imshow_with_format(sinogram,figure1,subplot1,frame_number,axis,norm=None):
        update_imshow(sinogram,figure1,subplot1,frame_number,axis=axis,title=True,norm=norm)
        format_tomo_plot(figure,subplot)



    def load_recon(dummy):

        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list

        if load_selection.value == "Original":
            savepath = global_dict["reconstruction_filepath"]
        elif load_selection.value == "Equalized":
            savepath = global_dict["reconstruction_equalized_filepath"]
        
        print('Loading 3D recon from: ',savepath)
        global reconstruction
        reconstruction = np.load(savepath)
        print('\t Loaded!')
        print(f'Max = {np.max(reconstruction)}, Min = {np.min(reconstruction)}, Mean = {np.mean(reconstruction)}')
      
        if direction_selector.value == "X":
            axis_direction = 0
        elif direction_selector.value == "Y":
            axis_direction = 1
        elif direction_selector.value == "Z":
            axis_direction = 2
      
        tomo_slice.widget.max, tomo_slice.widget.value = reconstruction.shape[axis_direction], reconstruction.shape[axis_direction]//2
        norm = colors.Normalize(vmin=np.min(reconstruction), vmax=np.max(reconstruction))

        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[0]), 'axis':fixed(axis_direction), 'frame_number': tomo_slice.widget,'norm':fixed(norm)})    
        subplot[1].imshow(np.sum(reconstruction,axis=axis_direction),cmap='gray')
        format_tomo_plot(figure,subplot)

        axs[0].clear()
        axs[0].hist(reconstruction.flatten(),bins=300)
        axs[0].set_title("Histogram: Original")
        axs[1].set_title("Histogram: Equalized")

    def equalize(dummy):

        print('Computing statistics...')
        statistics_raw = (np.max(reconstruction),np.min(reconstruction),np.mean(reconstruction),np.std(reconstruction))
        print('Raw data statistics: ',  f'\n\tMax = {statistics_raw[0]:.2e}\t Min = {statistics_raw[1]:.2e}\t Mean = {statistics_raw[2]:.2e}\t StdDev = {statistics_raw[3]:.2e}')

        if direction_selector.value == "X":
            axis_direction = 0
        elif direction_selector.value == "Y":
            axis_direction = 1
        elif direction_selector.value == "Z":
            axis_direction = 2

        global equalized_tomogram 
        equalized_tomogram = equalize_tomogram(reconstruction,statistics_raw[2],statistics_raw[3],remove_outliers=remove_outliers_slider.widget.value,threshold=float(tomo_threshold.widget.value),bkg_window=remove_local_offset_field.widget.value,axis_direction=axis_direction)#,mask_slice=mask_final_tomo.widget.value)

        statistics_equalized = (np.max(equalized_tomogram),np.min(equalized_tomogram),np.mean(equalized_tomogram),np.std(equalized_tomogram))
        print('Thresholded data statistics: ',f'\n\tMax = {statistics_equalized[0]:.2e}\n\t Min = {statistics_equalized[1]:.2e}\n\t Mean = {statistics_equalized[2]:.2e}\n\t StdDev = {statistics_equalized[3]:.2e}')

        flattened_tomogram = equalized_tomogram.flatten()
        axs[1].clear()
        axs[1].hist(np.where(flattened_tomogram==0,np.NaN,flattened_tomogram),bins=300,color='green')
        if hist_max.widget.value != 0:
            axs[0].set_ylim(0,hist_max.widget.value)
            axs[1].set_ylim(0,hist_max.widget.value)
        axs[0].set_title("Histogram: Original")
        axs[1].set_title("Histogram: Equalized")

        print('Saving equalized reconstruction...')
        np.save(global_dict["reconstruction_equalized_filepath"],equalized_tomogram)
        imsave(global_dict["reconstruction_equalized_filepath"][:-4] + '.tif',equalized_tomogram)
        print('\tSaved reconstruction at: ',global_dict["reconstruction_equalized_filepath"])

    direction_selector = widgets.RadioButtons(options=['X','Y','Z'], value='Z', style=style,layout=items_layout,description='Plot direction:',disabled=False)

    reg_checkbox    = Input(global_dict,"tomo_regularization",description = "Apply Regularization")
    reg_param       = Input(global_dict,"tomo_regularization_param",description = "Regularization Parameter",layout=items_layout)
    iter_slider     = Input(global_dict,"tomo_iterations",description = "Iterations", bounded=(1,200,2),slider=True,layout=slider_layout)
    widgets.interactive_output(update_cpus_GPUs,{"cpus":cpus_slider.widget,"GPUs":GPUs_slider.widget,"machine_selection":machine_selection})

    filename_field  = Input({"dummy_str":'reconstruction3Dphase'},"dummy_str",description = "Output Filename",layout=items_layout)
    tomo_threshold  = Input(global_dict,"tomo_threshold",description = "Value threshold",layout=items_layout)
    tomo_slice     = Input({"dummy_key":1},"dummy_key", description="Slice #", bounded=(1,10,1),slider=True,layout=slider_layout)
    algo_dropdown   = widgets.Dropdown(options=['EEM','EM', 'ART','FBP'], value='EEM',description='Algorithm:',layout=items_layout)
    load_selection  = widgets.RadioButtons(options=['Original', 'Equalized'], value='Original',style=style, layout=items_layout,description='Load:',disabled=False)

    widgets.interactive_output(update_paths,{'global_dict':fixed(global_dict),'dummy1':algo_dropdown,'dummy2':fixed(algo_dropdown)})

    load_recon_button = Button(description="Load recon slices",layout=buttons_layout,icon='play')
    load_recon_button.trigger(load_recon)

    plot_histogram_button = Button(description="Equalize Reconstruction",layout=buttons_layout,icon='play')
    plot_histogram_button.trigger(equalize)

    remove_outliers_slider    = Input(global_dict,"tomo_remove_outliers",  description="Remove Outliers", bounded=(0,10,1), slider=True,layout=slider_layout)
    remove_local_offset_field = Input(global_dict,"tomo_local_offset",  description='Bkg Offset [yu,yd,xl,xr]',layout=items_layout)
    # mask_final_tomo = Input(global_dict,"tomo_mask",  description='mask [yu,yd,xl,xr]',layout=items_layout)
    hist_max = Input({"dummy_key":0},"dummy_key",  description='Histogram Maximum',layout=items_layout)

    load_box = widgets.HBox([load_recon_button.widget,load_selection])
    threshold_box = widgets.VBox([remove_outliers_slider.widget,remove_local_offset_field.widget,tomo_threshold.widget,hist_max.widget,plot_histogram_button.widget])#, plot_histogram_button.widget])
    controls = widgets.VBox([reg_checkbox.widget,reg_param.widget,iter_slider.widget,direction_selector,hbar2,load_box,tomo_slice.widget,hbar2,threshold_box])
    box = widgets.HBox([controls,vbar,output])
    box = widgets.VBox([box,hbar,output2])
    return box 

def deploy_tabs(mafalda_session,tab1=folders_tab(),tab2=crop_tab(),tab3=unwrap_tab(),tab5=wiggle_tab(),tab6=tomo_tab(),tab7=equalizer_tab()):
    
    children_dict = {
    "Select and Sort"       : tab1,
    "Cropping"              : tab2,
    "Phase Unwrap"          : tab3,
    "Frame Equalizer"       : tab7,
    "Wiggle"                : tab5,
    "Tomography"            : tab6}
    
    
    def update_fields(global_dict,contrast_type,machine_selection):
        global_dict["contrast_type"] = contrast_type
        global_dict["machine_selection"]    = machine_selection
    

    def run_tomo(dummy,args=()):
        GPUs_slider, cpus_slider,jobname_field,queue_field, checkboxes = args

        global_dict["processing_steps"] = { "Sort":checkboxes[0].value , "Crop":checkboxes[1].value , "Unwrap":checkboxes[2].value, "Equalize Frames":checkboxes[3].value, "Wiggle":checkboxes[4].value, "Tomo":checkboxes[5].value, "Equalize Recon":checkboxes[6].value } # select steps when performing full recon
        
        slurm_filepath = os.path.join(global_dict["jupyter_folder"] ,'inputs',f'{username}_tomo_job.srm')

        jsonFile_path = os.path.join(global_dict["jupyter_folder"] ,'inputs', f'{username}_tomo_input.json')


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
            run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path=global_dict["jupyter_folder"] ,slurmFile = slurm_filepath,  jobName=jobname_field.widget.value,queue=queue_field.widget.value,GPUs=GPUs_slider.widget.value,cpus=cpus_slider.widget.value)

    def update_processing_steps(dictionary,sort_checkbox,crop_checkbox,unwrap_checkbox,wiggle_checkbox,tomo_checkbox,equalize_frames_checkbox,equalize_recon_checkbox):
        # "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "Wiggle":1, "Tomo":1 } # select steps when performing full recon
        dictionary["processing_steps"]["Sort"]            = sort_checkbox 
        dictionary["processing_steps"]["Crop"]            = crop_checkbox 
        dictionary["processing_steps"]["Unwrap"]          = unwrap_checkbox 
        dictionary["processing_steps"]["Equalize Frames"] = equalize_frames_checkbox 
        dictionary["processing_steps"]["Wiggle"]          = wiggle_checkbox 
        dictionary["processing_steps"]["Tomo"]            = tomo_checkbox 
        dictionary["processing_steps"]["Equalize Recon"]  = equalize_recon_checkbox 

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
                                os.path.join(global_dict["output_folder"],'03_frames_unwrapped')]
                                
        import shutil
        for folderpath in folderpaths_to_remove:
            print('Removing file/folder: ', folderpath)
            if os.path.isdir(folderpath) == True:
                shutil.rmtree(folderpath)
                print(f'Deleted {folderpath}\n')
            else:
                print(f'Directory {folderpath} does not exists. Skipping deletion...\n')



    def save_on_click(dummy):
        global_dict["contrast_type"]  = data_selection.value
        global_dict["tomo_algorithm"] = "EEM"#algo_dropdown.value
        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list
        json_filepath = os.path.join(global_dict["jupyter_folder"],'inputs',f'{username}_tomo_input.json') #INPUT
        print('Saving JSON input file at: ',json_filepath)
        with open(json_filepath, 'w') as file:
            json.dump(global_dict, file)
        
        from pprint import pprint
        pprint(global_dict,width=200,compact=True)
        print('\t Saved!')

    # def load_on_click(dummy):
    #     global global_dict
    #     from pprint import pprint
    #     pprint(global_dict,width=200,compact=True)
    #     json_filepath = os.path.join(global_dict["jupyter_folder"],'inputs', f'{username}_tomo_input.json') #INPUT
    #     with open(json_filepath) as json_file:
    #         global_dict = json.load(json_file)
    #     print("Inputs loaded from ",json_filepath)

    global mafalda
    mafalda = mafalda_session
    
    widgets.interactive_output(update_gpu_limits,{"machine_selection":machine_selection})

    global data_selection
    data_selection = widgets.RadioButtons(options=['Absorption', 'Phase'], value='Phase', layout={'width': '10%'},description='Contrast',disabled=False)
    widgets.interactive_output(update_fields,{'global_dict':fixed(global_dict),'contrast_type':data_selection,'machine_selection':machine_selection})
    widgets.interactive_output(update_paths, {'global_dict':fixed(global_dict),'dummy1':data_selection,'dummy2':machine_selection})

    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout_fixed,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    checkboxes      = [widgets.Checkbox(value=False, description=label,layout=checkbox_layout, style=style) for label in ["Select and Sort", "Cropping", "Phase Unwrap", "Frame Equalizer", "Wiggle", "Tomography", "Equalize Tomo"]]
    widgets.interactive_output(update_processing_steps,{'dictionary':fixed(global_dict),'sort_checkbox':checkboxes[0],'crop_checkbox':checkboxes[1],'unwrap_checkbox':checkboxes[2],'wiggle_checkbox':checkboxes[4],'tomo_checkbox':checkboxes[5],'equalize_frames_checkbox':checkboxes[3],'equalize_recon_checkbox':checkboxes[6]})

    queue_field     = Input({"dummy_str":'cat'},"dummy_str",description = "Machine Queue",layout=widgets.Layout( width='180px',border=standard_border))
    jobname_field   = Input({"dummy_str":f"{username}_tomo"},"dummy_str",description = "Slurm Job Name",layout=widgets.Layout( width='300px',border=standard_border))
    
    save_dict_button  = Button(description="Save inputs",layout=buttons_layout_fixed,icon='fa-floppy-o')
    save_dict_button.trigger(save_on_click)    

    # load_dict_button  = Button(description="Load inputs",layout=buttons_layout_fixed,icon='folder-open-o')
    # load_dict_button.trigger(load_on_click)  

    args = GPUs_slider,cpus_slider,jobname_field,queue_field, checkboxes
    start_tomo = Button(description="Start",layout=buttons_layout_fixed,icon='play')
    start_tomo.trigger(partial(run_tomo,args=args))

    slurm_box = widgets.HBox([data_selection,machine_selection,cpus_slider.widget,GPUs_slider.widget,queue_field.widget,jobname_field.widget])
    checkboxes_box  = widgets.HBox([widgets.HTML(f'<b><font size=2.0px>Steps to perform on cluster: \t \t  </b>' ),widgets.HBox([*checkboxes])])
    buttons_box = widgets.HBox([save_dict_button.widget,start_tomo.widget,delete_temporary_files_button.widget])
    box = widgets.VBox([hbar,slurm_box,checkboxes_box,buttons_box])
    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  


if __name__ == "__main__":
    pass
