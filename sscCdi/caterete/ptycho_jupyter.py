import os, json, ast
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import ipywidgets as widgets 
from ipywidgets import fixed 

from .ptycho_fresnel import create_propagation_video
from .ptycho_processing import masks_application
from .misc import miqueles_colormap
from .jupyter import call_and_read_terminal, monitor_job_execution, call_cmd_terminal, VideoControl, Button, Input, update_imshow

if 0: # paths for beamline use
    ptycho_folder     = "/ibira/lnls/beamlines/caterete/apps/ptycho-dev/" # folder with json template, and where to output jupyter files. path to output slurm logs as well
    pythonScript    = '/ibira/lnls/beamlines/caterete/apps/ssc-cdi/bin/sscptycho_main.py' # path with python script to run
else: # paths for GCC tests       
    ptycho_folder   = "/ibira/lnls/beamlines/caterete/apps/jupyter/" 
    pythonScript    = '~/ssc-cdi/bin/sscptycho_main.py' 

acquisition_folder = 'SS61'
output_folder = os.path.join('/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/', 'proc','recons',acquisition_folder) # changes with control

global_paths_dict = { "jupyter_folder"                   : "/ibira/lnls/beamlines/caterete/apps/jupyter/",
                      "ptycho_folder"                    : ptycho_folder,
                      "ptycho_script_path"               : pythonScript,
                      "template_file"                    : "000000_template.json",
                      "slurm_filepath"                   : os.path.join(ptycho_folder,'slurm_job.srm'), # path to create slurm_file
                      "json_filepath"                    : os.path.join(ptycho_folder,'user_input.json'), # path with input json to run
                      "path_to_npy_frames"               : os.path.join(output_folder,f'phase_{acquisition_folder}.npy'), # path to load npy with first reconstruction preview
                      "path_to_probefile"                : os.path.join(output_folder,f'probe_{acquisition_folder}.npy'), # path to load probe
                      "path_to_diffraction_pattern_file" : os.path.join(output_folder,'03_difpad_raw_flipped_3072.npy') # path to load diffraction pattern
                    }

global_dict = json.load(open(os.path.join(global_paths_dict["jupyter_folder"] ,global_paths_dict["template_file"]))) # load from template

output_dictionary = {}

############################################ Global Layout ###########################################################################

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

#######################################################################################################################


def write_slurm_file(python_script_path,json_filepath_path,output_path="",slurm_filepath = 'slurmJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
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

python3 {python_script_path} {json_filepath_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
# python3 {python_script_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}

"""
    
    with open(slurm_filepath,'w') as the_file:
        the_file.write(string)
    
    return slurm_filepath
       
def plotshow(figure,subplot,image,title="",figsize=(8,8),savepath=None,show=False):
    subplot.clear()
    cmap, colors, bounds, norm = miqueles_colormap(image)
    handle = subplot.imshow(image, interpolation='nearest', cmap = cmap, norm=norm)
    if title != "":
        subplot.set_title(title)
    if show:
        plt.show()
    figure.canvas.draw_idle()

def update_mask(figure, subplot,output_dictionary,image,key1,key2,key3,cx,cy,button,exposure,radius):
    output_dictionary[key1] = [cx,cy]
    output_dictionary[key2] = [button,radius]
    output_dictionary[key3] = [exposure,0.15]
    if exposure == True or button == True:
        image2, _ = masks_application(np.copy(image), output_dictionary)
        plotshow(figure,subplot,image2)
    else:
        plotshow(figure,subplot,image)
   
def update_values(n_frames,start_f,end_f,power):
    global starting_f_value
    global ending_f_value
    global number_of_frames
    starting_f_value=-start_f*10**power
    ending_f_value=-end_f*10**power
    number_of_frames=int(n_frames)
    
def update_dic(output_dictionary,key,boxvalue):
    output_dictionary[key]  = boxvalue 


############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################

def delete_files(dummy):
    sinogram_path = global_dict["sinogram_path"].rsplit('/',1)[0]

    filepaths_to_remove = [ global_dict["ordered_angles_filepath"],  
                            global_dict["ordered_object_filepath"] , 
                            global_dict["cropped_sinogram_filepath"],
                            global_dict["unwrapped_sinogram_filepath"],
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

def run_ptycho_from_jupyter(mafalda,python_script_path,json_filepath_path,output_path="",slurm_filepath = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    slurm_file = write_slurm_file(python_script_path,json_filepath_path,output_path,slurm_filepath,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)
    
def run_ptycho(dummy,args=()):
    mafalda,pythonScript,json_filepath,outputFolder,slurm_filepath,jobName,queue,gpus,cpus = args
    run_ptycho_from_jupyter(mafalda,pythonScript,json_filepath,output_path=outputFolder,slurm_filepath = slurm_filepath,jobName=jobName,queue=queue,gpus=gpus,cpus=cpus)

def load_json(dummy):
    json_path = os.path.join("/ibira/lnls/beamlines/caterete/apps/jupyter/" ,"000000_template.json")
    template_dict = json.load(open(json_path))
    for key in template_dict:
        global_dict[key] = template_dict[key]


############################################ INTERFACE / GUI : TABS ###########################################################################

def slurm_tab():

    jobNameField           = widgets.Text(value='myJobName',description="Insert slurm job name:",style=style)
    jobQueueField          = widgets.Text(value='cat-proc',description="Insert machine queue name:",style=style)
    gpusField              = widgets.BoundedIntText(value=1,min=0,max=4,description="Insert # of gpus to use:",style=style)
    cpusField              = widgets.BoundedIntText(value=32,min=1,max=128,description="Insert # of cpus to use:",style=style)

    box = widgets.VBox([jobNameField,jobQueueField,gpusField,cpusField])

    return box

def inputs_tab():


    def save_on_click(dummy,json_filepath="",dictionary={}):
        print(json_filepath)
        with open(json_filepath, 'w') as file:
            json.dump(dictionary, file)

    save_on_click_partial = partial(save_on_click,json_filepath=global_paths_dict["json_filepath"],dictionary=global_dict)

    global saveJsonButton
    saveJsonButton = Button(description="Save Dictinary to json",layout=buttons_layout)
    saveJsonButton.trigger(save_on_click_partial)

    keys_list = [key for key in global_dict]
    fields_dict = {}
    box = widgets.VBox([saveJsonButton.widget])
    for counter in range(0,len(keys_list)): # deployt all interactive fields    
        fields_dict_key = f'field_{counter}'
        fields_dict[fields_dict_key] = Input(global_dict,keys_list[counter],description=keys_list[counter])
        box = widgets.VBox([box,fields_dict[fields_dict_key].widget])

    return box

def center_tab():
    
    output = widgets.Output()
    image = np.load(global_paths_dict['path_to_diffraction_pattern_file'])
    centered_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',width='100%')

    with output:
        figure, subplot = plt.subplots(figsize=(8,8),constrained_layout=True)
        plotshow(figure,subplot,image,title="",show=True)
    

    """ Difpad center boxes """
    center_x_box = widgets.IntText(value=1400,min=0,max=3072, description='Center Row pixel:', disabled=False,layout=centered_box_layout)
    center_y_box = widgets.IntText(value=1400,min=0,max=3072, description='Center Column pixel:', disabled=False,layout=centered_box_layout)

    """ Central mask radius box """
    mask_size_box = widgets.BoundedIntText(value=50,min=0,max=3072, description='Mask radius (pixels):', disabled=False,layout=centered_box_layout)
    central_mask_checkbox = widgets.Checkbox(value=False,description='Central-mask')
    exposure_checkbox = widgets.Checkbox(value=False,description='Exposure')
    widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                            'output_dictionary':fixed(output_dictionary),'image':fixed(image),
                                            'key1':fixed('DifpadCenter'),'key2':fixed('CentralMask'),'key3':fixed('DetectorExposure'),
                                            'cx':center_x_box,'cy':center_y_box,
                                            'button':central_mask_checkbox,
                                            'exposure':exposure_checkbox,
                                            'radius':mask_size_box})


    box1 = widgets.VBox([center_x_box,center_y_box])
    box3 = widgets.VBox([central_mask_checkbox,exposure_checkbox])
    box2 = widgets.VBox([mask_size_box,box3])
    controls = widgets.VBox([box1,box2])
    box = widgets.HBox([output,controls])
    return box

def fresnel_tab():
        
    def on_click_propagate(dummy,args=()):
    
        path_to_probefile,starting_f_value,ending_f_value,number_of_frames,play,slider,fig, ax1 = args

        image_list, f1_list = create_propagation_video(path_to_probefile,
                                starting_f_value=starting_f_value,
                                ending_f_value=ending_f_value,
                                number_of_frames=number_of_frames,
                                jupyter=True)

        play.max = len(image_list)-1
        slider.max = len(image_list)-1

        widgets.interactive_output(update_imshow,{'fig':fixed(fig),'ax1':fixed(ax1),'image_list':fixed(image_list),'f1_list':fixed(f1_list),'index':play})



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
        plt.show()

    args = (global_paths_dict['path_to_probefile'],starting_f_value,ending_f_value,number_of_frames,play,slider,fig,ax1)
    start_ptycho_button.on_click(partial(on_click_propagate,args=args))

    box3 = widgets.VBox([play_box,output])
    box = widgets.HBox([box2,box3])
    return box

def ptycho_tab():

    run_button = Button(description='Run Ptycho',layout=buttons_layout,icon='play')
    run_button.trigger(run_ptycho)

    run_button = Button(description='Run Ptycho',layout=buttons_layout,icon='play')
    run_button.trigger(run_ptycho)

    use_obj_guess = Input({"dummy_key":False},"dummy_key",layout=items_layout,description='Use object reconstruction as initial guess')
    use_probe_guess = Input({"dummy_key":False},"dummy_key",layout=items_layout,description='Use probe reconstruction as initial guess')

    box = widgets.VBox([saveJsonButton.widget,use_obj_guess.widget,use_probe_guess.widget,run_button.widget])

    return box

def reconstruction_tab():
    
    frame_time_in_milisec = 100
    output = widgets.Output()

    centered_box_layout = widgets.Layout(flex_flow='column',align_items='center',width='30%')

    def update_imshow(fig,ax1,image_list,frame_list,index):
        ax1.clear()
        ax1.set_title(f'Frame #: {frame_list[index]:.1f}')
        ax1.imshow(image_list[index],cmap='gray')
        fig.canvas.draw_idle()

    def on_click_load(dummy,args=()): 

        path_to_npy_frames, fig, ax1, play, slider  = args

        matrix = np.load(path_to_npy_frames)

        image_list = [ matrix[i,:,:] for i in range(0,matrix.shape[0])]
        # image_list = [ matrix[:,i,:] for i in range(0,matrix.shape[1])]
        # image_list = [ matrix[:,:,i] for i in range(0,matrix.shape[2])]

        play.max = len(image_list)-1
        slider.max = len(image_list)-1

        frame_list = [ i for i in range(matrix.shape[0])]

        out = widgets.interactive_output(update_imshow,{'fig':fixed(fig),'ax1':fixed(ax1),'image_list':fixed(image_list),'frame_list':fixed(frame_list),'index':play})

    image_list = [np.ones((100,100))]

    play = widgets.Play(
        value=0,
        min=0,
        max=len(image_list)-1,
        step=1,
        interval=frame_time_in_milisec,
        description="Press play",
        disabled=False
    )

    slider = widgets.IntSlider(min=0,max=len(image_list)-1)
    widgets.jslink((play, 'value'), (slider, 'value'))
    play_box = widgets.HBox([play, slider])
        

    with output:
        fig = plt.figure(figsize=(10,5))
        ax1  = fig.add_subplot(1, 1, 1)
        ax1.imshow(image_list[0],cmap='gray') # initialize
        plt.show()

    args = (global_paths_dict['path_to_npy_frames'], fig, ax1,play,slider)

    loadButton = widgets.Button(description="Load Object Preview",layout=widgets.Layout(width='50%',height='50px'))
    on_click_load_partial = partial(on_click_load,args=args)
    loadButton.on_click(on_click_load_partial)
    box1 = widgets.VBox([play_box,output])
    box = widgets.VBox([loadButton,box1])

    return box

def merge_tab():

    box = widgets.VBox([])

    return box 

def deploy_tabs(mafalda_session,tab1=slurm_tab(),tab2=inputs_tab(),tab3=center_tab(),tab4=fresnel_tab(),tab5=ptycho_tab(),tab6=reconstruction_tab(),tab7=merge_tab()):
    
    children_dict = {
    "Job Inputs"        : tab1,
    "Ptycho Inputs"     : tab2,
    "Find Center"       : tab3,
    "Probe Propagation" : tab4,
    "Ptychography"      : tab5,
    "Reconstruction"    : tab6,
    "Merge sinograms"   : tab7
    }
    
    global mafalda
    mafalda = mafalda_session

    load_json_button  = Button(description="Reset JSON",layout=buttons_layout,icon='folder-open-o')
    load_json_button.trigger(load_json)
    
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Local', layout={'width': '70%'},description='Machine',disabled=False)

    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    box = widgets.HBox([machine_selection,load_json_button.widget,delete_temporary_files_button.widget])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  



if __name__ == "__main__":
    pass
