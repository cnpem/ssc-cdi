import os, json, ast
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import ipywidgets as widgets 
from ipywidgets import fixed 

from .ptycho_fresnel import create_propagation_video
from .ptycho_processing import masks_application
from .misc import miqueles_colormap
from .jupyter import call_and_read_terminal, monitor_job_execution, call_cmd_terminal, VideoControl, Button, Input, update_imshow, slide_and_play

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
standard_border='1px solid black'
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

def get_box_layout(width,flex_flow='column',align_items='center',border='1px none black'):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)

############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################

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
   
def update_dict_entry(dictionary,key,boxvalue):
    dictionary[key]  = boxvalue 

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
        fields_dict[fields_dict_key] = Input(global_dict,keys_list[counter],description=keys_list[counter],layout=items_layout)
        box = widgets.VBox([box,fields_dict[fields_dict_key].widget])

    return box

def center_tab():


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

    def load_difpad(dummy):
        image = np.load(global_paths_dict['path_to_diffraction_pattern_file'])
        widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                                'output_dictionary':fixed(output_dictionary),'image':fixed(image),
                                                'key1':fixed('DifpadCenter'),'key2':fixed('CentralMask'),'key3':fixed('DetectorExposure'),
                                                'cx':center_x_box.widget,'cy':center_y_box.widget,
                                                'button':central_mask_checkbox,
                                                'exposure':exposure_checkbox,
                                                'radius':mask_size_box.widget})

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
        figure,subplot.imshow(np.random.random((4,4)))
        subplot.set_title('Diffraction Pattern')
        figure.canvas.header_visible = False 
        plt.show()

    load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    load_difpad_button.trigger(load_difpad)

    """ Difpad center boxes """
    center_x_box  = Input({'dummy-key':1400},'dummy-key',bounded=(0,3072,1), description='Center Row pixel:',layout=items_layout)
    center_y_box  = Input({'dummy-key':1400},'dummy-key',bounded=(0,3072,1), description='Center Column pixel:',layout=items_layout)
    mask_size_box = Input({'dummy-key':5},   'dummy-key',bounded=(0,50,1), slider=True,description='Mask radius (pixels):',layout=items_layout)
    central_mask_checkbox = widgets.Checkbox(value=False,description='Central-mask',layout=checkbox_layout,style=style)
    exposure_checkbox     = widgets.Checkbox(value=False,description='Exposure',    layout=checkbox_layout,style=style)

    controls = widgets.Box([load_difpad_button.widget,center_x_box.widget,center_y_box.widget,mask_size_box.widget,central_mask_checkbox,exposure_checkbox],layout=box_layout)
    box = widgets.HBox([controls,vbar,output])
    return box

def fresnel_tab():
    
    image_list, f1_list = [np.random.random((5,5))], [0]

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(image_list[0],cmap='jet') # initialize
        subplot.set_title('Propagated Probe')
        figure.canvas.header_visible = False 
        plt.show()

    def update_probe_plot(fig,subplot,image_list,frame_list,index):
        subplot.clear()
        subplot.set_title(f'Frame #: {frame_list[index]:.1f}')
        subplot.imshow(image_list[index],cmap='jet')
        fig.canvas.draw_idle()

    def on_click_propagate(dummy):
    
        print('Propagating probe...')
        image_list, f1_list = create_propagation_video(global_paths_dict['path_to_probefile'],
                                                        starting_f_value=starting_f_value,
                                                        ending_f_value=ending_f_value,
                                                        number_of_frames=number_of_frames,
                                                        jupyter=True)
        
        play_control.widget.max, selection_slider.widget.max = len(image_list)-1, len(image_list)-1

        print(global_paths_dict['path_to_probefile'])
        print(f1_list)
        widgets.interactive_output(update_probe_plot,{'fig':fixed(figure),'subplot':fixed(subplot),'image_list':fixed(image_list),'frame_list':fixed(f1_list),'index':selection_slider.widget})
        print('\t Done!')

    def update_values(n_frames,start_f,end_f,power):
        global starting_f_value, ending_f_value, number_of_frames
        starting_f_value=-start_f*10**power
        ending_f_value=-end_f*10**power
        number_of_frames=int(n_frames)
        label.value = r"Propagating from f = {0}$\times 10^{{{1}}}$ to {2}$\times 10^{{{1}}}$".format(start_f,power,end_f)


    play_box, selection_slider,play_control = slide_and_play(label="")

    power   = Input( {'dummy-key':-4}, 'dummy-key', bounded=(-10,10,1),  slider=True, description=r'Exponent $10^n$'    ,layout=items_layout)
    start_f = Input( {'dummy-key':-1}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='Start f-value'   ,layout=items_layout)
    end_f   = Input( {'dummy-key':-9}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='End f-value'      ,layout=items_layout)
    n_frames= Input( {'dummy-key':100},'dummy-key', bounded=(10,200,10), slider=True, description='Number of Frames',layout=items_layout)

    label = widgets.Label(value=r"Propagating from f = {0} $\times 10^{{{1}}}$ to {2} $\times 10^{{{1}}}$".format(start_f,power,end_f),layout=items_layout)

    widgets.interactive_output(update_values,{'n_frames':n_frames.widget,'start_f':start_f.widget,'end_f':end_f.widget,'power':power.widget})
    propagate_button = Button(description=('Propagate Probe'),layout=buttons_layout)
    propagate_button.trigger(on_click_propagate)

    """ Fresnel Number box """
    fresnel_box = Input({'dummy-key':-0.001},'dummy-key', description='Chosen Fresnel Number (float)',layout=items_layout)
    widgets.interactive_output(update_dict_entry,{'dictionary':fixed(output_dictionary),'key':fixed('f1'),'boxvalue':fresnel_box.widget})
    
    box = widgets.VBox([n_frames.widget, power.widget, start_f.widget,end_f.widget,label,propagate_button.widget,fresnel_box.widget],layout=box_layout)
    play_box = widgets.VBox([play_box,output],layout=box_layout)
    box = widgets.HBox([box,vbar,play_box])
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

    box_layout = widgets.Layout(flex_flow='column',align_items='center',width='30%')

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

def deploy_tabs(mafalda_session,tab2=inputs_tab(),tab3=center_tab(),tab4=fresnel_tab(),tab5=ptycho_tab(),tab6=reconstruction_tab(),tab7=merge_tab()):
    
    children_dict = {
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

    jobNameField  = Input({'dummy_key':'CateretePtycho'},'dummy_key',description="Insert slurm job name:")
    jobQueueField = Input({'dummy_key':'cat-proc'},'dummy_key',description="Insert machine queue name:")
    gpusField     = Input({'dummy_key':1},'dummy_key',bounded=(0,4,1),  slider=True,description="Insert # of GPUs to use:")
    cpusField     = Input({'dummy_key':32},'dummy_key',bounded=(1,128,1),slider=True,description="Insert # of CPUs to use:")

    box = widgets.HBox([machine_selection,load_json_button.widget,delete_temporary_files_button.widget])
    boxSlurm = widgets.HBox([gpusField.widget,cpusField.widget,jobQueueField.widget,jobNameField.widget])
    box = widgets.VBox([box,boxSlurm])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  



if __name__ == "__main__":
    pass
