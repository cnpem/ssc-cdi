from calendar import c
import os, json, ast, h5py
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import ipywidgets as widgets 
from ipywidgets import fixed

from .ptycho_fresnel import create_propagation_video
from .ptycho_processing import masks_application
from .misc import miqueles_colormap
from .jupyter import monitor_job_execution, call_cmd_terminal, Button, Input, update_imshow, slide_and_play, call_and_read_terminal

if 0: # paths for beamline use
    ptycho_folder     = "/ibira/lnls/beamlines/caterete/apps/ptycho-dev/" # folder with json template, and where to output jupyter files. path to output slurm logs as well
    pythonScript    = '/ibira/lnls/beamlines/caterete/apps/ssc-cdi/bin/sscptycho_main.py' # path with python script to run
else: # paths for GCC tests       
    ptycho_folder   = "/ibira/lnls/beamlines/caterete/apps/jupyter/" 
    pythonScript    = '~/ssc-cdi/bin/sscptycho_main.py' 

acquisition_folder = 'SS61'
output_folder = os.path.join('/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/', 'proc','recons',acquisition_folder) # changes with control

global_paths_dict = { "jupyter_folder"       : "/ibira/lnls/beamlines/caterete/apps/jupyter/",
                    "ptycho_folder"            : ptycho_folder,
                    "ptycho_script_path"       : pythonScript,
                    "template_json"            : "000000_template.json",
                    "slurm_filepath"           : os.path.join(ptycho_folder,'slurm_job.srm'), # path to create slurm_file
                    "json_filepath"            : os.path.join(ptycho_folder,'user_input.json'), # path with input json to run
                    "sinogram_filepath"        : os.path.join(output_folder,f'object_{acquisition_folder}.npy'), # path to load npy with first reconstruction preview
                    "cropped_sinogram_filepath": os.path.join(output_folder,f'object_{acquisition_folder}_cropped.npy'),
                    "probe_filepath"           : os.path.join(output_folder,f'probe_{acquisition_folder}.npy'), # path to load probe
                    "difpad_raw_mean_filepath"  : os.path.join(output_folder,'03_difpad_raw_mean.npy'), # path to load diffraction pattern
                    "flipped_difpad_filepath"  : os.path.join(output_folder,'03_difpad_raw_flipped_3072.npy'), # path to load diffraction pattern
                    "output_folder"            : output_folder

                }

global_dict = json.load(open(os.path.join(global_paths_dict["jupyter_folder"] ,global_paths_dict["template_json"]))) # load from template

output_dictionary = {}

############################################ Global Layout ###########################################################################

""" Standard styling definitions """
standard_border='1px none black'
vbar = widgets.HTML(value="""<div style="border-left:2px solid #000;height:500px"></div>""")
vbar2 = widgets.HTML(value="""<div style="border-left:2px solid #000;height:1000px"></div>""")
hbar = widgets.HTML(value="""<hr class="solid" 2px #000>""")
hbar2 = widgets.HTML(value="""<hr class="solid" 2px #000>""")
slider_layout = widgets.Layout(width='90%',border=standard_border)
slider_layout2 = widgets.Layout(width='30%',flex='flex-grow',border=standard_border)
slider_layout3 = widgets.Layout(display='flex', flex_flow='row',  align_items='flex-start', width='70%',border=standard_border)
items_layout = widgets.Layout( width='90%',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
items_layout2 = widgets.Layout( width='50%',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
checkbox_layout = widgets.Layout( width='150px',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
buttons_layout = widgets.Layout( width='90%',height="40px")     # override the default width of the button to 'auto' to let the button grow
center_all_layout = widgets.Layout(align_items='center',width='100%',border=standard_border) #align_content='center',justify_content='center'
box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
sliders_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
style = {'description_width': 'initial'}

def get_box_layout(width,flex_flow='column',align_items='flex-start',border=standard_border):
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
   
def update_cpus_gpus(cpus,gpus):
    global_dict["Threads"] = cpus

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

def delete_files(dummy):
    sinogram_path = global_dict["sinogram_path"].rsplit('/',1)[0]

    filepaths_to_remove = [ global_paths_dict["flipped_difpad_filepath"],  
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
    # monitor_job_execution(given_jobID,mafalda)
    
def run_ptycho(dummy):
    pythonScript = global_paths_dict["ptycho_script_path"]
    json_filepath = global_paths_dict["json_filepath"]
    slurm_filepath = global_paths_dict["slurm_filepath"]

    print(f'Running ptycho with {machine_selection.value}...')
    if machine_selection.value == 'Local':
        cmd = f'python3 {pythonScript} {json_filepath}'
        print('Running command: ',cmd)               
        call_and_read_terminal(cmd,mafalda,use_mafalda=False)
    elif machine_selection.value == "Cluster": 
        global jobNameField, jobQueueField, cpus, gpus
        jobName_value = jobNameField.widget.value
        queue_value   = jobQueueField.widget.value
        cpus_value    = cpus.widget.value
        gpus_value    = gpus.widget.value
        run_ptycho_from_jupyter(mafalda,pythonScript,json_filepath,output_path=global_paths_dict["output_folder"],slurm_filepath = slurm_filepath,jobName=jobName_value,queue=queue_value,gpus=gpus_value,cpus=cpus_value)

def load_json(dummy):
    json_path = os.path.join(global_paths_dict["jupyter_folder" ] ,global_paths_dict["000000_template.json"])
    template_dict = json.load(open(json_path))
    for key in template_dict:
        global_dict[key] = template_dict[key]

def create_label_widget(text):
    # label = widgets.Label(value=text)
    label = widgets.HTML(value=f"<b style='color:#00008B;font-size:18px;'>{text}</b>")
    return label

############################################ INTERFACE / GUI : TABS ###########################################################################

def inputs_tab():

    global global_dict

    def save_on_click(dummy,json_filepath="",dictionary={}):
        print('Saving input json file at: ',json_filepath)
        with open(json_filepath, 'w') as file:
            json.dump(dictionary, file)                                                    
        print('\t Saved!')


    def update_global_dict(proposal_path_str,acquisition_folders,projections,centerx,centery,detector_ROI,save_or_load_difpads,CentralMask_bool,CentralMask_radius,ProbeSupport_radius,ProbeSupport_centerX,ProbeSupport_centerY,PhaseUnwrap,PhaseUnwrap_iter,top_crop,bottom_crop,left_crop,right_crop,use_obj_guess,use_probe_guess,fresnel_number,DetectorPileup):

        if type(acquisition_folders) == type([1,2]): # if list, correct data type of this input
            pass 
        else: # if string
            acquisition_folders = ast.literal_eval(acquisition_folders)
            projections = ast.literal_eval(projections)

        global global_dict
        global_dict["ProposalPath"]        = proposal_path_str
        global_dict["Acquisition_Folders"] = acquisition_folders
        global_dict["Projections"]         = projections

        output_folder = os.path.join( global_dict["ProposalPath"].rsplit('/',3)[0] , 'proc','recons',acquisition_folders[0]) # changes with control

        global_paths_dict["jupyter_folder"]            = "/ibira/lnls/beamlines/caterete/apps/jupyter/"
        global_paths_dict["ptycho_folder"]             = ptycho_folder
        global_paths_dict["ptycho_script_path"]        = pythonScript
        global_paths_dict["template_json"]             = "000000_template.json"
        global_paths_dict["slurm_filepath"]            = os.path.join(ptycho_folder,'slurm_job.srm') # path to create slurm_file
        global_paths_dict["json_filepath"]             = os.path.join(ptycho_folder,'user_input.json') # path with input json to run
        global_paths_dict["sinogram_filepath"]         = os.path.join(output_folder,f'object_{acquisition_folders[0]}.npy') # path to load npy with first reconstruction preview
        global_paths_dict["cropped_sinogram_filepath"] = os.path.join(output_folder,f'object_{acquisition_folders[0]}_cropped.npy')
        global_paths_dict["probe_filepath"]            = os.path.join(output_folder,f'probe_{acquisition_folders[0]}.npy') # path to load probe
        global_paths_dict["difpad_raw_mean_filepath"]  = os.path.join(output_folder,'03_difpad_raw_mean.npy') # path to load diffraction pattern
        global_paths_dict["flipped_difpad_filepath"]   = os.path.join(output_folder,'03_difpad_raw_flipped_3072.npy') # path to load diffraction pattern
    
        global_paths_dict["output_folder"]             = output_folder

        global_dict["DifpadCenter"] = [centerx,centery]

        global_dict["DetectorROI"] = detector_ROI

        if save_or_load_difpads == "Save Diffraction Pattern":
            global_dict["SaveDifpads"] = 1
            global_dict["ReadRestauredDifpads"] = 0
        elif save_or_load_difpads == "Load Diffraction Pattern":
            global_dict["SaveDifpads"] = 0
            global_dict["ReadRestauredDifpads"] = 1

        global_dict["CentralMask"] = [CentralMask_bool,CentralMask_radius]
        global_dict["DetectorExposure"][0] = DetectorPileup 

        global_dict["f1"] = fresnel_number
        global_dict["ProbeSupport"] = [ProbeSupport_radius, ProbeSupport_centerX, ProbeSupport_centerY]

        global_dict["Phaseunwrap"][0] = PhaseUnwrap
        global_dict["Phaseunwrap"][1] = PhaseUnwrap_iter

        if [top_crop,bottom_crop] == [0,0]:
            global_dict["Phaseunwrap"][2] =  []
        else:
            global_dict["Phaseunwrap"][2] = [top_crop,bottom_crop]
        if [left_crop,right_crop] == [0,0]:
            global_dict["Phaseunwrap"][3] =  []
        else:
            global_dict["Phaseunwrap"][3] = [left_crop,right_crop]

        if use_obj_guess:
            global_dict["InitialObj"] = global_paths_dict["sinogram_filepath"]
        else: 
            global_dict["InitialObj"] = ''
        if use_probe_guess:
            global_dict["InitialProbe"] = global_paths_dict["probe_filepath"]
        else: 
            global_dict["InitialProbe"] = ''

    save_on_click_partial = partial(save_on_click,json_filepath=global_paths_dict["json_filepath"],dictionary=global_dict)

    global saveJsonButton
    saveJsonButton = Button(description="Save Inputs",layout=buttons_layout)
    saveJsonButton.trigger(save_on_click_partial)

    label1 = create_label_widget("Data Selection")
    proposal_path_str     = Input(global_dict,"ProposalPath",description="Proposal Path",layout=items_layout2)
    acquisition_folders   = Input(global_dict,"Acquisition_Folders",description="Data Folders",layout=items_layout2)
    projections           = Input(global_dict,"Projections",description="Projections",layout=items_layout2)
    
    label2 = create_label_widget("Restauration")
    global centerx, centery
    centerx    = Input({'dummy-key':1345},'dummy-key',bounded=(0,3072,1),slider=True,description="Center row",layout=slider_layout)
    centery    = Input({'dummy-key':1375},'dummy-key',bounded=(0,3072,1),slider=True,description="Center column",layout=slider_layout)
    center_box = widgets.Box([centerx.widget,centery.widget],layout=slider_layout3)

    detector_ROI          = Input({'dummy-key':1280},'dummy-key',bounded=(0,1536,1),slider=True,description="Diamenter (pixels)",layout=slider_layout2)
    # binning             = Input(global_dict,"Binning",bounded=(1,4,1),slider=True,description="Binning factor",layout=slider_layout2)
    save_or_load_difpads  = widgets.RadioButtons(options=['Save Diffraction Pattern', 'Load Diffraction Pattern'], value='Save Diffraction Pattern', layout={'width': '50%'},description='Save or Load')

    label3 = create_label_widget("Diffraction Pattern Processing")
    autocrop           = Input(global_dict,"AutoCrop",description="Auto Crop borders",layout=items_layout2)
    global CentralMask_radius, CentralMask_bool, DetectorPileup
    CentralMask_bool   = Input({'dummy-key':False},'dummy-key',description="Use Central Mask",layout=items_layout2)
    CentralMask_radius = Input({'dummy-key':3},'dummy-key',bounded=(0,100,1),slider=True,description="Central Mask Radius",layout=slider_layout)
    central_mask_box   = widgets.Box([CentralMask_bool.widget,CentralMask_radius.widget],layout=slider_layout3)
    DetectorPileup   = Input({'dummy-key':False},'dummy-key',description="Ignore Detector Pileup",layout=items_layout2)

    label4 = create_label_widget("Probe Adjustment")
    ProbeSupport_radius   = Input({'dummy-key':300},'dummy-key',bounded=(0,1000,10),slider=True,description="Probe Support Radius",layout=slider_layout2)
    ProbeSupport_centerX  = Input({'dummy-key':0},'dummy-key',bounded=(-100,100,10),slider=True,description="Probe Center X",layout=slider_layout2)
    ProbeSupport_centerY  = Input({'dummy-key':0},'dummy-key',bounded=(-100,100,10),slider=True,description="Probe Center Y",layout=slider_layout2)
    probe_box = widgets.Box([ProbeSupport_radius.widget,ProbeSupport_centerX.widget,ProbeSupport_centerY.widget],layout=slider_layout3)

    global fresnel_number
    fresnel_number = Input(global_dict,"f1",description="Fresnel Number",layout=items_layout2)
    Modes = Input(global_dict,"Modes",bounded=(0,30,1),slider=True,description="Probe Modes",layout=slider_layout2)

    label5 = create_label_widget("Ptychography")
    global use_obj_guess, use_probe_guess
    use_obj_guess = Input({"dummy_key":False},"dummy_key",layout=items_layout,description='Use OBJECT reconstruction as initial guess')
    use_probe_guess = Input({"dummy_key":False},"dummy_key",layout=items_layout,description='Use PROBE reconstruction as initial guess')
    Algorithm1 = Input(global_dict,"Algorithm1",description="Recon Algorithm 1",layout=items_layout2)
    Algorithm2 = Input(global_dict,"Algorithm2",description="Recon Algorithm 2",layout=items_layout2)
    Algorithm3 = Input(global_dict,"Algorithm3",description="Recon Algorithm 3",layout=items_layout2)

    label6 = create_label_widget("Post-processing")
    Phaseunwrap      = Input({'dummy-key':False},'dummy-key',description="Phase Unwrap",layout=checkbox_layout)
    Phaseunwrap_iter = Input({'dummy-key':3},'dummy-key',bounded=(0,20,1),slider=True,description="Gradient Removal Iterations",layout=slider_layout2)
    phase_unwrap_box = widgets.Box([Phaseunwrap.widget,Phaseunwrap_iter.widget],layout=slider_layout3)
    global top_crop, bottom_crop,left_crop,right_crop # variables are reused in crop tab
    top_crop      = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1),  description="Top",   slider=True,layout=slider_layout)
    bottom_crop   = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1),  description="Bottom",   slider=True,layout=slider_layout)
    left_crop     = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1),description="Left", slider=True,layout=slider_layout)
    right_crop    = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1),description="Right", slider=True,layout=slider_layout)

    FRC = Input(global_dict,"FRC",description="FRC: Fourier Ring Correlation",layout=items_layout2)

    widgets.interactive_output(update_global_dict,{'proposal_path_str':proposal_path_str.widget,
                                                    'acquisition_folders': acquisition_folders.widget,
                                                    'projections': projections.widget,                                                    
                                                    'centerx':centerx.widget,
                                                    'centery':centery.widget,
                                                    'detector_ROI':detector_ROI.widget,
                                                    'save_or_load_difpads':save_or_load_difpads,
                                                    'CentralMask_bool': CentralMask_bool.widget,
                                                    'CentralMask_radius': CentralMask_radius.widget,
                                                    'ProbeSupport_radius': ProbeSupport_radius.widget,
                                                    'ProbeSupport_centerX': ProbeSupport_centerX.widget,
                                                    'ProbeSupport_centerY': ProbeSupport_centerY.widget,
                                                    'PhaseUnwrap': Phaseunwrap.widget,
                                                    'PhaseUnwrap_iter': Phaseunwrap_iter.widget,
                                                    'top_crop': top_crop.widget,
                                                    'bottom_crop': bottom_crop.widget,
                                                    'left_crop': left_crop.widget,
                                                    'right_crop': right_crop.widget,
                                                    "use_obj_guess": use_obj_guess.widget,
                                                    "use_probe_guess":use_probe_guess.widget,
                                                    "fresnel_number":fresnel_number.widget,
                                                    "DetectorPileup":DetectorPileup.widget
                                                     })

    box = widgets.Box([saveJsonButton.widget,label1,proposal_path_str.widget,acquisition_folders.widget,projections.widget,label2,center_box,detector_ROI.widget,save_or_load_difpads],layout=box_layout)
    box = widgets.Box([box,label3,autocrop.widget,central_mask_box,DetectorPileup.widget,label4,probe_box,fresnel_number.widget,Modes.widget,label5,use_obj_guess.widget,use_probe_guess.widget,Algorithm1.widget,Algorithm2.widget,Algorithm3.widget,label6,phase_unwrap_box,FRC.widget],layout=box_layout)

    return box

def mask_tab():
    
    initial_image = np.random.random((10,10)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        subplot.set_title('Raw')
        figure.canvas.header_visible = False 
        plt.show()

    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots()
        subplot3.imshow(initial_image,cmap='gray')
        subplot.set_title('Masked')
        figure3.canvas.header_visible = False 
        plt.show()


    def load_frames(dummy):
        global sinogram
        from matplotlib.colors import LogNorm
        print("Loading difpad from: ",global_paths_dict["difpad_raw_mean_filepath"] )
        difpad = np.load(global_paths_dict["difpad_raw_mean_filepath"] ) 
        masked_difpad = difpad
        mask = h5py.File(os.path.join(global_dict["ProposalPath"],global_dict["Acquisition_Folders"][0],'images','mask.hdf5'), 'r')['entry/data/data'][()][0, 0, :, :]
        masked_difpad[mask ==1] = -1 # Apply Mask
        subplot.imshow(difpad,cmap='jet',norm=LogNorm())
        subplot3.imshow(masked_difpad,cmap='jet',norm=LogNorm())


    load_frames_button  = Button(description="Load Diffraction Patterns",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    objects_box = widgets.HBox([output,output3])
    box = widgets.VBox([buttons_box,objects_box])

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

        print(global_dict)
        print(global_paths_dict)
        image = np.load(global_paths_dict['flipped_difpad_filepath'])
        widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                                'output_dictionary':fixed(output_dictionary),'image':fixed(image),
                                                'key1':fixed('DifpadCenter'),'key2':fixed('CentralMask'),'key3':fixed('DetectorExposure'),
                                                'cx':centerx.widget,'cy':centery.widget,
                                                'button':CentralMask_bool.widget,
                                                'exposure':DetectorPileup.widget,
                                                'radius':CentralMask_radius.widget})

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
    sliders_box = widgets.HBox([centerx.widget,centery.widget,CentralMask_radius.widget],layout=box_layout)
    controls = widgets.Box([load_difpad_button.widget,sliders_box,CentralMask_bool.widget,DetectorPileup.widget],layout=get_box_layout('500px'))
    box = widgets.HBox([controls,vbar,output])
    return box

def fresnel_tab():
    
    image_list, fresnel_number_list = [np.random.random((5,5))], [0]

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(image_list[0],cmap='jet') # initialize
        subplot.set_title('Propagated Probe')
        figure.canvas.header_visible = False 
        plt.show()

    def update_probe_plot(fig,subplot,image_list,frame_list,index):
        subplot.clear()
        subplot.set_title(f'Frame #: {frame_list[index]:.1e}')
        subplot.imshow(image_list[index],cmap='jet')
        fig.canvas.draw_idle()

    def on_click_propagate(dummy):
    
        print('Propagating probe...')
        image_list, fresnel_number_list = create_propagation_video(global_paths_dict['probe_filepath'],
                                                        starting_f_value=starting_f_value,
                                                        ending_f_value=ending_f_value,
                                                        number_of_frames=number_of_frames,
                                                        jupyter=True)
        
        play_control.widget.max, selection_slider.widget.max = len(image_list)-1, len(image_list)-1

        widgets.interactive_output(update_probe_plot,{'fig':fixed(figure),'subplot':fixed(subplot),'image_list':fixed(image_list),'frame_list':fixed(fresnel_number_list),'index':selection_slider.widget})
        print('\t Done!')

    def update_values(n_frames,start_f,end_f,power):
        global starting_f_value, ending_f_value, number_of_frames
        starting_f_value=-start_f*10**power
        ending_f_value=-end_f*10**power
        number_of_frames=int(n_frames)
        label.value = r"Propagating from f = {0}$\times 10^{{{1}}}$ to {2}$\times 10^{{{1}}}$".format(start_f,power,end_f)

    play_box, selection_slider,play_control = slide_and_play(label="")

    power   = Input( {'dummy-key':-4}, 'dummy-key', bounded=(-10,10,1),  slider=True, description=r'Exponent'       ,layout=items_layout)
    start_f = Input( {'dummy-key':-1}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='Start f-value'   ,layout=items_layout)
    end_f   = Input( {'dummy-key':-9}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='End f-value'     ,layout=items_layout)
    n_frames= Input( {'dummy-key':100},'dummy-key', bounded=(10,200,10), slider=True, description='Number of Frames',layout=items_layout)

    label = widgets.Label(value=r"Propagating from f = {0} $\times 10^{{{1}}}$ to {2} $\times 10^{{{1}}}$".format(start_f,power,end_f),layout=items_layout)

    widgets.interactive_output(update_values,{'n_frames':n_frames.widget,'start_f':start_f.widget,'end_f':end_f.widget,'power':power.widget})
    propagate_button = Button(description=('Propagate Probe'),layout=buttons_layout)
    propagate_button.trigger(on_click_propagate)

    box = widgets.Box([n_frames.widget, power.widget, start_f.widget,end_f.widget,label,propagate_button.widget,fresnel_number.widget],layout=get_box_layout('700px'))
    play_box = widgets.VBox([play_box,output],layout=box_layout)
    box = widgets.HBox([box,vbar,play_box])
    return box

def crop_tab():

    initial_image = np.random.random((100,100)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()
    
    def load_frames(dummy):
        global sinogram
        
        print("Loading sinogram from: ",global_paths_dict["sinogram_filepath"] )
        sinogram = np.load(global_paths_dict["sinogram_filepath"] ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        print(selection_slider.widget.max, selection_slider.widget.value )
        play_control.widget.max = selection_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
        print(top_crop.widget,left_crop.widget,right_crop.widget,bottom_crop.widget)
        # widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})

    def save_cropped_sinogram(dummy):
        cropped_sinogram = sinogram[:,top_crop.widget.value:-bottom_crop.widget.value,left_crop.widget.value:-right_crop.widget.value]
        print('Saving cropped frames...')
        np.save(global_paths_dict['cropped_sinogram_filepath'],cropped_sinogram)
        print('\t Saved!')

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    save_cropped_frames_button = Button(description="Save cropped frames",layout=buttons_layout,icon='fa-floppy-o') 
    save_cropped_frames_button.trigger(save_cropped_sinogram)
    
    buttons_box = widgets.Box([load_frames_button.widget,save_cropped_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar,output])
    return box

def ptycho_tab():

    def view_jobs(dummy):
        output = call_and_read_terminal('squeue',mafalda)
        print(output.decode("utf-8"))
    def cancel_job(dummy):
        print(f'Cancelling job {job_number.widget.value}')    
        call_and_read_terminal(f'scancel {job_number.widget.value}',mafalda)

    job_number = Input({"dummy-key":00000},"dummy-key",description="Job ID number",layout=items_layout)

    view_jobs_button = Button(description='List Jobs',layout=buttons_layout,icon='fa-eye')
    view_jobs_button.trigger(view_jobs)

    cancel_job_button = Button(description='Cancel Job',layout=buttons_layout,icon='fa-stop-circle')
    cancel_job_button.trigger(cancel_job)    

    run_button = Button(description='Run Ptycho',layout=buttons_layout,icon='play')
    run_button.trigger(run_ptycho)

    job_box = widgets.VBox([job_number.widget,view_jobs_button.widget,cancel_job_button.widget])
    box = widgets.Box([use_obj_guess.widget,use_probe_guess.widget,saveJsonButton.widget,run_button.widget],layout=get_box_layout('500px'))
    box = widgets.HBox([box,job_box])

    return box

def reconstruction_tab():
    
    initial_image = np.ones((100,100)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()

    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots()
        subplot3.imshow(initial_image,cmap='gray')
        figure3.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots()
        subplot2.imshow(initial_image,cmap='gray')
        figure2.canvas.header_visible = False 
        plt.show()

    def load_frames(dummy):
        global sinogram
        print("Loading sinogram from: ",global_paths_dict["sinogram_filepath"] )
        sinogram = np.load(global_paths_dict["sinogram_filepath"] ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.abs(sinogram)),'figure':fixed(figure3),'subplot':fixed(subplot3),'title':fixed(True), 'frame_number': selection_slider.widget})

        probe = np.abs(np.load(global_paths_dict["probe_filepath"]))[0] 
        selection_slider2.widget.max, selection_slider2.widget.value = probe.shape[0]-1, probe.shape[0]//2
        play_control2.widget.max = selection_slider2.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(probe),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed(True), 'cmap':fixed('jet'), 'frame_number': selection_slider.widget})


    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    play_box2, selection_slider2,play_control2 = slide_and_play(label="Probe Selector")

    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))

    controls_box = widgets.Box([play_box],layout=get_box_layout('500px'))
    objects_box = widgets.HBox([output,output3])
    object_box = widgets.VBox([controls_box,objects_box])
    controls_box2 = widgets.Box([play_box2],layout=get_box_layout('500px'))
    probe_box = widgets.VBox([controls_box2,output2])
    box = widgets.HBox([object_box,vbar,probe_box])
    box = widgets.VBox([buttons_box,box])

    return box

def deploy_tabs(mafalda_session,tab2=inputs_tab(),tab3=center_tab(),tab4=fresnel_tab(),tab5=ptycho_tab(),tab6=reconstruction_tab(),tab1=crop_tab(),tab7=mask_tab()):
    
    children_dict = {
    "Ptycho Inputs"     : tab2,
    "Mask"              : tab7,
    "Find Center"       : tab3,
    "Probe Propagation" : tab4,
    "Crop"              : tab1,
    "Ptychography"      : tab5,
    "Reconstruction"    : tab6
    }
    
    global mafalda
    mafalda = mafalda_session

    load_json_button  = Button(description="Reset JSON",layout=buttons_layout,icon='folder-open-o')
    load_json_button.trigger(load_json)
    
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Cluster', layout={'width': '70%'},description='Machine',disabled=False)

    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    global jobNameField, jobQueueField
    jobNameField  = Input({'dummy_key':'CateretePtycho'},'dummy_key',description="Insert slurm job name:")
    jobQueueField = Input({'dummy_key':'cat-proc'},'dummy_key',description="Insert machine queue name:")
    global cpus, gpus
    gpus = Input({'dummy_key':1},'dummy_key',bounded=(0,4,1),  slider=True,description="Insert # of GPUs to use:")
    cpus = Input({'dummy_key':32},'dummy_key',bounded=(1,128,1),slider=True,description="Insert # of CPUs to use:")
    widgets.interactive_output(update_cpus_gpus,{"cpus":cpus.widget,"gpus":gpus.widget})

    box = widgets.HBox([machine_selection,load_json_button.widget,delete_temporary_files_button.widget])
    boxSlurm = widgets.HBox([gpus.widget,cpus.widget,jobQueueField.widget,jobNameField.widget])
    box = widgets.VBox([box,boxSlurm])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  



if __name__ == "__main__":
    pass
