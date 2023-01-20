from calendar import c
import os, json, ast, h5py
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import ipywidgets as widgets 
from ipywidgets import fixed

import sscCdi, sscPimega, sscRaft, sscRadon, sscResolution

from .ptycho_fresnel import create_propagation_video
from .ptycho_processing import masks_application
from .misc import miqueles_colormap
from .jupyter import call_cmd_terminal, Button, Input, update_imshow, slide_and_play, call_and_read_terminal
from .unwrap import phase_unwrap

from .misc import create_directory_if_doesnt_exist

if 1: # paths for beamline use
    pythonScript    = '/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/ssc-cdi/bin/sscptycho_main.py' # path with python script to run
else: # paths for GCC tests       
    pythonScript    = '~/ssc-cdi/bin/sscptycho_main.py' 

import getpass
username = getpass.getuser()

jupyter_folder = "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/"

acquisition_folder = 'SS61'
output_folder = os.path.join('/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/', 'proc','recons',acquisition_folder) # changes with control

global_paths_dict = { "jupyter_folder"         : "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/",
                    "ptycho_script_path"       : pythonScript,
                    "template_json"            : "000000_template.json",
                    "slurm_filepath"           : os.path.join(jupyter_folder,'inputs',f'{username}_ptycho_job.srm'), # path to create slurm_file
                    "json_filepath"            : os.path.join(jupyter_folder,'inputs',f'{username}_ptycho_input.json'), # path with input json to run
                    "sinogram_filepath"        : os.path.join(output_folder,f'object_{acquisition_folder}.npy'), # path to load npy with first reconstruction preview
                    "cropped_sinogram_filepath": os.path.join(output_folder,f'object_{acquisition_folder}_cropped.npy'),
                    "probe_filepath"           : os.path.join(output_folder,f'probe_{acquisition_folder}.npy'), # path to load probe
                    "difpad_raw_mean_filepath" : os.path.join(output_folder,'02_difpad_raw_mean.npy'), # path to load diffraction pattern
                    "flipped_difpad_filepath"  : os.path.join(output_folder,'03_difpad_restaured_flipped.npy'), # path to load diffraction pattern
                    "output_folder"            : output_folder
                }


global_dict = json.load(open(os.path.join(global_paths_dict["jupyter_folder"] ,global_paths_dict["template_json"]))) # load from template

json_filepath = os.path.join(global_paths_dict["jupyter_folder"],'inputs', f'{username}_ptycho_input.json') #INPUT
if os.path.exists(json_filepath):  
    with open(json_filepath) as json_file:
        global_dict = json.load(json_file)

global_dict["00_versions"] = f"sscCdi={sscCdi.__version__},sscPimega={sscPimega.__version__},sscResolution={sscResolution.__version__},sscRaft={sscRaft.__version__},sscRadon={sscRadon.__version__}"


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

def write_slurm_file(python_script_path,json_filepath_path,output_path="",slurm_filepath = 'slurmJob.sh',jobName='jobName',queue='cat',gpus=1,cpus=32):
    # Create slurm file
    logfiles_path = slurm_filepath.rsplit('/',2)[0]
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o {logfiles_path}/logfiles/{username}_slurm.log        # Select output path of slurm file

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

python3 {python_script_path} {json_filepath_path} > {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_output.log')} 2> {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_error.log')}
"""
    
    with open(slurm_filepath,'w') as the_file:
        the_file.write(string)
    
    return slurm_filepath

def update_gpu_limits(machine_selection):

    if machine_selection == 'Cluster':
        gpus.widget.value = 0
        gpus.widget.max = 4
    elif machine_selection == 'Local':
        gpus.widget.value = 0
        gpus.widget.max = 1

def update_cpus_gpus(cpus,gpus):
    global_dict["Threads"] = cpus

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
]
                            
    import shutil
    for folderpath in folderpaths_to_remove:
        print('Removing file/folder: ', folderpath)
        if os.path.isdir(folderpath) == True:
            shutil.rmtree(folderpath)
            print(f'Deleted {folderpath}\n')
        else:
            print(f'Directory {folderpath} does not exists. Skipping deletion...\n')

def run_ptycho_from_jupyter(mafalda,python_script_path,json_filepath_path,output_path="",slurm_filepath = 'ptychoJob2.srm',jobName='jobName',queue='cat',gpus=1,cpus=32):
    slurm_file = write_slurm_file(python_script_path,json_filepath_path,output_path,slurm_filepath,jobName,queue,gpus,cpus)
    call_cmd_terminal(slurm_file,mafalda,remove=False)
    
def run_ptycho(dummy):
    pythonScript = global_paths_dict["ptycho_script_path"]
    json_filepath = global_paths_dict["json_filepath"]
    slurm_filepath = global_paths_dict["slurm_filepath"]

    print(f'Running ptycho with {machine_selection.value} machine...')

    global global_dict

    from pprint import pprint
    pprint(global_dict)

    print(f'\nWARNING: IS CENTER {global_dict["DifpadCenter"]} CORRECT?\n')

    if machine_selection.value == 'Local':
        cmd = f'python3 {pythonScript} {json_filepath}'
        # cmd = f'python3 ~/ssc-cdi/bin/sscptycho_main_test.py {json_filepath}'
        print('Running command: ',cmd)               
        output = call_and_read_terminal(cmd,mafalda,use_mafalda=False)
        print(output.decode("utf-8"))
    elif machine_selection.value == "Cluster": 
        global jobNameField, jobQueueField, cpus, gpus
        jobName_value = jobNameField.widget.value
        queue_value   = jobQueueField.widget.value
        cpus_value    = cpus.widget.value
        gpus_value    = gpus.widget.value
        create_directory_if_doesnt_exist(global_paths_dict["output_folder"])
        run_ptycho_from_jupyter(mafalda,pythonScript,json_filepath,output_path=global_paths_dict["output_folder"],slurm_filepath = slurm_filepath,jobName=jobName_value,queue=queue_value,gpus=gpus_value,cpus=cpus_value)

# def load_json(dummy):
#     global global_dict
#     json_filepath = os.path.join(global_paths_dict["jupyter_folder"],'inputs', f'{username}_ptycho_input.json') #INPUT
#     with open(json_filepath) as json_file:
#         global_dict = json.load(json_file)
#     print("Inputs loaded from ",json_filepath)

def create_label_widget(text):
    # label = widgets.Label(value=text)
    label = widgets.HTML(value=f"<b style='color:#00008B;font-size:18px;'>{text}</b>")
    return label

############################################ INTERFACE / GUI : TABS ###########################################################################

def inputs_tab():

    global global_dict

    def save_on_click(dummy,json_filepath="",dictionary={}):
        print('Saving input json file at: ',json_filepath)
        file = open(json_filepath,"w")
        file.write(json.dumps(dictionary,indent=3,sort_keys=True))
        file.close()
        print('\t Saved!')


    def update_global_dict(proposal_path_str,acquisition_folders,projections,center_y,center_x,detector_ROI,ChipBorderRemoval,save_or_load_difpads,CentralMask_bool,CentralMask_radius,ProbeSupport_radius,ProbeSupport_centerX,ProbeSupport_centerY,PhaseUnwrap,PhaseUnwrap_iter,top_crop,bottom_crop,left_crop,right_crop,use_obj_guess,use_probe_guess,fresnel_number,DetectorPileup):

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

        global_paths_dict["sinogram_filepath"]         = os.path.join(output_folder,f'{acquisition_folders[0]}_object.npy') # path to load npy with first reconstruction preview
        global_paths_dict["cropped_sinogram_filepath"] = os.path.join(output_folder,f'{acquisition_folders[0]}_object_cropped.npy')
        global_paths_dict["probe_filepath"]            = os.path.join(output_folder,f'{acquisition_folders[0]}_probe.npy') # path to load probe
        global_paths_dict["difpad_raw_mean_filepath"]  = os.path.join(output_folder,'02_difpad_raw_mean.npy') # path to load diffraction pattern
        global_paths_dict["flipped_difpad_filepath"]   = os.path.join(output_folder,'03_difpad_restaured_flipped.npy') # path to load diffraction pattern
        global_paths_dict["output_folder"]             = output_folder

        global_dict["DifpadCenter"] = [center_y,center_x]

        global_dict["DetectorROI"] = detector_ROI
        global_dict["ChipBorderRemoval"] = ChipBorderRemoval

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

        if [top_crop,bottom_crop] == [0,1]:
            global_dict["Phaseunwrap"][2] =  []
        else:
            global_dict["Phaseunwrap"][2] = [top_crop,bottom_crop]
        if [left_crop,right_crop] == [0,1]:
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
    saveJsonButton = Button(description="Save Inputs",layout=buttons_layout,icon='fa-floppy-o')
    saveJsonButton.trigger(save_on_click_partial)

    label1 = create_label_widget("Data Selection")
    proposal_path_str     = Input(global_dict,"ProposalPath",description="Proposal Path",layout=items_layout2)
    acquisition_folders   = Input(global_dict,"Acquisition_Folders",description="Data Folders",layout=items_layout2)
    projections           = Input(global_dict,"Projections",description="Projections",layout=items_layout2)
    
    label2 = create_label_widget("Restoration")
    global center_y, center_x
    center_x    = Input({'dummy-key':global_dict["DifpadCenter"][1]},'dummy-key',bounded=(0,3072,1),slider=True,description="Center column (x)",layout=slider_layout)
    center_y    = Input({'dummy-key':global_dict["DifpadCenter"][0]},'dummy-key',bounded=(0,3072,1),slider=True,description="Center row (y)   ",layout=slider_layout)
    center_box = widgets.Box([center_y.widget,center_x.widget],layout=slider_layout3)



    detector_ROI          = Input({'dummy-key':global_dict["DetectorROI"]},'dummy-key',bounded=(0,1536,1),slider=True,description="Diamenter (pixels)",layout=slider_layout2)
    suspect_pixels        = Input({'dummy-key':global_dict["ChipBorderRemoval"]},'dummy-key',bounded=(0,20,1),slider=True,description="Suppress pixels from chip border",layout=slider_layout2)
    # binning             = Input(global_dict,"Binning",bounded=(1,4,1),slider=True,description="Binning factor",layout=slider_layout2)
    save_or_load_difpads  = widgets.RadioButtons(options=['Save Diffraction Pattern', 'Load Diffraction Pattern'], value='Save Diffraction Pattern', layout={'width': '50%'},description='Save or Load')

    label3 = create_label_widget("Diffraction Pattern Processing")
    autocrop           = Input(global_dict,"AutoCrop",description="Auto Crop borders",layout=items_layout2)
    global CentralMask_radius, CentralMask_bool, DetectorPileup
    CentralMask_bool   = Input({'dummy-key':global_dict["CentralMask"][0]},'dummy-key',description="Use Central Mask",layout=items_layout2)
    CentralMask_radius = Input({'dummy-key':global_dict["CentralMask"][1]},'dummy-key',bounded=(0,100,1),slider=True,description="Central Mask Radius",layout=slider_layout)
    central_mask_box   = widgets.Box([CentralMask_bool.widget,CentralMask_radius.widget],layout=slider_layout3)
    DetectorPileup   = Input({'dummy-key':global_dict["DetectorExposure"][0]},'dummy-key',description="Ignore Detector Pileup",layout=items_layout2)

    label4 = create_label_widget("Probe Adjustment")
    ProbeSupport_radius   = Input({'dummy-key':global_dict["ProbeSupport"][0]},'dummy-key',bounded=(0,1000,10),slider=True,description="Probe Support Radius",layout=slider_layout2)
    ProbeSupport_centerX  = Input({'dummy-key':global_dict["ProbeSupport"][1]},'dummy-key',bounded=(-100,100,10),slider=True,description="Probe Center X",layout=slider_layout2)
    ProbeSupport_centerY  = Input({'dummy-key':global_dict["ProbeSupport"][2]},'dummy-key',bounded=(-100,100,10),slider=True,description="Probe Center Y",layout=slider_layout2)
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
    Phaseunwrap      = Input({'dummy-key':global_dict["Phaseunwrap"][0]},'dummy-key',description="Phase Unwrap",layout=checkbox_layout)
    Phaseunwrap_iter = Input({'dummy-key':global_dict["Phaseunwrap"][1]},'dummy-key',bounded=(0,20,1),slider=True,description="Gradient Removal Iterations",layout=slider_layout2)
    phase_unwrap_box = widgets.Box([Phaseunwrap.widget],layout=items_layout2)
    global top_crop, bottom_crop,left_crop,right_crop # variables are reused in crop tab
    top_crop      = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Top", slider=True,layout=slider_layout)
    bottom_crop   = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Bottom", slider=True,layout=slider_layout)
    left_crop     = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Left", slider=True,layout=slider_layout)
    right_crop    = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Right", slider=True,layout=slider_layout)

    FRC = Input(global_dict,"FRC",description="FRC: Fourier Ring Correlation",layout=items_layout2)

    widgets.interactive_output(update_global_dict,{'proposal_path_str':proposal_path_str.widget,
                                                    'acquisition_folders': acquisition_folders.widget,
                                                    'projections': projections.widget,                                                    
                                                    'center_y':center_y.widget,
                                                    'center_x':center_x.widget,
                                                    'detector_ROI':detector_ROI.widget,
                                                    'ChipBorderRemoval':suspect_pixels.widget,
                                                    'save_or_load_difpads':save_or_load_difpads,
                                                    'CentralMask_bool': CentralMask_bool.widget,
                                                    'CentralMask_radius': CentralMask_radius.widget,
                                                    'ProbeSupport_radius': ProbeSupport_radius.widget,
                                                    'ProbeSupport_centerX': ProbeSupport_centerX.widget,
                                                    'ProbeSupport_centerY': ProbeSupport_centerY.widget,
                                                    'PhaseUnwrap': Phaseunwrap.widget,
                                                    'PhaseUnwrap_iter': Phaseunwrap_iter.widget, # Remove this in the future
                                                    'top_crop': top_crop.widget,
                                                    'bottom_crop': bottom_crop.widget,
                                                    'left_crop': left_crop.widget,
                                                    'right_crop': right_crop.widget,
                                                    "use_obj_guess": use_obj_guess.widget,
                                                    "use_probe_guess":use_probe_guess.widget,
                                                    "fresnel_number":fresnel_number.widget,
                                                    "DetectorPileup":DetectorPileup.widget
                                                     })

    box = widgets.Box([label1,proposal_path_str.widget,acquisition_folders.widget,projections.widget,label2,center_box,detector_ROI.widget,suspect_pixels.widget,save_or_load_difpads],layout=box_layout)
    box = widgets.Box([box,label3,autocrop.widget,central_mask_box,DetectorPileup.widget,label4,probe_box,fresnel_number.widget,Modes.widget,label5,Algorithm1.widget,Algorithm2.widget,Algorithm3.widget,label6,phase_unwrap_box,FRC.widget],layout=box_layout)

    return box

def mask_tab():
    
    initial_image = np.random.random((5,5)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(initial_image,cmap='gray')
        subplot.set_title('Raw')
        figure.canvas.header_visible = False 
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(figsize=(4,4))
        subplot2.imshow(initial_image,cmap='gray')
        subplot2.set_title('Mask')
        figure2.canvas.header_visible = False 
        plt.show()


    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots(figsize=(4,4))
        subplot3.imshow(initial_image,cmap='gray')
        subplot3.set_title('Masked')
        figure3.canvas.header_visible = False 
        plt.show()



    def load_frames(dummy):
        global sinogram
        from matplotlib.colors import LogNorm
        print("Loading difpad from: ",global_paths_dict["difpad_raw_mean_filepath"] )
        difpad = np.load(global_paths_dict["difpad_raw_mean_filepath"] ) 
        masked_difpad = difpad.copy()
        mask = h5py.File(os.path.join(global_dict["ProposalPath"],global_dict["Acquisition_Folders"][0],'images','mask.hdf5'), 'r')['entry/data/data'][()][0, 0, :, :]
        masked_difpad[np.abs(mask) == 1] = -1 # Apply Mask
        subplot.imshow(difpad,cmap='jet',norm=LogNorm())
        subplot2.imshow(mask,cmap='gray')
        subplot3.imshow(masked_difpad,cmap='jet',norm=LogNorm())


    load_frames_button  = Button(description="Load Diffraction Patterns",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    objects_box = widgets.HBox([output,output2,output3])
    box = widgets.VBox([buttons_box,objects_box])

    return box

def center_tab():

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
        figure,subplot.imshow(np.random.random((4,4)))
        subplot.set_title('Diffraction Pattern')
        figure.canvas.header_visible = False 
        plt.show()


    def plotshow(figure,subplot,image,title="",figsize=(8,8),savepath=None,show=False):
        subplot.clear()
        cmap, colors, bounds, norm = miqueles_colormap(image)
        handle = subplot.imshow(image, interpolation='nearest', cmap = cmap, norm=norm)
        if title != "":
            subplot.set_title(title)
        if show:
            if __name__ == '__main__': 
                plt.show()

        figure.canvas.draw_idle()

    def update_mask(figure, subplot,output_dictionary,image,key1,key2,key3,cy,cx,button,exposure,exposure_time,radius):


        output_dictionary[key1] = [cy,cx]
        output_dictionary[key2] = [button,radius]
        output_dictionary[key3] = [exposure,exposure_time]
        if exposure == True or button == True:
            image2 = masks_application(np.copy(image), output_dictionary)
            plotshow(figure,subplot,image2)
        else:
            plotshow(figure,subplot,image)

    def load_difpad(dummy):

        mdata_filepath = os.path.join(global_dict["ProposalPath"],global_dict['Acquisition_Folders'][0],'mdata.json')
        input_dict = json.load(open(mdata_filepath))

        image = np.load(global_paths_dict['flipped_difpad_filepath'])
        widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                                'output_dictionary':fixed(global_dict),'image':fixed(image),
                                                'key1':fixed('DifpadCenter'),'key2':fixed('CentralMask'),'key3':fixed('DetectorExposure'),
                                                'cy':center_y.widget,'cx':center_x.widget,
                                                'button':CentralMask_bool.widget,
                                                'exposure':DetectorPileup.widget,
                                                'exposure_time':fixed(input_dict['/entry/beamline/detector']['pimega']["exposure time"]),
                                                'radius':CentralMask_radius.widget})

    load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    load_difpad_button.trigger(load_difpad)

    """ Difpad center boxes """
    sliders_box = widgets.HBox([center_y.widget,center_x.widget,CentralMask_radius.widget],layout=box_layout)
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

def cropunwrap_tab():

    output = widgets.Output()
    with output:
        figure_unwrap, subplot_unwrap = plt.subplots(figsize=(3,3))
        subplot_unwrap.imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap.set_title('Cropped image')
        figure_unwrap.canvas.draw_idle()
        figure_unwrap.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure_unwrap2, subplot_unwrap2 = plt.subplots(figsize=(3,3))
        subplot_unwrap2.imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap2.set_title('Unwrapped image')
        figure_unwrap2.canvas.draw_idle()
        figure_unwrap2.canvas.header_visible = False 
        plt.show()

    
    def load_frames(dummy):
        global sinogram
        
        print("Loading sinogram from: ",global_paths_dict["sinogram_filepath"] )
        sinogram = np.load(global_paths_dict["sinogram_filepath"] ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = selection_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap),'title':fixed(True),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})


    def preview_unwrap(dummy):
        cropped_frame = sinogram[selection_slider.widget.value,top_crop.widget.value:-bottom_crop.widget.value,left_crop.widget.value:-right_crop.widget.value]
        cropped_frame = cropped_frame[np.newaxis]
        unwrapped_frame = phase_unwrap(np.angle(cropped_frame),iterations=0,non_negativity=False,remove_gradient = False)
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_frame),'figure':fixed(figure_unwrap2),'subplot':fixed(subplot_unwrap2),'title':fixed(True),'frame_number': fixed(0)})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    preview_unwrap_button = Button(description="Preview Unwrap",layout=buttons_layout,icon='play') 
    preview_unwrap_button.trigger(preview_unwrap)
    
    buttons_box = widgets.Box([load_frames_button.widget,preview_unwrap_button.widget],layout=get_box_layout('100%',align_items='center'))
    sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar,output,output2])
    return box

def ptycho_tab():

    def view_jobs(dummy):
        output = call_and_read_terminal('squeue',mafalda)
        print(output.decode("utf-8"))
    def cancel_job(dummy):
        print(f'Cancelling job {job_number.widget.value}')    
        call_and_read_terminal(f'scancel {job_number.widget.value}',mafalda)

    job_number = Input({"dummy-key":00000},"dummy-key",description="Job ID",layout=items_layout)

    view_jobs_button = Button(description='List Jobs',layout=buttons_layout,icon='fa-eye')
    view_jobs_button.trigger(view_jobs)

    cancel_job_button = Button(description='Cancel Job',layout=buttons_layout,icon='fa-stop-circle')
    cancel_job_button.trigger(cancel_job)    

    run_button = Button(description='Run Ptycho',layout=buttons_layout,icon='play')
    run_button.trigger(run_ptycho)

    box = widgets.Box([saveJsonButton.widget,run_button.widget,view_jobs_button.widget,job_number.widget,cancel_job_button.widget],layout=get_box_layout('500px'))

    return box

def reconstruction_tab():
    
    initial_image = np.random.random((5,5)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()


    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots(figsize=(4,4))
        subplot3.imshow(initial_image,cmap='gray')
        figure3.canvas.header_visible = False 
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(figsize=(4,4))
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

        probe = np.abs(np.load(global_paths_dict["probe_filepath"]))[:,0,:,:] # get only 0th order 
        widgets.interactive_output(update_imshow, {'sinogram':fixed(probe),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed(True), 'cmap':fixed('jet'), 'frame_number': selection_slider.widget})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")

    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))

    controls_box = widgets.Box([play_box],layout=get_box_layout('500px'))
    objects_box = widgets.HBox([output,output3,output2])
    object_box = widgets.VBox([controls_box,objects_box])
    box = widgets.HBox([object_box])
    box = widgets.VBox([buttons_box,box])

    return box

def deploy_tabs(mafalda_session,tab2=inputs_tab(),tab3=center_tab(),tab4=fresnel_tab(),tab6=reconstruction_tab(),tab1=cropunwrap_tab(),tab7=mask_tab()):
    
    __name__ = "__main__"

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

    # load_json_button  = Button(description="Load inputs",layout=buttons_layout,icon='folder-open-o')
    # load_json_button.trigger(load_json)

    ptycho_box = widgets.Box([saveJsonButton.widget,run_button.widget,view_jobs_button.widget,cancel_job_button.widget,job_number.widget],layout=get_box_layout('1000px',flex_flow='row'))


    children_dict = {
    "Inputs"            : tab2,
    "Mask"              : tab7,
    "Find Center"       : tab3,
    "Probe Propagation" : tab4,
    "Crop and Unwrap"   : tab1,
    "Reconstruction"    : tab6
    }
    
    global mafalda
    mafalda = mafalda_session
    
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Cluster', layout={'width': '10%'},description='Machine',disabled=False)
    widgets.interactive_output(update_gpu_limits,{"machine_selection":machine_selection})

    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    if username == 'yuri.tonin' or username == 'julia.carvalho' or username == 'paola.ferraz' or username == 'eduardo.miqueles':
        slurmequeue = 'dev-gcc'
    else:
        slurmequeue = 'cat'

    global jobNameField, jobQueueField
    jobNameField  = Input({'dummy_key':f'{username}_ptycho'},'dummy_key',description="Insert slurm job name:")
    jobQueueField = Input({'dummy_key':slurmequeue},'dummy_key',description="Insert machine queue name:")
    global cpus, gpus
    gpus = Input({'dummy_key':1}, 'dummy_key',bounded=(0,5,1),  slider=True,description="Insert # of GPUs to use:")
    cpus = Input({'dummy_key':32},'dummy_key',bounded=(1,160,1),slider=True,description="Insert # of CPUs to use:")
    widgets.interactive_output(update_cpus_gpus,{"cpus":cpus.widget,"gpus":gpus.widget})

    boxSlurm = widgets.HBox([machine_selection,gpus.widget,cpus.widget,jobQueueField.widget,jobNameField.widget])
    box = widgets.VBox([boxSlurm,ptycho_box])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  



if __name__ == "__main__":
    pass
