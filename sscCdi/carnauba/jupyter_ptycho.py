from calendar import c
import os, json, ast, h5py
import numpy as np
from PIL.Image import open as tifOpen
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit

import ipywidgets as widgets 
from ipywidgets import fixed

import sscPimega, sscResolution

from .ptycho_fresnel import propagate_beam, fit_2Dgaussian, fit_2Dsinc
from ..caterete.ptycho_fresnel import  create_propagation_video_f1
from ..caterete.misc import miqueles_colormap
from .jupyter import monitor_job_execution, call_cmd_terminal, Button, Input, update_imshow, slide_and_play, call_and_read_terminal

from ..caterete.misc import create_directory_if_doesnt_exist



pythonScript = '/ibira/lnls/beamlines/carnauba/apps/ssc-cdi/bin/sscptycho_cnb.py'  # path with python script to run


jupyter_folder = "/ibira/lnls/beamlines/carnauba/apps/jupyter/"
foldername = '20220831_ANT1_ptycho__PiMega_001'
filename = '20220831_ANT1_ptycho__PiMega_001.hdf5'
output_folder = os.path.join('/ibira/lnls/beamlines/carnauba/apps/jupyter/00000000', 'proc', foldername) # changes with control

global_paths_dict = {"jupyter_folder"         : "/ibira/lnls/beamlines/carnauba/apps/jupyter/", 
                    "ptycho_script_path"       : pythonScript,
                    "template_json"            : "template.json",   
                    "slurm_filepath"           : os.path.join(jupyter_folder,'slurm_job.srm'), # path to create slurm_file
                    "json_filepath"            : os.path.join(jupyter_folder,'user_input.json'), # path with input json to run
                    "sinogram_filepath"        : os.path.join(output_folder,f'object_{foldername}.npy'), # path to load npy with first reconstruction preview
                    "cropped_sinogram_filepath": os.path.join(output_folder,f'object_{foldername}_cropped.npy'),
                    "probe_filepath"           : os.path.join(output_folder,f'probe_{foldername}.npy'), # path to load probe
                    "difpad_raw_mean_filepath" : os.path.join(output_folder,'02_difpad_raw_mean.npy'), # path to load diffraction pattern
                    "flipped_difpad_filepath"  : os.path.join(output_folder,'03_difpad_restaured_flipped.npy'), # path to load diffraction pattern/GCC/ssc-cdi/-/graphs/master
                    "output_folder"            : output_folder
                }

global_dict = json.load(open(os.path.join(global_paths_dict["jupyter_folder"], global_paths_dict["template_json"]))) # load from template

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
#SBATCH -o ./slurm.out      # Select output path of slurm file

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

python3 {python_script_path} {json_filepath_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
"""
    
    with open(slurm_filepath,'w') as the_file:
        the_file.write(string)
    
    return slurm_filepath

#
def update_gpu_limits(machine_selection):

    if machine_selection == 'Cluster':
        gpus.widget.value = 0
        gpus.widget.max = 4
    elif machine_selection == 'Local':
        gpus.widget.value = 0
        gpus.widget.max = 1

#============================================================================================================================================# 

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

#============================================================================================================================================# 

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

#============================================================================================================================================# 

def run_ptycho_from_jupyter(mafalda,python_script_path,json_filepath_path,output_path="",slurm_filepath = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    slurm_file = write_slurm_file(python_script_path,json_filepath_path,output_path,slurm_filepath,jobName,queue,gpus,cpus)
    call_cmd_terminal(slurm_file,mafalda,remove=False)
   
#============================================================================================================================================#   

def run_ptycho(dummy):
    pythonScript = global_paths_dict["ptycho_script_path"]
    json_filepath = global_paths_dict["json_filepath"]
    slurm_filepath = global_paths_dict["slurm_filepath"]

    print(f'Running ptycho with {machine_selection.value}...')
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

#============================================================================================================================================# 

def load_json(dummy):
    json_path = os.path.join(global_paths_dict["jupyter_folder"] ,global_paths_dict["template_json"])
    template_dict = json.load(open(json_path))
    for key in template_dict:
        global_dict[key] = template_dict[key]

def create_label_widget(text):
    # label = widgets.Label(value=text)
    label = widgets.HTML(value=f"<b style='color:#E1AD01;font-size:18px;'>{text}</b>")
    return label

############################################ INTERFACE / GUI : TABS ###########################################################################

def inputs_tab():

    global global_dict

    def save_on_click(dummy,json_filepath="",dictionary={}):

        print('save_on_click')

        print('Saving input json file at: ',json_filepath)
        with open(json_filepath, 'w') as file:
            json.dump(dictionary, file, separators=(',', ':'), indent = 2)                                                    
        print('\t Saved!')


    def update_global_dict(proposal_str, data_filename, position_folder, centerx, centery, detector_ROI, save_or_load_difpads, ProbeSupport_R, ProbeSupport_n, ProbeSupport_sigma, ProbeSupport_borderpx, object_magnification, fresnel_number):

        print('data_filename in update_global_dict: ', data_filename)
        
        global global_dict
        global_dict["Proposal"]      = proposal_str
        global_dict["Data_Filename"] = data_filename
        global_dict["BeamlineParameters_Filename"] = position_folder
        global_dict["object_magnification"] = object_magnification
        

        output_folder_name = (data_filename.split("."))[0]
        output_folder = os.path.join('/ibira/lnls/beamlines/carnauba/apps/jupyter/00000000', 'proc', output_folder_name) # changes with control
        
        global_paths_dict["output_folder"]             =  output_folder
        global_paths_dict["sinogram_filepath"]         = os.path.join(output_folder,f'object_{output_folder_name}.npy') # path to load npy with first reconstruction preview
        global_paths_dict["cropped_sinogram_filepath"] = os.path.join(output_folder,f'object_{output_folder_name}_cropped.npy')
        global_paths_dict["probe_filepath"]            = os.path.join(output_folder,f'probe_{output_folder_name}.npy') # path to load probe
        global_paths_dict["difpad_raw_mean_filepath"]  = os.path.join(output_folder,'03_difpad_raw_mean.npy') # path to load diffraction pattern
        global_paths_dict["flipped_difpad_filepath"]   = os.path.join(output_folder,'03_difpad_restaured_flipped.npy') # path to load diffraction pattern
        
        print('Output Folder in global_dict: ', global_dict['output_folder'])
        print('Centerx and centery: ', centerx, centery) 

        global_dict["DifpadCenter"] = [centerx,centery]

        global_dict["DetectorROI"] = detector_ROI

        if save_or_load_difpads == "Save Diffraction Pattern":
            global_dict["SaveDifpads"] = 1
            global_dict["ReadRestauredDifpads"] = 0
        elif save_or_load_difpads == "Load Diffraction Pattern":
            global_dict["SaveDifpads"] = 0
            global_dict["ReadRestauredDifpads"] = 1

       
        global_dict["f1"] = fresnel_number
        global_dict["R_parameter"] = ProbeSupport_R
        global_dict["n_parameter"] = ProbeSupport_n
        global_dict["sigma"] = ProbeSupport_sigma
        global_dict["border_px"] = ProbeSupport_borderpx

    global saveJsonButton
    saveJsonButton = Button(description="Save Inputs",layout=buttons_layout, icon='fa-floppy-o')
    save_on_click_partial = partial(save_on_click,json_filepath=global_paths_dict["json_filepath"],dictionary=global_dict)
    saveJsonButton.trigger(save_on_click_partial)

    label1 = create_label_widget("Data Selection")
    proposal_str          = Input(global_dict,"Proposal",description="Proposal",layout=items_layout2)
    data_filename         = Input(global_dict,"Data_Filename",description="Data Filename",layout=items_layout2)
    position_folder       = Input(global_dict,"BeamlineParameters_Filename",description="Beamline Parameters Filename",layout=items_layout2)
    mask_filename         = Input(global_dict,"mask_Filename",description="mask Filename",layout=items_layout2)
    
    
    label2 = create_label_widget("Restauration")
    global centerx, centery
    centerx    = Input({'dummy-key':832},'dummy-key',bounded=(0,1536,1),slider=True,description="Center row",layout=slider_layout)
    centery    = Input({'dummy-key':1062},'dummy-key',bounded=(0,1536,1),slider=True,description="Center column",layout=slider_layout)
    center_box = widgets.Box([centerx.widget,centery.widget],layout=slider_layout3)

    detector_ROI          = Input({'dummy-key':32},'dummy-key',bounded=(0,768,1),slider=True,description="Diamenter (pixels)",layout=slider_layout2)
    save_or_load_difpads  = widgets.RadioButtons(options=['Save Diffraction Pattern', 'Load Diffraction Pattern'], value='Save Diffraction Pattern', layout={'width': '50%'},description='Save or Load')

    label3 = create_label_widget("Diffraction Pattern Processing")
    global linearity_function
    linearity_function = Input(global_dict,"Linearity_Function",description="Apply Linearity Correction Function",layout=items_layout2)
    W_function = Input(global_dict, "W_Function",description="Apply W Function",layout=items_layout2)
    
    label4 = create_label_widget("Probe Adjustment")
    ProbeSupport_R = Input(global_dict, "R_parameter", description="R parameter", layout = items_layout2)
    ProbeSupport_n = Input(global_dict, "n_parameter", description="n parameter", layout = items_layout2)
    ProbeSupport_sigma = Input(global_dict, "sigma", description="Sigma", layout = items_layout2)
    ProbeSupport_borderpx = Input(global_dict, "border_px", description="Border Pixels", layout = items_layout2)
    probe_box1 = widgets.Box([ProbeSupport_R.widget, ProbeSupport_n.widget], layout = items_layout2)
    probe_box = widgets.Box([ProbeSupport_sigma.widget, ProbeSupport_borderpx.widget], layout = items_layout2)

    # To viasualize the probe support #
    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
        figure,subplot.imshow(np.random.random((4,4)))
        subplot.set_title('Probe Support')
        figure.canvas.header_visible = False 
        plt.show()


    def load_probe_support(dummy):
        from scipy.ndimage import gaussian_filter
        N = 2*detector_ROI.widget.value

        # dx = global_dict['object_pixel']
        # print('dx: ', dx)

        n = ProbeSupport_n.widget.value
        R = ProbeSupport_R.widget.value
        sigma = ProbeSupport_sigma.widget.value
        border_px = ProbeSupport_borderpx.widget.value
       
        x = np.linspace(0,N-1,N) - N//2
        x = x / np.max(np.abs(x))
        X, Y = np.meshgrid(x,x)
        X = X - np.max(x)
        Y = Y - np.max(x)
        
        f1 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

        x = np.linspace(0,N-1,N) - N//2
        x = x / np.max(np.abs(x))
        X, Y = np.meshgrid(x,x)
        X = X + np.max(x)
        Y = Y + np.max(x)
 
        f2 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

        x = np.linspace(0,N-1,N) - N//2
        x = x / np.max(np.abs(x))
        X, Y = np.meshgrid(x,x)
        X = X - np.max(x)
        Y = Y + np.max(x)

        f3 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

        x = np.linspace(0,N-1,N) - N//2
        x = x / np.max(np.abs(x))
        X, Y = np.meshgrid(x,x)
        X = X + np.max(x)
        Y = Y - np.max(x)

        f4 = np.where(np.abs(X)**n + np.abs(Y)**n < R**n,0,1)

        f = (f1 + f2 + f3 + f4)/4

        f[0:border_px,:] = 0
        f[:, 0:border_px] = 0
        f[f.shape[0] - border_px: f.shape[0], :] = 0
        f[:, f.shape[1] - border_px: f.shape[1]] = 0

        f = gaussian_filter(f, sigma)

        # probe_support = np.asarray([f for k in range(Modes)])
        probe_support = f
        subplot.imshow(probe_support, cmap='viridis')

    small_button_layout = widgets.Layout( width='27%',height="40px")     # override the default width of the button to 'auto' to let the button grow
    load_probe_support_button  = Button(description="Load Probe Support",layout=small_button_layout,icon='folder-open-o')
    load_probe_support_button.trigger(load_probe_support)


    global fresnel_number
    fresnel_number = Input(global_dict,"f1",description="Fresnel Number",layout=items_layout2)
    Modes = Input(global_dict,"Modes",bounded=(0,30,3),slider=True,description="Probe Modes",layout=slider_layout2)

    label5 = create_label_widget("Ptychography")
    object_magnification = Input({'dummy-key':4},'dummy-key',bounded=(1,20,1),slider=True,description="Object Magnification",layout=slider_layout2)
    Algorithm1 = Input(global_dict,"Algorithm1",description="Recon Algorithm 1",layout=items_layout2)
    Algorithm2 = Input(global_dict,"Algorithm2",description="Recon Algorithm 2",layout=items_layout2)
    Algorithm3 = Input(global_dict,"Algorithm3",description="Recon Algorithm 3",layout=items_layout2)
    Algorithm4 = Input(global_dict,"Algorithm4",description="Recon Algorithm 4",layout=items_layout2)

    label6 = create_label_widget("Post-processing")    
    FRC = Input(global_dict,"FRC",description="FRC: Fourier Ring Correlation",layout=items_layout2)

    widgets.interactive_output(update_global_dict, {'proposal_str':proposal_str.widget, 'data_filename': data_filename.widget, 'position_folder': position_folder.widget,'centerx':centerx.widget, 'centery':centery.widget, 'detector_ROI':detector_ROI.widget, 'save_or_load_difpads':save_or_load_difpads, 'ProbeSupport_R': ProbeSupport_R.widget, 'ProbeSupport_n': ProbeSupport_n.widget, 'ProbeSupport_sigma': ProbeSupport_sigma.widget, 'ProbeSupport_borderpx': ProbeSupport_borderpx.widget, 'object_magnification': object_magnification.widget, 'fresnel_number':fresnel_number.widget})

    box = widgets.Box([label1,proposal_str.widget,data_filename.widget,position_folder.widget, mask_filename.widget, label2,center_box,detector_ROI.widget,save_or_load_difpads],layout=box_layout)
    box = widgets.Box([box,label3,W_function.widget,linearity_function.widget,label4, probe_box1, probe_box,Modes.widget, load_probe_support_button.widget, output, fresnel_number.widget, label5, object_magnification.widget, Algorithm1.widget, Algorithm2.widget, Algorithm3.widget, Algorithm4.widget, label6, FRC.widget],layout=box_layout)

    return box

#==============================================================================================================================================================#

def mask_tab():
    
    initial_image = np.random.random((10,10)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        subplot.set_title('Raw')
        figure.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots()
        subplot2.imshow(initial_image,cmap='gray')
        subplot2.set_title('Flat')
        figure2.canvas.header_visible = False 
        plt.show()

    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots()
        subplot3.imshow(initial_image,cmap='gray')
        subplot3.set_title('Flat Applied')
        figure3.canvas.header_visible = False 
        plt.show()


    def load_frames(dummy):
        global sinogram
        from matplotlib.colors import LogNorm
        print("Loading difpad from: ",global_paths_dict["difpad_raw_mean_filepath"] )
        difpad = np.load(global_paths_dict["difpad_raw_mean_filepath"] ) 
        flat = tifOpen(global_dict['flatfield'])
        flat = np.array(flat)
        print('Loading flatfield from ', global_dict['flatfield'])
        flatted_difpad = flat * difpad# Apply mask
        subplot.imshow(difpad,cmap='jet',norm=LogNorm())
        subplot2.imshow(flat,cmap='jet')
        subplot3.imshow(flatted_difpad,cmap='jet',norm=LogNorm())


    load_frames_button  = Button(description="Load Diffraction Patterns",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    objects_box = widgets.HBox([output,output2,output3])
    box = widgets.VBox([buttons_box,objects_box])

    return box

#==============================================================================================================================================================#

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
            plt.show()
        figure.canvas.draw_idle()

    def update_mask(figure, subplot,output_dictionary,image,key1,cx,cy):
        output_dictionary[key1] = [cx,cy]
        plotshow(figure,subplot,image)

    def load_difpad(dummy):

        # input_dict = h5py.File(os.path.join(global_dict["ProposalPath"],global_dict["Position_Folders"][0], global_dict["Position_Filename"]), 'r')['entry/data/data'][()][0, 0, :, :]

        print(centerx.widget.value,centery.widget.value)
        image = np.load(global_paths_dict['flipped_difpad_filepath'])
        widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                                'output_dictionary':fixed(global_dict),'image':fixed(image),
                                                'key1':fixed('DifpadCenter'),
                                                'cx':centerx.widget,'cy':centery.widget})

    load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    load_difpad_button.trigger(load_difpad)

    """ Difpad center boxes """
    sliders_box = widgets.HBox([centerx.widget,centery.widget],layout=box_layout)
    controls = widgets.Box([load_difpad_button.widget,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls,vbar,output])
    return box

#==============================================================================================================================================================#

def support_tab():

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
        figure,subplot.imshow(np.random.random((4,4)))
        subplot.set_title('Probe Support')
        figure.canvas.header_visible = False 
        plt.show()

    def plotshow(figure,subplot,image,title="",figsize=(8,8),savepath=None,show=False):
        subplot.clear()
        cmap, colors, bounds, norm = miqueles_colormap(image)
        handle = subplot.imshow(image, interpolation='nearest', cmap = cmap, norm=norm)
        if title != "":
            subplot.set_title(title)
        if show:
            plt.show()
        figure.canvas.draw_idle()

    def update_support(figure, subplot,output_dictionary,image,key1,cx,cy):

        output_dictionary[key1] = [cx,cy]
       
        # if exposure == True or button == True:
        #     image2 = masks_application(np.copy(image), output_dictionary)
        #     plotshow(figure,subplot,image2)
        # else:
        plotshow(figure,subplot,image)

    def load_difpad(dummy):

        # input_dict = h5py.File(os.path.join(global_dict["ProposalPath"],global_dict["Position_Folders"][0], global_dict["Position_Filename"]), 'r')['entry/data/data'][()][0, 0, :, :]

        print(centerx.widget.value,centery.widget.value)
        image = np.load(global_paths_dict['flipped_difpad_filepath'])
        widgets.interactive_output(update_support,{'figure':fixed(figure), 'subplot': fixed(subplot),
                                                'output_dictionary':fixed(global_dict),'image':fixed(image),
                                                'key1':fixed('DifpadCenter'),
                                                'cx':centerx.widget,'cy':centery.widget})

    load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    load_difpad_button.trigger(load_difpad)

    """ Difpad center boxes """
    sliders_box = widgets.HBox([centerx.widget,centery.widget],layout=box_layout)
    controls = widgets.Box([load_difpad_button.widget,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls,vbar,output])
    return box

#==============================================================================================================================================================#

def fresnel_tab():
    
    image_list, fresnel_number_list = [np.random.random((5,5))], [0]

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(image_list[0],cmap='jet') # initialize
        subplot.set_title('Propagated Probe')
        figure.canvas.header_visible = False 
        plt.show()


    def update_probe_plot_f1(fig,subplot,image_list,frame_list,index):
        subplot.clear()
        subplot.set_title(f'Frame #: {frame_list[index]:.1e}')
        subplot.imshow(image_list[index],cmap='jet')
        fig.canvas.draw_idle()

    def on_click_propagate_f1(dummy):
    
        print('Propagating probe...')
        image_list, fresnel_number_list = create_propagation_video_f1(global_paths_dict['probe_filepath'],
                                                        starting_f_value=starting_f_value,
                                                        ending_f_value=ending_f_value,
                                                        number_of_frames=number_of_frames,
                                                        jupyter=True)
        
        play_control.widget.max, selection_slider.widget.max = len(image_list)-1, len(image_list)-1

        widgets.interactive_output(update_probe_plot_f1,{'fig':fixed(figure),'subplot':fixed(subplot),'image_list':fixed(image_list),'frame_list':fixed(fresnel_number_list),'index':selection_slider.widget})
        print('\t Done!')

    def update_values_f1(n_frames,start_f,end_f,power):
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

    widgets.interactive_output(update_values_f1,{'n_frames':n_frames.widget,'start_f':start_f.widget,'end_f':end_f.widget,'power':power.widget})
    propagate_button = Button(description=('Propagate Probe'),layout=buttons_layout)
    propagate_button.trigger(on_click_propagate_f1)

    box = widgets.Box([n_frames.widget, power.widget, start_f.widget,end_f.widget,label,propagate_button.widget,fresnel_number.widget],layout=get_box_layout('700px'))
    play_box = widgets.VBox([play_box,output],layout=box_layout)
    box = widgets.HBox([box,vbar,play_box])
    return box


#==============================================================================================================================================================#

def caustic_tab():

    def update_label(label, distance_L,step):
        label.value= "Propagating from z = - {0} to z = {0} in steps of {1} um".format(distance_L,step)

    def update_frame_slider(frames, step, distance_L):
        number_of_frames = 2*(int((distance_L)/step))
        frames.widget.max = number_of_frames


    def update_probe_plot(fig,subplot,image,step, frames, distance_L):
        distance_z = step*frames
        subplot.clear()
        subplot.set_title('Wavefront at z = ' + str(distance_z -distance_L) + ' mm')
        subplot.imshow(image[frames],cmap='jet')
        fig.canvas.draw_idle()      

        # to plot probe with gaussian contours
        # fig, ax = plt.subplots(1, 1)
        # ax.hold(True)
        # ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
        # ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
        # plt.show()

    def update_fitted_plot(fig3,subplot3,image,step, frames, distance_L):
        distance_z = step*frames
        subplot3.clear()
        subplot3.set_title('Fitted Probe at z = ' + str(distance_z -distance_L) + ' mm')
        subplot3.imshow(image[frames],cmap='jet')
        fig3.canvas.draw_idle()

    def update_vertical_caustic_plot(fig2,subplot2, v_caustic, distance_L, frames, step, vertical_waist_step):
        number_of_frames = 2*(int((distance_L)/step))
        frames = (frames -int(number_of_frames/2))*step
        subplot2.clear()
        subplot2.set_title('Vertical Caustic \n Minimum waist at frame ' + str(vertical_waist_step))
        x = [frames, frames]
        y = [-1, 1]
        subplot2.plot(x, y, color = "white", linewidth = 2)
        subplot2.imshow(v_caustic, cmap='jet',  aspect = 'equal', extent = [-distance_L, distance_L, -1, 1])
        fig2.canvas.draw_idle()


    def update_horizontal_caustic_plot(fig4,subplot4, distance_L, h_caustic, frames, step, horizontal_waist_step):
        number_of_frames = 2*(int((distance_L)/step))
        frame = (frames -int(number_of_frames/2))*step
        subplot4.clear()
        subplot4.set_title(f'Horizontal Caustic \n Minimum waist  at frame ' + str(horizontal_waist_step))
        x4 = [frame, frame]
        y4 = [-1, 1]
        subplot4.plot(x4, y4, color = "white", linewidth = 2)
        subplot4.imshow(h_caustic,cmap='jet',  aspect = 'equal', extent = [-distance_L, distance_L, -1, 1])
        fig4.canvas.draw_idle()


    def on_click_propagate(dummy):

            
        step_list = np.arange(1E-3*(-distance_L.widget.value), 1E-3*(distance_L.widget.value + 0.5), 1E-3*(step.widget.value))
        
        wave = np.load(global_paths_dict['probe_filepath'])[0,:,:,:]
        wavefront = (np.abs(np.sum(wave, axis=0)))**2
              
        probe_stack = np.zeros((len(step_list), wavefront.shape[0], wavefront.shape[1]))
        data_fitted = np.zeros((len(step_list), wavefront.shape[0], wavefront.shape[1]))
      
        # Calculating object pixel size dx =========================================================================================
        c = 299792458             # Speed of Light [m/s]
        planck = 4.135667662E-18  # Plank constant [keV*s]
        wavelength = planck * c / global_dict['energy'] # meters
        global_dict["wavelength"] = wavelength
        global_dict['RestauredPixelSize'] = 55.5E-6
        
        # Convert pixel size:
        dx = wavelength * global_dict['DetDistance'] / ( global_dict['Binning'] * global_dict['RestauredPixelSize'] * global_dict['DetectorROI'] * 2)
        global_dict['object_pixel'] = dx
        mesh_hsize = (wavefront.shape[0]/2)
        
        x = np.arange(-mesh_hsize, mesh_hsize, 1, dtype = int)
        y = np.arange(-mesh_hsize, mesh_hsize, 1, dtype=int)
        (xx,yy) = np.meshgrid(x, y)
        xdata_tuple = np.vstack((xx.ravel(), yy.ravel()))
        
        # ===========================================================================================================================

        sigma_x = []
        sigma_y = []
        center = []
            
        for count in range(len(step_list)):
            probe_stack[count,:,:] = np.abs(propagate_beam(wavefront, dx, wavelength, step_list[count], 'fresnel'))
            popt, pcov = curve_fit(fit_2Dgaussian, xdata_tuple, (probe_stack[count,:,:]).ravel(), p0 = None, maxfev=5000)
            sigma_x.append(popt[3])  
            sigma_y.append(popt[4])       
            center.append((popt[1], popt[2]))   
            data_fitted[count,:,:] = np.reshape(fit_2Dgaussian(xdata_tuple, *popt), (wavefront.shape[0], wavefront.shape[1]))


        vertical_waist = 1000
        horizontal_waist = 1000
        vertical_waist_step = 1000
        horizontal_waist_step = 1000
          
        #=== VERTICAL CAUSTIC ===#
        vertical_caustic = np.abs(probe_stack[:,:, (int(((probe_stack.shape[2]))/2)) ])

        for i in range(len(sigma_y)):
            if (sigma_y[i])**2 < vertical_waist:
                vertical_waist = (sigma_y[i])**2
                vertical_waist_step = i
        

        #=== HORIZONTAL CAUSTIC ===#
        horizontal_caustic = np.abs(probe_stack[:,(int(((probe_stack.shape[1]))/2)),:])

        for i in range(len(sigma_x)):
            if (sigma_x[i])**2 < horizontal_waist:
                horizontal_waist = (sigma_x[i])**2
                horizontal_waist_step = i


        horizontal_caustic = np.swapaxes(horizontal_caustic,0,1)
        vertical_caustic = np.swapaxes(vertical_caustic,0,1)

        widgets.interactive_output(update_vertical_caustic_plot, {'fig2':fixed(figure2),'subplot2':fixed(subplot2),'v_caustic':fixed(vertical_caustic), 'distance_L':distance_L.widget, 'frames': frames.widget, 'step': step.widget, 'vertical_waist_step': fixed(vertical_waist_step)})
        widgets.interactive_output(update_probe_plot, {'fig':fixed(figure),'subplot':fixed(subplot),'image':fixed(probe_stack),'step':step.widget,'frames':frames.widget, 'distance_L':distance_L.widget})

        widgets.interactive_output(update_horizontal_caustic_plot, {'fig4':fixed(figure4),'subplot4':fixed(subplot4), 'distance_L': distance_L.widget, 'h_caustic':fixed(horizontal_caustic), 'frames': frames.widget, 'step': step.widget, 'horizontal_waist_step': fixed(horizontal_waist_step)})
        widgets.interactive_output(update_fitted_plot, {'fig3':fixed(figure3),'subplot3':fixed(subplot3),'image':fixed(data_fitted),'step':step.widget,'frames':frames.widget, 'distance_L':distance_L.widget})
        
        global_dict['vertical_caustic_filepath'] = os.path.join( '/'.join( (global_paths_dict['probe_filepath'].split("/"))[:-1] ), global_dict['Data_Filename'][:-5] +'_vertical_caustic.npy')
        np.save(global_dict['vertical_caustic_filepath'], vertical_caustic)      

        global_dict['horizontal_caustic_filepath'] = os.path.join( '/'.join( (global_paths_dict['probe_filepath'].split("/"))[:-1] ), global_dict['Data_Filename'][:-5] +'_horizontal_caustic.npy')
        np.save(global_dict['horizontal_caustic_filepath'], horizontal_caustic)          

        widgets.interactive_output(update_frame_slider, {'frames': fixed(frames), 'step':(step.widget),'distance_L':(distance_L.widget)})
    
    #---------------------------------------------------------------------------------------------------------------------------   
       
    initial_caustic = np.random.random((4,4)) # dummy
    image_list = [np.random.random((4,4))]
    step   = Input( {'dummy-key':0.5},'dummy-key', bounded=(0,50,0.5), slider=True, description='Step (mm)',layout=items_layout)
    distance_L = Input( {'dummy-key':0.5},'dummy-key', bounded=(0,100,0.5), slider=True, description='Distance to propagate (mm) ', layout=items_layout)
    number_of_frames = int((distance_L.widget.value * 2)/step.widget.value)
    frames = Input( {'dummy-key':0},'dummy-key', bounded=(0,number_of_frames,1), slider=True, description='Frame ',layout=items_layout)
    # widgets.interactive_output(update_frame_slider, {'frames': fixed(frames), 'step':(step.widget),'distance_L':(distance_L.widget)})
    
    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(image_list[0],cmap='jet') # initialize
        subplot.set_title('Propagated Probe')
        figure.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(figsize=(4,4))
        subplot2.imshow(initial_caustic,cmap='jet') # initialize
        subplot2.set_title('Vertical Caustic')
        figure2.canvas.header_visible = False 
        plt.show()

    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots(figsize=(4,4))
        subplot3.imshow(initial_caustic,cmap='jet') # initialize
        subplot3.set_title('Fitted Probe')
        figure3.canvas.header_visible = False 
        plt.show()


    output4 = widgets.Output()
    with output4:
        figure4, subplot4 = plt.subplots(figsize=(4,4))
        subplot4.imshow(initial_caustic,cmap='jet') # initialize
        subplot4.set_title('Horizontal Caustic')
        figure4.canvas.header_visible = False 
        plt.show()

    label = widgets.Label(value=r"Propagating from z = - {0} to z = {0} in steps of {1} mm".format(distance_L.widget.value,step.widget.value),layout=items_layout)
    widgets.interactive_output(update_label,{'label':fixed(label), 'distance_L':distance_L.widget,'step':step.widget})


    propagate_button = Button(description=('Propagate Probe'),layout=buttons_layout)
    propagate_button.trigger(on_click_propagate)

    inputs_box = widgets.Box([step.widget, distance_L.widget,frames.widget, propagate_button.widget, label],layout=get_box_layout('700px'))
    wavefronts_box = widgets.HBox([output, output3])
    caustic_box = widgets.HBox([output2, output4])
    figures_box = widgets.VBox([wavefronts_box, caustic_box])
    box = widgets.VBox([inputs_box, figures_box])
    return box

#==============================================================================================================================================================#

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
        play_control.widget.max = selection_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})

    def save_cropped_sinogram(dummy):
        cropped_sinogram = sinogram[:,top_crop.widget.value:-bottom_crop.widget.value,left_crop.widget.value:-right_crop.widget.value]
        print('Saving cropped frames...')
        np.save(global_paths_dict['cropped_sinogram_filepath'],cropped_sinogram)
        print('\t Saved!')

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")

    top_crop      = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1),  description="Top",   slider=True,layout=slider_layout)
    bottom_crop   = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1),  description="Bottom",   slider=True,layout=slider_layout)
    left_crop     = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1),description="Left", slider=True,layout=slider_layout)
    right_crop    = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1),description="Right", slider=True,layout=slider_layout)

    
    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    save_cropped_frames_button = Button(description="Save cropped frames",layout=buttons_layout,icon='fa-floppy-o') 
    save_cropped_frames_button.trigger(save_cropped_sinogram)
    
    buttons_box = widgets.Box([load_frames_button.widget,save_cropped_frames_button.widget],layout=get_box_layout('100%',align_items='center'))
    sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar,output])
    return box

#==============================================================================================================================================================#

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
    box = widgets.Box([saveJsonButton.widget,run_button.widget],layout=get_box_layout('500px'))
    box = widgets.HBox([box,job_box])

    return box

#==============================================================================================================================================================#

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

        probe = np.abs(np.load(global_paths_dict["probe_filepath"]))[:,0,:,:] # get only 0th order 
        selection_slider2.widget.max, selection_slider2.widget.value = probe.shape[0]-1, probe.shape[0]//2
        play_control2.widget.max = selection_slider2.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(probe),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed(True), 'cmap':fixed('jet'), 'frame_number': selection_slider2.widget})


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

#==============================================================================================================================================================#

def deploy_tabs(mafalda_session,tab2=inputs_tab(),tab3=center_tab(),tab4=caustic_tab(),tab5=ptycho_tab(),tab6=reconstruction_tab(),tab1=crop_tab(), tab8=fresnel_tab()):

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

    load_json_button  = Button(description="Load inputs",layout=buttons_layout,icon='folder-open-o')
    load_json_button.trigger(load_json)
    
    ptycho_box = widgets.Box([saveJsonButton.widget,load_json_button.widget,run_button.widget,view_jobs_button.widget,cancel_job_button.widget,job_number.widget],layout=get_box_layout('1000px',flex_flow='row'))
    
    children_dict = {
    "Ptycho Inputs"     : tab2,
    "Find Center"       : tab3,
    "Find Fresnel Number": tab8,
    "Probe Propagation" : tab4,
    "Reconstruction"    : tab6,
    "Crop"              : tab1
    }
    
    global mafalda
    mafalda = mafalda_session   
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Cluster', layout={'width': '10%'},description='Machine',disabled=False)
    widgets.interactive_output(update_gpu_limits,{"machine_selection":machine_selection})



    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    global jobNameField, jobQueueField
    jobNameField  = Input({'dummy_key':'CarnaubaPtycho'},'dummy_key',description="Insert slurm job name:")
    jobQueueField = Input({'dummy_key':'cnb-proc'},'dummy_key',description="Insert machine queue name:")
    global cpus, gpus
    gpus = Input({'dummy_key':1},'dummy_key',bounded=(0,4,1),  slider=True,description="Insert # of GPUs to use:")
    cpus = Input({'dummy_key':32},'dummy_key',bounded=(1,128,1),slider=True,description="Insert # of CPUs to use:")
    widgets.interactive_output(update_cpus_gpus,{"cpus":cpus.widget,"gpus":gpus.widget})


    boxSlurm = widgets.HBox([machine_selection,gpus.widget,cpus.widget,jobQueueField.widget,jobNameField.widget])
    box = widgets.VBox([boxSlurm,ptycho_box])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  

#==============================================================================================================================================================#

if __name__ == "__main__":
    pass
