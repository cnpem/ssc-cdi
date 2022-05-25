import subprocess
from subprocess import Popen, PIPE, STDOUT
import time, os, json, ast
import numpy as np
import matplotlib.pyplot as plt
import paramiko
import getpass
import ipywidgets as widgets 
from ipywidgets import fixed 

from .fresnel import create_propagation_video
from .functions import masks_application
from .misc import miqueles_colormap

madalda_ip = "10.30.4.10" # Mafalda IP
mafalda_port = 22

def write_to_file(python_script_path,jsonFile_path,output_path="",slurmFile = 'slurmJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
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

python3 {python_script_path} {jsonFile_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
# python3 {python_script_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}

"""
    
    with open(slurmFile,'w') as the_file:
        the_file.write(string)
    
    return slurmFile

def call_and_read_terminal(cmd,mafalda):
    if 0:
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        terminal_output = p.stdout.read() # Read output from terminal
    else:
        stdin, stdout, stderr = mafalda.exec_command(cmd)
    terminal_output = stdout.read() 
    print('Output: ',terminal_output)
    print('Error:  ',stderr.read())
    return terminal_output

def call_cmd_terminal(filename,mafalda,remove=False):
    cmd = f'sbatch {filename}'
    terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
    given_jobID = terminal_output.rsplit("\n",1)[0].rsplit(" ",1)[1]
    if remove: # Remove file after call
        cmd = f'rm {filename}'
        subprocess.call(cmd, shell=True)
        
    return given_jobID
        
def monitor_job_execution(given_jobID,mafalda):
    sleep_time = 10 # seconds
    print(f'Starting job #{given_jobID}...')
    time.sleep(3) # sleep for a few seconds to wait for job to really start
    jobDone = False
    job_duration = 0
    while jobDone == False:
        time.sleep(sleep_time)
        job_duration += sleep_time
        cmd = f'squeue | grep {given_jobID}'
        terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
        if given_jobID not in terminal_output:
            jobDone = True
        else:
            print(f'\tWaiting for job {given_jobID} to finish. Current duration: {job_duration/60:.2f} minutes')
    return print(f"\t \t Job {given_jobID} done!")

def run_ptycho_from_jupyter(mafalda,python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    slurm_file = write_to_file(python_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

    
def connect_server():
    host = madalda_ip #"10.30.4.10" # Mafalda IP
    port = mafalda_port #22
    username = input("Username:")
    print("Password:")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, getpass.getpass())
    return ssh

def deploy_runPtycho_button(mafalda,auxiliary_dict):


    pythonScript = auxiliary_dict['pythonScript'] 
    jsonFile     = auxiliary_dict['jsonFile']
    outputFolder = auxiliary_dict['outputFolder']
    slurmFile    = auxiliary_dict['slurmFile']
    jobName      = auxiliary_dict['jobNameField']
    queue        = auxiliary_dict['jobQueueField']
    gpus         = auxiliary_dict['gpusField']
    cpus         = auxiliary_dict['cpusField']
    args = mafalda,pythonScript,jsonFile,outputFolder,slurmFile,jobName,queue,gpus,cpus

    from functools import partial

    def run_ptycho(dummy,args=()):
        mafalda,pythonScript,jsonFile,outputFolder,slurmFile,jobName,queue,gpus,cpus = args
        run_ptycho_from_jupyter(mafalda,pythonScript,jsonFile,output_path=outputFolder,slurmFile = slurmFile,jobName=jobName,queue=queue,gpus=gpus,cpus=cpus)

    run_ptycho_partial = partial(run_ptycho,args=args)
    button_layout = widgets.Layout(align_items='flex-end',width='30%', height='50px')
    run_button = widgets.Button(description=('RUN PTYCHO'),layout=button_layout,icon='play')
    run_button.on_click(run_ptycho_partial)
    # display(run_button)
    return run_button

class InputField(object):

    def __init__(self,dictionary,key):
        
        field_layout = widgets.Layout(align_items='flex-start',width='50%')
        field_style = {'description_width': 'initial'}
        
        self.dictionary = dictionary
        self.key = key

        if isinstance(self.dictionary[self.key],bool):
            self.field = widgets.Checkbox(description=f'{key} {str(type(self.dictionary[self.key]))}',value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],int):
            self.field = widgets.IntText( description=f'{key} {str(type(self.dictionary[self.key]))}',value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],float):
            self.field = widgets.FloatText(description=f'{key} {str(type(self.dictionary[self.key]))}',value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],list):
            self.field = widgets.Text(description=f'{key} {str(type(self.dictionary[self.key]))}',value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],str):
            self.field = widgets.Text(description=f'{key} {str(type(self.dictionary[self.key]))}',value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],dict):
            self.field = widgets.Text(description=f'{key} {str(type(self.dictionary[self.key]))}',value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        
        widgets.interactive_output(self.update_dict_value,{'value':self.field})

        
    def update_dict_value(self,value):
        if isinstance(self.dictionary[self.key],list):
            self.dictionary[self.key] = ast.literal_eval(value)
        elif isinstance(self.dictionary[self.key],dict):
            self.dictionary[self.key] = ast.literal_eval(value)
        else:
            self.dictionary[self.key] = value

def deploy_inputJson_fields(dictionary):
        
    keys_list = [key for key in dictionary]
    fields_dict = {}
    box = widgets.VBox()
    for counter in range(0,len(keys_list)): # deployt all interactive fields    
        fields_dict_key = f'field_{counter}'
        fields_dict[fields_dict_key] = InputField(dictionary,keys_list[counter])
#         display(fields_dict[fields_dict_key].field) 
        box = widgets.VBox([box,fields_dict[fields_dict_key].field])
#     display(box)
    return box

def deploy_framesVisualization(path_to_npy_frames):
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets 
    from ipywidgets import fixed, VBox
    from functools import partial


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

    args = (path_to_npy_frames, fig, ax1,play,slider)

    loadButton = widgets.Button(description="Load Object Preview",layout=widgets.Layout(width='50%',height='50px'))
    on_click_load_partial = partial(on_click_load,args=args)
    loadButton.on_click(on_click_load_partial)
    box1 = VBox([play_box,output])
    box1 = VBox([loadButton,box1])
    # display(loadButton)
    # display(box1)
    return box1

def global_deploy(dictionary,mafalda,auxiliary_dict):

    path_to_diffraction_pattern_file = auxiliary_dict['path_to_diffraction_pattern_file']
    path_to_probefile                = auxiliary_dict['path_to_probefile']
    path_to_npy_frames               = auxiliary_dict['path_to_npy_frames']
    
    # display(deploy_input_fields(dictionary,auxiliary_dict)[0])
    # display(deploy_input_fields(dictionary,auxiliary_dict)[1])
    # display(deploy_json_save_button(dictionary,auxiliary_dict))
    # display(deploy_inputJson_fields(dictionary))
    # display(deploy_runPtycho_button(mafalda,auxiliary_dict))
    display(deploy_center_interface(path_to_diffraction_pattern_file,dictionary))
    display(deploy_interface_fresnel(path_to_probefile,dictionary))
    display(deploy_framesVisualization(path_to_npy_frames))

def inputs_deploy(dictionary,mafalda,auxiliary_dict):

    display(deploy_json_save_button(dictionary,auxiliary_dict))
    display(deploy_inputJson_fields(dictionary))
    display(deploy_runPtycho_button(mafalda,auxiliary_dict))

def deploy_input_fields(dictionary,auxiliary_dict):

    field_layout = widgets.Layout(align_items='flex-start',width='70%')
    field_style = {'description_width': 'initial'}

    from functools import partial
    from ipywidgets import fixed
    import os

    def loadJson(dummy,user_folder="/ibira/lnls/beamlines/caterete/apps/jupyter/" ,dictionary={}):
        json_path = os.path.join(user_folder,"000000_template.json")
        template_dict = json.load(open(json_path))
        for key in template_dict:
            dictionary[key] = template_dict[key]
        
    loadJsonButton = widgets.Button(description="Load json template",layout=widgets.Layout(width='30%', height='100px',max_height='50px'),icon='play')

    ProposalPathField = widgets.Text(value='/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/data/ptycho2d/',description="Insert data path:",style=field_style,layout=field_layout)
    AcquisitionFolderField = widgets.Text(value='[ "SS61"]',description="Insert list of acquisiton folders inside the data path:",style=field_style,layout=field_layout)
    jobNameField  = widgets.Text(value='myJobName',description="Insert slurm job name:",style=field_style)
    jobQueueField = widgets.Text(value='cat-proc',description="Insert machine queue name:",style=field_style)
    gpusField = widgets.BoundedIntText(value=1,min=0,max=4,description="Insert # of gpus to use:",style=field_style)
    cpusField = widgets.BoundedIntText(value=32,min=1,max=128,description="Insert # of cpus to use:",style=field_style)

    # auxiliary_dict = { 'pythonScript':"",
    #                 'standard_folder': "",
    #                 'slurmFile': "",
    #                 'jobNameField':'',
    #                 'jobQueueField':'',
    #                 'gpusField':0,
    #                 'cpusField':0,
    #                 'path_to_diffraction_pattern_file':"",
    #                 'path_to_probefile':"",
    #                 'path_to_npy_frames':""}


    def update_inputs(dictionary,auxiliary_dict,ProposalPathField,AcquisitionFolderField,jobNameField,jobQueueField,gpusField,cpusField):
        
        """ HARD CODED PATHS """
        if 1: # paths for beamline use
            user_folder     = "/ibira/lnls/beamlines/caterete/apps/ptycho-dev/" # folder with json template, and where to output jupyter files. path to output slurm logs as well
            pythonScript    = '/ibira/lnls/beamlines/caterete/apps/ssc-cdi/bin/sscptycho_main.py' # path with python script to run
        else: # paths for GCC tests       
            user_folder     = "/ibira/lnls/beamlines/caterete/apps/jupyter/" 
            pythonScript    = '/ibira/lnls/beamlines/caterete/apps/jupyter/sscptycho_main.py' # path with python script to run
        
        slurmFile       = os.path.join(user_folder,'slurm_job.srm') # path to create slurm_file
        jsonFile        = os.path.join(user_folder,'user_input.json') # path with input json to run
       
        acquisition_folder = ast.literal_eval(AcquisitionFolderField)[0] # changes with control
        standard_folder = os.path.join(ProposalPathField.rsplit('/',3)[0], 'proc','recons',acquisition_folder) # changes with control
        path_to_diffraction_pattern_file = os.path.join(standard_folder,'03_difpad_raw_flipped_3072.npy') # path to load diffraction pattern
        path_to_probefile                = os.path.join(standard_folder,f'probe_{acquisition_folder}.npy') # path to load probe
        path_to_npy_frames               = os.path.join(standard_folder,f'phase_{acquisition_folder}.npy') # path to load npy with first reconstruction preview
        
        loadJsonPartial = partial(loadJson,user_folder=user_folder,dictionary=dictionary)
        loadJsonButton.on_click(loadJsonPartial)
        
        auxiliary_dict['standard_folder'] = standard_folder
        auxiliary_dict['pythonScript'] = pythonScript
        auxiliary_dict['slurmFile'] =slurmFile
        auxiliary_dict['outputFolder'] = user_folder
        auxiliary_dict['jsonFile'] = jsonFile
        auxiliary_dict['jobNameField']=jobNameField
        auxiliary_dict['jobQueueField']=jobQueueField
        auxiliary_dict['gpusField']=gpusField
        auxiliary_dict['cpusField']=cpusField
        auxiliary_dict['path_to_diffraction_pattern_file']=path_to_diffraction_pattern_file
        auxiliary_dict['path_to_probefile']=path_to_probefile
        auxiliary_dict['path_to_npy_frames']=path_to_npy_frames
        
        dictionary["ProposalPath"] = ProposalPathField
        dictionary["Acquisition_Folders"] = ast.literal_eval(AcquisitionFolderField)

        return dictionary
        
    widgets.interactive_output(update_inputs,{'dictionary':fixed(dictionary),'auxiliary_dict':fixed(auxiliary_dict),'ProposalPathField':ProposalPathField,'AcquisitionFolderField':AcquisitionFolderField,"jobNameField":jobNameField,"jobQueueField":jobQueueField,"gpusField":gpusField,"cpusField":cpusField})
        
    box2 = widgets.VBox([jobNameField,jobQueueField,gpusField,cpusField])
    box3 = widgets.VBox([ProposalPathField,AcquisitionFolderField,box2])
    box1 = widgets.VBox([loadJsonButton,box3])

    return loadJsonButton, box3

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
   
def deploy_center_interface(path_to_diffraction_pattern_file,output_dictionary):
        
    output = widgets.Output()
    image = np.load(path_to_diffraction_pattern_file)
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
    saida = widgets.HBox([output,controls])
    # display(saida)
    return saida


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
        plt.show()

    args = (path_to_probefile,starting_f_value,ending_f_value,number_of_frames,play,slider,fig,ax1)
    start_ptycho_button.on_click(partial(on_click_propagate,args=args))

    box3 = widgets.VBox([play_box,output])
    saida = widgets.HBox([box2,box3])

    # display(saida)
    return saida


def deploy_json_save_button(dictionary,auxiliary_dict):

    import json
    from functools import partial

    def save_on_click(dummy,jsonFile="",dictionary={}):
        print(jsonFile)
        with open(jsonFile, 'w') as file:
            json.dump(dictionary, file)

    save_on_click_partial = partial(save_on_click,jsonFile=auxiliary_dict["jsonFile"],dictionary=dictionary)
    button_layout = widgets.Layout(align_items='flex-end',width='30%', height='50px')

    saveJsonButton = widgets.Button(description="Save Dictinary to json",layout=button_layout)
    saveJsonButton.on_click(save_on_click_partial)
    return saveJsonButton




if __name__ == "__main__":

    python_script_path = 'testpy.py'
    jsonFile_path = ''
    run_ptycho_from_jupyter(python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32)