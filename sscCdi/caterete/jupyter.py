import subprocess
from subprocess import Popen, PIPE, STDOUT
import time, os, json, ast
import paramiko
import getpass
import ipywidgets as widgets 

from .fresnel import deploy_interface_fresnel
from .functions import deploy_interface

def write_to_file(python_script_path,jsonFile_path,output_path="",slurmFile = 'slurmJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    # Create slurm file
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o ./slurm.out        # Select output path of slurm file

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

# python3 {python_script_path} {jsonFile_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
python3 {python_script_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}

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
    # print('Output: ',terminal_output)
    # print('Error:  ',stderr.read())
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
    time.sleep(3) # sleep for a few seconds to wait for job to really start
    jobDone = False
    while jobDone == False:
        time.sleep(2)
        cmd = f'squeue | grep {given_jobID}'
        terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
        if given_jobID not in terminal_output:
            jobDone = True
        else:
            print(f'\tWaiting for job {given_jobID} to finish')
    return print(f"\t \t Job {given_jobID} done!")

def run_ptycho_from_jupyter(mafalda,python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    
    # python_script_path = 'testpy.py'
    jsonFile_path = ''
    slurm_file = write_to_file(python_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

    
def connect_mafalda():
    host = "10.30.4.10" # Mafalda IP
    port = 22
    username = input("Username:")
    print("Password:")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, getpass.getpass())
    return ssh

def deploy_runPtycho_button(args):

    from functools import partial

    def run_ptycho(dummy,args=()):
        mafalda,pythonScript,jsonFile,outputFolder,slurmFile,jobName,queue,gpus,cpus = args
        run_ptycho_from_jupyter(mafalda,pythonScript,jsonFile,output_path=outputFolder,slurmFile = slurmFile,jobName=jobName,queue=queue,gpus=gpus,cpus=cpus)

    run_ptycho_partial = partial(run_ptycho,args=args)
    button_layout = widgets.Layout(align_items='flex-end',width='20%')
    run_button = widgets.Button(description=('RUN PTYCHO'),layout=button_layout)
    run_button.on_click(run_ptycho_partial)
    display(run_button)

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
    for counter in range(0,len(keys_list)): # deployt all interactive fields    
        fields_dict_key = f'field_{counter}'
        fields_dict[fields_dict_key] = InputField(dictionary,keys_list[counter])
        display(fields_dict[fields_dict_key].field) 


def deploy_framesVisualization(path_to_npy_frames):
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets 
    from ipywidgets import fixed, VBox
   
    frame_time_in_milisec = 100
    output = widgets.Output()

    centered_box_layout = widgets.Layout(flex_flow='column',align_items='center',width='30%')

    def update_imshow(image_list,index):
        ax1.clear()
        ax1.set_title(f'Frame #: {frame_list[index]:.1f}')
        ax1.imshow(image_list[index],cmap='gray')
        fig.canvas.draw_idle()

    matrix = np.load(path_to_npy_frames)

    image_list = [ matrix[i,:,:] for i in range(0,matrix.shape[0])]
    # image_list = [ matrix[:,i,:] for i in range(0,matrix.shape[1])]
#     image_list = [ matrix[:,:,i] for i in range(0,matrix.shape[2])]

    frame_list = [ i for i in range(matrix.shape[0])]

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

    out = widgets.interactive_output(update_imshow,{'image_list':fixed(image_list),'index':play})
    with output:
        fig = plt.figure(figsize=(10,5))
        ax1  = fig.add_subplot(1, 1, 1)
        ax1.imshow(image_list[0],cmap='gray') # initialize

    box1 = VBox([play_box,output])
        
    display(box1)


def global_deploy(dictionary,mafalda,auxiliary_dict):



    pythonScript = auxiliary_dict['pythonScript'] 
    jsonFile     = auxiliary_dict['jsonFile']
    outputFolder = auxiliary_dict['outputFolder']
    slurmFile    = auxiliary_dict['slurmFile']
    jobName      = auxiliary_dict['jobNameField']
    queue        = auxiliary_dict['jobQueueField']
    gpus         = auxiliary_dict['gpusField']
    cpus         = auxiliary_dict['cpusField']

    path_to_diffraction_pattern_file = auxiliary_dict['path_to_diffraction_pattern_file']
    path_to_probefile                =  auxiliary_dict['path_to_probefile']
    path_to_npy_frames               = auxiliary_dict['path_to_npy_frames']
    args = mafalda,pythonScript,jsonFile,outputFolder,slurmFile,jobName,queue,gpus,cpus

    deploy_inputJson_fields(dictionary)
    deploy_runPtycho_button(args)
    deploy_interface(path_to_diffraction_pattern_file,dictionary)
    deploy_interface_fresnel(path_to_probefile,dictionary)
    deploy_runPtycho_button(args)
    deploy_framesVisualization(path_to_npy_frames)


    # jupyter.deploy_inputJson_fields(dictionary)
    # jupyter.deploy_runPtycho_button(args)
    # functions.deploy_interface(path_to_diffraction_pattern_file,dictionary)
    # fresnel.deploy_interface_fresnel(path_to_probefile,dictionary)
    # jupyter.deploy_runPtycho_button(args)
    # jupyter.deploy_framesVisualization(path_to_npy_frames)

    
def deploy_input_fields(dictionary,auxiliary_dict):

    field_layout = widgets.Layout(align_items='flex-start',width='70%')
    field_style = {'description_width': 'initial'}

    from functools import partial
    from ipywidgets import fixed
    import os

    def loadJson(dummy,user_folder="/ibira/lnls/beamlines/caterete/apps/jupyter-dev/" ,dictionary={}):
        json_path = os.path.join(user_folder,"000000_template.json")
        template_dict = json.load(open(json_path))
        for key in template_dict:
            dictionary[key] = template_dict[key]
        
    loadJsonButton = widgets.Button(description="Load json template",layout=widgets.Layout(width='50%', height='100px',max_height='50px'))

    ProposalPathField = widgets.Text(value='/ibira/lnls/beamlines/caterete/proposals/YYYYNNNN',description="Insert data path:",style=field_style,layout=field_layout)
    AcquisitionFolderField = widgets.Text(value='[ "dataFolder1","dataFolder2"]',description="Insert list of acquisiton folders inside the data path:",style=field_style,layout=field_layout)
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
        user_folder     = "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/" # folder with codes. path to output slurm logs as well
        pythonScript    = '/ibira/lnls/beamlines/caterete/apps/jupyter-dev/sscptycho_main.py' # path with python script to run
        slurmFile       = os.path.join(user_folder,'slurm_job.srm') # path to create slurm_file
        jsonFile        = os.path.join(user_folder,'user_input.json') # path with input json to run

        standard_folder = "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/" # folder with beamline outputs
        path_to_diffraction_pattern_file = os.path.join(standard_folder,'03_difpad_raw_flipped_3072.npy') # path to load diffraction pattern
        path_to_probefile                = os.path.join(standard_folder,'probe_SS61.npy') # path to load probe
        path_to_npy_frames               = os.path.join(standard_folder,'unwrap_microagg_SS61_ordered_phase.npy') # path to load npy with first reconstruction preview
        # acquisition_folder = ast.literal_eval(AcquisitionFolderField)[0] # changes with control
        # standard_folder = os.path.join(ProposalPathField.rsplit('/',3)[0], 'proc','recons',acquisition_folder) # changes with control
        # path_to_diffraction_pattern_file = os.path.join(standard_folder,'03_difpad_raw_flipped_3072.npy') # path to load diffraction pattern
        # path_to_probefile                = os.path.join(standard_folder,f'probe_{acquisition_folder}.npy') # path to load probe
        # path_to_npy_frames               = os.path.join(standard_folder,f'phase_{acquisition_folder}_01.npy') # path to load npy with first reconstruction preview
        
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
        dictionary["Acquisition_Folders"] = AcquisitionFolderField

        return dictionary
        
    widgets.interactive_output(update_inputs,{'dictionary':fixed(dictionary),'auxiliary_dict':fixed(auxiliary_dict),'ProposalPathField':ProposalPathField,'AcquisitionFolderField':AcquisitionFolderField,"jobNameField":jobNameField,"jobQueueField":jobQueueField,"gpusField":gpusField,"cpusField":cpusField})
        
    box2 = widgets.VBox([jobNameField,jobQueueField,gpusField,cpusField])
    box3 = widgets.VBox([ProposalPathField,AcquisitionFolderField,box2])
    box1 = widgets.VBox([loadJsonButton,box3])
    display(box1)








if __name__ == "__main__":

    python_script_path = 'testpy.py'
    jsonFile_path = ''
    run_ptycho_from_jupyter(python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32)