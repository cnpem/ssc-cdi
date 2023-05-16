import subprocess
from subprocess import Popen, PIPE, STDOUT
import time, os
import ast

import ipywidgets as widgets 
from ipywidgets import fixed 

from matplotlib import colors

field_style = {'description_width': 'initial'}


def call_and_read_terminal(cmd,mafalda,use_mafalda=True):
    if use_mafalda == False:
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
    print('Terminal output:',terminal_output)
    if remove: # Remove file after call
        cmd = f'rm {filename}'
        subprocess.call(cmd, shell=True)
        
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

def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title=False,clear_axis=True,cmap='gray',norm=None):
    subplot.clear()
    if bottom == None or right == None:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap=cmap,norm=norm)
        elif axis == 1:
            subplot.imshow(sinogram[top:bottom,frame_number,left:right],cmap=cmap,norm=norm)
        elif axis == 2:
            subplot.imshow(sinogram[top:bottom,left:right,frame_number],cmap=cmap,norm=norm)
    else:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap=cmap,norm=norm)
        elif axis == 1:
            subplot.imshow(sinogram[top:-bottom,frame_number,left:-right],cmap=cmap,norm=norm)
        elif axis == 2:
            subplot.imshow(sinogram[top:-bottom,left:-right,frame_number],cmap=cmap,norm=norm)
    if title == True:
        subplot.set_title(f'Frame #{frame_number}')
    if clear_axis == True:
        subplot.set_xticks([])
        subplot.set_yticks([])    
    figure.canvas.draw_idle()

class VideoControl:
    
    def __init__ (self,slider,step,interval,description):
    
        value, minimum, maximum = slider.widget.value,slider.widget.min,slider.widget.max

        self.widget = widgets.Play(value=value,
                            min=minimum,
                            max=maximum,
                            step=step,
                            interval=interval,
                            description=description,
                            disabled=False )

        widgets.jslink((self.widget, 'value'), (slider.widget, 'value'))

class Button:

    def __init__(self,description="DESCRIPTION",layout=widgets.Layout(),icon=""):

        self.button_layout = layout
        self.widget = widgets.Button(description=description,layout=self.button_layout,icon=icon,style=field_style)

    def trigger(self,func):
        self.widget.on_click(func)

class Input(object):

    def __init__(self,dictionary,key,description="",layout=None,bounded=(),slider=False):
        
        self.dictionary = dictionary
        self.key = key
        
        if layout == None:
            self.items_layout = widgets.Layout()
        else:
            self.items_layout = layout
   
        field_description = description

        if isinstance(self.dictionary[self.key],bool):
            self.widget = widgets.Checkbox(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],int):
            if bounded == ():
                self.widget = widgets.IntText( description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
            else:
                if slider:
                    self.widget = widgets.IntSlider(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
                else:
                    self.widget = widgets.BoundedIntText(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],float):
            if bounded == ():
                self.widget = widgets.FloatText(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
            else:
                self.widget = widgets.BoundedFloatText(min=bounded[0],max=bounded[1],step=bounded[2],description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],list):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],str):
            self.widget = widgets.Text(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],dict):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        
        widgets.interactive_output(self.update_dict_value,{'value':self.widget})

    def update_dict_value(self,value):
        if isinstance(self.dictionary[self.key],list) or isinstance(self.dictionary[self.key],dict):
            if isinstance(value,str):
                self.dictionary[self.key] = ast.literal_eval(value)
            else :
                self.dictionary[self.key] = value    
        else:
            self.dictionary[self.key] = value    

def slide_and_play(slider_layout=widgets.Layout(width='90%'),label="",description="",frame_time_milisec = 0):

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

def get_box_layout(width,flex_flow='column',align_items='center',border='1px none black'):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)


def write_slurm_file(python_script_path,json_filepath_path,slurm_filepath,jobName='jobName',queue='cat',gpus=1,cpus=32):
    logfiles_path = slurm_filepath.rsplit('/',2)[0]
    username = os.getlogin()
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o {logfiles_path}/logfiles/{username}_slurm.log        # Select output path of slurm file

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.2_parallel

python3 {python_script_path} {json_filepath_path} > {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_output.log')} 2> {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_error.log')}
"""
    
    with open(slurm_filepath,'w') as the_file:
        the_file.write(string)
    
def run_ptycho_at_cluster(mafalda,json_filepath_path,queue='cat',gpus=[0],cpus=32, slurm_path = '/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/inputs/'):
    
    user = os.getlogin()
    
    python_script_path = "/ibira/lnls/labs/tepui/home/yuri.tonin/ssc-cdi/bin/caterete_ptycho.py"
    slurm_filepath = os.path.join(slurm_path,f'{user}_ptycho_job.srm')
    jobName = user+'_ptycho'
    
    gpus = len(gpus)
    write_slurm_file(python_script_path,json_filepath_path,slurm_filepath,jobName,queue,gpus,cpus)
    
    call_cmd_terminal(slurm_filepath,mafalda,remove=False)