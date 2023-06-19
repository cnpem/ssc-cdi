import subprocess
from subprocess import Popen, PIPE, STDOUT
import time, os
import getpass

import ipywidgets as widgets 
from ipywidgets import fixed 

from matplotlib import colors

field_style = {'description_width': 'initial'}


def call_and_read_terminal(cmd,mafalda,use_mafalda=True):
    """_summary_

    Args:
        cmd (_type_): _description_
        mafalda (_type_): _description_
        use_mafalda (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
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

def write_slurm_file(python_script_path,json_filepath_path,slurm_filepath,jobName='jobName',queue='cat',gpus=1,cpus=32):
    logfiles_path = slurm_filepath.rsplit('/',2)[0]
    username = getpass.getuser()
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o {logfiles_path}/logfiles/{username}_slurm.log        # Select output path of slurm file

if [ ${{SLURMD_NODENAME}} = "bertha" ]; then
    HOME=/home/ABTLUS/${{USER}}
    source /etc/profile
    source /etc/bash.bashrc
    source ${{HOME}}/.bashrc
    cd $HOME
fi

source /etc/profile.d/modules.sh # need this to load the correct python version from modules

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.2_parallel

python3 {python_script_path} {json_filepath_path} > {os.path.join(logfiles_path,'logfiles',f'{username}_output.log')} 2> {os.path.join(logfiles_path,'logfiles',f'{username}_error.log')}
"""
    
    with open(slurm_filepath,'w') as the_file:
        the_file.write(string)
    
def run_at_cluster(mafalda,json_filepath_path,queue='cat',gpus=[0],cpus=32, jobName='job', slurm_path = '/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/inputs/',script_path = "/ibira/lnls/labs/tepui/home/yuri.tonin/ssc-cdi/bin/caterete_ptycho.py"):
    
    user = getpass.getuser()
    
    slurm_filepath = os.path.join(slurm_path,f'{user}_job.srm')
    jobName = user+'_'+jobName
    
    gpus = len(gpus)
    write_slurm_file(script_path,json_filepath_path,slurm_filepath,jobName,queue,gpus,cpus)
    call_cmd_terminal(slurm_filepath,mafalda,remove=False)

############################ PLOTS ####################################

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


def plot_DPs_with_slider(data,axis=0):

    colornorm=colors.Normalize(vmin=data.min(), vmax=data.max())
    cmap = 'viridis'
    
    def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title=False,clear_axis=False,cmap=cmap,norm=colors.LogNorm()):
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
            subplot.set_title(f'#{frame_number}')
        if clear_axis == True:
            subplot.set_xticks([])
            subplot.set_yticks([])    
        figure.canvas.draw_idle()
    
    output = widgets.Output()
    
    with output:
        figure, ax = plt.subplots(dpi=150)
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False
        figure.colorbar(matplotlib.cm.ScalarMappable(norm=colornorm, cmap=cmap))
        plt.show()   

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)

    selection_slider.widget.max, selection_slider.widget.value = data.shape[0] - 1, data.shape[0]//2
    play_control.widget.max =  selection_slider.widget.max
    widgets.interactive_output(update_imshow, {'sinogram':fixed(data),'figure':fixed(figure),'title':fixed(True),'subplot':fixed(ax),'axis':fixed(axis), 'norm':fixed(colors.LogNorm()),'frame_number': selection_slider.widget})    
    box = widgets.VBox([play_box,output])
    return box

def plot_flipped_full_DP(restored_full_DP):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(restored_full_DP,norm=LogNorm()), ax.set_title("Average of DPs")