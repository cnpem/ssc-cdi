from cmath import sin
import ipywidgets as widgets
from ipywidgets import fixed
import ast 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os, time
import json
from tqdm import tqdm
from skimage.io import imsave

from sscRadon import radon
from sscRaft import parallel

from .jupyter_recon2D import call_cmd_terminal, monitor_job_execution, call_and_read_terminal
from .unwrap import unwrap_in_parallel
from .misc import list_files_in_folder
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from skimage.morphology import square, erosion, opening, convex_hull_image, dilation


global sinogram
sinogram = np.random.random((2,2,2)) # dummy sinogram

""" Standard dictionary definition """
global_dict = {"ibira_data_path": "/ibira/lnls/beamlines/caterete/proposals/20210177/data/ptycho3d/",
               "folders_list": ["microagg_P2_01"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/proc/recons/microagg_P2_01/object_microagg_P2_01.npy",
               "jupyter_folder":"/ibira/lnls/beamlines/caterete/apps/jupyter/"  , # FIXED PATH FOR BEAMLINE
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_list": [7,20,36,65,94,123,152,181,210,239,268,296,324],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,
               "bad_frames_list2": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               
               "wiggle_reference_frame": 0,
               "wiggle_cpus": 32,
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "tomo_n_of_gpus": [0],
               "tomo_threshold" : float(100.0), # max value to be left in reconstructed matrix
               "run_all_tomo_steps":False,
               "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "ConvexHull":1, "Wiggle":1, "Tomo":1 }, # select steps when performing full recon
               "contrast_type": "Phase", # Phase or Absolute
}


""" Standard folders definitions"""
tomo_script_path = '~/ssc-cdi/bin/sscptycho_raft.py' # NEED TO CHANGE FOR EACH USER? 

angles_filename = 'dummy_angles_filename'
object_filename = 'dummy_object_filename'

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

############################################ PROCESSING FUNCTIONS ###########################################################################

def create_directory_if_doesnt_exist(*args):
    for arg in args:
        if os.path.isdir(arg) == False:
            os.mkdir(arg)

def angle_mesh_organize( mdata, angles, use_max=True ): 
        """ Project angles to regular mesh and pad it to run from 0 to 180

        Args:
            mdata (_type_): _description_
            angles (_type_): _description_
            use_max (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        angles_list = []

        starta = angles[:,1].min()
        enda   = angles[:,1].max()
        rangea = enda - starta
        forw = np.roll(angles[:,1],1,0) - angles[:,1]
        forw[-1] = forw[-2]
        forw[0] = forw[1]
        maxdth = abs(forw).max() 
        mindth = abs(forw).min()
        if use_max == False:
            nangles = int( (np.pi)/mindth ) 
        else:
            nangles = int( (np.pi)/maxdth ) 
        dth = np.pi / (nangles-1)
        ndata = np.zeros([nangles,mdata.shape[1],mdata.shape[2]])
        idx = np.zeros([nangles], dtype=np.int)
        for k in range(nangles):
            angle = -np.pi/2.0 + k*dth
            if angle > enda or angle < starta:
                idx[k] = -1
                ndata[k,:,:] = np.zeros([mdata.shape[1],mdata.shape[2]])
            else:
                idx[k] = int( np.argmin( abs(angle - angles[:,1]) ) )
                ndata[k,:,:] = mdata[idx[k],:]
            angles_list.append(angle*180/np.pi)
        first = np.argmin((idx < 0)) - 1
        angles_array = np.asarray(angles_list) - np.min(angles_list) # convert values to range 0 - 180
        return ndata, idx, first, angles_array 

def tomography(algorithm,data_selection,angles_filename,iterations,GPUs,do_regularization,regularization_parameter,use_regularly_spaced_angles=True):
    
    data = np.load(os.path.join(output_folder,f'{data_selection}_wiggle_sinogram.npy'))

    if use_regularly_spaced_angles == True:
        angles_filename = angles_filename[:-4]+'_projected.npy'

    angles_filepath = os.path.join(output_folder,angles_filename)

    angles = np.load(angles_filepath) # sorted angles?

    """ ######################## Regularization ################################ """

    if 0: # Paola's approach to correcting angles
        # Padded zeros for completion of missing wedge:  from (-70,70) - 140 degrees, to (-90,90) - 180 degrees
        angles = angles[:,1] # get the angles
        anglesmax, anglesmin = angles[-1],  angles[0]     # max and min angles
        angles = np.insert(angles, 0, -90)     # Insert the first angle as -90. Why I do that? Beacause I assume that the first angles is always zero, in order to correctly find the angle step size inside the EM algorithm fro all angles.
        data = np.pad(data,((1,0),(0,0),(0,0)),'constant') # Pad zeros corresponding to the extra -90 value
        angles = (angles + 90) # Transform the angles from (-90,90) to (0,180)

    if do_regularization == True and algorithm == "EEM": # If which_reconstruction == "EEM" MIQUELES
        print('\tBegin Regularization')
        for k in range(data.shape[1]):
            data[:,k,:] = regularization( data[:,k,:], regularization_parameter)

        print('\tRegularization Done')

    """ ######################## RECON ################################ """
    print('Starting tomographic algorithm: ',algorithm)
    if algorithm == "TEM" or algorithm == "EM":
        data = np.exp(-data)
    elif algorithm == "ART":
        flat = np.ones([1,data.shape[-2],data.shape[-2]],dtype=np.uint16)
        dark = np.zeros(flat.shape[1:],dtype=np.uint16)
        centersino1 = Centersino(frame0=data[0,:,:], frame1=data[-1,:,:], flat=flat[0], dark=dark, device=0) 

    if algorithm != "EEM": # for these
        rays, slices = data.shape[-1], data.shape[-2]
        reconstruction3D = np.zeros((rays,slices,rays))
        for i in range(slices):
            sinogram = data[:,i,:]
            if algorithm == "ART":
                reconstruction3D[:,i,:]= MaskedART( sino=sinogram,mask=flat,niter=iterations ,device=GPUs)
            elif algorithm == "FBP": 
                reconstruction3D[:,i,:]= FBP( sino=sinogram,angs=angles,device=GPUs,csino=centersino1)
            elif algorithm == "RegBackprojection":
                reconstruction3D[:,i,:]= Backprojection( sino=sinogram,device=GPUs)
            elif algorithm == "EM":
                reconstruction3D[:,i,:]= EM(sinogram, flat, iter=iterations, pad=2, device=GPUs, csino=0)
            elif algorithm == "SIRT":
                reconstruction3D[:,i,:]= SIRT_FST(sinogram, iter=iterations, zpad=2, step=1.0, csino=0, device=GPUs, art_alpha=0.2, reg_mu=0.2, param_alpha=0, supp_reg=0.2, img=None)
    elif algorithm == "EEM": #data é o que sai do wiggle! 
        data = np.swapaxes(data, 0, 1) #tem que trocar eixos 0,1 - por isso o swap.
        nangles = data.shape[1]
        recsize = data.shape[2]
        iterations_list = [iterations,3,8] # [# iterations globais, # iterations EM, # iterations TV total variation], para o EM-TV
        dic = {'gpu': GPUs, 'blocksize':20, 'nangles': nangles, 'niterations': iterations_list,  'regularization': 0.0001,  'epsilon': 1e-15, 'method': 'eEM','angles':angles}
        reconstruction3D = parallel.emfs( data, dic )
    else:
        import sys
        sys.exit('Select a proper reconstruction method')
    print('\t Tomography done!')

    print('Saving tomography logfile...')
    save_json_logfile(global_dict)
    print('\tSaved!')

    return reconstruction3D

def save_json_logfile(jason):
    """Save a copy of the json input file with datetime at the filename

    Args:
        path (string): output folder path 
        jason (dic): jason dictionary
    """    
    import json, os
    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d-%Hh%Mm")
    
    name = jason["folders_list"][0]

    name = dt_string + "_" + name+".json"

    filepath = os.path.join(output_folder,name)
    file = open(filepath,"w")
    file.write(json.dumps(jason,indent=3,sort_keys=True))
    file.close()

def _operator_T(u):
    d   = 1.0
    uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
    uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
    uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
    uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
    delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
    z = np.sqrt( delta )
    return z

def do_chull(sinogram,invert,tolerance,opening_param,erosion_param,chull_param,frame):
    img = sinogram[frame,:,:] 
    where = _operator_T(img).real
    new = np.copy(img)
    if invert:
        new[ new > 0] = _operator_T(new).real[ img > 0]
    else:
        new[ new < 0] = _operator_T(new).real[ img < 0]

    mask = (np.abs( new - img) < tolerance) * 1.0
    mask2 = opening(mask, square(opening_param))
    mask3 = erosion(mask2, square(erosion_param))
    chull = dilation( convex_hull_image(mask3), square(chull_param) ) # EXPAND CASCA DA MASCARA
    img_masked = np.copy(img * chull)  #nova imagem apenas com o suporte
    # sinogram[frame,:,:] = img_masked
    return new,mask,mask2,mask3,chull,img_masked

def apply_chull_parallel(sinogram,invert=True,tolerance=1e-5,opening_param=10,erosion_param=30,chull_param=50):
    if sinogram.ndim == 2:
        sinogram = np.expand_dims(sinogram, axis=0) # add dummy dimension to get 3d array
    chull_sinogram = np.empty_like(sinogram)
    do_chull_partial = partial(do_chull,sinogram,invert,tolerance,opening_param,erosion_param,chull_param)
    frames = [f for f in range(sinogram.shape[0])]
    processes = min(os.cpu_count(),32)
    print(f'Using {processes} parallel processes')
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(tqdm(executor.map(do_chull_partial,frames),total=sinogram.shape[0]))
        for counter, result in enumerate(results):
            new,mask,mask2,mask3,chull,img_masked = result
            chull_sinogram[counter,:,:] = img_masked
    return [new,mask,mask2,mask3,chull,img_masked,chull_sinogram]

def sort_frames_by_angle(ibira_path,foldernames):
    rois = []
    counter = -1 
    for folder in foldernames:

        print(f"Sorting data for {folder} folder")

        filepaths, filenames = list_files_in_folder(os.path.join(ibira_path, folder,'positions'), look_for_extension=".txt")
        # filepaths, filenames = list_files_in_folder(os.path.join(ibira_path, folder), look_for_extension="1_001.txt") # use for old file standard

        print('\t # of files in folder:',len(filenames))

        for filepath in filepaths:
            roisname = filepath  
            if roisname == os.path.join(ibira_path,folder, 'positions', folder + '_Ry_positions.txt'): # ignore this file, to use only the positions file inside /positions/ folder
            # if roisname == os.path.join(ibira_path, folder) + '/Ry_positions.txt': # use for old file standard
                continue
            else:
                counter += 1 
                posfile = open(roisname)
                a = 0
                for line in posfile:
                    line = str(line)
                    if a < 1: # get value from first line of the file only
                        angle = line.split(':')[1].split('\t')[0]
                        rois.append([int(counter),float(angle)])
                    a += 1

    rois = np.asarray(rois)
    rois = rois[rois[:,1].argsort(axis=0)]
    return rois 

def reorder_slices_low_to_high_angle(object, rois):
    object_temporary = np.zeros_like(object)

    for k in range(object.shape[0]): # reorder slices from lowest to highest angle
            # print(f'New index: {k}. Old index: {int(rois[k,0])}')
            object_temporary[k,:,:] = object[int(rois[k,0]),:,:] 

    return object_temporary

def regularization(sino, L):
    a = 1
    R = sino.shape[1]
    V = sino.shape[0]
    th = np.linspace(0, np.pi, V, endpoint=False)
    t  = np.linspace(-a, a, R)
    dt = (2*a)/float((R-1))
    wc = 1.0/(2*dt)
    w = np.linspace(-wc, wc, R)
    if 1: # two options
        h = np.abs(w) / (1 + 4 * np.pi * L * (w**2) )
    else:
        h = 1 / (1 + 4 * np.pi * L * (w**2) )
    G = np.fft.fftshift(np.transpose(np.kron(np.ones((V, 1)), h))).T
    B = np.fft.fft(sino, axis=1)
    D = np.fft.ifft(B * G, axis=1).real
    return D

############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################

def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title=False,clear_axis=True):
    subplot.clear()
    if bottom == None or right == None:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap='gray')
        elif axis == 1:
            subplot.imshow(sinogram[top:bottom,frame_number,left:right],cmap='gray')
        elif axis == 2:
            subplot.imshow(sinogram[top:bottom,left:right,frame_number],cmap='gray')
    else:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap='gray')
        elif axis == 1:
            subplot.imshow(sinogram[top:-bottom,frame_number,left:-right],cmap='gray')
        elif axis == 2:
            subplot.imshow(sinogram[top:-bottom,left:-right,frame_number],cmap='gray')
    if title == True:
        subplot.set_title(f'Frame #{frame_number}')
    if clear_axis == True:
        subplot.set_xticks([])
        subplot.set_yticks([])    
    figure.canvas.draw_idle()

def write_to_file(tomo_script_path,jsonFile_path,output_path="",slurmFile = 'tomoJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
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

python3 {tomo_script_path} {jsonFile_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
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

def run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    slurm_file = write_to_file(tomo_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

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
        self.widget = widgets.Button(description=description,layout=self.button_layout,icon=icon,style=style)

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
        field_style = {'description_width': 'initial'}
   
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
        if isinstance(self.dictionary[self.key],list):
            self.dictionary[self.key] = ast.literal_eval(value)
        elif isinstance(self.dictionary[self.key],dict):
            self.dictionary[self.key] = ast.literal_eval(value)
        else:
            self.dictionary[self.key] = value            

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

############################################ INTERFACE / GUI : TABS ###########################################################################
            
def folders_tab():
    global output_folder, angles_filename, object_filename
    angles_filename = 'dummy2_angles_filename'
    object_filename = 'dummy2_object_filename'

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(np.random.random((4,4)),cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()

    def update_fields(ibira_data_path,folders_list,sinogram_path):
        global output_folder, angles_filename, object_filename
        global_dict["ibira_data_path"] = ibira_data_path
        global_dict["folders_list"]    = folders_list
        global_dict["sinogram_path"]   = sinogram_path
        output_folder = global_dict["sinogram_path"].rsplit('/',1)[0]

        if type(global_dict["folders_list"][0]) == type([1,2]):
            angles_filename = global_dict["folders_list"][0] + '_ordered_angles.npy'
            object_filename = global_dict["folders_list"][0]  + '_ordered_object.npy'
        else:
            angles_filename = ast.literal_eval(global_dict["folders_list"])[0] + '_ordered_angles.npy'
            object_filename = ast.literal_eval(global_dict["folders_list"])[0] + '_ordered_object.npy'        


    def sort_frames(dummy):
        global object

        save_path = sinogram_path.widget.value.rsplit('/',1)[0]
        print(f'Saving sorted frames to: {save_path}')

        global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"])

        complex_object_file  = os.path.join(output_folder, 'object_' + global_dict["folders_list"][0] + '.npy') #hard coded path
        
        print('Loading sinogram: ',complex_object_file)
        object = np.load(complex_object_file)
        print('\t Loaded!')

        rois = sort_frames_by_angle(ibira_data_path.widget.value,global_dict["folders_list"])

        global object_filename, angles_filename
        object_filename = global_dict["folders_list"][0]  + '_ordered_object.npy'
        angles_filename = global_dict["folders_list"][0] + '_ordered_angles.npy'

        object = reorder_slices_low_to_high_angle(object, rois)

        print(f'Extracting sinogram {data_selection.value}...')
        global_dict["contrast_type"] = data_selection.value
        if data_selection.value == 'Magnitude':
            object = np.abs(object)
        elif data_selection.value == "Phase":
            object = np.angle(object)
        print('\t Extraction done!')

        print('Saving angles file: ',os.path.join(save_path,angles_filename))
        np.save(os.path.join(save_path,angles_filename),rois)
        print('Saving ordered sinogram: ', os.path.join(save_path,object_filename))
        np.save(os.path.join(save_path,object_filename), object) 
        print('\tSaved! Sinogram shape: ',object.shape)
        selection_slider.widget.max, selection_slider.widget.value = object.shape[0] - 1, object.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max

        widgets.interactive_output(update_imshow, {'sinogram':fixed(object),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})  


    ibira_data_path = Input(global_dict,"ibira_data_path",layout=items_layout,description='Ibira Datapath (str)')
    folders_list    = Input(global_dict,"folders_list",layout=items_layout,description='Ibira Datafolders (list)')
    sinogram_path   = Input(global_dict,"sinogram_path",layout=items_layout,description='Ptycho sinogram path (str)')
    widgets.interactive_output(update_fields, {'ibira_data_path':ibira_data_path.widget,'folders_list':folders_list.widget,'sinogram_path':sinogram_path.widget})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")

    sort_button = Button(description="Sort frames",layout=buttons_layout, icon="fa-sort-numeric-asc")
    sort_button.trigger(sort_frames)

    controls_box = widgets.Box(children=[sort_button.widget,play_box], layout=get_box_layout('500px',align_items='center'))

    paths_box = widgets.VBox([ibira_data_path.widget,folders_list.widget,sinogram_path.widget])
    box = widgets.HBox([controls_box,vbar,output])
    box = widgets.VBox([paths_box,box])

    return box


def crop_tab():

    initial_image = np.ones((100,100)) # dummy
    vertical_max, horizontal_max = initial_image.shape[0]//2, initial_image.shape[1]//2

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()
    
    def load_frames(dummy, args = ()):
        global sinogram
        top_crop, bottom_crop, left_crop, right_crop, selection_slider, play_control = args
        
        print(output_folder)
        if type(global_dict['folders_list']) == type('a'): # if string
            sinogram_path = os.path.join(output_folder, ast.literal_eval(global_dict['folders_list'])[0] + '_ordered_object.npy')
        else: # if list
            sinogram_path = os.path.join(output_folder, global_dict['folders_list'][0] + '_ordered_object.npy')

        print("Loading sinogram from: ",sinogram_path)
        sinogram = np.load(sinogram_path) 
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
        np.save(os.path.join(output_folder,f'{data_selection.value}_cropped_sinogram.npy'),cropped_sinogram)
        print('\t Saved!')

    top_crop      = Input(global_dict,"top_crop"   ,description="Top",   bounded=(0,vertical_max,1),  slider=True,layout=slider_layout)
    bottom_crop   = Input(global_dict,"bottom_crop",description="Bottom",bounded=(1,vertical_max,1),  slider=True,layout=slider_layout)
    left_crop     = Input(global_dict,"left_crop"  ,description="Left",  bounded=(0,horizontal_max,1),slider=True,layout=slider_layout)
    right_crop    = Input(global_dict,"right_crop" ,description="Right", bounded=(1,horizontal_max,1),slider=True,layout=slider_layout)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
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
        cropped_sinogram = np.load(os.path.join(output_folder,f'{data_selection.value}_cropped_sinogram.npy'))
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

    def save_sinogram(dummy):
        global unwrapped_sinogram
        if np.isnan(unwrapped_sinogram).any() == True:
            print('Removing NaN values from unwrapped sinogram...')
            unwrapped_sinogram = np.where(np.isnan(unwrapped_sinogram),0,unwrapped_sinogram)

        print('Saving unwrapped sinogram...')
        savepath = os.path.join(output_folder,f'{data_selection.value}_unwrapped_sinogram.npy')
        np.save(savepath,unwrapped_sinogram)
        print('\tSaved sinogram at: ',savepath)


    load_cropped_frames_button = Button(description="Load cropped frames",layout=buttons_layout,icon='folder-open-o')

    bad_frames_list  = Input(global_dict,"bad_frames_list", description = 'Bad frames',layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_list1":bad_frames_list.widget})
    
    iterations_slider = Input(global_dict,"unwrap_iterations",bounded=(0,10,1),slider=True, description='Unwrap Iterations',layout=slider_layout)
    non_negativity_checkbox = Input(global_dict,"unwrap_non_negativity",layout=items_layout,description='Non-negativity')
    gradient_checkbox = Input(global_dict,"unwrap_gradient_removal",layout=items_layout,description='Gradient')
    preview_unwrap_button = Button(description="Perform Unwrap",layout=buttons_layout,icon='play')
    preview_unwrap_button.trigger(phase_unwrap)
    
    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector",frame_time_milisec=300)

    args = (selection_slider,play_control)
    load_cropped_frames_button.trigger(partial(load_cropped_frames,args=args))

    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    save_unwrapped_button = Button(description="Save unwrapped frames",layout=buttons_layout,icon='fa-floppy-o') 
    save_unwrapped_button.trigger(save_sinogram)
    
    unwrap_params_box = widgets.Box([iterations_slider.widget,non_negativity_checkbox.widget,gradient_checkbox.widget],layout=get_box_layout('100%'))
    controls_box = widgets.Box([load_cropped_frames_button.widget,correct_bad_frames_button.widget,preview_unwrap_button.widget,save_unwrapped_button.widget,play_box, unwrap_params_box,bad_frames_list.widget],layout=get_box_layout('500px'))
    plot_box = widgets.VBox([output])
        
    box = widgets.HBox([controls_box,vbar,plot_box])
    
    return box


def chull_tab():
    
    def format_chull_plot(figure,subplots):
        subplots[0,0].set_title('Original')
        subplots[0,1].set_title('Threshold')
        subplots[0,2].set_title('Opening')
        subplots[1,0].set_title('Erosion')
        subplots[1,1].set_title('Convex Hull')
        subplots[1,2].set_title('Masked Image')

        for subplot in subplots.reshape(-1):
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
    
    output = widgets.Output()
    
    with output:
        figure, subplots = plt.subplots(2,3)
        subplots[0,0].imshow(np.random.random((4,4)),cmap='gray')
        subplots[0,1].imshow(np.random.random((4,4)),cmap='gray')
        subplots[0,2].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,0].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,1].imshow(np.random.random((4,4)),cmap='gray')
        subplots[1,2].imshow(np.random.random((4,4)),cmap='gray')
        format_chull_plot(figure,subplots)
        plt.show()
    

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots()
        subplot2.imshow(np.random.random((3,3)),cmap='gray')
        figure2.canvas.header_visible = False 
        plt.show()

    def load_unwrapped_sinogram(dummy,args=()):
        global unwrapped_sinogram
        print('Loading unwrapped sinogram: ',os.path.join(output_folder,f'{data_selection.value}_unwrapped_sinogram.npy'))
        unwrapped_sinogram = np.load(os.path.join(output_folder,f'{data_selection.value}_unwrapped_sinogram.npy'))
        print('\t Loaded!')
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = unwrapped_sinogram.shape[0] - 1, unwrapped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure),'subplot':fixed(subplots[0,0]), 'title':fixed(True),'frame_number': selection_slider.widget})    
        format_chull_plot(figure,subplots)

    def preview_cHull(dummy,args=()):
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram[selection_slider.widget.value,:,:],invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        cHull_sinogram = output_list[-1]
        selection_slider.widget.max, selection_slider.widget.value = cHull_sinogram.shape[0] - 1, cHull_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        for subplot, image in zip(subplots.reshape(-1),output_list[0::]):
            image = np.expand_dims(image, axis=0)
            widgets.interactive_output(update_imshow, {'sinogram':fixed(image),'figure':fixed(figure),'subplot':fixed(subplot), 'title':fixed(True),'frame_number': selection_slider.widget})    
        format_chull_plot(figure,subplots)
        print('\tDone with convex hull...')

    def complete_cHull(dummy,args=()):
        print('Applying complete Convex Hull...')
        invert,tolerance,opening_param,erosion_param,chull_param,selection_slider = args
        output_list = apply_chull_parallel(unwrapped_sinogram,invert=invert.widget.value,tolerance=tolerance.widget.value,opening_param=opening_param.widget.value,erosion_param=erosion_param.widget.value,chull_param=chull_param.widget.value)
        cHull_sinogram = output_list[-1]
        print('Saving cHull sinogram...')
        np.save(os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'),cHull_sinogram)
        print('\tSaved!')
    
    def load_chull_sinogram(dummy,args=()):
        global cHull_sinogram
        print('Loading cHull sinogram...')
        cHull_sinogram = np.load(os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'))
        print('\t Loaded!')
        selection_slider, play_control = args
        selection_slider.widget.max, selection_slider.widget.value = cHull_sinogram.shape[0] - 1, cHull_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(cHull_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2), 'title':fixed(True),'frame_number': selection_slider2.widget})    

    def correct_bad_frames(dummy):
        print('Zeroing frames: ', bad_frames2)
        global cHull_sinogram
        cHull_sinogram_corrected = np.load(os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'))
        unwrapped_sinogram[bad_frames2,:,:]   = np.zeros((unwrapped_sinogram.shape[1],unwrapped_sinogram.shape[2]))
        cHull_sinogram_corrected[bad_frames2,:,:]       = np.zeros((unwrapped_sinogram.shape[1],unwrapped_sinogram.shape[2]))
        print('\t Done!')
        widgets.interactive_output(update_imshow, {'sinogram':fixed(cHull_sinogram_corrected),'figure':fixed(figure2),'subplot':fixed(subplot2), 'title':fixed(True),'frame_number': selection_slider2.widget})    
        print('Saving corrected cHull sinogram:',os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'))
        np.save(os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'),cHull_sinogram_corrected)
        print('\tSaved!')

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_list2):
        global bad_frames2
        bad_frames2 = ast.literal_eval(bad_frames_list2)

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_button = Button(description="Load unwrapped sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(partial(load_unwrapped_sinogram,args=(selection_slider,play_control)))

    invert_checkbox  = Input(global_dict,"chull_invert",    description='Invert')
    tolerance        = Input(global_dict,"chull_tolerance", description='Threshold')
    opening_slider   = Input(global_dict,"chull_opening",   description="Opening",     bounded=(1,100,1),slider=True)
    erosion_slider   = Input(global_dict,"chull_erosion",   description="Erosion",     bounded=(1,100,1),slider=True)
    param_slider     = Input(global_dict,"chull_param",     description="Convex Hull", bounded=(1,200,1),slider=True)
    bad_frames_list2 = Input(global_dict,"bad_frames_list2",description='Bad Frames',  layout=items_layout)
    widgets.interactive_output(update_lists,{ "bad_frames_list2":bad_frames_list2.widget})

    preview_button = Button(description="Convex Hull Preview",layout=buttons_layout,icon='play')
    preview_button.trigger(partial(preview_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    correct_bad_frames_button = Button(description='Remove Bad Frames',layout=buttons_layout,icon='fa-check-square-o')
    correct_bad_frames_button.trigger(correct_bad_frames)

    start_button = Button(description="Do complete Convex Hull",layout=buttons_layout,icon='play')
    start_button.trigger(partial(complete_cHull,args=(invert_checkbox,tolerance,opening_slider,erosion_slider,param_slider,selection_slider)))
    
    play_box2, selection_slider2,play_control2 = slide_and_play(label="Frame Selector")

    load_button2 = Button(description="Load cHull sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button2.trigger(partial(load_chull_sinogram,args=(selection_slider2,play_control2)))

    cHull_controls = widgets.Box([load_button2.widget,correct_bad_frames_button.widget,bad_frames_list2.widget,play_box2],layout=box_layout)

    controls0 = widgets.Box([invert_checkbox.widget,tolerance.widget,opening_slider.widget,erosion_slider.widget,param_slider.widget],layout=box_layout)
    controls_box = widgets.Box([load_button.widget,preview_button.widget,start_button.widget,play_box,controls0],layout=get_box_layout('500px'))
    controls_box = widgets.VBox([controls_box,hbar2,cHull_controls])

    box = widgets.HBox([controls_box,vbar, output,vbar,output2])
    
    return box


def wiggle_tab():
    
    def format_wiggle_plot(figure,subplots):
        subplots[0,0].set_title('Pre-wiggle')
        subplots[0,1].set_title('Post_wiggle')
        subplots[0,0].set_ylabel('XY')
        subplots[1,0].set_ylabel('XZ')

        for subplot in subplots.reshape(-1):
            subplot.set_aspect('auto')
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
        figure.tight_layout()
    
    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(3,3))
        subplot.imshow(np.random.random((4,4)),cmap='gray')
        subplot.set_title('Reference Frame')
        figure.canvas.draw_idle()
        figure.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(2,2)
        subplot2[0,0].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[0,1].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[1,0].imshow(np.random.random((4,4)),cmap='gray')
        subplot2[1,1].imshow(np.random.random((4,4)),cmap='gray')
        format_wiggle_plot(figure2,subplot2)
        plt.show()
    
    def load_sinogram(dummy,args=()):
        selection_slider, play_control,sinogram_selection = args
        
        if sinogram_selection.value == "unwrapped":
            file = f'{data_selection.value}_unwrapped_sinogram.npy'
        elif sinogram_selection.value == "convexHull":
            file = f'{data_selection.value}_chull_sinogram.npy'

        global sinogram
        print('Loading sinogram',os.path.join(output_folder,file))
        sinogram = np.load(os.path.join(output_folder,file))
        print('\t Loaded!')
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})    
    
    def update_imshow_with_format(sinogram,figure,subplot,frame_number,axis):
        update_imshow(sinogram,figure,subplot,frame_number,axis=axis)
        format_wiggle_plot(figure2,subplot2)

    global wiggled_sinogram
  
    def start_wiggle(dummy,args=()):

        global sinogram # [7,20,36,65,94,123,152,181,210,239,268,296,324]
        sinogram_selection,sinogram_slider1,sinogram_slider2,cpus_slider,selection_slider = args
        
        global savepath
        savepath = os.path.join(output_folder,f'{data_selection.value}_wiggle_sinogram.npy')


        listOfGlobals = globals()
        print(listOfGlobals['angles_filename'])
        print(listOfGlobals['object_filename'])

        print('Projecting angles to regular mesh...')
        angles  = np.load( os.path.join(output_folder, angles_filename))
        # print('angles:', angles)
        angles = (np.pi/180.) * angles
        sinogram, _, _, projected_angles = angle_mesh_organize(sinogram, angles)
        print(f'Sinogram max = {np.max(sinogram)} \t Sinogram min = {np.min(sinogram)}')

        #TODO: BUG to fix: after projection of angles, the reference frame is on the new projected frames! selection slider should be adjusted before starting wiggle!
        # selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0] - 1, sinogram.shape[0]//2
        # play_control.widget.max =  selection_slider.widget.max

        print(f' Sinogram shape {sinogram.shape} \n Number of Original Angles: {angles.shape} \n Number of Projected Angles: {projected_angles.shape}')
        global_dict['NumberOriginalAngles'] = angles.shape
        global_dict['NumberUsedAngles']     = projected_angles.shape
        projected_angles_filename = angles_filename[:-4]+'_projected.npy'
        np.save(os.path.join(output_folder, projected_angles_filename),projected_angles)

        print("Starting wiggle...")
        global wiggled_sinogram
        wiggled_sinogram = radon.get_wiggle( sinogram,  'vertical', cpus_slider.widget.value, selection_slider.widget.value)
        wiggled_sinogram = radon.get_wiggle( wiggled_sinogram, 'horizontal', cpus_slider.widget.value, selection_slider.widget.value)
        print("\t Wiggle done!")
        
        print("Saving wiggle sinogram to: ", savepath)
        np.save(savepath,wiggled_sinogram)
        print("\t Saved!")

    def load_wiggle(dummy):
        global wiggled_sinogram
        wiggle_datapath = os.path.join(output_folder,f'{data_selection.value}_wiggle_sinogram.npy')
        wiggled_sinogram = np.load(wiggle_datapath)
        print('Loading wiggled frames from:',wiggle_datapath)
        sinogram_slider1.widget.max, sinogram_slider1.widget.value = wiggled_sinogram.shape[1] - 1, wiggled_sinogram.shape[1]//2
        sinogram_slider2.widget.max, sinogram_slider2.widget.value = wiggled_sinogram.shape[2] - 1, wiggled_sinogram.shape[2]//2
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[0,0]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(sinogram),        'figure':fixed(figure2),'subplot':fixed(subplot2[1,0]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[0,1]), 'axis':fixed(1),'frame_number': sinogram_slider1.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(wiggled_sinogram),'figure':fixed(figure2),'subplot':fixed(subplot2[1,1]), 'axis':fixed(2),'frame_number': sinogram_slider2.widget})    
        print('\tLoaded!')

    def save_inverted_sinogram(dummy):
        print('Multiplying sinogram by -1 and saving at:',os.path.join(output_folder,f'{data_selection.value}_wiggle_sinogram.npy'))
        savepath = os.path.join(output_folder,f'{data_selection.value}_wiggle_sinogram.npy')
        global wiggled_sinogram
        wiggled_sinogram = -1*wiggled_sinogram
        np.save(savepath,wiggled_sinogram)
        print(f'\t Saved! New max = {np.max(wiggled_sinogram)}. New min = {np.min(wiggled_sinogram)}')

    play_box, selection_slider,play_control = slide_and_play(label="Reference Frame")

    cpus_slider      = Input(global_dict,"wiggle_cpus", description="# of CPUs", bounded=(1,128,1),slider=True,layout=slider_layout)

    wiggle_button = Button(description='Perform Wiggle',icon='play',layout=buttons_layout)
    load_wiggle_button   = Button(description='Load Wiggle',icon='folder-open-o',layout=buttons_layout)

    sinogram_selection = widgets.RadioButtons(options=['unwrapped', 'convexHull'], value='unwrapped', style=style,layout=items_layout,description='Sinogram to import:',disabled=False)
    sinogram_slider1   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Y", bounded=(1,10,1),slider=True,layout=slider_layout)
    sinogram_slider2   = Input({"dummy_key":1},"dummy_key", description="Sinogram Slice Z", bounded=(1,10,1),slider=True,layout = slider_layout)
    
    load_button = Button(description="Load sinogram",layout=buttons_layout,icon='folder-open-o')
    load_button.trigger(partial(load_sinogram,args=(selection_slider,play_control,sinogram_selection)))
    
    args2 = (sinogram_selection,sinogram_slider1,sinogram_slider2,cpus_slider,selection_slider)
    wiggle_button.trigger(partial(start_wiggle,args=args2))
    load_wiggle_button.trigger(load_wiggle)

    invert_sinogram_buttom = Button(description='Invert Sinogram',icon='undo',layout=buttons_layout)
    invert_sinogram_buttom.trigger(save_inverted_sinogram)


    controls = widgets.VBox([sinogram_selection,load_button.widget,play_box,cpus_slider.widget,wiggle_button.widget,load_wiggle_button.widget,sinogram_slider1.widget,sinogram_slider2.widget,invert_sinogram_buttom.widget])
    output = widgets.Box([output],layout=widgets.Layout(align_content='center'))#,align_items='center',justify_content='center'))
    box = widgets.HBox([controls,vbar,output,vbar,output2])
    
    return box


def tomo_tab():
    
    def format_tomo_plot(figure,subplots):
        # subplots[0].set_title('YZ')
        # subplots[1].set_title('XZ')
        # subplots[2].set_title('XY')

        for subplot in subplots.reshape(-1):
            subplot.set_aspect('equal')
            subplot.set_xticks([])
            subplot.set_yticks([])
        figure.canvas.header_visible = False 
        figure.tight_layout()


    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(1,3)
        subplot[0].imshow(np.random.random((4,4)),cmap='gray')
        subplot[1].imshow(np.random.random((4,4)),cmap='gray')
        subplot[2].imshow(np.random.random((4,4)),cmap='gray')
        format_tomo_plot(figure,subplot)
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure2, axs = plt.subplots(2,2,figsize=(10,5))
        axs[0,0].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[0,1].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[1,0].hist(np.random.random((10,10)).flatten(),bins=100)
        axs[1,1].hist(np.random.random((10,10)).flatten(),bins=100)
        figure2.canvas.header_visible = False 
        figure2.tight_layout()
        plt.show()

    def update_imshow_with_format(sinogram,figure1,subplot1,frame_number,axis):
        update_imshow(sinogram,figure1,subplot1,frame_number,axis=axis,title=True)
        format_tomo_plot(figure,subplot)

    def run_tomo(dummy,args=()):
        iter_slider,gpus_field,filename_field, cpus_field,jobname_field,queue_field, checkboxes = args

        global_dict["processing_steps"] = { "Sort":checkboxes[0].value , "Crop":checkboxes[1].value , "Unwrap":checkboxes[2].value, "ConvexHull":checkboxes[3].value, "Wiggle":checkboxes[4].value, "Tomo":checkboxes[5].value } # select steps when performing full recon

        output_path = global_dict["jupyter_folder"] 
        
        slurm_filepath = os.path.join(output_path,'tomo_job.srm')

        jsonFile_path = os.path.join(output_path,'user_input_tomo.json')

        n_gpus = len(ast.literal_eval(gpus_field.widget.value))

        global machine_selection
        print(f'Running tomo with {machine_selection.value}...')
        if machine_selection.value == 'Local':               
            reconstruction3D = tomography(global_dict["tomo_algorithm"],data_selection.value,angles_filename,global_dict["tomo_iterations"],global_dict["tomo_n_of_gpus"],global_dict["tomo_regularization"],global_dict["tomo_regularization_param"])
            print('\t Done! Please, load the reconstruction with the button...')
            reconstruction3D = reconstruction3D.astype(np.float32)
            print('Saving 3D recon...')
            if type(global_dict["folders_list"]) == type('a'):
                global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list
            savepath = os.path.join(output_folder, f'{data_selection.value}_{global_dict["folders_list"][0]}_reconstruction3D_' +  algo_dropdown.value  + '.npy' )
            np.save(savepath,reconstruction3D)
            imsave(savepath[:-4] + '.tif',reconstruction3D)
            print('\t Saved!')

        elif machine_selection.value == "Cluster": 
            run_job_from_jupyter(mafalda,tomo_script_path,jsonFile_path,output_path=output_path,slurmFile = slurm_filepath,  jobName=jobname_field.widget.value,queue=queue_field.widget.value,gpus=n_gpus,cpus=cpus_field.widget.value)

    def load_recon(dummy):

        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list

        if load_selection.value == "Original":
            savepath = os.path.join(output_folder, f'{data_selection.value}_{global_dict["folders_list"][0]}_reconstruction3D_' +  algo_dropdown.value  + '.npy' )
        elif load_selection.value == "Threshold":
            savepath = os.path.join(output_folder, f'{data_selection.value}_{global_dict["folders_list"][0]}_reconstruction3D_' +  algo_dropdown.value  + '_thresholded.npy' )
        
        print('Loading 3D recon from: ',savepath)
        time.sleep(0.5)
        global reconstruction
        reconstruction = np.load(savepath)
        print('\t Loaded!')
        print(f'Max = {np.max(reconstruction)}, Min = {np.min(reconstruction)}, Mean = {np.mean(reconstruction)}')
        tomo_sliceX.widget.max = reconstruction.shape[0]
        tomo_sliceY.widget.max = reconstruction.shape[1]
        tomo_sliceZ.widget.max = reconstruction.shape[2]
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[0]), 'axis':fixed(0), 'frame_number': tomo_sliceX.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[1]), 'axis':fixed(1), 'frame_number': tomo_sliceY.widget})    
        widgets.interactive_output(update_imshow_with_format, {'sinogram':fixed(reconstruction),'figure1':fixed(figure),'subplot1':fixed(subplot[2]), 'axis':fixed(2), 'frame_number': tomo_sliceZ.widget})    

    def save_thresholded_tomo(dummy):
        print(f'Applying threshold value of {tomo_threshold.widget.value} to reconstruction')
        thresholded_recon = np.where( np.abs(reconstruction) > tomo_threshold.widget.value,0,reconstruction)
        print('\t Done!')
        savepath = os.path.join(output_folder, f'{data_selection.value}_{global_dict["folders_list"][0]}_reconstruction3D_' + algo_dropdown.value + '_thresholded.npy' )
        print('Saving thresholded reconstruction...')
        np.save(savepath,thresholded_recon)
        imsave(savepath[:-4] + '.tif',thresholded_recon)
        print('\tSaved reconstruction at: ',savepath)

    def plot_histograms(dummy):

        n_bins=100
        threshold = tomo_threshold.widget.value
        
        raw_data = np.load(os.path.join(output_folder,f'{data_selection.value}_{global_dict["folders_list"][0]}_reconstruction3D_thresholded.npy'))

        raw_data = raw_data.flatten()
        data = raw_data
        data = np.where(np.abs(data)>threshold,0,data)

        print('Computing statistics...')
        statistics_raw         = (np.max(raw_data),np.min(raw_data),np.mean(raw_data),np.std(raw_data))
        label_raw = f'\n\tMax = {statistics_raw[0]:.2e}\n\t Min = {statistics_raw[1]:.2e}\n\t Mean = {statistics_raw[2]:.2e}\n\t StdDev = {statistics_raw[3]:.2e}'
        statistics_thresholded = (np.max(data),np.min(data),np.mean(data),np.std(data))
        label_thresh = f'\n\tMax = {statistics_thresholded[0]:.2e}\n\t Min = {statistics_thresholded[1]:.2e}\n\t Mean = {statistics_thresholded[2]:.2e}\n\t StdDev = {statistics_thresholded[3]:.2e}'
        print('Raw data statistics: ', label_raw)
        print('Thresholded data statistics: ',label_thresh)


        print('Plotting histograms...')
        with output2:
            for ax in axs.reshape(-1):
                ax.clear()
            try:
                axs[0,0].hist(raw_data,bins=n_bins)
                axs[0,1].hist(raw_data,bins=n_bins)
            except:
                print('Problem found when plotting raw data! Check values!')
            axs[1,0].hist(data,bins=n_bins)
            axs[1,1].hist(data,bins=n_bins)
            axs[0,0].set_title('Raw histogram')
            axs[1,0].set_title('Threshold')
            axs[0,1].set_title('Log(Raw)')
            axs[1,1].set_title('Log(Threshold)')
            axs[0,1].set_yscale('log')
            axs[1,1].set_yscale('log')
        print('\t Done!')

    reg_checkbox    = Input(global_dict,"tomo_regularization",description = "Apply Regularization")
    reg_param       = Input(global_dict,"tomo_regularization_param",description = "Regularization Parameter",layout=items_layout)
    iter_slider     = Input(global_dict,"tomo_iterations",description = "Iterations", bounded=(1,200,2),slider=True,layout=slider_layout)
    cpus_field      = Input(global_dict,"wiggle_cpus",description = "# of CPUs",layout=items_layout)
    gpus_field      = Input(global_dict,"tomo_n_of_gpus",description = "GPUs list",layout=items_layout)
    queue_field     = Input({"dummy_str":'cat-proc'},"dummy_str",description = "Machine Queue",layout=items_layout)
    jobname_field   = Input({"dummy_str":'myTomography'},"dummy_str",description = "Slurm Job Name",layout=items_layout)
    filename_field  = Input({"dummy_str":'reconstruction3Dphase'},"dummy_str",description = "Output Filename",layout=items_layout)
    tomo_threshold  = Input(global_dict,"tomo_threshold",description = "Value threshold for recon",layout=items_layout)
    tomo_sliceX     = Input({"dummy_key":1},"dummy_key", description="Slice X", bounded=(1,10,1),slider=True,layout=slider_layout)
    tomo_sliceY     = Input({"dummy_key":1},"dummy_key", description="Slice Y", bounded=(1,10,1),slider=True,layout=slider_layout)
    tomo_sliceZ     = Input({"dummy_key":1},"dummy_key", description="Slice Z", bounded=(1,10,1),slider=True,layout=slider_layout)
    algo_dropdown   = widgets.Dropdown(options=['EEM','EM', 'ART','FBP'], value='EEM',description='Algorithm:',layout=items_layout)
    load_selection  = widgets.RadioButtons(options=['Original', 'Threshold'], value='Original',style=style, layout=items_layout,description='Load:',disabled=False)
    checkboxes      = [widgets.Checkbox(value=False, description=label,layout=checkbox_layout, style=style) for label in ["Sort", "Crop", "Unwrap", "ConvexHull", "Wiggle", "Tomo"]]
    checkboxes_box  = widgets.VBox(children=checkboxes)

    def update_processing_steps(dictionary,sort_checkbox,crop_checkbox,unwrap_checkbox,chull_checkbox,wiggle_checkbox,tomo_checkbox):
        # "processing_steps": { "Sort":1 , "Crop":1 , "Unwrap":1, "ConvexHull":1, "Wiggle":1, "Tomo":1 } # select steps when performing full recon
        dictionary["processing_steps"]["Sort"]       = sort_checkbox 
        dictionary["processing_steps"]["Crop"]       = crop_checkbox 
        dictionary["processing_steps"]["Unwrap"]     = unwrap_checkbox 
        dictionary["processing_steps"]["ConvexHull"] = chull_checkbox 
        dictionary["processing_steps"]["Wiggle"]     = wiggle_checkbox 
        dictionary["processing_steps"]["Tomo"]       = tomo_checkbox 
    widgets.interactive_output(update_processing_steps,{'dictionary':fixed(global_dict),'sort_checkbox':checkboxes[0],'crop_checkbox':checkboxes[1],'unwrap_checkbox':checkboxes[2],'chull_checkbox':checkboxes[3],'wiggle_checkbox':checkboxes[4],'tomo_checkbox':checkboxes[5]})

    start_tomo = Button(description="Start",layout=buttons_layout,icon='play')
    args = iter_slider,gpus_field,filename_field,cpus_field,jobname_field,queue_field, checkboxes
    start_tomo.trigger(partial(run_tomo,args=args))
    start_tomo_box = widgets.Box([start_tomo.widget],layout=center_all_layout)

    load_recon_button = Button(description="Load recon slices",layout=buttons_layout,icon='play')
    load_recon_button.trigger(load_recon)

    save_thresholded_tomo_button = Button(description="Save thresholded tomo",layout=buttons_layout,icon='play')
    save_thresholded_tomo_button.trigger(save_thresholded_tomo)

    plot_histogram_button = Button(description="Plot Histograms",layout=buttons_layout,icon='play')
    plot_histogram_button.trigger(plot_histograms)

    def save_on_click(dummy):
        print('Saving JSON input file...')
        global_dict["contrast_type"] = data_selection.value
        if type(global_dict["folders_list"]) == type('a'):
            global_dict["folders_list"] = ast.literal_eval(global_dict["folders_list"]) # convert string to literal list
        json_filepath = os.path.join(global_dict["jupyter_folder"],'user_input_tomo.json') #INPUT
        with open(json_filepath, 'w') as file:
            json.dump(global_dict, file)
        print('\t Saved!')

    save_dict_button  = Button(description="Save JSON",layout=buttons_layout,icon='fa-floppy-o')
    save_dict_button.trigger(save_on_click)    
    
    load_box = widgets.HBox([load_recon_button.widget,load_selection])
    start_box = widgets.HBox([checkboxes_box,start_tomo_box])#,layout=widgets.Layout(flex_flow='row',width='100%',border=standard_border))
    threshold_box = widgets.HBox([save_thresholded_tomo_button.widget])#, plot_histogram_button.widget])
    slurm_box = widgets.VBox([cpus_field.widget,gpus_field.widget,queue_field.widget,jobname_field.widget])
    controls = widgets.VBox([algo_dropdown,reg_checkbox.widget,reg_param.widget,iter_slider.widget,slurm_box,hbar2,save_dict_button.widget,start_box,hbar2,load_box,tomo_sliceX.widget,tomo_sliceY.widget,tomo_sliceZ.widget,hbar2,tomo_threshold.widget,threshold_box])
    box = widgets.HBox([controls,vbar2,output])#widgets.VBox([output,hbar,output2])])
    
    return box 


def deploy_tabs(mafalda_session,tab1=folders_tab(),tab2=crop_tab(),tab3=unwrap_tab(),tab4=chull_tab(),tab5=wiggle_tab(),tab6=tomo_tab()):
    
    children_dict = {
    "Select and Sort"       : tab1,
    "Cropping"              : tab2,
    "Phase Unwrap"          : tab3,
    "Convex Hull"           : tab4,
    "Wiggle"                : tab5,
    "Tomography"            : tab6}
    
    def load_json(dummy,dictionary={}):
        template_dict = {"ibira_data_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/data/ptycho2d/",
               "folders_list": ["SS61"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter/00000000/proc/recons/SS61/phase_microagg_P2_01.npy",
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_list": [7,20,36,65,94,123,152,181,210,239,268,296,324],
               "unwrap_iterations": 0,
               "unwrap_non_negativity": False,
               "unwrap_gradient_removal": False,
               "bad_frames_list2": [],
               "chull_invert": False,
               "chull_tolerance": 1e-5,
               "chull_opening": 10,
               "chull_erosion": 10,
               "chull_param": 10,               
               "wiggle_reference_frame": 0,
               "wiggle_cpus": 32,
               "tomo_regularization": True,
               "tomo_regularization_param": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "tomo_n_of_gpus": [0],
               "tomo_threshold" : float(0.0), # max value to be left in reconstructed absorption
               "run_all_tomo_steps":False}
    
        for key in template_dict:
            dictionary[key] = template_dict[key]
    
    global mafalda
    mafalda = mafalda_session

    load_json_button  = Button(description="Load JSON template",layout=buttons_layout,icon='folder-open-o')
    load_json_button.trigger(partial(load_json,dictionary=global_dict))
    
    global machine_selection
    machine_selection = widgets.RadioButtons(options=['Local', 'Cluster'], value='Local', layout={'width': '30%'},description='Machine',disabled=False)

    global data_selection
    data_selection = widgets.RadioButtons(options=['Magnitude', 'Phase'], value='Phase', layout={'width': '30%'},description='Visualize',disabled=False)


    
    def delete_files(dummy):
        sinogram_path = global_dict["sinogram_path"].rsplit('/',1)[0]

        filepaths_to_remove = [os.path.join(output_folder,f'{data_selection.value}_chull_sinogram.npy'),
        os.path.join(output_folder,f'{data_selection.value}_wiggle_sinogram.npy'),
        os.path.join(output_folder,f'{data_selection.value}_unwrapped_sinogram.npy'),
        os.path.join(output_folder,f'{data_selection.value}_cropped_sinogram.npy'),
        os.path.join(output_folder, angles_filename[:-4]+'_projected.npy'),
        os.path.join(sinogram_path,angles_filename),
        os.path.join(sinogram_path,object_filename)]

        for filepath in filepaths_to_remove:
            print('Removing file/folder: ', filepath)
            if os.path.exists(filepath) == True:
                os.remove(filepath)
                print(f'Deleted {filepath}\n')
            else:
                print(f'Directory {filepath} does not exists. Skipping deletion...\n')

        folderpaths_to_remove =[os.path.join(output_folder,'00_frames_original'),
                                os.path.join(output_folder,'01_frames_ordered'),
                                os.path.join(output_folder,'02_frames_cropped'),
                                os.path.join(output_folder,'03_frames_unwrapped'),
                                os.path.join(output_folder,'04_frames_convexHull')]
                                
        import shutil
        for folderpath in folderpaths_to_remove:
            print('Removing file/folder: ', folderpath)
            if os.path.isdir(folderpath) == True:
                shutil.rmtree(folderpath)
                print(f'Deleted {folderpath}\n')
            else:
                print(f'Directory {folderpath} does not exists. Skipping deletion...\n')


    delete_temporary_files_button = Button(description="Delete temporary files",layout=buttons_layout,icon='folder-open-o')
    delete_temporary_files_button.trigger(partial(delete_files))

    box = widgets.HBox([machine_selection,data_selection,load_json_button.widget,delete_temporary_files_button.widget])
    display(box)

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return tab, global_dict  
