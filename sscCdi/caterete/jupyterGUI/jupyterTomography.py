import ipywidgets as widgets
from ipywidgets import fixed
import ast 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os

from sscCdi import unwrap_in_parallel

sinogram = np.random.random((2,2,2))

global_dict = {"ibira_data_path": "path/to/ibira/difpads",
               "folders_list": ["folder1","folder2"],
               "sinogram_path": "/ibira/lnls/beamlines/caterete/apps/jupyter-dev/00000000/proc/recons/SS61/phase_microagg_P2_01.npy",
               "top_crop": 0,
               "bottom_crop":0,
               "left_crop":0,
               "right_crop":0,
               "bad_frames_list": [],
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
               "wiggle_regularization": 0.001, # arbitrary value
               "tomo_iterations": 25,
               "tomo_algorithm": "EEM", # "ART", "EM", "EEM", "FBP", "RegBackprojection"
               "tomo_n_of_gpus": [0,1,2,3],
               "threshold_abs" : 0, # max value to be left in reconstructed absorption
               "threshold_phase" : 0, # max value to be left in reconstructed absorption
}

output_folder = global_dict["sinogram_path"].rsplit('/',1)[0]  #os.path.join(global_dict["ibira_data_path"], 'proc','recons',global_dict["folders_list"][0]) # changes with control

class VideoControl:
    
    def __init__ (self,slider,value,minimum,maximum,step,interval,description):
    
        self.widget = widgets.Play(value=value,
                            min=minimum,
                            max=maximum,
                            step=step,
                            interval=interval,
                            description=description,
                            disabled=False )

        widgets.jslink((self.widget, 'value'), (slider, 'value'))

class Button:

    def __init__(self,description="DESCRIPTION",width="50%",height="50px",icon=""):

        self.button_layout = widgets.Layout(width=width, height=height)
        self.widget = widgets.Button(description=description,layout=self.button_layout,icon=icon)

    def trigger(self,func):
        self.widget.on_click(func)

class Input(object):

    def __init__(self,dictionary,key,description="",layout=None,bounded=(),slider=False):
        
        self.dictionary = dictionary
        self.key = key
        
        if layout == None:
            field_layout = widgets.Layout(align_items='flex-start',width='50%')
        else:
            field_layout = layout
        field_style = {'description_width': 'initial'}
        

        if description == "":
            field_description = f'{key}{str(type(self.dictionary[self.key]))}'
        else:
            field_description = description

        if isinstance(self.dictionary[self.key],bool):
            self.widget = widgets.Checkbox(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],int):
            if bounded == ():
                self.widget = widgets.IntText( description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
            else:
                if slider:
                    self.widget = widgets.IntSlider(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key])
                else:
                    self.widget = widgets.BoundedIntText(min=bounded[0],max=bounded[1],step=bounded[2], description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],float):
            if bounded == ():
                self.widget = widgets.FloatText(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
            else:
                self.widget = widgets.BoundedFloatText(min=bounded[0],max=bounded[1],step=bounded[2],description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],list):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],str):
            self.widget = widgets.Text(description=field_description,value=self.dictionary[self.key],layout=field_layout, style=field_style)
        elif isinstance(self.dictionary[self.key],dict):
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=field_layout, style=field_style)
        
        widgets.interactive_output(self.update_dict_value,{'value':self.widget})

    def update_dict_value(self,value):
        if isinstance(self.dictionary[self.key],list):
            self.dictionary[self.key] = ast.literal_eval(value)
        elif isinstance(self.dictionary[self.key],dict):
            self.dictionary[self.key] = ast.literal_eval(value)
        else:
            self.dictionary[self.key] = value            

import asyncio

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
            
def folders_tab():

    output = widgets.Output()
    
    def sort_frames(dummy):
        with output:
            print('Still need to implement sorting function!')
    
    ibira_data_path = Input(global_dict,"ibira_data_path")
    folders_list    = Input(global_dict,"folders_list")
    sinogram_path   = Input(global_dict,"sinogram_path")
    
    sort_button = Button(description="Sort frames",icon="fa-sort-numeric-asc")
    sort_button.trigger(sort_frames)
    
    box = widgets.VBox([ibira_data_path.widget,folders_list.widget,sinogram_path.widget,sort_button.widget,output])
    return box


def crop_tab():

    # make sure dimension is (F,N,M) always! (1,N,M) for single frame!
    
    initial_image = np.ones((100,100)) # dummt
    vertical_max, horizontal_max = initial_image.shape[0]//2, initial_image.shape[1]//2

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots()
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()
    
    def load_frames(dummy, args = ()):
        global sinogram
        top_crop, bottom_crop, left_crop, right_crop, select_slider, play_control = args
        sinogram = np.load(global_dict["sinogram_path"])
        select_slider.widget.max, select_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = select_slider.widget.max
        vertical_max, horizontal_max = sinogram.shape[1]//2, sinogram.shape[2]//2
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
        plot = widgets.interactive_output(update_imshow, {'sinogram':fixed(sinogram),'figure':fixed(figure),'subplot':fixed(subplot),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': select_slider.widget})

    def save_cropped_sinogram(dummy,args=()):
        top,bottom,left,right = args
        cropped_sinogram = sinogram[:,top.value:-bottom.value,left.value:-right.value]
        np.save(os.path.join(output_folder,'cropped_sinogram.npy'),cropped_sinogram)

            
    top_crop      = Input(global_dict,"top_crop"   ,description="Top",   bounded=(0,vertical_max,1),  slider=True)
    bottom_crop   = Input(global_dict,"bottom_crop",description="Bottom",bounded=(1,vertical_max,1),  slider=True)
    left_crop     = Input(global_dict,"left_crop"  ,description="Left",  bounded=(0,horizontal_max,1),slider=True)
    right_crop    = Input(global_dict,"right_crop" ,description="Right", bounded=(1,horizontal_max,1),slider=True)
    select_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,100,1),slider=True)

    play_control = VideoControl(select_slider.widget,select_slider.widget.value,select_slider.widget.min,select_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([select_slider.widget,play_control.widget])
    
    load_frames_button  = Button(description="Load Frames",width='50%', height='50px',icon='fa-file-o')
    args = (top_crop, bottom_crop, left_crop, right_crop, select_slider, play_control)
    load_frames_button.trigger(partial(load_frames,args=args))

    save_cropped_frames_button = Button(description="Save cropped frames",width='70%', height='50px',icon='fa-floppy-o') 
    args2 = (top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget)
    save_cropped_frames_button.trigger(partial(save_cropped_sinogram,args=args2))
    
    sliders_box = widgets.VBox([load_frames_button.widget,play_box,top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget,save_cropped_frames_button.widget])
    box = widgets.HBox([sliders_box,output])
    return box

def update_imshow(sinogram,figure,subplot,top, bottom,left,right,frame_number):
    subplot.clear()
    if bottom == None or right == None:
        subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap='gray')
    else:
        subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap='gray')
    figure.canvas.draw_idle()

def show_selected_slice(figure,subplot,sinogram,frame_number):
    subplot.clear()
    subplot.imshow(sinogram[frame_number,:,:],cmap='gray')
    figure.canvas.draw_idle()

def update_image(image):
    subplot.clear()
    subplot.imshow(image,cmap='gray')
    figure.canvas.draw_idle()


    
def unwrap_tab():
    
    global unwrapped_sinogram
    unwrapped_sinogram = np.empty_like(sinogram)
    
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
        with output: print('Performing phase unwrap...')
        unwrapped_sinogram = unwrap_in_parallel(cropped_sinogram,iterations_slider.widget.value,non_negativity=non_negativity_checkbox.widget.value,remove_gradient = gradient_checkbox.widget.value)
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_sinogram),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap[1]),'top': fixed(0), 'bottom': fixed(None), 'left': fixed(0), 'right': fixed(None), 'frame_number': selection_slider.widget})    
        
    def load_cropped_frames(dummy,args=()):
        global cropped_sinogram
        selection_slider, play_control = args
        cropped_sinogram = np.load(os.path.join(output_folder,'cropped_sinogram.npy'))
        selection_slider.widget.max, selection_slider.widget.value = cropped_sinogram.shape[0] - 1, cropped_sinogram.shape[0]//2
        play_control.widget.max =  selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(cropped_sinogram),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap[0]),'top': fixed(0), 'bottom': fixed(None), 'left': fixed(0), 'right': fixed(None), 'frame_number': selection_slider.widget})    

    def correct_bad_frames(dummy,bad_frames1=[],bad_frames2=[]):
        bad_frames = bad_frames1 + bad_frames2 # concatenate lists
        with output: print('Zeroing frames: ', bad_frames)
        cropped_sinogram[bad_frames,:,:]   = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        unwrapped_sinogram[bad_frames,:,:] = np.zeros((cropped_sinogram.shape[1],cropped_sinogram.shape[2]))
        save_unwrapped_button.trigger(partial(save_sinogram,sinogram_to_save=unwrapped_sinogram,filename='unwrapped_sinogram.npy'))

    @debounce(0.5) # check changes every 0.5sec
    def update_lists(bad_frames_list1,bad_frames_list2):
        bad_frames_list1 = ast.literal_eval(bad_frames_list1)
        bad_frames_list2 = ast.literal_eval(bad_frames_list2)
        correct_bad_frames_partial = partial(correct_bad_frames,bad_frames1=bad_frames_list1,bad_frames2=bad_frames_list2)
        correct_bad_frames_button.trigger(correct_bad_frames_partial)
   
    def save_sinogram(dummy,sinogram_to_save=np.ones((2,2,2)),filename="dummy.npy"):
        np.save(os.path.join(output_folder,filename),sinogram_to_save)
        with output: print('Saved sinogram at: ',os.path.join(output_folder,filename))

    load_cropped_frames_button = Button(description="Load cropped frames",width='50%', height='50px',icon='fa-file-o')

    bad_frames_list = Input(global_dict,"bad_frames_list", description = 'Bad frames',layout=widgets.Layout(align_items='flex-start',width='80%'))
    bad_frames_list2 = Input(global_dict,"bad_frames_list2",description='Bad Frames after Unwrap',layout=widgets.Layout(align_items='flex-start',width='80%'))
    widgets.interactive_output(update_lists,{ "bad_frames_list1":bad_frames_list.widget,"bad_frames_list2":bad_frames_list2.widget})
    
    iterations_slider = Input(global_dict,"unwrap_iterations",bounded=(0,10,1),slider=True, description='Iterations')
    non_negativity_checkbox = Input(global_dict,"unwrap_non_negativity",layout=widgets.Layout(align_items='flex-start',width='40%'),description='Non-negativity')
    gradient_checkbox = Input(global_dict,"unwrap_gradient_removal",layout=widgets.Layout(align_items='flex-start',width='40%'),description='Gradient')
    preview_unwrap_button = Button(description="Preview unwrap",width='50%', height='50px',icon='play')
    preview_unwrap_button.trigger(phase_unwrap)
    
    selection_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,10,1),slider=True)
    play_control = VideoControl(selection_slider.widget,selection_slider.widget.value,selection_slider.widget.min,selection_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([selection_slider.widget,play_control.widget])
    
    args = (selection_slider,play_control)
    load_cropped_frames_button.trigger(partial(load_cropped_frames,args=args))

    correct_bad_frames_button = Button(description='Correct Bad Frames',icon='fa-check-square-o')
    
    save_unwrapped_button = Button(description="Save unwrapped frames",icon='fa-floppy-o') 
    
    unwrap_params_box = widgets.VBox([iterations_slider.widget,non_negativity_checkbox.widget,gradient_checkbox.widget])
    controls_box = widgets.VBox([load_cropped_frames_button.widget,correct_bad_frames_button.widget,preview_unwrap_button.widget,save_unwrapped_button.widget,play_box, unwrap_params_box,bad_frames_list.widget,bad_frames_list2.widget])
    plot_box = widgets.VBox([output])
        
    box = widgets.HBox([controls_box,plot_box])
    
    return box
    











from concurrent.futures import ProcessPoolExecutor
from functools import partial
from skimage.morphology import square, erosion, opening, convex_hull_image, dilation
def _operator_T(u):
    d   = 1.0
    uxx = (np.roll(u,1,1) - 2 * u + np.roll(u,-1,1) ) / (d**2)
    uyy = (np.roll(u,1,0) - 2 * u + np.roll(u,-1,0) ) / (d**2)
    uyx = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,1,1),-1,0) - np.roll(np.roll(u,1,0),-1,1) + np.roll(np.roll(u,-1,1),-1,0)  )/ (2 * d**2) 
    uxy = (np.roll(np.roll(u,1,1),1,1) - np.roll(np.roll(u,-1,1),1,0) - np.roll(np.roll(u,-1,0),1,1) + np.roll(np.roll(u,-1,1),-1,0)   )/ (2 * d**2)
    delta = (uxx + uyy)**2 - 4 * (uxx * uyy - uyx * uxy)
    z = np.sqrt( delta )
    return z

def do_chull(sinogram,frame):
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
    return img_masked

def apply_chull_parallel(sinogram,invert=True,tolerance=1e-5,opening_param=10,erosion_param=30,chull_param=50):
    print('Perfoming convex hull...')
    chull_sinogram = np.empty_like(sinogram)
    do_chull_partial = partial(do_chull,sinogram)
    frames = [f for f in range(sinogram.shape[0])]
    with ProcessPoolExecutor() as executor:
        results = executor.map(do_chull_partial,frames)
        for counter, result in enumerate(results):
            chull_sinogram[counter,:,:] = result
    return chull_sinogram
    
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
    plt.show()
    
def chull_tab():
    
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
        
    load_button = Button(description="Load unwrapped sinogram",icon='fa-file-o')
    preview_button = Button(description="Convex Hull Preview",icon='play')
    start_button = Button(description="Do complete Convex Hull",icon='play')
    save_button = Button(description="Save CHull sinogram",icon='fa-floppy-o')

    selection_slider = Input({"dummy_key":1},"dummy_key",description="Select Frame", bounded=(0,10,1),slider=True)
    play_control = VideoControl(selection_slider.widget,selection_slider.widget.value,selection_slider.widget.min,selection_slider.widget.max,1,300,"Play Button")
    play_box = widgets.HBox([selection_slider.widget,play_control.widget])
    
    
    invert_checkbox = Input(global_dict,"chull_invert",description='Invert')
    tolerance = Input(global_dict,"chull_tolerance",description='Threshold')
    opening_slider = Input(global_dict,"chull_opening",description="Opening", bounded=(0,20,1),slider=True)
    erosion_slider = Input(global_dict,"chull_erosion",description="Erosion", bounded=(0,20,1),slider=True)
    param_slider   = Input(global_dict,"chull_param",description="Convex Hull", bounded=(0,20,1),slider=True)

    controls0 = widgets.VBox([invert_checkbox.widget,tolerance.widget,opening_slider.widget,erosion_slider.widget,param_slider.widget])
    controls_box = widgets.VBox([load_button.widget,preview_button.widget,start_button.widget,save_button.widget,play_box,controls0])
    
    box = widgets.HBox([controls_box,output])
    
    return box













def deploy_tabs(tab1=folders_tab(),tab2=crop_tab(),tab3=unwrap_tab(),tab4=chull_tab()):
    
    children_dict = {
    "Select Folders" : tab1,
    "Cropping"       : tab2,
    "Phase Unwrap"   : tab3,
    "Convex Hull"    : tab4,
    "Wiggle" : widgets.Text(description="Wiggle"),
    "Tomography"     : widgets.Text(description="Tomography")}
    
    button_layout = widgets.Layout(width='30%', height='100px',max_height='50px')
    load_json_button  = Button(description="Load JSON template",width='50%', height='50px',icon='fa-file-o')
    run_ptycho_button = Button(description="Run Ptycho",width='50%', height='50px',icon='play')
    save_dict_button  = Button(description="Save Dictionary",width='50%', height='50px',icon='fa-floppy-o')
    box = widgets.HBox([load_json_button.widget,run_ptycho_button.widget,save_dict_button.widget])
    display(box)
    
    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return tab, global_dict  