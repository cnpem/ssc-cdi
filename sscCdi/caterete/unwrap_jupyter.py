import matplotlib.pyplot as plt
import numpy

from IPython.display import display
from ipywidgets import *
from skimage.io import imsave

from .unwrap import phase_unwrap

def unwrapInterface(recon_folder,recon_filename,frame_number):

    path_to_recon = os.path.join(recon_folder,recon_filename)
    image = numpy.load(path_to_recon)[frame_number]
    
    vsize, hsize = image.shape[0], image.shape[1]

    fig = plt.figure(figsize=(10,5))
    ax1  = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax2  = fig.add_subplot(1, 2, 2)
    ax2.imshow(image)
    ax1.set_title('Original image')
    ax2.set_title('Unwrapped image')

    def update(top, bottom,left,right):
        ax1.clear()
        ax1.set_title('Original image')
        ax1.imshow(image[top:-bottom,left:-right])
        fig.canvas.draw_idle()
        return top, bottom,left,right

    def on_button_clicked(b):
        global unwrapped_image 
        unwrapped_image = phase_unwrap(image[top.value:-bottom.value,left.value:-right.value],iterations.value,non_negativity=non_negativity_checkbox,remove_gradient = remove_gradient_checkbox)
        ax2.imshow(unwrapped_image)
        return unwrapped_image

    def savefig_button(b):
        fig.savefig('figure.png',dpi=300)
        numpy.save('unwrapped.npy',unwrapped_image)
        imsave('unwrapped.tif',unwrapped_image)


    # Sliders     
    top    = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Top')
    bottom = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Bottom')
    left   = widgets.IntSlider(min=1, max=hsize//2, step=1, value = 1,description='Left')
    right  = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Right')
    ui = widgets.HBox([top, bottom, left, right])    
    out = widgets.interactive_output(update, {'top': top, 'bottom': bottom, 'left': left, 'right': right})

    # Button 1
    button = widgets.Button(description="Unwrap!")
    output = widgets.Output()
    button.on_click(on_button_clicked)

    # Button 2
    save_button = widgets.Button(description="Save figure")
    output2 = widgets.Output()
    save_button.on_click(savefig_button)

    # CheckBox 1 
    non_negativity_checkbox = widgets.Checkbox(value=False,description='Non-negativity')
    # CheckBox 2
    remove_gradient_checkbox= widgets.Checkbox(value=False,description='Remove Gradient')

    # Input 1
    iterations = widgets.BoundedIntText(value=0,  min=0,step=1, description='Iterations:', disabled=False)

    # DISPLAY
    display(ui, out)
    display(iterations)
    display(non_negativity_checkbox)
    display(remove_gradient_checkbox)
    display(button)
    display(save_button)