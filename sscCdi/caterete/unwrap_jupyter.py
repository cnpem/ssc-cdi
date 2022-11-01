{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named '__main__.unwrap'; '__main__' is not a package",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-c3065cb3f415>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mskimage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mio\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mimsave\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0munwrap\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mphase_unwrap\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0munwrapInterface\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecon_folder\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mrecon_filename\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mframe_number\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named '__main__.unwrap'; '__main__' is not a package"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy\n",
    "\n",
    "from IPython.display import display\n",
    "from ipywidgets import *\n",
    "from skimage.io import imsave\n",
    "\n",
    "from .unwrap import phase_unwrap\n",
    "\n",
    "def unwrapInterface(recon_folder,recon_filename,frame_number):\n",
    "\n",
    "    path_to_recon = os.path.join(recon_folder,recon_filename)\n",
    "    image = numpy.load(path_to_recon)[frame_number]\n",
    "    \n",
    "    vsize, hsize = image.shape[0], image.shape[1]\n",
    "\n",
    "    fig = plt.figure(figsize=(10,5))\n",
    "    ax1  = fig.add_subplot(1, 2, 1)\n",
    "    ax1.imshow(image)\n",
    "    ax2  = fig.add_subplot(1, 2, 2)\n",
    "    ax2.imshow(image)\n",
    "    ax1.set_title('Original image')\n",
    "    ax2.set_title('Unwrapped image')\n",
    "\n",
    "    def update(top, bottom,left,right):\n",
    "        ax1.clear()\n",
    "        ax1.set_title('Original image')\n",
    "        ax1.imshow(image[top:-bottom,left:-right])\n",
    "        fig.canvas.draw_idle()\n",
    "        return top, bottom,left,right\n",
    "\n",
    "    def on_button_clicked(b):\n",
    "        global unwrapped_image \n",
    "        unwrapped_image = phase_unwrap(image[top.value:-bottom.value,left.value:-right.value],iterations.value,non_negativity=non_negativity_checkbox,remove_gradient = remove_gradient_checkbox)\n",
    "        ax2.imshow(unwrapped_image)\n",
    "        return unwrapped_image\n",
    "\n",
    "    def savefig_button(b):\n",
    "        fig.savefig('figure.png',dpi=300)\n",
    "        numpy.save('unwrapped.npy',unwrapped_image)\n",
    "        imsave('unwrapped.tif',unwrapped_image)\n",
    "\n",
    "\n",
    "    # Sliders     \n",
    "    top    = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Top')\n",
    "    bottom = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Bottom')\n",
    "    left   = widgets.IntSlider(min=1, max=hsize//2, step=1, value = 1,description='Left')\n",
    "    right  = widgets.IntSlider(min=1, max=vsize//2, step=1, value = 1,description='Right')\n",
    "    ui = widgets.HBox([top, bottom, left, right])    \n",
    "    out = widgets.interactive_output(update, {'top': top, 'bottom': bottom, 'left': left, 'right': right})\n",
    "\n",
    "    # Button 1\n",
    "    button = widgets.Button(description=\"Unwrap!\")\n",
    "    output = widgets.Output()\n",
    "    button.on_click(on_button_clicked)\n",
    "\n",
    "    # Button 2\n",
    "    save_button = widgets.Button(description=\"Save figure\")\n",
    "    output2 = widgets.Output()\n",
    "    save_button.on_click(savefig_button)\n",
    "\n",
    "    # CheckBox 1 \n",
    "    non_negativity_checkbox = widgets.Checkbox(value=False,description='Non-negativity')\n",
    "    # CheckBox 2\n",
    "    remove_gradient_checkbox= widgets.Checkbox(value=False,description='Remove Gradient')\n",
    "\n",
    "    # Input 1\n",
    "    iterations = widgets.BoundedIntText(value=0,  min=0,step=1, description='Iterations:', disabled=False)\n",
    "\n",
    "    # DISPLAY\n",
    "    display(ui, out)\n",
    "    display(iterations)\n",
    "    display(non_negativity_checkbox)\n",
    "    display(remove_gradient_checkbox)\n",
    "    display(button)\n",
    "    display(save_button)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
