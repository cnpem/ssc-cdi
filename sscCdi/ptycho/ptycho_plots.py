# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.colors as colors

import numpy as np

from ..misc import convert_complex_to_RGB


def plot_ptycho_scan_points(positions,pixel_size=None):

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title('Scan points')
    if pixel_size is None:
        ax.set_ylabel('Y [pxls]')
        ax.set_xlabel('X [pxls]')

    else:
        ax.set_ylabel('Y [m]')
        ax.set_xlabel('X [m]')
        
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.grid()
    ax.set_aspect(1)

    
    if pixel_size is None:
        pixel_size = 1
    else:
        pass
    
    positions = positions*pixel_size
    ax.plot(positions[:,1],positions[:,0],'o-',color='gray')
    ax.plot(positions[0,1],positions[0,0],'X',color='green',label='start')
    ax.plot(positions[-1,1],positions[-1,0],'X',color='red',label='end')
    ax.legend(loc='best')
    plt.show()


def plot_ptycho_corrected_scan_points(positions, positions2, pixel_size=None):

    if positions2 is None:
        positions2 = positions

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title('Scan points')
    if pixel_size is None:
        ax.set_ylabel('Y [pxls]')
        ax.set_xlabel('X [pxls]')
    else:
        ax.set_ylabel('Y [m]')
        ax.set_xlabel('X [m]')
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.grid()
    ax.set_aspect(1)

    if pixel_size is None:
        pixel_size = 1
    
    positions = positions * pixel_size
    positions2 = positions2 * pixel_size

    # Plot the points
    ax.plot(positions[:,1], positions[:,0], 'o', color='gray', label='Original')
    ax.plot(positions2[:,1], positions2[:,0], 'o', color='orange', label='Corrected')
    ax.legend(loc='best')

    # Add connectors between the points
    # for (y1, x1), (y2, x2) in zip(positions, positions2):
        # ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)

    plt.show()


def plot_object_spectrum(data,pixel_size=1,cmap='viridis',figsize=(10, 10)):
    
    magnitude_spectrum = np.abs(np.fft.fftshift(np.fft.fftn(data)))
    
    # Create frequency axis
    rows, cols = data.shape
    row_freqs = np.fft.fftfreq(rows,d=pixel_size)
    col_freqs = np.fft.fftfreq(cols,d=pixel_size)
    row_freqs_shifted = np.fft.fftshift(row_freqs)
    col_freqs_shifted = np.fft.fftshift(col_freqs)

    # Plot the magnitude spectrum
    plt.figure(figsize=figsize)
    plt.imshow(magnitude_spectrum, extent=(col_freqs_shifted[0], col_freqs_shifted[-1], row_freqs_shifted[0], row_freqs_shifted[-1]), cmap=cmap,norm=LogNorm())
    plt.xlabel('Spatial Frequency (1/m)')
    plt.ylabel('Spatial Frequency (1/m)')
    plt.title('2D Fourier Transform - Magnitude Spectrum')
    plt.colorbar(label='Log Magnitude')
    plt.show()

def plot_probe_modes(probe,extent=None):

    from matplotlib.colors import hsv_to_rgb

    if len(probe.shape) == 2:
        probe = np.expand_dims(probe, axis=0)
        print(probe.shape)
    
    N, Y, X = probe.shape  # N modes
    
    fig = plt.figure(figsize=(20+N, 7+N))
    gs = plt.GridSpec(1, N + 1, width_ratios=[1] + [9] * N)

    # Plot the color map on the left
    ax_cbar = fig.add_subplot(gs[0, 0])
    
    # Create a custom colorbar
    # Generate a 2D array where the first dimension is phase and the second dimension is amplitude
    phase = np.linspace(0, 1, 256)  # Normalized phase
    amplitude = np.linspace(0, 1, 256)  # Normalized amplitude
    phase_grid, amplitude_grid = np.meshgrid(phase, amplitude)
    color_map = hsv_to_rgb(np.dstack((phase_grid, np.ones_like(phase_grid), amplitude_grid)))

    # Plot the color map as an image
    ax_cbar.imshow(color_map, aspect='auto', origin='lower')
    ax_cbar.set_xticks([0, 255])
    ax_cbar.set_xticklabels(['-π', 'π'])
    ax_cbar.set_yticks([0, 255])
    ax_cbar.set_yticklabels(['0', 'Max'])
    ax_cbar.set_xlabel('Phase')
    ax_cbar.set_ylabel('Amplitude')
    ax_cbar.set_title('')

    ax_cbar.set_aspect((X / Y) * 9)

    powers = calculate_mode_powers(probe)
    
    for i in range(N):
        rgb_probe = convert_complex_to_RGB(probe[i], bias=0.01)
        
        ax_main = fig.add_subplot(gs[0, i + 1])
        
        im = ax_main.imshow(rgb_probe,extent=extent)
        if extent is None:
            ax_main.set_ylabel('Y [pxls]')
            ax_main.set_xlabel('X [pxls]')
        else:
            ax_main.set_ylabel('Y [m]')
            ax_main.set_xlabel('X [m]')
        ax_main.set_title(f'Mode {i+1}. Power = {powers[i]*100:.2f}%')

    plt.tight_layout()
    plt.show()  

def plot_probe_support(support,extent=None,cmap='gray'):
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(support,extent=extent,cmap=cmap)
    if extent is None:
        ax.set_ylabel('Y [pxls]')
        ax.set_xlabel('X [pxls]')
    else:
        ax.set_ylabel('Y [m]')
        ax.set_xlabel('X [m]')
    ax.set_title('Probe Support')
    plt.show()  


def get_plot_extent_from_positions(positions):
    y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
    x_min, x_max = positions[:, 1].min(), positions[:, 1].max()
    return [x_min,x_max,y_min,y_max]

def get_extent_from_pixel_size(array_shape,pixel_size):
    sy, sx = array_shape
    
    x_min = (-sx//2)*pixel_size
    x_max = (sx//2)*pixel_size
    y_min = (-sy//2)*pixel_size
    y_max = (sy//2)*pixel_size    
    return [x_min,x_max,y_min,y_max] 

def plot_iteration_error(errors):
    fig, ax1 = plt.subplots(figsize=(13, 6))

    # Create a second y-axis (ax2)
    ax2 = ax1.twinx()

    # Create a third y-axis (ax3), by offsetting ax2
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward by 60 points

    # Normalize and plot errors on the respective axes
    lines = []
    labels = []

    for i in range(errors.shape[1]):
        error = errors[:, i]
        if i == 0:
            line, = ax1.plot(error, '.-', label='R-factor', color='tab:blue')
            lines.append(line)
            labels.append('R-factor')
        elif i == 1:
            line, = ax2.plot(error, '.-', label='NMSE', color='tab:orange')
            lines.append(line)
            labels.append('NMSE')
        elif i == 2:
            line, = ax3.plot(error, '.-', label='LLK', color='tab:green')
            lines.append(line)
            labels.append('LLK')

    # Labels for the axes
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('R-factor')
    ax1.grid()

    ax2.set_ylabel('NMSE')
    ax3.set_ylabel('LLK')

    # Place the legends below the plot
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=3)

    # Adjust the layout to avoid overlap
    fig.tight_layout()
    plt.show()

def object_slice_visualizer(data, positions=None, axis=0, title='', cmap1='viridis', cmap2='viridis', aspect_ratio='', norm="normalize", vmin=None, vmax=None, extent=None):
    """
    data (ndarray): complex valued data
    positions (ndarray): 2D array of pixel values with shape (N, 2), where the first column is Y and the second column is X
    axis (int): slice direction
    extent (tuple): extent of the images in the format (xmin, xmax, ymin, ymax)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm

    import ipywidgets as widgets
    from ipywidgets import fixed

    def get_vol_slice(volume, axis, frame):
        selection = [slice(None)] * 3
        selection[axis] = frame
        frame_data = volume[tuple(selection)]
        return frame_data

    def get_colornorm(frame, vmin, vmax, norm):
        if norm is None:
            return None
        elif norm == "normalize":
            if vmin is not None or vmax is not None:
                return colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                return colors.Normalize(vmin=frame.min(), vmax=frame.max())
        elif norm == "LogNorm":
            return colors.LogNorm()
        else:
            raise ValueError("Invalid norm value: {}".format(norm))

    def draw_rectangle(ax, x_min, x_max, y_min, y_max):
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if positions is not None:
        y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
        x_min, x_max = positions[:, 1].min(), positions[:, 1].max()

        if extent is not None:
            ymin_extent, ymax_extent = extent[2], extent[3]
            xmin_extent, xmax_extent = extent[0], extent[1]
            pixel_size_y = (ymax_extent - ymin_extent) / data.shape[1]
            pixel_size_x = (xmax_extent - xmin_extent) / data.shape[2]

            y_min = ymin_extent + y_min * pixel_size_y
            y_max = ymin_extent + y_max * pixel_size_y
            x_min = xmin_extent + x_min * pixel_size_x
            x_max = xmin_extent + x_max * pixel_size_x

    output = widgets.Output()
    with output:
        volume_slice_amplitude = np.abs(get_vol_slice(data, axis, 0))
        volume_slice_phase = np.angle(get_vol_slice(data, axis, 0))

        figure, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(15, 7))

        im1 = ax1.imshow(volume_slice_amplitude, cmap=cmap1, norm=get_colornorm(volume_slice_amplitude, vmin, vmax, norm), extent=extent, origin='lower')
        ax1.set_title('Amplitude')
        if extent is None:
            ax1.set_ylabel('Y [pxls]')
            ax1.set_xlabel('X [pxls]')
        else:
            ax1.set_ylabel('Y [m]')
            ax1.set_xlabel('X [m]')
        cbar1 = figure.colorbar(im1, ax=ax1, format='%.2e')
        if positions is not None:
            draw_rectangle(ax1, x_min, x_max, y_min, y_max)

        im2 = ax2.imshow(volume_slice_phase, cmap=cmap2, norm=get_colornorm(volume_slice_phase, vmin, vmax, norm), extent=extent, origin='lower')
        ax2.set_title('Phase')
        if extent is None:
            ax2.set_ylabel('Y [pxls]')
            ax2.set_xlabel('X [pxls]')
        else:
            ax2.set_ylabel('Y [m]')
            ax2.set_xlabel('X [m]')
        cbar2 = figure.colorbar(im2, ax=ax2, format='%.2e')
        if positions is not None:
            draw_rectangle(ax2, x_min, x_max, y_min, y_max)

        figure.canvas.draw_idle()
        plt.show()

    def update_imshow(frame_number, axis=0, cmap1='viridis', cmap2='hsv', aspect_ratio='auto', norm=None, extent=None):
        nonlocal im1, im2, cbar1, cbar2

        ax1.clear()
        ax2.clear()

        volume_slice_amplitude = np.abs(get_vol_slice(data, axis, frame_number))
        volume_slice_phase = np.angle(get_vol_slice(data, axis, frame_number))

        im1 = ax1.imshow(volume_slice_amplitude, cmap=cmap1, norm=get_colornorm(volume_slice_amplitude, vmin, vmax, norm), extent=extent, origin='lower')
        ax1.set_title('Amplitude')
        if extent is None:
            ax1.set_ylabel('Y [pxls]')
            ax1.set_xlabel('X [pxls]')
        else:
            ax1.set_ylabel('Y [m]')
            ax1.set_xlabel('X [m]')
        if positions is not None:
            draw_rectangle(ax1, x_min, x_max, y_min, y_max)

        im2 = ax2.imshow(volume_slice_phase, cmap=cmap2, norm=get_colornorm(volume_slice_phase, vmin, vmax, norm), extent=extent, origin='lower')
        ax2.set_title('Phase')
        if extent is None:
            ax2.set_ylabel('Y [pxls]')
            ax2.set_xlabel('X [pxls]')
        else:
            ax2.set_ylabel('Y [m]')
            ax2.set_xlabel('X [m]')
        if positions is not None:
            draw_rectangle(ax2, x_min, x_max, y_min, y_max)

        # Update the colorbars
        cbar1.update_normal(im1)
        cbar2.update_normal(im2)

        figure.canvas.draw_idle()

        if aspect_ratio != '':
            ax1.set_aspect(aspect_ratio)
            ax2.set_aspect(aspect_ratio)

    slider_layout = widgets.Layout(width='50%')
    selection_slider = widgets.IntSlider(min=0, max=data.shape[axis] - 1, step=1, description="Slice", value=data.shape[axis] // 2, layout=slider_layout)

    interactive_output = widgets.interactive_output(update_imshow, {
        'frame_number': selection_slider,
        'axis': fixed(axis),
        'cmap1': fixed(cmap1),
        'cmap2': fixed(cmap2),
        'aspect_ratio': fixed(aspect_ratio),
        'norm': fixed(norm),
        'extent': fixed(extent)
    })

    box = widgets.VBox([selection_slider, output])
    return box



def plot_amplitude_and_phase(data, positions=None, title='', cmap1='viridis', cmap2='viridis', aspect_ratio='', norm="normalize", vmin=None, vmax=None, extent=None):
    """
    data (ndarray): 2D complex valued data
    positions (ndarray): 2D array of pixel values with shape (N, 2), where the first column is Y and the second column is X
    extent (tuple): extent of the images in the format (xmin, xmax, ymin, ymax)
    """

    def get_colornorm(frame, vmin, vmax, norm):
        if norm is None:
            return None
        elif norm == "normalize":
            if vmin is not None or vmax is not None:
                return colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                return colors.Normalize(vmin=frame.min(), vmax=frame.max())
        elif norm == "LogNorm":
            return colors.LogNorm()
        else:
            raise ValueError("Invalid norm value: {}".format(norm))

    def draw_rectangle(ax, x_min, x_max, y_min, y_max):
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if positions is not None:
        y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
        x_min, x_max = positions[:, 1].min(), positions[:, 1].max()

        if extent is not None:
            ymin_extent, ymax_extent = extent[2], extent[3]
            xmin_extent, xmax_extent = extent[0], extent[1]
            pixel_size_y = (ymax_extent - ymin_extent) / data.shape[0]
            pixel_size_x = (xmax_extent - xmin_extent) / data.shape[1]

            y_min = ymin_extent + y_min * pixel_size_y
            y_max = ymin_extent + y_max * pixel_size_y
            x_min = xmin_extent + x_min * pixel_size_x
            x_max = xmin_extent + x_max * pixel_size_x

    amplitude = np.abs(data)
    phase = np.angle(data)

    figure, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(15, 7))

    im1 = ax1.imshow(amplitude, cmap=cmap1, norm=get_colornorm(amplitude, vmin, vmax, norm), extent=extent, origin='lower')
    ax1.set_title('Amplitude')
    if extent is None:
        ax1.set_ylabel('Y [pxls]')
        ax1.set_xlabel('X [pxls]')
    else:
        ax1.set_ylabel('Y [m]')
        ax1.set_xlabel('X [m]')
    cbar1 = figure.colorbar(im1, ax=ax1, format='%.2e')
    if positions is not None:
        draw_rectangle(ax1, x_min, x_max, y_min, y_max)

    im2 = ax2.imshow(phase, cmap=cmap2, norm=get_colornorm(phase, vmin, vmax, norm), extent=extent, origin='lower')
    ax2.set_title('Phase')
    if extent is None:
        ax2.set_ylabel('Y [pxls]')
        ax2.set_xlabel('X [pxls]')
    else:
        ax2.set_ylabel('Y [m]')
        ax2.set_xlabel('X [m]')
    cbar2 = figure.colorbar(im2, ax=ax2, format='%.2e')
    if positions is not None:
        draw_rectangle(ax2, x_min, x_max, y_min, y_max)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def calculate_mode_powers(modes):
    """
    Calculate the power of each mode in an incoherent sum.

    Parameters:
    modes (ndarray): 3D complex-valued array with shape (N, Y, X) where N is the number of modes.

    Returns:
    ndarray: 1D array of power values for each mode.
    """
    powers = np.sum(np.abs(modes) ** 2, axis=(1, 2))
    # Normalize the power values
    total_power = np.sum(powers)
    normalized_powers = powers / total_power
    return normalized_powers

def plot_probe_modes_multiple(probe, extent=None):
    from matplotlib.colors import hsv_to_rgb

    if len(probe.shape) == 2:
        probe = np.expand_dims(probe, axis=0)
    
    N, Y, X = probe.shape  # N modes
    
    fig = plt.figure(figsize=(20+N, 7+N))
    gs = plt.GridSpec(1, N + 1, width_ratios=[1] + [9] * N)

    # Plot the color map on the left
    ax_cbar = fig.add_subplot(gs[0, 0])
    
    # Create a custom colorbar
    # Generate a 2D array where the first dimension is phase and the second dimension is amplitude
    phase = np.linspace(0, 1, 256)  # Normalized phase
    amplitude = np.linspace(0, 1, 256)  # Normalized amplitude
    phase_grid, amplitude_grid = np.meshgrid(phase, amplitude)
    color_map = hsv_to_rgb(np.dstack((phase_grid, np.ones_like(phase_grid), amplitude_grid)))

    # Plot the color map as an image
    ax_cbar.imshow(color_map, aspect='auto', origin='lower')
    ax_cbar.set_xticks([0, 255])
    ax_cbar.set_xticklabels(['-π', 'π'])
    ax_cbar.set_yticks([0, 255])
    ax_cbar.set_yticklabels(['0', 'Max'])
    ax_cbar.set_xlabel('Phase')
    ax_cbar.set_ylabel('Amplitude')
    ax_cbar.set_title('')

    ax_cbar.set_aspect((X / Y) * 9)

    powers = calculate_mode_powers(probe)
    
    for i in range(N):
        rgb_probe = convert_complex_to_RGB(probe[i], bias=0.01)
        
        ax_main = fig.add_subplot(gs[0, i + 1])
        
        im = ax_main.imshow(rgb_probe, extent=extent)
        if extent is None:
            ax_main.set_ylabel('Y [pxls]')
            ax_main.set_xlabel('X [pxls]')
        else:
            ax_main.set_ylabel('Y [m]')
            ax_main.set_xlabel('X [m]')
        ax_main.set_title(f'Mode {i+1}. Power = {powers[i]*100:.2f}%')

    plt.tight_layout()
    plt.show()

def plot_probe_modes_interactive(probes, extent=None):
    """
    Display an interactive plot to visualize different slices of multiple probes.

    Parameters:
    probes (ndarray): 4D complex-valued array with shape (M, N, Y, X) where M is the number of probes,
                      N is the number of modes, Y and X are the dimensions of each mode.
    extent (tuple): Extent of the plot for x and y axes. Default is None.
    """
    num_probes = probes.shape[0]
    from ipywidgets import interact, IntSlider, Play, jslink, VBox
    from IPython.display import display

    def update_plot(probe_index):
        plot_probe_modes_multiple(probes[probe_index], extent)
    
    slider = IntSlider(min=0, max=num_probes-1, step=1, description='Probe Index')
    play = Play(value=0, min=0, max=num_probes-1, step=1, interval=500)
    jslink((play, 'value'), (slider, 'value'))
    
    display(VBox([play]))
    interact(update_plot, probe_index=slider)


import matplotlib.patches as patches

def draw_rectangle(ax, x_min, x_max, y_min, y_max):
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def plot_objects_multiple(obj, extent=None, positions=None):
    if len(obj.shape) == 2:
        obj = np.expand_dims(obj, axis=0)
    
    N, Y, X = obj.shape  # N modes
    
    fig = plt.figure(figsize=(20+N, 7+N))
    gs = plt.GridSpec(1, 2 * N, width_ratios=[9] * 2 * N)

    for i in range(N):
        abs_obj = np.abs(obj[i])
        angle_obj = np.angle(obj[i])
        
        ax_abs = fig.add_subplot(gs[0, 2 * i])
        im_abs = ax_abs.imshow(abs_obj, cmap='viridis', extent=extent)
        fig.colorbar(im_abs, ax=ax_abs, orientation='vertical')
        if extent is None:
            ax_abs.set_ylabel('Y [pxls]')
            ax_abs.set_xlabel('X [pxls]')
        else:
            ax_abs.set_ylabel('Y [m]')
            ax_abs.set_xlabel('X [m]')
        ax_abs.set_title(f'Mode {i+1} Magnitude')
        
        if positions is not None:
            y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
            x_min, x_max = positions[:, 1].min(), positions[:, 1].max()

            if extent is not None:
                ymin_extent, ymax_extent = extent[2], extent[3]
                xmin_extent, xmax_extent = extent[0], extent[1]
                pixel_size_y = (ymax_extent - ymin_extent) / abs_obj.shape[0]
                pixel_size_x = (xmax_extent - xmin_extent) / abs_obj.shape[1]

                y_min = ymin_extent + y_min * pixel_size_y
                y_max = ymin_extent + y_max * pixel_size_y
                x_min = xmin_extent + x_min * pixel_size_x
                x_max = xmin_extent + x_max * pixel_size_x

            draw_rectangle(ax_abs, x_min, x_max, y_min, y_max)

        ax_angle = fig.add_subplot(gs[0, 2 * i + 1])
        im_angle = ax_angle.imshow(angle_obj, cmap='viridis', extent=extent)
        fig.colorbar(im_angle, ax=ax_angle, orientation='vertical')
        if extent is None:
            ax_angle.set_ylabel('Y [pxls]')
            ax_angle.set_xlabel('X [pxls]')
        else:
            ax_angle.set_ylabel('Y [m]')
            ax_angle.set_xlabel('X [m]')
        ax_angle.set_title(f'Mode {i+1} Phase')

        if positions is not None:
            draw_rectangle(ax_angle, x_min, x_max, y_min, y_max)

    plt.tight_layout()
    plt.show()

def plot_objects_interactive(objects, extent=None, positions=None):
    """
    Display an interactive plot to visualize different slices of multiple objects.

    Parameters:
    objects (ndarray): 4D complex-valued array with shape (M, N, Y, X) where M is the number of objects,
                      N is the number of modes, Y and X are the dimensions of each mode.
    extent (tuple): Extent of the plot for x and y axes. Default is None.
    positions (ndarray): 2D array with shape (P, 2) containing the positions to draw the rectangle. Default is None.
    """
    num_objects = objects.shape[0]
    from ipywidgets import interact, IntSlider, Play, jslink, VBox, HBox
    from IPython.display import display

    def update_plot(obj_index):
        plot_objects_multiple(objects[obj_index], extent, positions)
    
    slider = IntSlider(min=0, max=num_objects-1, step=1, description='Probe Index')
    play = Play(value=0, min=0, max=num_objects-1, step=1, interval=500)
    jslink((play, 'value'), (slider, 'value'))
    
    display(HBox([play]))
    interact(update_plot, obj_index=slider)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import ipywidgets as widgets
from IPython.display import display

def fresnel_propagate(complex_beam, wavelength, z, dx):
    k = 2 * np.pi / wavelength  # Wavenumber
    N, M = complex_beam.shape  # Size of the input field
    Lx = N * dx  # Physical size of the field in the x direction
    Ly = M * dx  # Physical size of the field in the y direction
    
    # Frequency coordinates
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # Fourier transform of the initial field
    complex_beam_ft = np.fft.fft2(complex_beam)
    
    # Fresnel transfer function
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # Multiply the transfer function with the Fourier transformed field
    Uz_ft = complex_beam_ft * H
    
    # Inverse Fourier transform to get the propagated field
    Uz = np.fft.ifft2(Uz_ft)
    
    return Uz

def propagate_and_slice(complex_beam, wavelength, dz, dz_range, dx, slice_idx=None, direction='vertical'):
    z_steps = int(dz_range / dz)  # Number of steps in each direction
    slice_z = np.linspace(-dz_range, dz_range, 2 * z_steps + 1)
    slice_I = np.zeros((complex_beam.shape[0], 2 * z_steps + 1))

    Uz_slices = []
    for i, z in enumerate(slice_z):
        print(f'Propagating beam to slice at z={z:.5f}', end='\r')
        Uz = fresnel_propagate(complex_beam, wavelength, z, dx)
        Uz_slices.append(Uz)
        
        if direction == 'horizontal':
            if slice_idx is None:
                slice_idx = complex_beam.shape[0] // 2
            slice_I[:, i] = np.abs(Uz[slice_idx, :])**2  # Take the central slice in x
        elif direction == 'vertical':
            if slice_idx is None:
                slice_idx = complex_beam.shape[1] // 2
            slice_I[:, i] = np.abs(Uz[:, slice_idx])**2  # Take the central slice in y
        else:
            raise ValueError('Select correct slice direction')
                
    return slice_z, slice_I, Uz_slices

def plot_caustic(complex_beam,wavelength, dz, dz_range, dx,direction='vertical'):
    def gaussian(z, a, z0, w):
        """Gaussian function for fitting."""
        return a * np.exp(-2 * (z - z0)**2 / w**2)


    N = complex_beam.shape[0]
    x = np.linspace(-N//2, N//2, N+1)*dx*1e6
    y = np.linspace(-N//2, N//2, N+1)*dx*1e6
    X, Y = np.meshgrid(x, y)

    slice_z, slice_I, Uz_slices = propagate_and_slice(complex_beam, wavelength, dz, dz_range, dx, direction=direction)
    slice_z_mm = slice_z * 1e3  # Convert z to millimeters

    fig = plt.figure(figsize=(25, 6))
    gs = gridspec.GridSpec(1, 3)

    # Plot the original transversal slice
    ax1 = plt.subplot(gs[0])
    cax1 = ax1.imshow(np.abs(complex_beam)**2, extent=[x.min(), x.max(), y.min(), y.max()], cmap='cividis')
    ax1.set_title('Initial Beam Intensity')
    ax1.set_xlabel('x (µm)')
    ax1.set_ylabel('y (µm)')
    fig.colorbar(cax1, ax=ax1, orientation='vertical', label='Intensity')

    from matplotlib.colors import LogNorm
    # Plot the longitudinal slice
    ax2 = plt.subplot(gs[1])
    cax2 = ax2.imshow(slice_I, extent=[slice_z_mm.min(), slice_z_mm.max(), x.min(), x.max()], aspect='auto', cmap='cividis')
    ax2.set_title('Longitudinal Beam Intensity')
    ax2.set_xlabel('Propagation direction z (mm)')
    if direction == 'vertical':
        ax2.set_ylabel('Transverse direction y (µm)')
    elif direction == 'horizontal':
        ax2.set_ylabel('Transverse direction x (µm)')
    fig.colorbar(cax2, ax=ax2, orientation='vertical', label='Intensity')
    line = ax2.axvline(x=0, color='r', linestyle='--')

    # Create a third plot
    ax3 = plt.subplot(gs[2])
    ax3.set_title('XY Slice at Selected z')
    ax3.set_xlabel('x (µm)')
    ax3.set_ylabel('y (µm)')
    cax3 = ax3.imshow(np.abs(complex_beam)**2, extent=[x.min(), x.max(), y.min(), y.max()], cmap='cividis')
    fig.colorbar(cax3, ax=ax3, orientation='vertical', label='Intensity')

    # Slider widget for selecting the z slice
    slider_z = widgets.FloatSlider(
        value=0,
        min=slice_z_mm.min(),
        max=slice_z_mm.max(),
        step=dz * 1e3,
        description='z (mm)',
        continuous_update=False
    )

    # Range slider for selecting vmax and vmin
    range_slider = widgets.FloatRangeSlider(
        value=[slice_I.min(), slice_I.max()],
        min=slice_I.min(),
        max=slice_I.max(),
        step=(slice_I.max() - slice_I.min()) / 100,
        description='MinMax',
        continuous_update=False
    )

    def update_plot(z, intensity_range):
        idx = (np.abs(slice_z_mm - z)).argmin()
        Uz = Uz_slices[idx]
        
        # Update the third plot
        cax3.set_data(np.abs(Uz)**2)
        ax3.set_title(f'XY Slice at z = {z:.3f} mm')
        
        # Update the vertical line in the second plot
        line.set_xdata(z)
        
        # Update the color limits for the second plot
        cax2.set_clim(*intensity_range)
        
        fig.canvas.draw_idle()

    widgets.interactive(update_plot, z=slider_z, intensity_range=range_slider)
    display(slider_z, range_slider)
    plt.show()