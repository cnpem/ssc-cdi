import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    ax.plot(positions[:,0],positions[:,1],'o-',color='gray')
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
    ax.plot(positions[:,0], positions[:,1], 'o', color='gray', label='Original')
    ax.plot(positions2[:,0], positions2[:,1], 'o', color='orange', label='Corrected')
    ax.legend(loc='best')

    # Add connectors between the points
    for (x1, y1), (x2, y2) in zip(positions, positions2):
        ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)

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
    
    fig = plt.figure(figsize=(15, 8))
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
        ax_main.set_title(f'Mode {i+1}')

    plt.tight_layout()
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

def plot_iteration_error(error):

    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(error,'.-',color='black')
    ax.grid()
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')


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
