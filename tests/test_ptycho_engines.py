import pytest
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import ndimage

from sscCdi.ptycho.ptychography import call_ptychography

error_threshold = 1e-6 # Error threshold for the tests

model_obj = np.load('object.npy')
model_probe = np.load('probe.npy')
data = np.load('diff_patterns.npy')
positions = np.load('positions.npy')
positions = np.roll(positions, 1, axis=1)

output_folder = "./" # Paths to output

path_npy_obj = 'object.npy'
path_npy_probe = 'probe.npy'

path_hdf5_obj = 'object.h5'
path_hdf5_probe = 'probe.h5'

def convert_npy_to_hdf5(npy_path, hdf5_path):
    data = np.load(npy_path)
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('data', data=data)

@pytest.fixture(scope="session", autouse=True)
def manage_temporary_files():

    convert_npy_to_hdf5(path_npy_obj, path_hdf5_obj)
    convert_npy_to_hdf5(path_npy_probe, path_hdf5_probe)

    yield

    os.remove(path_hdf5_obj)
    os.remove(path_hdf5_probe)

raar_default_params = {
    'name': 'RAAR',  # Relaxed Averaged Alternating Reflections
    'batch': 8,  # Number of data arrays for GPU; reduce if memory is an issue
    'iterations': 5,
    'beta': 0.5,
    'step_object': 0.5,
    'step_probe': 0.9,
    'regularization_object': 0.1,
    'regularization_probe': 0.1,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0,  # 0: no correction, N: correction every N iterations
}

pie_default_params = {
    'name': 'PIE',  # Ptychographic Iterative Engine
    'iterations': 5,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.5,
    'regularization_probe': 0.5,
    'momentum_obj': 0.0,  # > 0 for mPIE with the given friction value
    'momentum_probe': 0.0,  # > 0 for mPIE with the given friction value
    'position_correction': 0,  # 0: no correction, N: correction every N iterations
}

ap_default_params = {
    'name': 'AP',  # Alternating Projections
    'batch': 8,  # Number of data arrays for GPU; reduce if memory is an issue
    'iterations': 5,
    'step_object': 1.0,
    'step_probe': 1.0,
    'regularization_object': 0.01,
    'regularization_probe': 0.01,
    'momentum_obj': 0.0,
    'momentum_probe': 0.0,
    'position_correction': 0,  # 0: no correction, N: correction every N iterations
}

input_dict = {
    'algorithms': {  # Sub-dictionary for algorithms
        '1': { **raar_default_params },
        '2': { **pie_default_params },
        '3': { **ap_default_params }
    },
    'CPUs': 32,  # Number of CPUs for parallel execution
    'GPUs': [0],  # List of GPU indices
    'hdf5_output': output_folder + 'test_output.h5',  # Output HDF5 file path
    'regime': 'fraunhoffer',  # Propagation regime: 'fraunhoffer' or 'fresnel'
    'binning': 1,  # Binning factor (1 for no binning)
    'n_of_positions_to_remove': 1,  # Randomly remove this many scan positions
    'energy': 10,  # Energy in keV
    'detector_distance': 15,  # Detector distance in meters
    'detector_pixel_size': 55e-6,  # Detector pixel size in meters
    'object_padding': 10,  # Pixels to pad object array
    'incoherent_modes': 1,  # Number of incoherent modes
    'fourier_power_bound': 0,  # Relaxed Fourier magnitude constraint
    'clip_object_magnitude':
    False,  # Clip object magnitude between 0 and 1 if True
    'free_log_likelihood': 0,  # Points for error computation
    'position_rotation': 0,  # Points for error computation
    'fresnel_regime': 0,  # Points for error computation
    'distance_sample_focus': 0,  # Distance from sample to focus in meters
    'probe_support': {
        "type": "circular",  # Type of probe support
        "radius": 100,
        "center_y": 0,
        "center_x": 0
    },
    'initial_obj': { "obj": 'random' },  # Initial object guess
    'initial_probe': { "probe": 'inverse' },  # Initial probe guess
    'save_restored_data': False
}

def shift_ctr_of_mass_to_img_ctr(image):
    image = np.abs(image)
    height, width = image.shape
    com = ndimage.center_of_mass(image)
    shift_y = int(height / 2 - com[0])
    shift_x = int(width / 2 - com[1])
    shifted_image = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
    return shifted_image, shift_y, shift_x

def mean_squared_error(data, datashift):
    return np.abs(np.real(np.sum((data - datashift)**2)))/np.prod(data.shape)

def compare_model_to_recon(obj, probe, model_obj, model_probe, N=50):
    shifted_probe, shift_y, shift_x = shift_ctr_of_mass_to_img_ctr(probe[0])
    shifted_obj = np.roll(obj, shift=(shift_y, shift_x), axis=(0, 1))

    shifted_obj = shifted_obj[N:-N,N:-N]
    model_obj = model_obj[N:-N,N:-N]

    mean_squared_error_obj = mean_squared_error(model_obj, shifted_obj)
    mean_squared_error_probe = mean_squared_error(model_probe, shifted_probe)

    return mean_squared_error_obj, mean_squared_error_probe


def save_results_as_pngs(obj, probe, prefix, output_folder):
    probe = np.squeeze(probe)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    img_plots = [('Object Magnitude', np.abs(obj)),
                 ('Object Phase', np.angle(obj)),
                 ('Probe Magnitude', np.abs(probe)),
                 ('Probe Phase', np.angle(probe))]
    for i, (title, img) in enumerate(img_plots):
        img_normalized = (255 * (img - img.min()) / (img.ptp())).astype(np.uint8)
        axes[i].imshow(img_normalized, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{prefix}_results.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Results saved as {output_path}")

@pytest.mark.parametrize('algo_params', [
    raar_default_params,
    pie_default_params,
    ap_default_params
])
def test_quality(algo_params):
    test_input_dict = {
        **input_dict,
        'algorithms': { '1': { **algo_params, 'iterations': 200 } }
    }
    obj, probe, _, _, _ = call_ptychography(test_input_dict, data, positions)
    save_results_as_pngs(obj, probe, algo_params['name'], output_folder)
    error_obj, _ = compare_model_to_recon(obj, probe, model_obj, model_probe, N=50)
    assert error_obj < error_threshold, "Bad recon for {} engine.".format(algo_params['name'])

# the "complete run" tests only check if engines can run without crash

@pytest.mark.parametrize("probe_support", [
    { "type": "circular",  "radius": 100,  "center_y": 0, "center_x": 0 },
    { "type": "cross",  "center_width": 300,  "cross_width": 0, "border_padding": 0 },
    { "type": "array",  "data": np.ones_like(model_probe, dtype=np.float32) }
])
def test_probe_support_complete_run(probe_support):
    try:
        call_ptychography({**input_dict, 'probe_support': probe_support}, data, positions)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize("initial_obj", [
    { "obj": "random" },
    { "obj": np.ones_like(model_obj) },
    { "obj": path_npy_obj },
    { "obj": path_hdf5_obj,'h5_tree_path':'data' },
    { "obj": 'constant' } #object with 1s
])
def test_initial_object_complete_run(initial_obj):
    try:
        call_ptychography({ **input_dict , 'initial_obj': initial_obj }, data, positions)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

@pytest.mark.parametrize("initial_probe", [
    { "probe": 'inverse' },  # IFT of the average data
    { "probe": np.ones_like(model_probe) },
    { "probe": path_npy_probe },
    { "probe": 'random' },
    { "probe": 'circular', "radius": 100, "distance": 0 },
    { "probe": 'constant' }, # constant matrix of 1s
    {"probe": path_hdf5_probe,"h5_tree_path": 'data' }, # path to .hdf5 of previous recon containing the reconstructed object in 'recon/object'
    { "probe": 'fzp', 'beam_type': 'disc', # or 'gaussian',
     'distance_sample_fzpf': 2.9e-3, # in meters,
     'fzp_diameter': 50e-6, # in meters,
     'fzp_outer_zone_width': 50e-9, # in meters,
     'beamstopper_diameter': 20e-6, # in meters (0 if no beamstopper used),
     'probe_diameter': 50e-6,
     'probe_normalize': False }
])
def test_initial_probe_complete_run(initial_probe):
    try:
        call_ptychography({ **input_dict , 'initial_probe': initial_probe }, data, positions)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
