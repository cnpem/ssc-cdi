(dictionary)=
# Dictionary

## General dictionary parameters

```python
input_dict = {
    "hdf5_output": "path/to/output/file.hdf5", 
    "CPUs": 32,      
    "GPUs": [0], 
    "energy": energy, 
    "detector_distance": dist, 
    'distance_sample_focus': 0,
    "detector_pixel_size": pixeldet, 
    "binning": binning,    
    "object_padding": obj_pad,
    "incoherent_modes": modes, 
    "initial_obj": obj_init,   
    "initial_probe": probe_init,
    "probe_support": probe_supp,
    'total_flux': None,                    
    "positions_unit": None,
    'regime': 'fraunhoffer',
    'position_rotation': 0,
    'n_of_positions_to_remove': 0,  
    'clip_object_magnitude': False,
    'free_log_likelihood': 0,
    'fourier_power_bound': 0,
    'save_restored_data': False,
    "algorithms": {"1": engine}
}
```

| Input Dictionary Parameter | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `hdf5_output` | `None` | (`str`) Path to `hdf5` output file | `None` does not save output; `<path/to/output/file.hdf5>` |
| `CPUs` | 32 | (`int`) Number of cpus to use in parallel execution | {math}`1,2,\cdots,N` |
| `GPUs` | `[0]` | (`list of ints`) List of numbers (e.g. [0,1,2]) containg the number of GPUs | {math}`[0,1,2,\cdots,N]` |
| `energy` | 1.0 | (`float`) Energy in keV | {math}`> 0.0 \, keV` |
| `detector_distance` | 1.0 | (`float`) Sample to detector distance in meters | {math}`> 0.0 \, m` |
| `distance_sample_focus` | 0.0 | (`float`) Focus to sample distance in meters | {math}`> 0.0 \, m` |
| `detector_pixel_size` | {math}`55e-6` | (`float`) Detector pixel in meters | {math}`> 0.0 \, m` |
| `binning` | 1 | (`int`) Downscale the data: how many times to bin data; Must be even number; | {math}`N` even; if {math}`1`, no binning occurs; |
| `object_padding` | 0 | (`int`) Number of pixels to padd the object matrix | {math}`1,2,\cdots,N` |
| `incoherent_modes` | 1 | (`int`) Number of incoherent probe modes | {math}`1,2,\cdots,N` |
| `initial_obj` | `None` | (`dict`) Dictionary for setting inital object | See options here |
| `initial_probe` | `None` | (`dict`) Dictionary for setting inital probe | See options here |
| `probe_support` | `None` | (`dict`) Dictionary for setting probe support | See options here |
| `total_flux` | `None` | (`float/str`) Total photon flux | `None`, `initial_probe`, float value |
| `positions_unit` | `None` | (`str`) Positions unit  | `meters`/`m`; `milimeters`/`mm`; `microns`/`um`; `pixels`;  |
| `regime` | `fraunhoffer` | (`str`) Propagation regime  | `fraunhoffer`; `fresnel` |
| `position_rotation` | 0 | (`int`)  | --- |
| `n_of_positions_to_remove` | 0 | (`int`) Number of positions to remove at random | {math}`1,2,\cdots,N` |
| `clip_object_magnitude` | `False` | (`bool`)  | `True` or `False` |
| `free_log_likelihood` | 0 | (`int`)  | --- |
| `fourier_power_bound` | 0 | (`int`)  | --- |
| `save_restored_data` | `False` | (`bool`) Save output data with restored diffraction patterns | `True` or `False` |
| `algorithms` | `None` | (`dict`) Dictionary for setting engines | See options here |

### Initial Object dictionary

`input_dict['initial_obj'] = {"obj": "random"}`


| Initial Object Options | Description | Entry |
| :--- | :--- | :--- |
| `random` | Generate a random matrix with values between 0 and 1 | {"obj": "random"} |
| `constant` | Generate a constant matrix of 1s | {"obj": "constant"} |

### Initial Probe dictionary


### Probe Support dictionary


### Algorithms dictionary

The algorithms dictionary `input_dict['algorithms']` is a collection of ptychography engines that are applied sequentially on the measured data as: `input_dict['algorithms'] = {"1": engine_1, "2": engine_2}`.
The engines `engine_1`, `engine_2` are dictionaries that contains the parameters for each engine as below.

```python
engine_1 = {
    'name': 'AP',                 # Engine name;
    'batch': 0,                   # Number of data arrays to fit into the GPU. If 0 it runs automatically;
    'iterations':50,              # Number of iterations;
    'beta': 0.9,                  # float; 
    'step_object': 1.0,           # float (min: 0, max: 1); 
    'step_probe': 1.0,            # float (min: 0, max: 1); 
    'regularization_object': 0.0, # float (min: 0, max: 1); 
    'regularization_probe': 0.0,  # float (min: 0, max: 1); 
    'momentum_obj': 0.0,          # float; 
    'momentum_probe': 0.0,        # float; 
    'position_correction': 0,     # 0: no correction; N: performs correction every N iterations;
}
```


| Engine Dictionary Parameter |  Default Value | Description | Options |
| :--- | :--- | :--- | :--- |
| `name` | `AP` | (`str`) engine descriptor | `PIE`, `AP`, `RAAR`, `ML_python` |
| `batch` | 0 | (`int`) Number of data arrays to fit into the GPU. If 0 it runs automatically. | {math}`0,1,2,\cdots,N` |
| `iterations` | 50 | (`int`) Number of iterations | {math}`1,2,\cdots,N` |
| `beta` | {math}`0.0` | (`float`) RAAR {math}`\beta` parameter | {math}`0 \leq \cdot \leq 1` | 
| `step_object` | {math}`1.0` | (`float`) Object gradient descent step | {math}`\geq 0` |
| `step_probe` | {math}`1.0` | (`float`) Probe gradient descent step | {math}`\geq 0` |
| `regularization_object` | {math}`0.0` | (`float`) Object update preconditioner parameter | {math}`0 \leq \cdot \leq 1` |
| `regularization_probe` | {math}`0.0` | (`float`) Probe update preconditioner parameter | {math}`0 \leq \cdot \leq 1` |
| `momentum_obj`| {math}`0.0` | (`float`) Object momentum parameter; for `AP`, `RAAR` | {math}`\geq 0` |
| `momentum_probe` | {math}`0.0` | (`float`) Probe momentum parameter; for `AP`, `RAAR` | {math}`\geq 0` |
| `position_correction` | 0 (no correction) | (`int`) Performs position correction every N iterations. | {math}`0, 1,2,\cdots,N` |
| `probe_cycles` | {math}`1` | (`int`) Probe cycles update; for `RAAR` | {math}`1,2,\cdots,N` |  