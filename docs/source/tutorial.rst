Usage
=====

The ``ssc-cdi`` package is imported as ``sscCdi``. 

Ptychographic reconstructions are performed by calling ``sscCdi.call_ptychography()`` with the experimental data and a configuration dictionary.

.. code-block:: python

    import sscCdi

    # Load the experimental data
    difpads = load_diffraction_patterns()
    positions = load_positions()

    # Optional initial estimates
    initial_object = load_initial_object()
    initial_probe = load_initial_probe()

    input_dict = {
    "hdf5_output": "path/to/output/file.hdf5", 
    "CPUs": 2,      
    "GPUs": [0], 
    "energy": 12, # keV
    "detector_distance": 1..0, # meters
    'distance_sample_focus': 0, # meters
    "detector_pixel_size": 55e-6, # meters
    "binning": 1,    
    "object_padding": 0,
    "incoherent_modes": 1, 
    "initial_obj": None,   
    "initial_probe": None,
    "probe_support": None,
    "positions_unit": "meters",
    'regime': 'fraunhoffer',
    'position_rotation': 0,
    'n_of_positions_to_remove': 0,  
    'clip_object_magnitude': False,
    'free_log_likelihood': 0,
    'fourier_power_bound': 0,
    'save_restored_data': False,
    "algorithms": {"1": engine = {  'name': 'AP',          
                                    'batch': 0,                   # number of data arrays to fit into the GPU. 0: automatically
                                    'iterations':400,
                                    'regularization_object': 1e-3, # float (min: 0, max: 1); 
                                    'regularization_probe': 1e-3,  # float (min: 0, max: 1); 
                                    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations
                                }}
    }
    input_dict["wavelength"] = sscCdi.wavelength_meters_from_energy_keV(input_dict["energy"])

    recon_obj, recon_probe, recon_positions, output_dict, recon_error = sscCdi.call_ptychography(input_dict,
                                                                                                difpads, 
                                                                                                positions, 
                                                                                                initial_object, 
                                                                                                initial_probe, 
                                                                                                plot=False)


See the :ref:`dictionary` page and :ref:`API reference <apicallengines>` for a complete description of all available parameters.


.. toctree::
    :maxdepth: 1
    :caption: Contents

    tutorial/dictionary
    tutorial/pie
    tutorial/ap
    tutorial/raar
    tutorial/ml
    tutorial/errors

