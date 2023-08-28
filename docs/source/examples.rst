Examples
========

Tomography at CARNAUBA
******************************        

.. code-block:: python3

    To be added

Ptychography at CATERETE
******************************

.. code-block:: python3

    input_dict = {
        
    'beamline': 'CAT',                                              # CAT or CNB 
    'detector': '540D',                                             # 135D or 540D

    """ DATA SELECTION """
    "data_folder": "/ibira/lnls/beamlines/caterete/apps/gcc-jupyter/00000000/data/ptycho2d/",
    "acquisition_folders": ["SS03112022_02"],                       # list of foldernames in data_folder
    "projections": [0],                                             # global index to select which projections from the above folders to reconstruct
    "flatfield": "",                                                # path to .npy flatfield file. If empty string "", uses standard flatfield for the dataset.

    """ RESOURCE SELECTION """
    'CPUs': 96,                                                     # number of CPUs for processing
    'GPUs': [0,1,2],                                                # number of GPUs for processing

    """ RESTORATION """"
    'binning': 1,                                                   # binning factor for restored data. Must be even number.
    'using_direct_beam': False,                                     # [NOT WORKING CORRECTLY]. if True, converts DP_center coordinates from raw to restored coordinates
    'DP_center': [1392,1405],                                       # [center_y, center_x]. Diffraction pattern center selected for restoration in pixels
    'detector_ROI_radius': 1350,                                    # pixels. Half-size of restored diffraction data
    'fill_blanks': False,                                           # interpolate blank lines after restoration
    'keep_original_negative_values': False,                         # whether to turn all negative values to -1
    'suspect_border_pixels': 3,                                     # number of fat pixels at the chip border to ignore after restoration 

    """ PTYCHOGRAPHY """
    'position_rotation': -0.003,                                    # radians. Angle between detector and probe coordinate system
    'object_padding': 50,                                           # pixels. Size of null border of object array; needs to be expanded to accomodate probe positions thorughout the scan
    'incoherent_modes': 0,                                          # number of modes to decompose probe.
    'probe_support': [ "circular", 300,0,0 ],                       # ["circular",radius_pxls,center_y, center_x]; (0,0) is the center of the image
    'fresnel_number': -0.001,                                       # fresnel number of the system to adjust propagation in the algorithm
    "initial_obj": ["random"],                                      # options are: path to .npy, path to .hdf5, ["random"], ["constant"]
    "initial_probe": ["inverse"],                                   # options are: path to .npy, path to .hdf5, ["inverse"], ["random"], ["constant"], ["circular",radius]

    'Algorithm1': {
                    'Name': 'RAAR',                                 # Relaxed Averaged Alternating Reflections
                    'Batch': 64,
                    'Iterations': 70,
                    'Beta': 0.995,
                    'Epsilon': 0.01,
                    'ProbeCycles': 4,
                    'TV': 0},

    'Algorithm2': {
                    'Name': 'GL',                                   # Alternating Projections (Griffin-Lim) algorithm
                    'Batch': 64,
                    'Iterations': 50,
                    'ObjBeta': 0.97,
                    'ProbeBeta': 0.95,
                    'Epsilon': 0.01,
                    'TV': 0.0001},
\
        }


Tomography at CATERETE
******************************        

.. code-block:: python3

    To be added