Examples
========

****************
JSON input file
****************

We explain below all the inputs present in the Json file. 

    - LogfilePath: a path to save a json logfile when the sscptycho script is run. The json contains all the inputs from the logfile, plus a few extra ones calculated during runtime, such as object pixel size and object FRC resolution.
    - data_folder: path to the folder where data is located
    - mask: path to a predefined mask to ignore bad pixels
    - flatfield: path to DET flatfield in numpy format
    - empty_frame: path to empty frame hdf5 file
    - 3D_Acquistion_Folders: name of the folder containg all hdf5 frames for a 3D reconstruction
    - Frames: list of integers containing all frames to be used for the 3D reconstruction. If empty, uses all frames. 
    - SingleMeasurement: if not a 3D reconstruction, select the measurement hdf5 to reconstruct. If this is to be used, 3D_Acquistion_Folders should be empty.
    - autocrop: true/false. If true, will automatically crop the "noisy" borders from the reconstructed object. This is necessary if you want to properly perform phase unwrapping and resolution estimation via FRC.
    - phase_unwrap: [true/false,iterations, [upper_crop,lower_crop],[left_crop,right_crop]]. If true, will phase unwrap the object amplitude and phase. It is vital that the noisy border is completely removed for the unwrapping to work properly. Therefore, you can select manually the number of pixels to further crop in each direction with the [upper_crop,lower_crop],[left_crop,right_crop] parameters. Iterations may help to optimize the unwrapping for some samples, but start and use iterations=0 as a standard and increase it to 1, 2, 3 gradually if needed.
    - FRC: true/false. If true, will calculate Fourier Ring Correlation. IMPORTANT: for a proper estimate of the resolution, one must perform the manual cropping in the PhaseUnwrap input to select only a local region of interest, for instance, the center of a Siemens Star.
    - detector_exposure: [true/false, detector_exposure_time]. If true, will consider invalid all pixels with more than 300000 counts/sec for a detector_exposure_time of 0.15s.
    - Automaticcentral_mask: [T/F,radius,center_x,center_y,reference_frame]. If true, will insert a circular mask with a certain radius. In that case, center_x and center_y variables have no use. Referece_frame selects the diffraction pattern to be used as a reference for the automatic center selection. If the algorithm does not get the center correctly, try use a different integer value for reference_frame. If false, the central mask is inserted manual at position (center_x,center_r) with size = radius. If no central mask is needed, simply use 'Automaticcentral_mask' = []
    - Probe Support: (radius, center_x, center_y). Region for support that is used in the probe reconstruction. (center_x,center_y)=(0,0) means the center of the circular probe is at the center of the image
    - GPUs: [0,1,2,3]. Numeric label of GPUs to be used. For instance, if you want 2 GPUs to be used, choose [0,1]. If you want 3 GPUs, choose [0,1,2]
    - CPUs: [int] number of threads to be used
    - binning: [int]. binning of the diffraction pattern
    - Seed: [int] Seed for random number generator. Use 10 as standard.
    - incoherent_modes: [int]. Number of probe modes to be reconstructed 
    - energy: [float]. Beam energy for the experiment
    - detector_distance: [float] Detector distance for the experiment
    - restored_pixel_size: [float] Detector pixel size. Use 55.55e-6
    - f1: [float] Fresnel number. Adjusts the detector-sample distance. This is the parameter to be tweaked with for fine probe/pinhole adjustment. IMPORTANT: always use a negative value for f1! ​Reference values: 1e-4 for 10 micron pinhole. 1e-3 for 5 micron pinhole at Cateretê.
    - OldRestauration: if True, uses restauration/binning procedure by Giovanni. If false, uses new procedure by Miqueles
    - ChipBorderRemoval: only used when OldRestauration=False. Removes border pixels of the detector chips, where photon count is problematic due to bigger pixel size at the borders.
    - detector_ROI_radius: use 1280 for binning = 4
    - PreviewFolder: path to the folder where preview image of the object and probe will be saved, together with preview of diffraction pattern, empty and flatfield, among others.
    - SaveObj: if true, will save the reconstruction object with name "SaveObjName" at folder "ObjPath"
    - initial_obj: path to file with an initial guess for the object
    - SaveProbe and initial_probe: analog to SaveObj and initial_obj, but for the Probe
    - SaveBkg and InitialBkg: analog to SaveObj and initial_obj, but for the background.
    - AlgorithmX: X -> integer value. Dictionary with all algorithms to be used for reconstruction (in order). Examples for RAAR an GL (Alternate projections Griffin-Lim):
         - "Algorithm1": {"Name": "RAAR", "Iterations": 100, "TV": 0, "Epsilon": 1E-2, "Beta": 0.995, "ProbeCycles": 4, "Batch": 64 }
         - "Algorithm2": {"Name": "GL", "Iterations": 150,  "TV": 1E-4, "Epsilon": 1E-2, "ObjBeta": 0.97, "ProbeBeta": 0.95, "Batch": 64 }

