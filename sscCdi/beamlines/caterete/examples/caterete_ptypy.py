import ptypy
import time
from ptypy import utils as u
from sscCdi import CAT_create_ptyd_file, create_ptypy_parameter_tree

restored_data_path =  "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/restored_data/0000_SS03112022_02_001.npy"
positions_path     = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/data/ptycho2d/SS03112022_02/positions/0000_SS03112022_02_001.txt"
metadata_path      = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/data/ptycho2d/SS03112022_02/mdata.json"
ptyd_output_path   = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/SS03112022_02.ptyd"
io_home_path       = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/"
ptycho_final_path  = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/recons/%(run)s_%(engine)s_%(iterations)04d.ptyr"
run_ID = "withPC2"
frames_per_block = 10


# CAT_create_ptyd_file(restored_data_path, positions_path, metadata_path,ptyd_output_path,pack_size=100)

# parameter_tree = create_ptypy_parameter_tree(io_home_path, ptycho_final_path,ptyd_output_path,frames_per_block,run_ID)

ptypy.load_gpu_engines("cuda") # comment if not using CUDA

# Create parameter tree
parameter_tree = u.Param()

# Set verbose level, can be "interactive", "info" or "debug"
parameter_tree.verbose_level = "interactive"

# Run label (ID)
parameter_tree.run = run_ID

# I/O settings
parameter_tree.io = u.Param()

# Set the root path for all input/output operations
# Change this to wherever you would like 
parameter_tree.io.home = io_home_path

# Path to final .ptyr output file 
# using variables parameter_tree.run, engine name and total nr. of iterations
parameter_tree.io.rfile =  ptycho_final_path

# Turn off interaction server
# parameter_tree.io.interaction = u.Param(active=False)

# Use non-threaded live plotting
# parameter_tree.io.autoplot = u.Param()
# parameter_tree.io.autoplot.active=False
# parameter_tree.io.autoplot.threaded = True
# parameter_tree.io.autoplot.layout = "default"
# parameter_tree.io.autoplot.interval = 10

# Save intermediate .ptyr files (dumps) every 10 iterations
parameter_tree.io.autosave = u.Param()
parameter_tree.io.autosave.active = False
# parameter_tree.io.autosave.interval = 10
# parameter_tree.io.autosave.rfile = 'dumps/%(run)s_%(engine)s_%(iterations)04d.ptyr'

# Define the scan model
parameter_tree.scans = u.Param()
parameter_tree.scans.CateretePtyPyScan = u.Param()
parameter_tree.scans.CateretePtyPyScan.data = u.Param()
parameter_tree.scans.CateretePtyPyScan.name = "BlockFull"
parameter_tree.frames_per_block = frames_per_block

parameter_tree.scans.CateretePtyPyScan.data.name = 'PtydScan'
parameter_tree.scans.CateretePtyPyScan.data.source = 'file'
parameter_tree.scans.CateretePtyPyScan.data.dfile = ptyd_output_path

# Define reconstruction engine
parameter_tree.engines = u.Param()

# parameter_tree.scans.CateretePtyPyScan.illumination = u.Param()
# parameter_tree.scans.CateretePtyPyScan.illumination.model = None
# parameter_tree.scans.CateretePtyPyScan.illumination.photons = None
# parameter_tree.scans.CateretePtyPyScan.illumination.aperture = u.Param()
# parameter_tree.scans.CateretePtyPyScan.illumination.aperture.form = "circ"
# parameter_tree.scans.CateretePtyPyScan.illumination.aperture.size = 333e-6
# parameter_tree.scans.CateretePtyPyScan.illumination.propagation = u.Param()
# parameter_tree.scans.CateretePtyPyScan.illumination.propagation.focussed = 13.725e-3
# parameter_tree.scans.CateretePtyPyScan.illumination.propagation.parallel = 45e-6
# parameter_tree.scans.CateretePtyPyScan.illumination.propagation.antialiasing = 1
# parameter_tree.scans.CateretePtyPyScan.illumination.diversity = u.Param()
# parameter_tree.scans.CateretePtyPyScan.illumination.diversity.power = 0.1
# parameter_tree.scans.CateretePtyPyScan.illumination.diversity.noise = [0.5,1.0]
# parameter_tree.scans.CateretePtyPyScan.coherence = u.Param()
# parameter_tree.scans.CateretePtyPyScan.coherence.num_probe_modes = 5
# parameter_tree.scans.CateretePtyPyScan.coherence.num_object_modes = 1

# parameter_tree.engines.engine00 = u.Param()
# parameter_tree.engines.engine00.name = "DM_pycuda"
# parameter_tree.engines.engine00.numiter = 100
# parameter_tree.engines.engine00.numiter_contiguous = 1
# parameter_tree.engines.engine00.probe_support = 0.5
# parameter_tree.engines.engine00.alpha = 0.9

# parameter_tree.engines.engine01 = u.Param()
# parameter_tree.engines.engine01.name = "DM_pycuda"
# parameter_tree.engines.engine01.numiter = 50
# parameter_tree.engines.engine01.numiter_contiguous = 1
# parameter_tree.engines.engine01.probe_support = 0.5
# parameter_tree.engines.engine01.alpha = 0.1

parameter_tree.engines.engine01 = u.Param()
parameter_tree.engines.engine01.name = "RAAR_pycuda"
parameter_tree.engines.engine01.numiter = 100
parameter_tree.engines.engine01.beta = 0.9
parameter_tree.engines.engine01.probe_support = 0.5

parameter_tree.engines.engine01.position_refinement = u.Param()
parameter_tree.engines.engine01.position_refinement.method = "Annealing"
parameter_tree.engines.engine01.position_refinement.start = 25
parameter_tree.engines.engine01.position_refinement.stop = 100
parameter_tree.engines.engine01.position_refinement.interval = 10
parameter_tree.engines.engine01.position_refinement.nshifts = 8
parameter_tree.engines.engine01.position_refinement.amplitude = 50.0e-9
parameter_tree.engines.engine01.position_refinement.max_shift = 100.0e-9
parameter_tree.engines.engine01.position_refinement.record = True

parameter_tree.engines.engine02 = u.Param()
parameter_tree.engines.engine02.name = "RAAR_pycuda"
parameter_tree.engines.engine02.numiter = 100
parameter_tree.engines.engine02.beta = 0.5
parameter_tree.engines.engine02.probe_support = 0.5

parameter_tree.engines.engine02.position_refinement = u.Param()
parameter_tree.engines.engine02.position_refinement.method = "Annealing"
parameter_tree.engines.engine02.position_refinement.start = 100
parameter_tree.engines.engine02.position_refinement.stop = 200
parameter_tree.engines.engine02.position_refinement.interval = 10
parameter_tree.engines.engine02.position_refinement.nshifts = 8
parameter_tree.engines.engine02.position_refinement.amplitude = 50.0e-9
parameter_tree.engines.engine02.position_refinement.max_shift = 100.0e-9
parameter_tree.engines.engine02.position_refinement.record = True

# parameter_tree.engines.engine03 = u.Param()
# parameter_tree.engines.engine03.name = 'ML_pycuda'
# parameter_tree.engines.engine03.numiter = 50
# parameter_tree.engines.engine03.numiter_contiguous = 10
# parameter_tree.engines.engine03.reg_del2 = True 
# parameter_tree.engines.engine03.reg_del2_amplitude = 1.
# parameter_tree.engines.engine03.scale_precond = True
# parameter_tree.engines.engine03.scale_probe_object = 1.

t1 = time.time()
P = ptypy.core.Ptycho(parameter_tree,level=5)
elapsed = time.time()-t1
print(f'Time elapsed: {elapsed/60} min')