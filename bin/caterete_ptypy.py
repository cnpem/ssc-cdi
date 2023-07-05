import ptypy
import time
from sscCdi import CAT_create_ptyd_file, create_ptypy_parameter_tree

restored_data_path =  "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/restored_data/0000_SS03112022_02_001.npy"
positions_path     = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/data/ptycho2d/SS03112022_02/positions/0000_SS03112022_02_001.txt"
metadata_path      = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/data/ptycho2d/SS03112022_02/mdata.json"
ptyd_output_path   = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/SS03112022_02.ptyd"
io_home_path       = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/"
ptycho_final_path  = "/ibira/lnls/labs/tepui/home/yuri.tonin/00000000/proc/recons/SS03112022_02/ptypy/recons/%(run)s_%(engine)s_%(iterations)04d.ptyr"
run_ID = "mySample"
frames_per_block = 30


CAT_create_ptyd_file(restored_data_path, positions_path, metadata_path,ptyd_output_path,pack_size=100)

parameter_tree = create_ptypy_parameter_tree(io_home_path, ptycho_final_path,ptyd_output_path,frames_per_block,run_ID)

ptypy.load_gpu_engines("cuda") # comment if not using CUDA

# parameter_tree.engines.engine00 = u.Param()
# parameter_tree.engines.engine00.name = "DM_pycuda"
# parameter_tree.engines.engine00.numiter = 80
# parameter_tree.engines.engine00.numiter_contiguous = 1
# parameter_tree.engines.engine00.probe_support = 0.05

parameter_tree.engines.engine01 = u.Param()
parameter_tree.engines.engine01.name = "RAAR_pycuda"
parameter_tree.engines.engine01.numiter = 50
parameter_tree.engines.engine01.beta = 0.9
parameter_tree.engines.engine01.probe_support = 0.1

parameter_tree.engines.engine02 = u.Param()
parameter_tree.engines.engine02.name = "RAAR_pycuda"
parameter_tree.engines.engine02.numiter = 50
parameter_tree.engines.engine02.beta = 0.5
parameter_tree.engines.engine02.probe_support = 0.1

# parameter_tree.engines.engine03 = u.Param()
# parameter_tree.engines.engine03.name = 'ML_pycuda'
# parameter_tree.engines.engine03.numiter = 20

t1 = time.time()
P = ptypy.core.Ptycho(parameter_tree,level=5)
elapsed = time.time()-t1
print(f'Time elapsed: {elapsed/60} min')