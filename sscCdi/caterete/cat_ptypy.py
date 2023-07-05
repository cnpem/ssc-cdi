import numpy as np
import json, os
import time

import ptypy
from ptypy.core.data import PtyScan
from ptypy import utils as u
from ptypy import defaults_tree

@defaults_tree.parse_doc('scandata.numpyscan') # decorator to get defaults from docstring
class caterete_data_for_ptypy(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = numpyscan
    help =

    [auto_center]
    default = False
    """
    
    def __init__(self, restored_data_path, positions_path, metadata_path,output_path, pars=None, **kwargs):
        
        p = self.DEFAULT.copy(depth=2) # build a default parameter structure to ensure that all input parameters are available
        p.update(pars) # updates this structure to overwrite the entries specified by the user

        self.restored_data_path = restored_data_path
        self.positions_path = positions_path
        self.metadata_path = metadata_path
        mdata_dict = json.load(open(self.metadata_path))

        p["energy"]   = mdata_dict['/entry/beamline/experiment']["energy"]
        p["distance"] = mdata_dict['/entry/beamline/experiment']["distance"]*1e-3 # convert to meters
        p["psize"]    = mdata_dict['/entry/beamline/detector']['pimega']["pixel size"]*1e-6 # convert to microns
        
        dataset = np.load(self.restored_data_path)
        
        p["shape"] = dataset.shape[1]
        self.number_of_DPs = dataset.shape[0]
        
        super(caterete_data_for_ptypy, self).__init__(p, **kwargs)

        self.info.dfile = output_path 
        
        
    def load_positions(self):
        
        position_path = self.positions_path
        
        pos = []
        with open(position_path) as f: 
            for line in f:
                if 'Ry:' in line: continue
                y, x = line.strip().split()
                factor = 1e-6 # to convert to meters
                pos.append((eval(y)*factor, eval(x)*factor))
        positions = np.asarray(pos)

        return positions

    def load(self, indices):
        
        raw = {}
        filepath = self.restored_data_path
        
        data = np.load(filepath)
        for idx in indices:
            raw[idx] = data[idx] 
        return raw, {}, {}


def CAT_create_ptyd_file(restored_data_path, positions_path, metadata_path,output_path,pack_size=20):    

    data_object = caterete_data_for_ptypy(restored_data_path, positions_path, metadata_path,output_path)

    print('Number of DPs: ',data_object.number_of_DPs)
    
    data = data_object.DEFAULT.copy(depth=2)
    data.save = 'append' # set parameter to save data
    data_object = caterete_data_for_ptypy(restored_data_path, positions_path, metadata_path,output_path,pars=data)
    data_object.initialize()

    for i in range(1000): 
        print('Creating ptyd block #',i)
        msg = data_object.auto(pack_size) # process data for N frames; will create data in chunks of N frames
        if msg == data_object.EOS:
            print('All frames have been processed')
            break
            
    print("ptyd datafile created")

def create_ptypy_parameter_tree(io_home_path, ptycho_final_path,path_ptyd_datafile,frames_per_block, run_ID="mySample"):

    # Create parameter tree
    p = u.Param()

    # Set verbose level, can be "interactive", "info" or "debug"
    p.verbose_level = "interactive"

    # Run label (ID)
    p.run = run_ID

    # I/O settings
    p.io = u.Param()

    # Set the root path for all input/output operations
    # Change this to wherever you would like 
    p.io.home = io_home_path

    # Path to final .ptyr output file 
    # using variables p.run, engine name and total nr. of iterations
    p.io.rfile =  ptycho_final_path

    # Turn off interaction server
    # p.io.interaction = u.Param(active=False)

    # Use non-threaded live plotting
    # p.io.autoplot = u.Param()
    # p.io.autoplot.active=False
    # p.io.autoplot.threaded = True
    # p.io.autoplot.layout = "default"
    # p.io.autoplot.interval = 10

    # Save intermediate .ptyr files (dumps) every 10 iterations
    # p.io.autosave = u.Param()
    # p.io.autosave.active = True
    # p.io.autosave.interval = 10
    # p.io.autosave.rfile = 'dumps/%(run)s_%(engine)s_%(iterations)04d.ptyr'

    # Define the scan model
    p.scans = u.Param()
    p.scans.CateretePtyPyScan = u.Param()
    p.scans.CateretePtyPyScan.data = u.Param()
    p.scans.CateretePtyPyScan.name = "BlockFull"
    p.frames_per_block = frames_per_block

    p.scans.CateretePtyPyScan.data.name = 'PtydScan'
    p.scans.CateretePtyPyScan.data.source = 'file'
    p.scans.CateretePtyPyScan.data.dfile = path_ptyd_datafile

    # Define reconstruction engine
    p.engines = u.Param()

    return p